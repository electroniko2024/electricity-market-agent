import { useRef, useState } from "react";
import { createConversation, streamTurn } from "./api";

type Item = { text: string; refs: number[] };
type Citation = {
  n: number;
  kind: "data" | "policy";
  tool?: string;
  arguments?: Record<string, unknown>;
  field?: string;
  value?: unknown;
  row?: Record<string, unknown>;
  values?: { field: string; value: unknown; row?: Record<string, unknown> }[];
  document?: string;
  page?: number | null;
  effective_from?: string | null;
  effective_to?: string | null;
  excerpt?: string;
};
type Presentation = {
  tier: "quick" | "analysis";
  headline: Item[];
  sections: { title: string; items: Item[] }[];
  notes: string[];
  verified: boolean;
  citations: Citation[];
};
type SessionReport = {
  overview: string;
  topics: { title: string; entries: { turn: number; question: string; items: Item[]; verified: boolean }[] }[];
  citations: Citation[];
  markdown: string;
};
type Message = {
  role: "user" | "assistant";
  text?: string;
  interpretation?: string;
  clarification?: string;
  presentation?: Presentation;
  report?: SessionReport;
  progress?: string;
  error?: string;
  runId?: string;
  reused?: number;
};

const TOOL_NAMES: Record<string, string> = {
  query_price: "价格明细", query_stats: "价格统计", query_load: "负荷与出力", query_weather: "天气",
  detect_anomaly: "异常检测", explain_drivers: "成因解释", search_policy: "政策检索", forecast_price: "价格预测",
};

function CitationDetail({ citation }: { citation: Citation }) {
  if (citation.kind === "policy") {
    return (
      <div className="cite-detail">
        <strong>{citation.document}</strong>
        {citation.page ? ` · 第 ${citation.page} 页` : ""}
        {citation.effective_from ? ` · ${citation.effective_from} 起${citation.effective_to ? `至 ${citation.effective_to}` : ""}` : ""}
        <blockquote>{citation.excerpt}</blockquote>
      </div>
    );
  }
  const args = Object.entries(citation.arguments ?? {})
    .filter(([k]) => k !== "as_of")
    .map(([k, v]) => `${k}=${String(v)}`).join("，");
  const values = citation.values ?? ("value" in citation ? [{ field: citation.field ?? "", value: citation.value, row: citation.row }] : []);
  return (
    <div className="cite-detail">
      <strong>{TOOL_NAMES[citation.tool ?? ""] ?? citation.tool}</strong>（{args}）
      {values.map((v, index) => (
        <div key={index}>
          {Object.values(v.row ?? {}).length > 0 && <span>{Object.values(v.row ?? {}).join(" ")} · </span>}
          <code>{v.field}</code> = <code>{formatValue(v.value)}</code>
        </div>
      ))}
    </div>
  );
}

function formatValue(value: unknown): string {
  return typeof value === "number" && !Number.isInteger(value) ? String(Number(value.toFixed(4))) : String(value);
}

function ItemText({ item, citations }: { item: Item; citations: Citation[] }) {
  const [open, setOpen] = useState<number | null>(null);
  const shown = citations.find((c) => c.n === open);
  return (
    <>
      <span>{item.text}</span>
      {item.refs.map((n) => (
        <button key={n} type="button" className={`cite-mark${open === n ? " active" : ""}`}
          onClick={() => setOpen(open === n ? null : n)} aria-label={`出处 ${n}`}>
          {n}
        </button>
      ))}
      {shown && <CitationDetail citation={shown} />}
    </>
  );
}

function PresentationView({ p }: { p: Presentation }) {
  return (
    <div className="chat-answer">
      {p.headline.map((item, index) => (
        <p key={index} className="chat-headline"><ItemText item={item} citations={p.citations} /></p>
      ))}
      {p.sections.map((section) => (
        <details key={section.title} className="chat-section">
          <summary>{section.title}（{section.items.length}）</summary>
          <ul>{section.items.map((item, index) => <li key={index}><ItemText item={item} citations={p.citations} /></li>)}</ul>
        </details>
      ))}
      {p.notes.length > 0 && (
        <details className="chat-section chat-notes">
          <summary>数据说明与局限</summary>
          <ul>{p.notes.map((note, index) => <li key={index}>{note}</li>)}</ul>
        </details>
      )}
      {!p.verified && <p className="chat-unverified">本回答未完全通过核验，请结合出处判断。</p>}
    </div>
  );
}

function ReportView({ report, conversationId }: { report: SessionReport; conversationId: string }) {
  return (
    <div className="chat-answer chat-report">
      <h3>会话分析报告</h3>
      <p>{report.overview}</p>
      {report.topics.map((topic) => (
        <section key={topic.title}>
          <h4>{topic.title}</h4>
          {topic.entries.map((entry) => (
            <div key={entry.turn} className="report-entry">
              <p className="report-entry-q">第 {entry.turn} 问：{entry.question}</p>
              <ul>{entry.items.map((item, index) => <li key={index}><ItemText item={item} citations={report.citations} /></li>)}</ul>
            </div>
          ))}
        </section>
      ))}
      <details className="chat-section">
        <summary>出处（{report.citations.length}，每个来源只列一次）</summary>
        <ol>{report.citations.map((c) => <li key={c.n} value={c.n}><CitationDetail citation={c} /></li>)}</ol>
      </details>
      <a className="report-download" href={`/conversations/${conversationId}/report.md`} target="_blank" rel="noreferrer">下载 Markdown</a>
    </div>
  );
}

export function Conversation() {
  const [conversationId, setConversationId] = useState<string | null>(null);
  const [messages, setMessages] = useState<Message[]>([]);
  const [input, setInput] = useState("");
  const [asOf, setAsOf] = useState("2026-01-15T00:00");
  const [busy, setBusy] = useState(false);
  const abortRef = useRef<AbortController | null>(null);

  const patchLast = (patch: Partial<Message>) =>
    setMessages((current) => [...current.slice(0, -1), { ...current[current.length - 1], ...patch }]);

  const send = async (text: string) => {
    const message = text.trim();
    if (!message || busy) return;
    setBusy(true);
    setInput("");
    setMessages((current) => [...current, { role: "user", text: message }, { role: "assistant", progress: "理解问题…" }]);
    const controller = new AbortController();
    abortRef.current = controller;
    try {
      const id = conversationId ?? (await createConversation());
      setConversationId(id);
      let steps = 0;
      await streamTurn(id, { message, as_of: new Date(asOf).toISOString(), mode: "historical_analysis" }, {
        onEvent: (event) => {
          if (event.type === "rewrite") {
            if (event.action === "clarify") patchLast({ clarification: String(event.clarification), progress: undefined });
            else patchLast({ interpretation: String(event.standalone_question), progress: "查询数据…" });
          } else if (event.type === "agent_event") {
            const inner = event.event as Record<string, unknown>;
            if (inner.type === "tool_call") steps += 1;
            if (inner.type === "tool_call") patchLast({ progress: `查询数据…（第 ${steps} 次调用）` });
            if (inner.type === "reflection") patchLast({ progress: "撰写并核验回答…" });
          } else if (event.type === "answer") {
            patchLast({ presentation: event.presentation as Presentation, runId: String(event.run_id), progress: undefined });
          } else if (event.type === "session_report") {
            patchLast({ report: event.report as SessionReport, progress: undefined });
          } else if (event.type === "turn") {
            const turn = event.turn as { cache?: { hits?: number } };
            patchLast({ reused: turn.cache?.hits ?? 0, progress: undefined });
          }
        },
        onError: (error) => patchLast({ error, progress: undefined }),
      }, controller.signal);
    } catch (error) {
      patchLast({ error: (error as Error).message, progress: undefined });
    } finally {
      setBusy(false);
    }
  };

  const reset = () => {
    abortRef.current?.abort();
    setConversationId(null);
    setMessages([]);
    setBusy(false);
  };

  return (
    <section className="chat" aria-label="多轮对话">
      <div className="analysis-heading">
        <h2>多轮对话</h2>
        <span className="analysis-status">可以追问“那前一天呢”“这两天哪天高”，需要时说“出个报告”</span>
      </div>
      <div className="chat-log" aria-live="polite">
        {messages.length === 0 && (
          <p className="chat-empty">例如：2025年8月4日山东实时电价的日均价是多少？然后追问“那7月31日呢？”“这两天哪天贵，差多少？”</p>
        )}
        {messages.map((m, index) => m.role === "user" ? (
          <div key={index} className="chat-msg chat-user">{m.text}</div>
        ) : (
          <div key={index} className="chat-msg chat-assistant">
            {m.interpretation && messages[index - 1]?.text !== m.interpretation && (
              <p className="chat-echo">我理解为：{m.interpretation}</p>
            )}
            {m.clarification && <p className="chat-clarify">{m.clarification}</p>}
            {m.presentation && <PresentationView p={m.presentation} />}
            {m.report && conversationId && <ReportView report={m.report} conversationId={conversationId} />}
            {m.progress && <p className="chat-progress">{m.progress}</p>}
            {m.error && <p className="chat-error">{m.error}</p>}
            {(m.runId || (m.reused ?? 0) > 0) && (
              <p className="chat-meta">
                {(m.reused ?? 0) > 0 && <span>复用了前面查过的 {m.reused} 份数据 · </span>}
                {m.runId && <a href={`/trace/${m.runId}`} target="_blank" rel="noreferrer">执行轨迹</a>}
              </p>
            )}
          </div>
        ))}
      </div>
      <div className="analysis-controls chat-controls">
        <label className="analysis-question">
          <span>你的问题</span>
          <textarea value={input} rows={2} maxLength={4000} disabled={busy}
            onChange={(event) => setInput(event.target.value)}
            onKeyDown={(event) => { if (event.key === "Enter" && !event.shiftKey) { event.preventDefault(); void send(input); } }} />
        </label>
        <label>
          <span>历史截止时间</span>
          <input type="datetime-local" value={asOf} onChange={(event) => setAsOf(event.target.value)} disabled={busy || messages.length > 0} />
        </label>
        <div className="chat-buttons">
          <button type="button" onClick={() => void send(input)} disabled={busy || !input.trim()}>发送</button>
          <button type="button" onClick={() => void send("出个报告")} disabled={busy || !conversationId}>出个报告</button>
          <button type="button" className="secondary" onClick={reset} disabled={messages.length === 0}>新会话</button>
        </div>
      </div>
    </section>
  );
}
