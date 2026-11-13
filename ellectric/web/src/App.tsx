import { useRef, useState } from "react";
import { streamAnalysis } from "./api";
import { AnalysisReport } from "./AnalysisReport";
import "./styles.css";

export default function App() {
  const [analysisQuestion, setAnalysisQuestion] = useState("");
  const [analysisAsOf, setAnalysisAsOf] = useState("2026-01-15T00:00");
  const [analysisMode, setAnalysisMode] = useState<"historical_analysis" | "historical_forecast">("historical_analysis");
  const [analysisEvents, setAnalysisEvents] = useState<Record<string, unknown>[]>([]);
  const [analysisBusy, setAnalysisBusy] = useState(false);
  const analysisAbortRef = useRef<AbortController | null>(null);

  const startAnalysis = () => {
    if (!analysisQuestion.trim() || analysisBusy) return;
    const controller = new AbortController();
    analysisAbortRef.current = controller;
    setAnalysisEvents([]);
    setAnalysisBusy(true);
    void streamAnalysis(
      {
        question: analysisQuestion.trim(),
        as_of: new Date(analysisAsOf).toISOString(),
        mode: analysisMode,
      },
      {
        onEvent: (event) => setAnalysisEvents((current) => [...current, event]),
        onError: (message) => setAnalysisEvents((current) => [...current, { type: "client_error", message }]),
      },
      controller.signal,
    ).finally(() => setAnalysisBusy(false));
  };

  const stopAnalysis = () => {
    analysisAbortRef.current?.abort();
    analysisAbortRef.current = null;
    setAnalysisBusy(false);
  };

  return (
    <div className="app">
      <header className="app-header">
        <div>
          <h1>山东电力市场分析 Agent</h1>
          <p className="header-sub-info">规划 → 执行 → 反思 → 报告 → 校验 · 只使用历史数据，不执行交易</p>
        </div>
      </header>

      <main className="dashboard-main">
        <section className="market-analysis" aria-labelledby="market-analysis-title">
          <div className="analysis-heading">
            <h2 id="market-analysis-title">山东市场分析</h2>
            <span className="analysis-status">历史数据 · 不执行交易</span>
          </div>
          <div className="analysis-controls">
            <label className="analysis-question">
              <span>分析问题</span>
              <textarea
                value={analysisQuestion}
                onChange={(event) => setAnalysisQuestion(event.target.value)}
                placeholder="例如：分析 2025-10-01 的实时价格异常"
                rows={2}
                maxLength={4000}
                disabled={analysisBusy}
              />
            </label>
            <label>
              <span>历史截止时间</span>
              <input type="datetime-local" value={analysisAsOf} onChange={(event) => setAnalysisAsOf(event.target.value)} disabled={analysisBusy} />
            </label>
            <label>
              <span>模式</span>
              <select value={analysisMode} onChange={(event) => setAnalysisMode(event.target.value as typeof analysisMode)} disabled={analysisBusy}>
                <option value="historical_analysis">历史分析</option>
                <option value="historical_forecast">历史时点 D+1 预测</option>
              </select>
            </label>
            {analysisBusy ? (
              <button type="button" onClick={stopAnalysis}>停止</button>
            ) : (
              <button type="button" onClick={startAnalysis} disabled={!analysisQuestion.trim()}>运行分析</button>
            )}
          </div>
          <AnalysisReport events={analysisEvents} />
          {analysisEvents.length > 0 && (
            <details className="analysis-trace" open={!analysisEvents.some((event) => event.type === "report")}>
              <summary>执行轨迹（计划、工具、反思、校验）· {analysisEvents.length} 个事件</summary>
              <div className="analysis-results" aria-live="polite">
                {analysisEvents.map((event, index) => {
                  const kind = String(event.type ?? "event");
                  const details = event.report ?? event.reflection ?? event.validation ?? event.result ?? event.steps ?? event.message ?? event;
                  return (
                    <details key={`${kind}-${index}`} className={`analysis-event analysis-event-${kind}`} open={kind === "report" || kind === "client_error"}>
                      <summary>{kind.replace(/_/g, " ")}{typeof event.tool_name === "string" ? ` · ${event.tool_name}` : ""}</summary>
                      <pre>{typeof details === "string" ? details : JSON.stringify(details, null, 2)}</pre>
                    </details>
                  );
                })}
              </div>
            </details>
          )}
        </section>
      </main>
    </div>
  );
}
