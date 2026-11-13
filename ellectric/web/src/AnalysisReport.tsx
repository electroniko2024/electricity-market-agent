type Binding = { evidence_id: string; json_pointer: string; value: unknown };
type Derived = { operation: "sum" | "difference" | "ratio"; operands: Binding[]; value: number };
type Finding = {
  category?: string;
  statement: string;
  evidence_ids?: string[];
  value_bindings?: Binding[];
  derived_values?: Derived[];
  limitations?: string[];
  basis?: string;
};
type Answer = Finding & { item_id: string; status: "answered" | "unanswerable" };
type Report = {
  answers?: Answer[];
  summary?: string;
  anomalies?: Finding[];
  causes?: Finding[];
  outlook?: Finding | null;
  policy_basis?: { evidence_id: string; json_pointer?: string; statement: string }[];
  data_notes?: string[];
  known_limitations?: string[];
  unresolved_issues?: string[];
  verified?: boolean;
  validation?: { status?: string; errors?: string[] };
};
type ForecastRow = {
  slot_id: number;
  time_label: string;
  LightGBM: number;
  XGBoost: number;
  PatchTST: number;
  model_std?: number;
  high_divergence?: boolean | null;
};
type Forecast = {
  target_date?: string;
  snapshot_mode?: string;
  predictions?: ForecastRow[];
  divergence?: { status?: string; high_threshold_yuan_per_mwh?: number; high_divergence_slot_count?: number; mean_model_std?: number };
};

const MODEL_COLORS: Record<string, string> = { LightGBM: "#7fb7ff", XGBoost: "#f2b45c", PatchTST: "#9fd18b" };
const OPERATION_NAMES: Record<string, string> = { sum: "合计", difference: "差值", ratio: "比值" };

function FindingList({ title, items }: { title: string; items: Finding[] }) {
  if (!items.length) return null;
  return (
    <section className="report-block">
      <h4>{title}</h4>
      <ul className="report-findings">
        {items.map((finding, index) => (
          <li key={index}>
            <p>{finding.statement}</p>
            {finding.value_bindings && finding.value_bindings.length > 0 && (
              <div className="report-bindings">
                {finding.value_bindings.map((binding, bindingIndex) => (
                  <code key={bindingIndex} title={`${binding.evidence_id} ${binding.json_pointer}`}>
                    {String(binding.value)}
                  </code>
                ))}
              </div>
            )}
            {finding.derived_values && finding.derived_values.length > 0 && (
              <div className="report-bindings report-derived">
                {finding.derived_values.map((derived, derivedIndex) => (
                  <code
                    key={derivedIndex}
                    title={derived.operands.map((operand) => `${operand.evidence_id} ${operand.json_pointer}`).join("\n")}
                  >
                    {`${derived.value} = ${OPERATION_NAMES[derived.operation] ?? derived.operation}（${derived.operands
                      .map((operand) => String(operand.value))
                      .join("、")}）`}
                  </code>
                ))}
              </div>
            )}
            {finding.basis && <p className="report-finding-limit">口径：{finding.basis}</p>}
            {finding.limitations?.map((limitation, limitationIndex) => (
              <p key={limitationIndex} className="report-finding-limit">限制：{limitation}</p>
            ))}
          </li>
        ))}
      </ul>
    </section>
  );
}

function ForecastChart({ forecast }: { forecast: Forecast }) {
  const rows = forecast.predictions ?? [];
  if (rows.length === 0) return null;
  const width = 640;
  const height = 200;
  const pad = { left: 44, right: 10, top: 10, bottom: 22 };
  const values = rows.flatMap((row) => [row.LightGBM, row.XGBoost, row.PatchTST]);
  const low = Math.min(0, ...values);
  const high = Math.max(...values);
  const x = (slot: number) => pad.left + (slot / 95) * (width - pad.left - pad.right);
  const y = (value: number) => pad.top + (1 - (value - low) / (high - low || 1)) * (height - pad.top - pad.bottom);
  const path = (model: keyof ForecastRow) =>
    rows.map((row, index) => `${index ? "L" : "M"}${x(row.slot_id).toFixed(1)},${y(Number(row[model])).toFixed(1)}`).join("");
  const divergence = forecast.divergence;
  return (
    <section className="report-block">
      <h4>
        D+1 三模型预测 {forecast.target_date} · {forecast.snapshot_mode === "rolling" ? "滚动快照" : "固定快照"}
      </h4>
      <svg viewBox={`0 0 ${width} ${height}`} className="report-chart" role="img" aria-label="三模型 96 点预测曲线">
        {rows.filter((row) => row.high_divergence).map((row) => (
          <rect key={row.slot_id} x={x(row.slot_id) - 2} y={pad.top} width={4} height={height - pad.top - pad.bottom} fill="#bd9a59" opacity={0.25} />
        ))}
        {low < 0 && <line x1={pad.left} x2={width - pad.right} y1={y(0)} y2={y(0)} stroke="#5b6b70" strokeDasharray="3 3" />}
        {(["LightGBM", "XGBoost", "PatchTST"] as const).map((model) => (
          <path key={model} d={path(model)} fill="none" stroke={MODEL_COLORS[model]} strokeWidth={1.5} />
        ))}
        <text x={pad.left - 6} y={y(high) + 4} textAnchor="end" className="report-chart-label">{Math.round(high)}</text>
        <text x={pad.left - 6} y={y(low) + 4} textAnchor="end" className="report-chart-label">{Math.round(low)}</text>
        {[0, 32, 64, 95].map((slot) => (
          <text key={slot} x={x(slot)} y={height - 6} textAnchor="middle" className="report-chart-label">{rows[slot]?.time_label}</text>
        ))}
      </svg>
      <p className="report-chart-legend">
        {Object.entries(MODEL_COLORS).map(([model, color]) => (
          <span key={model}><i style={{ background: color }} />{model}</span>
        ))}
        {divergence?.status === "calibrated" && (
          <span>高分歧时段 {divergence.high_divergence_slot_count} 个（阈值 {divergence.high_threshold_yuan_per_mwh} 元/MWh，验证期 P95）</span>
        )}
      </p>
      <p className="report-finding-limit">模型分歧表示意见不一致，不是概率区间。</p>
    </section>
  );
}

function Bullets({ title, items, tone }: { title: string; items: string[]; tone?: "warn" }) {
  return (
    <section className={`report-block${tone ? ` report-${tone}` : ""}`}>
      <h4>{title}</h4>
      {items.length ? <ul>{items.map((item, index) => <li key={index}>{item}</li>)}</ul> : <p className="report-empty">无</p>}
    </section>
  );
}

export function AnalysisReport({ events }: { events: Record<string, unknown>[] }) {
  const reportEvent = [...events].reverse().find((event) => event.type === "report" && event.run_id);
  if (!reportEvent) return null;
  const report = reportEvent.report as Report;
  const runId = String(reportEvent.run_id);
  const forecast = [...events]
    .reverse()
    .map((event) => (event.type === "tool_result" ? (event.result as { tool?: string; data?: Forecast } | undefined) : undefined))
    .find((result) => result?.tool === "forecast_price")?.data;
  const validation = report.validation?.status ?? "not_run";
  return (
    <article className="analysis-report">
      <header className="analysis-report-header">
        <span className={`report-badge ${report.verified ? "report-badge-ok" : "report-badge-warn"}`}>
          {report.verified ? "已验证" : "未验证"}
        </span>
        <span className="report-badge">报告校验 {validation}</span>
        <a className="report-export" href={`/trace/${runId}/report.md`} download={`${runId}.md`}>导出 Markdown</a>
      </header>
      {report.summary && <p className="analysis-report-summary">{report.summary}</p>}
      <FindingList
        title="直接回答"
        items={(report.answers ?? []).map((answer) => ({
          ...answer,
          statement: (answer.status === "unanswerable" ? "（无法回答）" : "") + answer.statement,
        }))}
      />
      <FindingList title="异常" items={report.anomalies ?? []} />
      <FindingList title="原因" items={report.causes ?? []} />
      {report.outlook && <FindingList title="D+1 研判" items={[report.outlook]} />}
      {forecast && <ForecastChart forecast={forecast} />}
      <Bullets title="政策依据" items={(report.policy_basis ?? []).map((item) => item.statement)} />
      <div className="report-columns">
        <Bullets title="已知限制" items={report.known_limitations ?? []} />
        <Bullets title="未解决问题" items={report.unresolved_issues ?? []} tone={report.unresolved_issues?.length ? "warn" : undefined} />
      </div>
      {report.validation?.errors && report.validation.errors.length > 0 && (
        <Bullets title="校验错误" items={report.validation.errors} tone="warn" />
      )}
    </article>
  );
}
