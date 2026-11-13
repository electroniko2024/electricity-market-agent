# 山东电力市场分析 Agent

基于山东电力现货市场公开数据的分析智能体：用户用自然语言提问（"9 月 25 日电价最低的那一刻风光出力多少？""负电价最多的那个月里哪天最严重、为什么？"），智能体先判断问题能不能用现有数据回答，再规划并调用 8 个确定性工具取数、预测、归因、检索政策，最后输出**每个数值都绑定到工具返回证据**的结构化报告，并由校验器逐项核对。

- 数据：山东 15 分钟实时市场数据 71,520 个时段（2024-01-01 — 2026-01-14），济南逐时气象 17,904 点，17 份政策附件。
- 只做历史分析和历史时点的 D+1 预测，不执行交易。

## 架构

```
问题 ─▶ scope（可答性判断：对照数据目录拆出要回答的各项）
          │ 全部不可答 → 直接出报告说明缺什么
          ▼
       planner（计划，参数可引用前一步结果 $ref）─▶ executor（调用工具）
          ▲                                              │
          └──── replanner ◀── reflection（证据是否充分） ◀┘
                                       │
                                       ▼
                    reporter（answers + 异常 + 成因 + 政策依据）
                                       │
                                       ▼
          validator（数值绑定、行一致性、方向、政策时效、直接回答完整性）
                     不通过 → 带错误修订，最多 2 次
```

LangGraph 实现，提供四种配置用于对照实验：

| 配置 | 说明 |
|---|---|
| E0 | ReAct：逐步调用工具，无显式计划、无反思 |
| E1 | 规划 + 执行，无反思 |
| E2 | 规划 + 执行 + 反思 + 重规划（完整系统） |
| E3 | E2 去掉政策生效区间过滤（消融） |

### 工具（`ellectric/service/market_tools.py`）

| 工具 | 作用 |
|---|---|
| `query_price` / `query_load` / `query_weather` | 明细查询（单次 ≤ 32 天，可按 `slot_id` 或 `time_label` 取某个时段） |
| `query_stats` | 单次 ≤ 366 天的汇总：均值、极值及其所在行、负价/尖峰时段数，按日/月/时段分组，`group_extremes` 直接给出最大最小的分组 |
| `forecast_price` | LightGBM / XGBoost / PatchTST 三模型 D+1 电价预测与模型分歧 |
| `detect_anomaly` | 负价、尖峰、跳变、数据质量、预测残差事件 |
| `explain_drivers` | 事件时点的负荷、风电、光伏、联络线对竞价空间的贡献，附 SHAP 和相关政策 |
| `search_policy` | 政策混合检索：BM25 + bge-m3 双路召回、RRF 融合、bge-reranker-v2-m3 重排、按生效区间过滤 |

所有工具都用 Pydantic 校验参数，并按 `as_of` 截断数据，不会读到截止时点之后的数据。

## 正式评测结果（test_v3）

代码冻结后（哈希记录见 `ellectric/eval/market/cases/test_v3.freeze.json`）用新随机种子生成 237 题，日期窗口与开发集不重叠，只跑一次。E0、E2 每题 3 次，E1 只跑故障注入层，E3 只跑政策版本类。

| 配置 | pass@1 | 95% CI | pass^3 |
|---|---|---|---|
| **E2 规划 + 反思（完整系统）** | **83.3%** | 80.3%–85.8% | 71.3% |
| E0 ReAct | 80.7% | 77.7%–83.5% | 66.7% |

- 长链路任务（5 步以上）：E2 83.7% vs E0 75.2%，按题配对差值 +8.5pp（95% CI +1.3～+15.7）。
- 工具故障后的恢复：E2 66.7% vs 无反思的 E1 34.0%，差值 +32.7pp（95% CI +21.3～+44.0）。
- 不可答问题正确拒答 30/30，部分可答 36/36，零误答、零误拒；取错行 0 次。
- 政策检索：Recall@5 95.7%、MRR 0.935；三模型点预测 MAE 比最好的基线低 34.8%。

- 项目概览、改进历程和可引用的数字：[`docs/市场Agent改进与进度记录.md`](docs/市场Agent改进与进度记录.md)（建议先读）
- 评测设计：[`docs/评测方案v2.md`](docs/评测方案v2.md)
- 每轮改动和评测的原始数字：[`docs/开发过程详细记录.md`](docs/开发过程详细记录.md)
- 完整打分输出：`ellectric/eval/market/results/test_v3/summary_v2.md`

## 快速开始

环境：Python 3.12（开发在 WSL Ubuntu 24.04 上进行），本地 Ollama 提供 `bge-m3` 向量，`bge-reranker-v2-m3` 权重放在本地目录，LLM 走任意 OpenAI 兼容接口。

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r ellectric/requirements-market.lock
cp docker/market/market.env.example .env      # 填写 ELLECTRIC_LLM_API_KEY 等
```

### 数据

原始数据不随仓库发布（见 `ellectric/data/shandong/README.md`）。把 `山东_2024-2026_15min.csv` 和 `weather_2024-2026.parquet` 放进 `ellectric/data/shandong/`，政策附件放进仓库同级的 `../data/policy/`，然后：

```bash
python -m ellectric.scripts.prepare_market_data        # 标准化 → DuckDB / Parquet + 质量报告
python -m ellectric.scripts.train_market_models        # 训练三模型，输出 model_run_2025-10-01
python -m ellectric.scripts.catalog_policies           # 政策清单
python -m ellectric.scripts.parse_policies             # 政策解析与切块
python -m ellectric.scripts.build_policy_index         # BM25 + 向量索引
```

### 运行

```bash
uvicorn ellectric.api.server:app --host 0.0.0.0 --port 8000
```

打开 `http://localhost:8000` 使用网页（`ellectric/web`，构建产物已在 `ellectric/api/static`）。接口：`POST /analyze`、`POST /analyze/stream`（SSE）、`GET /trace/{id}`、`GET /trace/{id}/report.md`、`POST /market-tools/{name}`。

Docker：

```bash
cp docker/market/market.env.example docker/market/market.env
docker compose -f docker-compose.market.yml up -d --build
```

### 测试与评测

```bash
python -m pytest ellectric/tests
python -m ellectric.eval.market.run_eval --set test_v3 --variants E0 E2 --runs 3 --out ellectric/eval/market/results/test_v3
python -m ellectric.eval.market.score_v2 --set test_v3 --results ellectric/eval/market/results/test_v3 --pairs E0:E2
```

## 目录

```
ellectric/
  api/          FastAPI 服务与前端构建产物
  llm/          智能体（market_analysis.py）与报告 Markdown 导出
  service/      8 个市场工具（market_tools.py）、分析运行与 trace 存储（analysis.py）
  pipeline/     数据标准化、特征、三模型训练与推理（pipeline/market）
  policy/       政策清单、解析、混合检索
  eval/market/  题集生成、评测运行、v1/v2 判分、题集与结果
  scripts/      数据准备、模型训练与回测、政策索引
  tests/        单元测试
  web/          React 前端源码
docker/         市场服务 Dockerfile
docs/           改进与进度概览、开发过程详细记录、评测方案、政策元数据核对表
```

## 已知限制

- 超出 token 预算或报告输出被截断时，整次运行报错退出，已取到的证据不会写成报告（test_v3 中 13 次运行因此计为失败）。
- 政策对比类问题较弱（test_v3 上 E2 3/15）。
- 题目由模板生成，句式比真实用户提问单一。

## 许可

见 [LICENSE](LICENSE)。
