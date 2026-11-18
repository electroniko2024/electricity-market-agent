# 多轮评测（题集 multiturn_dev_v1，sha256 b7300357253c）

## 总体

- 追问轮通过率（第 2 轮起）：37/44 = 84.1% [71%, 92%]
- 全部轮次通过率：77/84 = 91.7% [84%, 96%]
- 追问改写正确率（日期/月份完全一致）：36/40 = 90.0% [77%, 96%]
- 追问答案通过率（单轮判分）：33/40 = 82.5% [68%, 91%]
- 整段对话通过率：33/40 = 82.5% [68%, 91%]
- 该反问时反问（召回）：4/4 = 100.0% [51%, 100%]
- 反问是否必要（精确）：4/8 = 50.0% [22%, 78%]
- 追问轮平均证据复用次数：0.20（有复用的轮次 5/40）
- 每轮平均 token：13171
- 上限（同一批追问直接问标准独立问题）：38/40 = 95.0% [83%, 99%]

## 按模板（追问轮）

| 模板 | 通过 | 改写正确 | 答案通过 |
|---|---|---|---|
| ambiguous | 4/4 = 100.0% [51%, 100%] | - | - |
| correction | 3/4 = 75.0% [30%, 95%] | 4/4 = 100.0% [51%, 100%] | 3/4 = 75.0% [30%, 95%] |
| cross_turn_compare | 7/8 = 87.5% [53%, 98%] | 8/8 = 100.0% [68%, 100%] | 7/8 = 87.5% [53%, 98%] |
| found_entity | 3/4 = 75.0% [30%, 95%] | 4/4 = 100.0% [51%, 100%] | 3/4 = 75.0% [30%, 95%] |
| last_year | 0/4 = 0.0% [0%, 49%] | 0/4 = 0.0% [0%, 49%] | 0/4 = 0.0% [0%, 49%] |
| next_month | 4/4 = 100.0% [51%, 100%] | 4/4 = 100.0% [51%, 100%] | 4/4 = 100.0% [51%, 100%] |
| same_slot_prev_day | 4/4 = 100.0% [51%, 100%] | 4/4 = 100.0% [51%, 100%] | 4/4 = 100.0% [51%, 100%] |
| shift_day | 4/4 = 100.0% [51%, 100%] | 4/4 = 100.0% [51%, 100%] | 4/4 = 100.0% [51%, 100%] |
| swap_metric | 4/4 = 100.0% [51%, 100%] | 4/4 = 100.0% [51%, 100%] | 4/4 = 100.0% [51%, 100%] |
| topic_switch | 4/4 = 100.0% [51%, 100%] | 4/4 = 100.0% [51%, 100%] | 4/4 = 100.0% [51%, 100%] |

## 失败的追问轮

- MV012_t2 found_entity expect=proceed type=question rewrite_ok=True stage=evidence periods=['2024-03-04'] 2024年3月4日，山东全网实际负荷最高是多少？
- MV017_t2 last_year expect=proceed type=clarify rewrite_ok=False stage=clarified periods=[] 
- MV018_t2 last_year expect=proceed type=clarify rewrite_ok=False stage=clarified periods=[] 
- MV019_t2 last_year expect=proceed type=clarify rewrite_ok=False stage=clarified periods=[] 
- MV020_t2 last_year expect=proceed type=clarify rewrite_ok=False stage=clarified periods=[] 
- MV021_t2 correction expect=proceed type=question rewrite_ok=True stage=aborted periods=['2024-09-01'] 2024年9月1日山东实时电价的日均价是多少？
- MV036_t3 cross_turn_compare expect=proceed type=question rewrite_ok=True stage=aborted periods=['2025-07-16', '2025-08-28'] 2025年7月16日和2025年8月28日山东实时电价的日均价相比，哪天更贵，相差多少元/兆瓦时？
