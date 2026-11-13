# Market Processed Data

Generate local artifacts from the read-only source files with:

```bash
python -m ellectric.scripts.prepare_market_data
```

The pipeline writes `sd_rt_15min.parquet`, `weather_actual_hourly.parquet`,
`sd_market.duckdb`, `quality_report.json`, and `dataset_manifest.json` here.
Use `--overwrite` only when intentionally rebuilding these generated outputs.

Market rows are keyed by `trade_date` and `slot_id` (`00:15` maps to slot 0;
`24:00` maps to slot 95). `time_label` is preserved, and `label_at_local` /
`label_at_utc` describe that clock label only. The interval start/end convention
has not been verified, so these fields are not interval boundaries.

Weather rows contain Jinan fields only. The cached timestamp basis, units, and
observation provenance are unverified; the pipeline preserves stored values
without shifting, resampling, unit conversion, or alignment.

## Policy corpus

The 17 local policy attachments are parsed into `policy_documents.json` and
`policy_chunks.json`. Their publication dates, effective periods, replacement
relationships, and overall historical coverage have not been verified. Search
results are therefore local text candidates, not legal conclusions or eligible
policy citations. Formula OCR blocks are not indexed.

Build the local BM25 + bge-m3 candidate index with:

```bash
python -m ellectric.scripts.build_policy_index
```

The command writes `policy_index/`, which is derived from the local attachments
and ignored by Git. Dense vectors come from `bge-m3` served by the local Ollama
(`ELLECTRIC_OLLAMA_URL`, default `http://localhost:11434`) and are stored in a
Chroma collection (`--vector-backend json` stores them as JSON instead). Queries
fuse BM25 and dense recall with RRF, then rerank the top 20 with the local
`bge-reranker-v2-m3` weights at `ELLECTRIC_RERANKER_PATH` (default
`/mnt/d/资料/bge-reranker-v2-m3`, loaded offline).

## Analysis service

From the repository root in the approved WSL environment:

```bash
export ELLECTRIC_LLM_BASE_URL=https://aispot.swj0227.icu/v1
export ELLECTRIC_LLM_MODEL=deepseek-v4.1-flash
export ELLECTRIC_LLM_THINKING=disabled   # default; low/medium/high map to reasoning_effort
read -rsp "API key: " ELLECTRIC_LLM_API_KEY; echo
export ELLECTRIC_LLM_API_KEY
python -m ellectric.scripts.build_policy_index
uvicorn ellectric.api.server:app --host 0.0.0.0 --port 8000
```

The new interfaces are `POST /analyze`, `POST /analyze/stream`, `GET /trace/{id}`,
and `POST /market-tools/{name}`. The API never reads other applications' auth
files. Keep `ELLECTRIC_LLM_API_KEY` in the private shell environment; do not put
it in this repository.

The React production bundle is built from `ellectric/web/` with the existing
lock file and pnpm. Historical analysis requires an explicit timezone-aware
`as_of`. It does not imply that inputs were actually available at that time.
Market intervals retain the confirmed key `trade_date + slot_id` and unknown
boundary semantics. Weather timestamps retain their unknown time basis.
