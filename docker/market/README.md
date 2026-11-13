# 市场分析服务 Docker 部署

镜像只打包 `ellectric/` 代码和 Python 依赖（`ellectric/requirements-docker.lock`，即已验证的 WSL 环境 `/home/max/venvs/electric-market` 的完整冻结）。数据、模型快照、政策索引、reranker 权重和密钥都在运行时挂载或注入，不进镜像。

## 运行前提

| 组成 | 位置 | 说明 |
|---|---|---|
| 市场数据与模型快照 | `ellectric/data/market_processed/` | 挂载到容器 `/app/ellectric/data`，分析 trace 也写在这里 |
| 政策索引 | `ellectric/data/market_processed/policy_index/` | 需先在 WSL 用 `python -m ellectric.scripts.build_policy_index` 以 bge-m3 建好（Chroma 路径为相对路径，容器内可直接用） |
| bge-m3 embedding | WSL 内的 Ollama（`localhost:11434`） | compose 使用 host 网络直接访问；镜像内不含 Ollama |
| bge-reranker-v2-m3 | 默认 `/mnt/d/资料/bge-reranker-v2-m3` | 只读挂载到 `/models/bge-reranker-v2-m3`；位置变了用 `RERANKER_DIR` 覆盖 |
| LLM 密钥 | `docker/market/market.env` | 从 `market.env.example` 复制，已加入 `.gitignore` |

## 启动

在 `electric/` 目录（WSL 内 Docker）：

```bash
cp docker/market/market.env.example docker/market/market.env   # 填 ELLECTRIC_LLM_API_KEY
docker compose -f docker-compose.market.yml up -d --build
curl http://localhost:8000/health
```

使用 Windows 上的 Docker Desktop 时，reranker 路径要写成 Windows 路径：`RERANKER_DIR=D:/资料/bge-reranker-v2-m3`。另外，Docker Desktop 的 host 网络连到的是 Docker Desktop 虚拟机，不是 Ubuntu WSL，所以需要让 Ollama 监听可达地址，并把 `ELLECTRIC_OLLAMA_URL` 改成对应地址。

## LLM 思考开关

`ELLECTRIC_LLM_THINKING` 取值：

- `disabled`（默认）：请求只带 `thinking: {"type": "disabled"}`。2026-11-09 晚用项目真实请求体实测（3 个提示词 × 3 次）：只发 `thinking` 9 次全部成功且没有思考；同时加 `reasoning_effort: "none"` 时 9 次里 5 次 HTTP 400；只发 `reasoning_effort: "none"` 时 9 次里 4 次仍在思考。中转站行为会变，出现 400 或思考内容时需重测。
- `low` / `medium` / `high`：请求带 `reasoning_effort`。实测 `low` 仍会产生约 250 个思考 token。
- `default`：不传参数，由端点决定（默认会思考）。

## 资源

WSL 当前约 6 GB 内存。reranker 以 fp32 加载约 2.3 GB，Ollama 中的 bge-m3 约 1.2 GB，再加上 PatchTST/树模型推理，内存偏紧；内存不够时可在 `.wslconfig` 中调高 WSL 内存。
