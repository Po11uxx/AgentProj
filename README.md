# AI Personal Productivity Agent

一个 Agent 项目，覆盖：

- Agent Core（LangGraph 状态机）
- Tool 调用（天气/搜索/地图POI/日程/交通/Python 执行器）
- Skill 封装（任务拆解/旅行规划/总结）
- RAG（FAISS 检索 + rerank）
- Memory（短期会话 + 长期用户偏好）
- Backend（FastAPI）
- Frontend（Streamlit Chat UI）

## 1. 项目结构

```text
app/
  agent/graph.py              # Agent 状态机流程
  tools/                      # Tool 层
  skills/                     # Skill 层
  rag.py                      # 向量化与检索
  memory.py                   # 记忆模块
  main.py                     # FastAPI 服务
  data/
    knowledge/*.md            # RAG 知识库文档
    vector_store.faiss        # FAISS 索引产物
    vector_store_meta.json    # chunk 元数据
    vector_store_embeddings.npy # faiss 不可用时的向量回退文件
    memory/user_preferences.json
  scripts/build_kb.py         # 重建知识库索引
frontend/
  streamlit_app.py            # 前端演示页面
requirements.txt
.env.example
README.md
```

## 2. 架构与执行流程

1. 用户输入请求（如洛杉矶一日游）
2. Agent 进入 `plan` 节点，做任务拆解
3. Agent 进入 `retrieve` 节点，执行 RAG 检索
4. Agent 进入 `execute` 节点，调度 Travel Planning Skill（内部调用 Weather/Search/Calendar）
5. Agent 进入 `summarize` 节点，输出结构化总结
6. Agent 进入 `remember` 节点，提取并存储用户偏好（预算/偏好场景）

LangGraph 不可用时自动降级为顺序执行（保证项目可跑）。

## 3. 快速启动

### 3.1 安装依赖

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

### 3.2 配置环境变量

```bash
cp .env.example .env
```

默认 `USE_MOCK_LLM=true`，不填 Gemini Key 也可运行。

### 3.3 构建知识库索引

```bash
python -m app.scripts.build_kb
```

如果本机安装了 `faiss/torch` 且出现 `segmentation fault`，先使用安全模式：

```env
RAG_USE_FAISS=false
RAG_USE_TRANSFORMERS=false
```

### 3.4 启动后端

```bash
uvicorn app.main:app --reload --host 0.0.0.0 --port 8000
```

访问健康检查：`GET http://127.0.0.1:8000/health`

### 3.5 启动前端

```bash
streamlit run frontend/streamlit_app.py
```

## 4. API 使用

### 4.1 Chat 接口

`POST /chat`

请求示例：

```json
{
  "session_id": "demo-session",
  "user_id": "demo-user",
  "message": "帮我规划一个周末洛杉矶一日游，考虑天气、人流，并生成行程表。我的预算是200。"
}
```

返回包含：

- `answer`：最终方案文本
- `plan`：拆解步骤
- `used_tools`：本轮调用工具
- `retrieved_docs`：RAG 命中文档
- `debug`：调试信息（工具输出、记忆状态）

## 5. 模块设计说明

### 5.1 Agent Core

- 文件：`app/agent/graph.py`
- 基于状态机节点：`plan -> retrieve -> execute -> summarize -> remember`
- 支持多阶段决策、动态工具编排、记忆写回

### 5.2 Tools

- `weather_tool.py`：优先调用 Open-Meteo，失败自动 mock
- `search_tool.py`：优先调用 DuckDuckGo API，失败自动 mock
- `map_tool.py`：调用 OpenStreetMap Nominatim 实时检索城市 POI（地理编码+地点搜索）
- `calendar_tool.py`：按时间槽生成结构化日程（支持地点坐标/地址）
- `transport_tool.py`：按相邻地点生成交通建议，并结合实时搜索提示具体路线
- `python_executor.py`：受限表达式计算
- `knowledge.py`：当本地 knowledge 缺少某城市时，运行时自动联网检索并生成 `knowledge/*.md`，随后刷新向量库供下次复用

### 5.3 Skills

- `TaskDecompositionSkill`：复杂任务拆步骤
- `TravelPlanningSkill`：组合多个工具产出旅行计划
- `SummarySkill`：统一格式化输出

### 5.4 RAG

- 文件：`app/rag.py`
- 流程：`query -> embedding -> FAISS 召回(candidate_k) -> rerank -> top_k -> context`
- embedding：`sentence-transformers`（默认 `BAAI/bge-small-en-v1.5`）
- rerank：`cross-encoder/ms-marco-MiniLM-L-6-v2`（不可用时自动回退）
- query expansion：对中英文城市名/地标做扩展（如 `杭州 -> Hangzhou`, `西湖 -> West Lake`）
- 可实验参数：`RAG_CHUNK_SIZE/RAG_CHUNK_OVERLAP/RAG_TOP_K/RAG_CANDIDATE_K`
- 原生库开关：`RAG_USE_FAISS`、`RAG_USE_TRANSFORMERS`（建议先 `false` 验证流程，再逐步打开）

### 5.5 Memory

- 短期记忆：`ConversationMemory`（按 session 窗口存储）
- 长期记忆：`UserPreferenceMemory`（JSON 持久化用户偏好）

## 6. 你需要手动修改/替换的地方

这是项目落地到生产或面试进阶时必须做的改动：

1. LLM 接入（推荐）
- 文件：`app/llm.py`
- 操作：设置 `.env` 中 `USE_MOCK_LLM=false`，并填 `GEMINI_API_KEY`
- 建议：模型参数使用 `GEMINI_MODEL`（默认 `gemini-1.5-flash`）

2. RAG 参数调优（推荐）
- 文件：`app/rag.py`
- 操作：调整 `.env` 中 `RAG_CHUNK_SIZE/RAG_CHUNK_OVERLAP/RAG_TOP_K/RAG_CANDIDATE_K`
- 建议：固定问集下记录命中率与时延，比较不同参数组合

3. Tool 真 API 化
- 文件：`app/tools/*.py`
- 操作：将 mock fallback 改为真实业务 API（地图、票务、人流、日历服务）

4. Memory 存储升级
- 文件：`app/memory.py`
- 操作：JSON 持久化替换为 Redis/Postgres

5. 安全与治理
- 文件：`app/tools/python_executor.py`
- 操作：生产环境建议关闭执行器，或放入独立沙箱

6. 观测与日志
- 文件：`app/main.py` + `app/agent/graph.py`
- 操作：接入 structured logging / tracing（如 OpenTelemetry）

