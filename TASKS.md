# Execution Plan & Atomic Commits (TASKS.md)

> **Agent Instruction (Antigravity):** Execute **only one task at a time**. Upon finishing the implementation of a task, invoke the `tdd-commit-gate` skill and run the corresponding test suite with `pytest -v`. Only after 100% of tests pass, request approval to execute the specified semantic commit before moving to the next task.

## [x] Task 0: Data Profiling, Project Setup, Ruff, CI/CD & Base Dockerization
**Objective:** Inspect the actual datasets provided for the challenge and scaffold the Python project foundation with linting, CI, and containerization.
- [x] Inspect all files in `data/*.csv` (list exact filenames, column names, data types, and first 5 rows) and extract the topic structure from `data/políticas_da_loja.pdf`.
- [x] Update `docs/architecture.md` with section "2.1. Mapped Data Dictionary" containing the discovered real schema.
- [x] Create `pyproject.toml` (compatible with `uv` and `pip`, including `ruff` and `pytest` configurations), `.python-version` (3.11), `.env.example`, and `.gitignore`.
- [x] Create `.github/workflows/ci.yml` to automatically run `ruff check .` and `pytest -v` on every push and merge to `main`.
- [x] Create a lightweight `Dockerfile` (`python:3.11-slim` with `uv`) and `docker-compose.yml` exposing the Streamlit UI (`8501`) and passing `.env` variables for zero-friction evaluation.
- **Validation:** Virtual environment created, dependencies installed, and `uv run ruff check .` passes with zero errors.
- **Commit:** `chore(setup): initialize project structure, ruff, github actions ci, docker config and data profiling`

## [x] Task 1: Core Configuration & `LLMFactory` Pattern
**Objective:** Decouple LLM providers and centralize environment settings.
- [x] Implement `src/config.py` using `pydantic-settings` (supporting `LLM_PROVIDER`, `GEMINI_API_KEY`, `OPENAI_API_KEY`, `OLLAMA_BASE_URL`, rate limit thresholds, and semantic cache similarity thresholds).
- [x] Implement `src/agent/llm_factory.py` with the `LLMFactory` class providing:
  - `get_router_llm(temperature=0.0)`: Fast, deterministic model for intent classification, guardrails, and tool routing.
  - `get_synthesizer_llm(temperature=0.2)`: Consultative model for final customer responses aligned with the store persona.
  - `get_embeddings()`: Embedding model instance matching the active provider.
- [x] Create unit tests in `tests/test_llm_factory.py` validating provider instantiation and configuration switching.
- **Validation:** `pytest tests/test_llm_factory.py -v`
- **Commit:** `feat(core): implement centralized config and multi-provider LLMFactory`

## [x] Task 2: DuckDB Ingestion & Deterministic CSV Tools
**Objective:** Load operational CSV files into an in-memory relational database and expose strictly typed tools.
- [x] Implement `src/database/duckdb_manager.py` to automatically ingest `data/*.csv` into DuckDB tables with text normalization (accent stripping and whitespace trimming).
- [x] Implement `src/tools/catalog_tools.py` with strict Pydantic v2 schemas:
  - `search_products(query, category, max_price, in_stock_only)`
  - `get_product_details(product_name_or_id)`
  - `get_active_promotions(category_or_product)`
- [x] Implement `src/tools/order_tools.py` with strict Pydantic v2 schemas:
  - `get_order_status(order_id, customer_identifier)`
- [x] Create deterministic unit tests in `tests/test_duckdb_tools.py` covering real queries (e.g., acoustic guitars/violões up to R$ 1000, Takamine GD20 lookup, non-existent item handling, and order status lookup).
- **Validation:** `pytest tests/test_duckdb_tools.py -v`
- **Commit:** `feat(data): implement DuckDB ingestion and deterministic catalog and order tools`

## [x] Task 3: Hybrid RAG Pipeline for Store Policies PDF
**Objective:** Implement hybrid retrieval (lexical + dense semantic search) over `data/políticas_da_loja.pdf`.
- [x] Implement `src/rag/pdf_loader.py` for clean PDF text extraction and semantic/topic-based chunking (exchanges, returns, warranty, opening hours, store address, payment methods).
- [x] Implement `src/rag/hybrid_retriever.py` combining `BM25Retriever` (weight 0.4 for exact lexical matches like deadlines and address) and `FAISS` (weight 0.6 for semantic similarity).
- [x] Implement the `consult_store_policies(question)` tool in `src/tools/policy_tools.py` returning relevant chunks with page/section metadata.
- [x] Create unit tests in `tests/test_rag_policies.py` validating retrieval accuracy for the Campo Grande/MS store address, opening hours, and return/refund rules.
- **Validation:** `pytest tests/test_rag_policies.py -v`
- **Commit:** `feat(rag): build hybrid retriever (BM25 + FAISS) for store policies PDF`

## [x] Task 4: Defense-in-Depth (Guardrails, Rate Limiter, Semantic Cache & PII Masking)
**Objective:** Ensure runtime protection against Prompt Injection/Jailbreak, LGPD PII compliance, and FinOps token savings.
- [x] Implement `src/security/rate_limiter.py` (in-memory Sliding Window per `session_id` + maximum character payload limit).
- [x] Implement `src/security/guardrails.py` blocking SQL Injection patterns, Jailbreak attempts, System Prompt extraction, and out-of-scope topics while preserving the musical specialist persona.
- [x] Implement `src/security/pii_anonymizer.py` (compatible with Microsoft Presidio + regex recognizers for Brazilian CPF, email, and phone numbers) to sanitize logs and telemetry without breaking in-memory DuckDB lookups.
- [x] Implement `src/cache/semantic_cache.py` in memory (cosine similarity threshold >= 0.92) with a **strict Transactional/PII Bypass Rule**: never cache queries or responses containing CPF, order IDs, personal data, or live order/stock lookups.
- [x] Create unit tests in `tests/test_security_and_cache.py` verifying jailbreak blocking, CPF masking, Cache Hit on institutional questions, and Cache Bypass on order queries.
- **Validation:** `pytest tests/test_security_and_cache.py -v`
- **Commit:** `feat(security): add input guardrails, rate limiter, context-aware semantic cache and PII masking`

## [x] Task 5: LangGraph Orchestrator, Persona, HITL & FinOps Telemetry
**Objective:** Build the stateful agent graph integrating all tools, human escalation, and observability metrics.
- [x] Implement `escalate_to_human(reason, case_summary, urgency_level)` in `src/tools/hitl_tools.py` for policy exceptions, complex warranty claims, or dissatisfied customers.
- [x] Implement `src/observability/telemetry.py` tracking total latency (ms), Semantic Cache status (`HIT`, `MISS`, `BYPASS_TRANSACTIONAL`), invoked tools, token usage, and estimated cost in USD.
- [x] Implement `src/agent/prompts.py` defining the Empório da Música specialist System Prompt and explicit negative constraints against hallucination.
- [x] Implement `src/agent/state.py` and `src/agent/graph.py` using LangGraph with `MemorySaver` (`thread_id`) and `recursion_limit=6`.
- [x] Create integration tests in `tests/test_agent_routing.py` validating routing across CSV tools, PDF RAG, combined CSV+PDF reasoning, HITL escalation, and out-of-scope refusal.
- **Validation:** `pytest tests/test_agent_routing.py -v`
- **Commit:** `feat(agent): implement LangGraph orchestrator, HITL handoff and FinOps telemetry`

## [x] Task 6: Interaction Interfaces (Rich CLI & Streamlit UI)
**Objective:** Deliver clean user interfaces with real-time architectural observability.
- [x] Implement `cli.py` using `rich` with session memory, live metadata display (invoked tools, cache status, latency), and commands `/clear`, `/metrics`, and `/exit`.
- [x] Implement `app.py` in Streamlit featuring:
  - Interactive chat interface with the Empório da Música specialist persona.
  - **Observability & FinOps Sidebar** displaying Semantic Cache status, executed tools, turn latency, cumulative token cost, and a visual alert card when a Human-in-the-Loop (HITL) handoff ticket is triggered.
- **Validation:** Smoke test both interfaces and verify clean imports.
- **Commit:** `feat(ui): add interactive Rich CLI and Streamlit interface with FinOps telemetry panel`

## [x] Task 7: Model Resiliency (Fallback Model) & Automated Generation of 5 Conversation Scenarios (`transcripts/`)
**Objective:** Enhance agent robustness with automatic provider failover (Fallback Model) and generate the mandatory interaction deliverables using the real agent pipeline.
- [x] Implement Fallback Model support in `src/agent/llm_factory.py` (via LangChain's `.with_fallbacks()` across primary and secondary LLM providers, e.g. Gemini -> OpenAI/Ollama) with graceful degradation to HITL if all providers fail.
- [x] Update `src/observability/telemetry.py` and `src/agent/graph.py` to record `fallback_used: bool` and the actual provider resolving the inference turn.
- [x] Add tests in `tests/test_llm_factory.py` and `tests/test_agent_routing.py` verifying fallback model activation on simulated API failures/rate limits (HTTP 429/500/timeout).
- [x] Create `scripts/generate_transcripts.py` to execute 5 realistic multi-turn conversations against the agent and export formatted Markdown files (including tool call traces and FinOps telemetry):
  1. `transcripts/01_catalogo_violoes_ate_1000.md` (Filtered catalog search + instrument specification advice).
  2. `transcripts/02_preco_especifico_takamine_gd20.md` (Exact price lookup, stock availability, and payment rules).
  3. `transcripts/03_informacoes_gerais_e_endereco.md` (Store address in Campo Grande/MS, opening hours, and demonstration of a Semantic Cache `HIT`).
  4. `transcripts/04_politica_devolucao_pedido_e_hitl.md` (Non-trivial multi-source scenario: order status lookup in CSV + return policy lookup in PDF + exception request triggering HITL escalation).
  5. `transcripts/05_guardrails_fora_de_escopo_e_jailbreak.md` (Polite refusal of out-of-domain questions and active blocking of a Prompt Injection attempt).
- **Validation:** Confirm fallback resilience tests pass and all 5 files exist in `transcripts/` strictly reflecting real data from `data/`.
- **Commit:** `feat(resilience): implement fallback model mechanism and generate 5 conversation scenarios with telemetry`

## [ ] Task 8: Enterprise Backend — FastAPI REST & SSE Streaming API + Dockerization
**Objective:** Deliver an enterprise-grade API tier aligning with production JD expectations (FastAPI backend wrapping LangGraph, Server-Sent Events streaming compatible with modern React/Next.js Vercel AI SDK, and 1-command Docker containerization).
- [ ] Implement `src/api/server.py` with FastAPI:
  - `POST /api/chat`: synchronous turn execution returning structured JSON with response, FinOps telemetry, tools called, and HITL tickets.
  - `POST /api/chat/stream`: Server-Sent Events (SSE) streaming endpoint returning chunked tokens and graph step events.
  - `GET /api/health`: production readiness/liveness probe checking DuckDB, Hybrid RAG, and LLM configuration.
  - `GET /api/metrics`: session-aggregated FinOps and telemetry metrics.
  - Interactive OpenAPI Swagger documentation (`/docs` and `/redoc`).
- [ ] Create `Dockerfile` (optimized multi-stage or Python 3.11-slim) and `docker-compose.yml` supporting both FastAPI and Streamlit services.
- [ ] Add unit and integration tests in `tests/test_api.py` validating all endpoints, status codes, and streaming responses using FastAPI `TestClient`.
- **Validation:** Confirm 100% of test suite passes and Docker image builds successfully.
- **Commit:** `feat(api): implement enterprise FastAPI REST and SSE streaming server with Docker containerization`

## [ ] Task 9: Executive & Technical `README.md`
**Objective:** Consolidate all architectural justifications and operational instructions required by Artefact.
- [ ] Write a comprehensive `README.md` in Portuguese (matching the challenge language) covering:
  - Executive summary, architecture topology, and Mermaid diagram.
  - Quickstart in 3 local commands (`uv`/`pip` with Streamlit or CLI) and 1 command via Docker (`docker compose up`).
  - OpenAPI Swagger documentation overview and React/Next.js Vercel AI SDK frontend integration example snippet.
  - Technical justifications (`LLMFactory`, DuckDB vs. Text-to-SQL, Hybrid RAG BM25+FAISS, Context-Aware Semantic Cache, Guardrails, Presidio PII Masking, HITL, and Resilient Fallbacks).
  - FinOps cost breakdown, latency metrics, and LangSmith observability guide.
  - AI Coding Assistant Workflow disclosure (Spec-Driven Development + TDD with custom skills in Google Antigravity).
  - Production roadmap (MCP servers, distributed Redis cache, dedicated Llama Prompt Guard 2 via vLLM, CI/CD evaluation with Ragas/DeepEval).
- **Validation:** Run full `pytest -v` suite to confirm 100% pass rate before final submission.
- **Commit:** `docs(readme): complete technical documentation, FinOps analysis and AI workflow disclosure`
