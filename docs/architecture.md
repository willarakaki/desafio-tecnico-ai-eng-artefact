# System Architecture — Empório da Música Intelligent Assistant

## 1. Executive Overview & Engineering Assumptions
This system is engineered under Artefact's core operational principle (*"If not used, it is useless"*): bridging the conversational empathy of Large Language Models (LLMs) with the deterministic reliability of in-memory relational queries, hybrid information retrieval, and runtime safety guardrails.

### Core Engineering Assumptions
- **Latency & FinOps Efficiency:** Recurrent institutional questions (opening hours, store address, general return rules) must be served in under 50ms at zero token cost after the initial lookup via a **Context-Aware Semantic Cache**.
- **Transactional Accuracy:** Product pricing, stock availability, promotions, and order status lookups require 100% deterministic accuracy, handled via parameterized **DuckDB** SQL queries over the operational CSV files.
- **Data Governance & Privacy (LGPD):** Customer Personally Identifiable Information (PII — Brazilian CPF, email, phone number) is kept strictly in ephemeral memory during DuckDB tool execution and automatically redacted via **Presidio-compatible PII Masking** before reaching logs or telemetry panels.
- **Provider Agnosticism:** The inference layer is decoupled through the **`LLMFactory` pattern**, supporting Google Gemini (`gemini-2.5-flash`), OpenAI (`gpt-4o-mini`), or local open-weights models via Ollama (`qwen2.5`).

---

## 2. System Topology & Layered Execution Flow

```mermaid
flowchart TD
    User([Empório da Música Customer]) --> RateLimit[1. Rate Limiter & Input Guardrail]
    RateLimit -->|Blocked: Jailbreak / SQLi / Abuse| Reject[Safe Standardized Refusal]
    RateLimit -->|Approved| SemCache{2. Context-Aware<br/>Semantic Cache}
    SemCache -->|Cache Hit: Institutional FAQ| FastOut[Immediate Response < 50ms]
    SemCache -->|Cache Miss or Transactional| Orchestrator[3. LangGraph Orchestrator<br/>via LLMFactory]
    
    Orchestrator -->|Catalog / Prices / Promotions| DuckCatalog[Tool: DuckDB Catalog Queries]
    Orchestrator -->|Order Status / Customer Lookup| DuckOrders[Tool: DuckDB Order Queries]
    Orchestrator -->|Returns / Hours / Address / Payment| HybridRAG[Tool: Hybrid RAG BM25 + FAISS]
    Orchestrator -->|Policy Exception / Critical Case| HITL[Tool: Human-in-the-Loop Escalation]
    
    DuckCatalog --> Synthesizer[4. Consultative Synthesis & Grounding Check]
    DuckOrders --> Synthesizer
    HybridRAG --> Synthesizer
    HITL --> Synthesizer
    
    Synthesizer --> PIIMask[5. PII Anonymizer & FinOps Telemetry]
    FastOut --> PIIMask
    Reject --> PIIMask
    PIIMask --> Output([Final Customer Response + Audit Metadata])
```

### 2.1. Mapped Data Dictionary (To be populated in Task 0)

> **Note for Agent:** During Task 0, inspect `data/*.csv` and `data/políticas_da_loja.pdf` and document the exact table schemas, column names, data types, and PDF section headers here before implementing Task 2 and Task 3.

## 3. Detailed Architectural Decisions

### 3.1. LLMFactory Pattern & Cost-Aware Routing
To prevent vendor lock-in and optimize inference spend, `src/agent/llm_factory.py` centralizes model instantiation across distinct operational profiles:

- **Router / Guardrail LLM (`temperature=0.0`):** Optimized for low latency, deterministic tool selection, and structured intent classification.
- **Synthesizer LLM (`temperature=0.2`):** Optimized for empathetic, consultative responses in Brazilian Portuguese aligned with the Empório da Música musical specialist persona, bound by strict negative constraints against hallucination.
- **Embedding Model (`get_embeddings()`):** Provides dense vector representations for both the FAISS policy index and the in-memory Semantic Cache.

### 3.2. Context-Aware Semantic Cache (With Transactional & PII Bypass)
Reduces token consumption and latency by caching validated (query_embedding, response) pairs using cosine similarity (`threshold >= 0.92`).

- **Cache-Eligible Queries:** Store address in Campo Grande/MS, opening hours, payment methods, and general warranty/return policy explanations.
- **Strict Bypass Rule (`BYPASS_TRANSACTIONAL`):** Any user message containing order identifiers, CPF, email, personal data, or live catalog/order status intents automatically bypasses cache read/write operations. This guarantees real-time accuracy and prevents cross-customer data leakage.

### 3.3. Structured Data Engineering (DuckDB + Typed Tools)
Instead of injecting raw CSV files into the prompt context (which wastes tokens and degrades recall) or using unconstrained Text-to-SQL (prone to syntax errors and SQL injection), operational tables are loaded into an in-memory DuckDB instance (`src/database/duckdb_manager.py`):

- Text columns are normalized (case-insensitive and accent-insensitive comparisons via `ILIKE` and `strip_accents`) so queries like "violao" or "Takamine GD20" match reliably.
- Tools are exposed with strict Pydantic v2 schemas (`search_products`, `get_product_details`, `get_active_promotions`, `get_order_status`).

### 3.4. Unstructured Policy Pipeline (Hybrid RAG: BM25 + FAISS)
The internal manual `data/políticas_da_loja.pdf` contains both conceptual guidelines (exchange eligibility, warranty coverage) and exact lexical entities (deadlines in days, percentages, PIX/installment rules, physical address in Campo Grande/MS).

- **Topic-Aware Chunking:** Extracts text preserving logical policy sections with controlled overlap.
- **Ensemble Hybrid Retriever:** Combines sparse lexical search via BM25 (weight = 0.4 for exact keyword/number matching) with dense semantic search via FAISS (weight = 0.6 for paraphrased customer questions).

### 3.5. Defense-in-Depth: Rate Limiting, Guardrails & LGPD PII Masking
- **Sliding-Window Rate Limiter (`src/security/rate_limiter.py`):** Restricts excessive requests per `session_id` and enforces a maximum input character length to prevent Denial-of-Wallet attacks.
- **Input Guardrails (`src/security/guardrails.py`):** Two-tier barrier combining fast deterministic heuristics (blocking SQL injection keywords, prompt extraction, and DAN/Jailbreak signatures) with domain scope validation. Out-of-scope questions (e.g., cooking recipes, politics, general coding) are politely declined in-persona without invoking downstream tools.
- **PII Anonymizer (`src/security/pii_anonymizer.py`):** Implements Microsoft Presidio-compatible recognizers tailored for Brazilian entities (`CPF`, `EMAIL_ADDRESS`, `PHONE_NUMBER`), masking sensitive strings (`***.***.***-**`) prior to logging or telemetry export.

### 3.6. Human-in-the-Loop (HITL) — Smart Escalation
Because the agent's mission is to assist the human store team, `src/tools/hitl_tools.py` provides the `escalate_to_human(reason, case_summary, urgency_level)` tool:

- Triggered when a customer requests an exception outside standard policy (e.g., return after the statutory deadline claiming hidden defect), high-value custom negotiation, or expresses strong dissatisfaction.
- Sets `requires_human_intervention = True` in the LangGraph state, outputs a structured Handoff Ticket in the UI/CLI, and reassures the customer that a human specialist has received the context.

### 3.7. Observability & FinOps Telemetry
Every turn processed by the agent records a structured `TurnTelemetry` object (`src/observability/telemetry.py`) capturing:

- End-to-end execution latency (ms).
- Semantic Cache status (`HIT`, `MISS`, or `BYPASS_TRANSACTIONAL`).
- List of deterministic tools invoked (`tools_called`).
- Input/output token counts and estimated inference cost (USD).
- HITL escalation status.