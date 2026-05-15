# Blog Writing Agent

> Type a topic. Get a publication-ready technical blog post.

Most LLM-powered writing tools are single-prompt wrappers. BWA is a **multi-agent pipeline** that mirrors how an experienced technical writer actually works: decide whether research is needed, gather it, plan a coherent structure, then write each section in parallel with full context. Every stage is a deliberate node with a specific contract.

---

## Pipeline

```
                  topic
                    │
                    ▼
             ┌──────────────┐
             │    Router    │
             └──────┬───────┘
          closed_book│  hybrid / open_book
                     │         │
                     │         ▼
                     │  ┌──────────────┐
                     │  │   Research   │
                     │  │   (Tavily)   │
                     │  └──────┬───────┘
                     │         │
                     └────┬────┘
                          ▼
                   ┌──────────────┐
                   │ Orchestrator │
                   │  (Planner)   │
                   └──────┬───────┘
                          │  fan-out via Send
                          ▼
             ┌────────────────────────┐
             │  Worker · Worker · N   │  ← parallel
             └────────────┬───────────┘
                          ▼
                   ┌──────────────┐
                   │    Merge     │ → files/*.md
                   └──────────────┘
```

---

## How It Works

**Router** classifies the topic into one of three modes before anything else runs. Evergreen concepts skip research entirely. Time-sensitive topics trigger targeted Tavily queries rather than passing the raw topic string - "AI" as a search query returns noise, scoped queries return signal.

**Research** collects results across all queries, then runs a synthesis pass to deduplicate by URL, normalize dates, and discard low-quality sources. Workers downstream receive a clean evidence pack, not raw search output.

**Orchestrator** produces a fully structured plan using Pydantic-enforced schemas - section goals, non-overlapping bullet points, word targets, and per-section flags for citations and code. This contract is what makes parallel writing safe: workers cannot contradict each other when they share a strict plan.

**Workers** run concurrently via LangGraph's `Send` API, one per section. Each receives the full plan for context but writes only its assigned section. In research-backed modes, citation grounding is enforced at the prompt level - workers can only reference URLs from the evidence pack.

**Merge** collects results from all parallel branches, restores section order by task ID (parallel execution does not guarantee order), and assembles the final Markdown document.

---

## Stack

| Layer | Choice |
|---|---|
| Orchestration | LangGraph - typed state machine with native fan-out |
| LLM | Gemini 2.0 Flash - fast and cost-effective across 8+ calls per run |
| Web search | Tavily - built for LLM pipelines, structured results with dates |
| Schema enforcement | Pydantic v2 - validated contracts at every node boundary |
| Frontend | Streamlit - two-panel layout with history browser and streaming output |

---



## Key Concepts

* Agentic routing with dynamic pipeline paths based on topic classification
* Schema-first planning that enforces section contracts before writing begins
* Fan-out parallelism with ordered merge using LangGraph `Send` and annotated reducers
* Evidence grounding that ties citations to verified retrieved URLs only
