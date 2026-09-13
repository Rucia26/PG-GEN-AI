# The Agentic AI Engineer Path — Zero to Senior

A from-scratch curriculum for a working software engineer with no LLM background, aimed at senior-engineer depth: not just building an agent that works once, but knowing why it works, when it will fail, and what to do differently in production.

Live version with the full visual layout, module cards, and a junior-vs-senior comparison table: https://claude.ai/code/artifact/09033507-9aca-4a65-ab8b-c38f679534b9

**Assumes:** a working software engineer comfortable with Python, APIs, git, and basic cloud/database concepts — "zero" means zero LLM/agent background, not zero programming.

**Scope:** 6 phases, ~34 weeks at 8–10 hrs/week, one capstone system. Vendor-neutral — frameworks (LangGraph, CrewAI, AutoGen, etc.) are taught as interchangeable implementations of the same underlying patterns, not as the curriculum itself.

## What "senior" means here

Junior-level agentic AI work is getting a demo to run: an agent that answers three questions correctly. Senior-level work is different in kind, not degree — it's knowing which of several architectures fits a given problem, why a naive RAG pipeline will silently degrade at scale, how to bound an agent's cost and blast radius before it ships, and how to debug a system whose failures are probabilistic rather than deterministic.

Every phase ends with a deliverable you build, not a notebook you run — and each includes a "senior lens" naming the judgment calls a senior engineer is expected to make that a junior engineer typically hasn't been burned by yet.

---

## Phase 1 (Weeks 1–4): LLM Foundations & Prompting as Engineering

Before any agent, understand the thing agents are built on: a stateless text-completion API with a context window, a cost, and no memory of its own.

- **How LLMs actually work as an API call** — tokens, context windows, temperature/top-p, statelessness, autoregressive generation and why it explains both fluency and failure modes.
- **Prompting as a versioned engineering artifact** — system/user/assistant roles, few-shot examples, chain-of-thought and when it's worth the tokens, prompts as tested and versioned code.
- **Structured output & schema validation** — JSON mode, schema-constrained generation, Pydantic validation, retry-on-failure as a reusable pattern.
- **Cost, latency, and model selection** — token economics, model tiering, streaming vs. blocking, logging cost per call from day one.

**Deliverable:** A CLI tool that classifies unstructured text into a validated, schema-constrained JSON output, with logged cost per call and automatic retry on schema failure.

**Senior lens:** Build a labeled eval set on day one instead of eyeballing three examples — prompt "improvements" often regress cases you didn't check. Pick the model based on what a wrong answer costs, not out of habit.

---

## Phase 2 (Weeks 5–10): Retrieval, Grounding & RAG

Agents are only as good as the information they can reach. Covers getting correct, current, private-data answers — and all the ways that pipeline quietly degrades.

- **Embeddings & vector search** — what embeddings encode, cosine vs. dot product, ANN search (HNSW, IVF), and why "similar" isn't always "relevant."
- **Chunking strategy** — fixed-size vs. semantic vs. structure-aware chunking, overlap, and how naive splitting destroys document structure.
- **Building a full RAG pipeline** — ingest → chunk → embed → index → retrieve → augment → generate, with citations back to source chunks.
- **Why naive RAG breaks — and how to fix it** — recall failures, the lost-in-the-middle effect, hybrid search (keyword + vector via RRF), cross-encoder reranking, query rewriting, metadata filtering.

**Deliverable:** A RAG service over a real, messy document set with hybrid retrieval, reranking, and inline citations, plus a written failure log of 10+ queries it got wrong.

**Senior lens:** Treat retrieval quality as an ongoing measurement problem (recall@k, precision on a real query log), not a solved primitive. Know when RAG is the wrong tool — a long context window or fine-tuning can beat retrieval's operational complexity for small, stable knowledge bases.

---

## Phase 3 (Weeks 11–17): Tool Use & Single-Agent Architecture

The core idea of "agentic": an LLM deciding which action to take next, in a loop, until the task is done. Build this from raw primitives before naming any framework.

- **Function/tool calling from first principles** — how tool calling works under the hood; build the loop by hand with raw API calls first.
- **Reasoning & planning patterns** — ReAct, plan-and-execute, reflection/self-critique, and their real tradeoffs (adaptability vs. token cost vs. brittleness).
- **State & memory design** — short-term (buffer, summarization, sliding window) vs. long-term memory, and what should persist, be discarded, or never be stored.
- **Graph-based agent orchestration** — modeling an agent as an explicit state graph: nodes, edges, conditional branching, parallel fan-out/fan-in, cycles with exit conditions.
- **Framework survey** — LangGraph, CrewAI, AutoGen, and a code-execution framework (e.g. smolagents), mapped back to the patterns above rather than memorized as separate APIs.

**Deliverable:** A single agent with 3+ real tools (one live external API, one RAG query, one that can fail), built first as a hand-rolled loop, then reimplemented in a graph-based framework, with a comparison write-up.

**Senior lens:** Being able to build the loop by hand is what makes debugging the framework possible later — there's no magic left to be confused by. The first design question is always: what happens when a tool call fails, times out, or returns garbage?

---

## Phase 4 (Weeks 18–23): Multi-Agent Systems & Orchestration

Splitting one agent into several only pays for itself sometimes. Most systems calling themselves "multi-agent" would be simpler and more reliable as one well-designed agent.

- **Why (and when) to go multi-agent** — real justifications (separating concerns, parallelizing independent subtasks, isolating risky actions) vs. non-justifications (it sounded more impressive).
- **Orchestration topologies** — sequential, hierarchical, peer-to-peer debate, blackboard/shared-state — build the same task three ways to feel the tradeoffs.
- **Handoffs, routing & shared context** — what transfers across a handoff, why over-sharing context is its own failure mode, explicit vs. LLM-decided routing.
- **Cross-agent protocols** — MCP for tool/context exposure, A2A-style protocols for agent interoperability, and why standardizing this matters once agents call agents built by other teams.
- **Failure modes unique to multi-agent systems** — infinite handoff loops, compounding hallucination, combinatorial cost/latency blowup.

**Deliverable:** A 3–5 agent system solving a task your Phase 3 single agent genuinely couldn't do well alone, with a design doc justifying the topology and instrumented loop/cost limits.

**Senior lens:** The most senior-differentiating skill here is being willing to say "this should be one agent" and cut a multi-agent design down. Design a hard ceiling on agent-to-agent turns and cost before shipping, not after a runaway bill.

---

## Phase 5 (Weeks 24–28): Evaluation, Observability & Reliability

The phase most tutorials skip and most production incidents trace back to. Agentic systems fail probabilistically and silently — you need to see inside them and measure continuously.

- **Tracing every step** — instrumenting every LLM call, tool call, and state transition (LangSmith, Langfuse, OpenTelemetry-based tracing) so a failed run can be replayed step by step.
- **Offline evaluation** — labeled golden datasets, non-LLM metrics and their limits, regression testing prompts/agents like code before every deploy.
- **LLM-as-judge — and its failure modes** — calibrating a judge model against human ratings, correcting for position bias, verbosity bias, self-preference bias.
- **Online monitoring & guardrails** — PII/toxicity/off-topic guardrails, anomaly detection on cost/latency/loop-count, human-in-the-loop escalation paths.
- **Agent-specific failure taxonomy** — infinite tool-call loops, hallucinated tool arguments, silent partial completion, context poisoning from a bad early result.

**Deliverable:** Full tracing and an automated eval suite wired into your Phase 4 system, plus a monitoring dashboard tracking cost/latency/failure rate with justified alert thresholds.

**Senior lens:** Ask "how will I know when this silently breaks in three weeks, for a case I never tested?" before shipping, not after. Distrust a single aggregate eval score — insist on segment-level breakdowns, since a 92% average often hides a subgroup failing 100% of the time.

---

## Phase 6 (Weeks 29–34): Production Engineering & Senior-Level Judgment

The phase that's actually about being senior: security, cost discipline, deployment architecture, and the judgment to know when an agent is the wrong solution.

- **Security: prompt injection & tool sandboxing** — direct/indirect prompt injection, the confused-deputy problem, sandboxing code-execution tools, least-privilege tool design.
- **Deployment architecture** — sync vs. async/queued execution, streaming partial results, idempotency and retries for actions with real side effects, scaling stateful agent runs.
- **Cost engineering at scale** — prompt caching, model routing (cheap model first, escalate on low confidence), batching, per-user/per-tenant cost ceilings.
- **Human-in-the-loop & escalation design** — approval gates for high-stakes actions, confidence-based escalation, and designing the handoff back to the agent after human intervention.
- **System design for agentic products** — practicing single- vs. multi-agent, RAG vs. fine-tuning vs. long-context, sync vs. async tradeoffs against a skeptical reviewer.

**Deliverable:** A production readiness review of your capstone system: a written threat model for prompt injection, a cost ceiling implementation with a test proving it triggers, and an architecture decision record defending every major choice against a stated alternative.

**Senior lens:** Be able to name, unprompted, three scenarios where the right answer is "don't use an agent for this" — a deterministic script or rules engine is often more reliable, cheaper, and easier to audit. Treat the threat model and cost ceiling as part of the deliverable, not hardening bolted on after a security review.

---

## Junior vs. senior, in one table

| Dimension | Junior-level habit | Senior-level habit |
|---|---|---|
| Evaluation | Eyeballs a few outputs, ships if they look right. | Maintains a versioned eval set; treats prompt changes like code changes requiring regression tests. |
| Architecture | Defaults to the most sophisticated pattern available regardless of need. | Starts from the simplest architecture that could work; justifies every added layer of complexity. |
| Failure handling | Designs the happy path; tool failures crash or hang silently. | Designs failure paths first: timeouts, retries, fallback tools, a defined escalate-to-human state. |
| Cost | Discovers the bill after it's large. | Sets per-request/per-tenant cost ceilings before launch, with alerts before the ceiling. |
| Security | Trusts retrieved documents and tool output as safe input. | Treats all retrieved/tool content as untrusted; designs against indirect prompt injection. |
| Debugging | Re-runs the whole pipeline and hopes the bug reproduces. | Pulls the exact trace of the failed run and inspects the step that diverged. |

---

## Capstone: ship one real agentic system, end to end

Pick a problem with real stakes, not another customer-support demo — something with genuine tool actions (booking, filing, purchasing, editing production data) where a wrong action has a real cost.

- [ ] **Multi-step, multi-tool agent** with at least one tool call that has a genuine external side effect, not just reads.
- [ ] **Grounded in retrieval** from a real, non-trivial knowledge source, with citations.
- [ ] **Full tracing and an automated eval suite** that runs on every change before deploy.
- [ ] **A hard cost ceiling and a loop/turn limit**, with a test proving both actually trigger.
- [ ] **A written threat model** covering prompt injection and confused-deputy risk for its specific tools.
- [ ] **A human-in-the-loop approval gate** for its highest-stakes action.
- [ ] **An architecture decision record** defending single- vs. multi-agent, and RAG vs. alternatives, against a stated rejected alternative.

---

## Reference shelf

Not required reading before starting — pull these in per-phase as primary sources, since specific tools and frameworks will keep changing under these concepts.

- **Phase 1–2:** Your model provider's own prompting and structured-output docs — they change faster than any book.
- **Phase 2:** "Retrieval-Augmented Generation for Knowledge-Intensive NLP Tasks," plus recent surveys on hybrid retrieval and reranking.
- **Phase 3:** The original ReAct paper, plus LangGraph/CrewAI/AutoGen architecture docs read as pattern references.
- **Phase 4:** The MCP specification and any published agent-to-agent interoperability spec, read as engineering documents.
- **Phase 5:** LangSmith or Langfuse documentation, read cover to cover once.
- **Phase 6:** OWASP's LLM application security guidance, plus published indirect prompt-injection case studies.

---

*A phase-based curriculum, not a fixed calendar — move faster through what you already half-know, and slower through Phase 5, which is where most self-taught engineers are weakest.*
