# GenAI Engineering Syllabus

A study guide for the 18 notebooks in this folder — covering image generation, retrieval-augmented generation (RAG), tool-calling agents, and evaluation/tracing.

Live version with cards, tags, and a suggested 4-week cadence: https://claude.ai/code/artifact/30f416c3-70ee-40ff-a752-6ed6d5ddc534

## How to work through this

The notebooks are ordered so each one assumes the last. Don't jump to the agent notebooks before finishing Track B — every agent below calls a RAG tool it expects you to already understand.

1. **Run it before you read it.** Open the notebook in Colab, run every cell top to bottom once with no edits. Get it working before studying why.
2. **Re-run cell by cell, reading the markdown between them.** Most notebooks explain their own code in prose cells — that's the primary material.
3. **Break it on purpose.** Change the prompt, swap the document, delete a tool. Notebooks in Track C and D include explicit student exercises — do those before moving on.
4. **Get five free API keys before Track B:** OpenRouter, SerpAPI, OpenWeatherMap, LangSmith, and Langfuse. Every notebook past Track A requests one of these via `getpass` at runtime; none are hardcoded.

**GPU note:** Notebooks 01 and 02 (Stable Diffusion) need a GPU runtime — in Colab, Runtime → Change runtime type → T4 GPU. Everything from Track B onward runs fine on CPU.

## Suggested four-week cadence

| Week | Track | Notebooks | You should be able to say, by the end |
|---|---|---|---|
| 1 | A — Foundations | 01, 02 | "I can explain the diffusion loop and steer output with prompt structure alone." |
| 2 | B — Retrieval & RAG | 03, 04, 05, 06, 18 | "I can build a FAISS-backed RAG chain and explain why a cross-encoder reranks better than a bi-encoder alone." |
| 3 | C — Tools & Agents | 08, 09, 10, 11, 12, 16, 17 | "I can wire a tool-calling loop by hand, and I know when to reach for LangGraph vs. CrewAI vs. AutoGen vs. smolagents." |
| 4 | D — Tracing & Evaluation | 07, 13, 14, 15 | "I can trace an agent run end-to-end and score it with local metrics and an LLM judge." |

---

## Track A — Generative Foundations

Two short notebooks on image generation. Not required for the RAG/agent tracks, but they establish the "text in, media out" mental model that reappears later as "the LLM call."

### 01 — Introduction to Image Generation with Stable Diffusion Pipeline
Loads a pretrained Stable Diffusion pipeline from Hugging Face `diffusers`, moves it to GPU, and generates images from a list of text prompts.
- **Needs:** GPU runtime (T4). No API key.
- **Installs:** `diffusers[torch] transformers accelerate safetensors xformers`

### 02 — Prompt Engineering with Stable Diffusion
Same pipeline as 01, varying prompt wording, style keywords, and negative prompts side by side to show how phrasing changes output.
- **Needs:** GPU runtime. No API key.
- Pairs with 01 — run in the same session so the pipeline stays loaded.

---

## Track B — Retrieval & RAG

The backbone of everything that follows: embeddings → vector stores → retrieval → reranking → full RAG chains → conversational memory. Every agent in Track C uses a RAG tool built from these pieces.

### 03 — Understanding Vector Db
Ground-up explanation of embeddings and vector stores: what an embedding is, why similarity search works, hands-on similarity search with relevance scores using Chroma and FAISS. Includes hands-on exercises.
- **Installs:** `chromadb faiss-cpu sentence-transformers langchain`

### 04 — LangChain Basics – From LLM Calls to RAG
The core LangChain on-ramp: bare LLM call → prompt template → LCEL chain → output parser → a full document-answering RAG chain. The single most important notebook for LangChain syntax. Includes 4 hands-on exercises.
- **Installs:** `langchain langchain-core langchain-community langchain_openai faiss-cpu`

### 05 — Cohere Embeddings with FAISS Vector Store and QA
A second, self-contained RAG build using Cohere embeddings instead of Hugging Face sentence-transformers: load a PDF, chunk it, embed with Cohere, index in FAISS, answer questions. Compare against notebook 04's embedding choice.
- **Needs:** Cohere API key.
- **Installs:** `cohere faiss-cpu langchain-community pypdf PyPDF2`

### 06 — Bi-Encoder and Cross-Encoder
Answers the question notebook 04 leaves open: naive similarity search is fast but imprecise. Shows a bi-encoder for first-pass retrieval, a cross-encoder for reranking, and a hybrid BM25 + bi-encoder search combined by reciprocal rank fusion (RRF).
- **Needs:** No API key.
- **Installs:** `rank_bm25 sentence-transformers`

### 18 — LangChain Memory
Surveys LangChain's conversational memory types — buffer, summary, windowed — and the context/performance tradeoff each implies. Its ideas resurface directly in notebook 10's chat-history variations.
- **Installs:** `langchain-community langchain-core langchain-openai`

---

## Track C — Tools & Agents

From a single hand-rolled tool call to four different multi-agent frameworks. Read notebook 08 first regardless of order — it's the only one that builds the tool-calling loop from raw HTTP, before any framework hides it from you.

### 08 — OpenRouter API: JSON Output, Pydantic Validation, and Weather Tool Calling
Builds from a basic OpenRouter chat call, to JSON-mode output, to a Pydantic-validated schema, then implements the full tool-calling loop by hand: question → LLM picks a tool → Python executes it → result returns to the LLM → final answer. No framework — this is what every later notebook automates.
- **Needs:** OpenRouter API key.
- **Installs:** `openai pydantic requests groq`

### 09 — LangChain Agent Class Demo — Base with Student Exercise
A LangChain agent choosing between three tools: web search (SerpAPI), live weather (OpenWeatherMap), and RAG over an uploaded PDF (FAISS). Ends with demo queries, a student exercise, and a teaching explanation of the agent's internal decision process.
- **Needs:** OpenRouter, SerpAPI, and OpenWeatherMap keys.
- **Installs:** `langchain-community langchain-openrouter langchain-huggingface faiss-cpu google-search-results`

### 10 — LangChain Agent Class Demo — Chat History Variations
Identical setup to 09, but the second half explores chat history handling instead of demo queries — a direct continuation of notebook 18's memory concepts inside a live tool-using agent. Run right after 09; same setup cells, different ending.

### 11 — CrewAI Travel Planner — Three Orchestration Styles
The same travel-planning problem solved three ways: a Sequential crew (fixed task order), a Hierarchical crew (a manager delegates to specialists, with an optional SerpAPI tool), and a Flow with an explicit router. Ends with a one-line mental map of when to use each.
- **Installs:** `crewai crewai-tools[serpapi]`

### 12 — LangGraph Agentic Workflow: Product Description Generator
Introduces LangGraph's vocabulary — shared state, nodes, edges — first as a sequential graph, then rebuilt with parallel fan-out nodes, a fan-in merge, conditional edges, and a loop. Finishes with a Gradio UI. This exact "advanced" graph is what notebooks 13, 14, and 15 trace and evaluate.
- **Installs:** `langgraph langchain gradio`

### 16 — AutoGen: Basic to Advanced — Classroom Annotated Notebook
The longest notebook in the set. UserProxyAgent talking to an AssistantAgent, then human-in-the-loop troubleshooting, dynamically configured agents, GroupChat multi-agent collaboration, an AutoGen+LangGraph integration for a retail-support workflow, a tool-enabled agent, and an OCR-based multi-agent contract analysis pipeline.
- **Installs:** `pyautogen langgraph pytesseract pdf2image`

### 17 — smolagents Agentic Architecture — From One Agent to Multi-Agent
The most beginner-friendly agent framework tour. Explains agent architecture conceptually, builds custom tools with `@tool`, a `CodeAgent`, compares it to a `ToolCallingAgent`, adds conversational memory, and builds a manager–specialist multi-agent system. Closes with a production checklist and 5 graded exercises.
- **Needs:** A Hugging Face-compatible LLM endpoint.

---

## Track D — Tracing & Evaluation

Do these last — every notebook here evaluates or traces an agent built earlier. Notebooks 13, 14, and 15 all instrument the same "advanced" LangGraph workflow from notebook 12, so the fairest comparison is tracing one run through all three back to back.

### 07 — RAG Evaluation with Local Metrics, Custom LLM Judge, and RAGAS
Evaluates one retrieval system three ways, in this order: (1) local non-LLM metrics — semantic similarity, ROUGE, string similarity, context hit; (2) a hand-built GPT-4o-mini judge via OpenRouter; (3) the RAGAS framework's LLM-based metrics — faithfulness, answer relevancy, answer correctness, context precision, context recall. The ordering shows what a framework buys you over a hand-rolled judge.
- **Installs:** `ragas rouge_score rapidfuzz sentence-transformers`

### 13 — Langsmith Tracing
Pure code, no commentary cells. Re-runs notebook 12's advanced parallel/looping LangGraph workflow with LangSmith tracing wired in, so every node execution appears as a trace in the LangSmith dashboard.
- **Needs:** LangSmith API key.
- Read alongside notebook 15, which explains LangSmith's dataset/target/evaluator model in prose.

### 14 — Langfuse Tracing
Same base workflow as 13, swapping in a Langfuse `CallbackHandler` instead of LangSmith. Run both back to back to compare the two platforms on identical input.
- **Needs:** Langfuse public/secret keys.

### 15 — Evaluating the Advanced LangGraph Product Agent with LangSmith
Where 13 just traces, this one evaluates: builds a small LangSmith dataset, defines a target function wrapping the agent, writes four custom evaluators, runs the evaluation, and walks through reading the resulting scores in the LangSmith UI. Ends with an optional classroom experiment.
- **Installs:** `langgraph langchain-openai langsmith`

---

*Compiled from each notebook's own markdown cells. Source notebooks live in this folder.*
