# RAG Pipeline for 10M+ Documents With Near-Zero Hallucination

A single, end-to-end, **fully-local** agentic RAG pipeline that turns hallucination into a measurable, calibrated abstention and scales retrieval to **10M+ vectors**. Built and run on **one NVIDIA H100 80GB**, with no external model APIs.

The whole thing is one idea carried end to end: **retrieve, constrain, verify, abstain.** An answer only ever reaches the user if the system can ground every claim in retrieved text, and otherwise it says "I do not have enough evidence."

> **Thesis.** Literal zero hallucination is impossible. *Near-zero on the answerable subset* is achievable by turning unsupported claims into calibrated abstentions: **retrieve → constrain → verify → abstain.**

## Results (measured on one H100)

| Metric | Value |
|---|---|
| Hallucination on unanswerable | **2%** (2 / 100) |
| Faithfulness | **0.908** |
| Context recall@k | **0.97** |
| Coverage on answerable | 0.46 |
| Verifier AUROC (HaluBench) | 0.702 |
| End-to-end latency (p50 / p95) | 4.0 s / 17.7 s |
| **10M-vector ANN search (p95)** | **18 ms** |

The 11-page write-up with all figures and an explicit Scope & Limitations section is in [`report/RAG_Near_Zero_Hallucination_Report.pdf`](report/RAG_Near_Zero_Hallucination_Report.pdf).

## What the pipeline does

Top to bottom, the notebook:

1. **Builds a corpus** from HotpotQA, with SQuAD v2 plus a hand-built false-premise set as the unanswerable stratum, and HaluBench to validate the verifier.
2. **Cleans** it with unicode normalization and MinHash-LSH near-duplicate removal.
3. **Chunks** it structure-aware (whole sentences, 256 tokens / 32 overlap) and adds a one-line **contextual** prefix to each chunk.
4. **Indexes** it as a **hybrid** store: dense vectors in **LanceDB** (on-disk) and sparse **BM25** (bm25s).
5. **Retrieves** with **reciprocal rank fusion**, then **reranks** 150 candidates down to 20 with Qwen3-Reranker.
6. **Routes and decomposes** each query into `no_retrieval`, `single_hop`, or `multi_hop`.
7. **Generates** strictly from the context, with an inline **citation** on every sentence, or an abstain token.
8. **Verifies** every atomic **claim** against its cited context with a faithfulness judge.
9. **Abstains** through a calibrated policy when support is missing.
10. Wires it all into a **LangGraph CRAG / Self-RAG agent** that grades its own evidence and re-retrieves on weak context.
11. **Evaluates** on a 200-question golden set (a 2x2 confusion matrix, a risk-coverage curve, and verifier AUROC on HaluBench).
12. **Scales** the index in a real ANN benchmark from 100k to 10M vectors, with a 100M projection.

**Local model stack (all on the GPU, no external APIs):** Qwen3-32B (generator, served by vLLM), Qwen3-Embedding-4B, Qwen3-Reranker-4B, and the 32B model itself as the faithfulness judge.

## Repository layout

```
rag-zero-hallucinations/
├── notebooks/
│   ├── rag_zero_hallucination.ipynb     # THE notebook: code + the final run's outputs, tables, and plots
│   └── earlier_runs/                     # earlier runs of the same notebook (an NLI-verifier run + a smoke test)
├── figures/                              # the 5 result figures (PNG)
├── results/
│   ├── run_manifest.json                 # config + headline metrics
│   ├── scale_lab.json                    # measured 10M-vector ANN results
│   ├── judge.log                         # the final run log
│   ├── scale.log                         # the 10M scale-benchmark log
│   └── logs/                             # setup and serving logs
├── report/
│   └── RAG_Near_Zero_Hallucination_Report.pdf
└── requirements.txt
```

## Running it

- **To read everything:** open [`notebooks/rag_zero_hallucination.ipynb`](notebooks/rag_zero_hallucination.ipynb). It *is* the project, with every code cell and the final run's outputs, tables, and plots embedded inline.
- **To re-execute it:** clear the outputs and Run All. It needs the full local stack, an NVIDIA GPU (ideally an 80 GB H100), the dependencies in [`requirements.txt`](requirements.txt), the open Qwen3 models pulled from HuggingFace, and a vLLM server for the generator. It is a **GPU-host notebook, not a laptop notebook.** The key knobs live in the `parameters` cell near the top: `SMOKE_TEST`, `SLICE_SIZE`, `GEN_MODEL`, `N_EVAL_ANSWERABLE` / `N_EVAL_UNANSWERABLE`, and `RUN_SCALE_LAB`.

> Secrets (API keys, SSH keys) are intentionally **not** included in this repository.
