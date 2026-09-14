# SOGR: Semantic Object Graph Runtime

**Language:** **English** · [简体中文](README.zh.md)

**Paper:** [English Markdown](paper/SOGR.md) · [简体中文 Markdown](paper/SOGR.zh.md) · [PDF](paper/SOGR.pdf)

A Structured Incremental Architecture for Long-Context Language Modeling  
Independent Research Draft · Version 0.3 · September 2026

| | |
|---|---|
| **Name** | SOGR (Semantic Object Graph Runtime) |
| **Author** | Zeping Tu, frontend engineer |
| **Type** | Research proposal + CPU-laptop baseline |
| **Not** | A GPU-scale training result or a production LLM |
| **Repo** | [github.com/heiyantutu/sogr](https://github.com/heiyantutu/sogr) |

**SOGR** is a proposed language-model runtime that treats semantic structure as a **cacheable intermediate representation**. Instead of rediscovering relationships over tokens at every Transformer layer, the model would compile a **Semantic Object Graph (SOG)**, execute along that graph, and recompute only a **dirty dependency subgraph** when something changes.

The idea comes from **frontend engineering**. **React** and **Vue** compile a structured view, dirty only the dependent subtree, and persist **objects** rather than pixels. SOGR asks whether a language model can use the same contract: compile once, execute incrementally, persist the object rather than a token-level key–value (KV) cache.

> DistilGPT-2 and two 0.5B instruction models on a notebook **without a discrete GPU** are the present experimental floor. They are not evidence about 7B-class models or 16K–128K context.

---

## FAQ

### What is SOGR?

SOGR (Semantic Object Graph Runtime) is an architectural hypothesis for long-context language models. A low-frequency global-attention stage constructs or repairs a Semantic Object Graph. High-frequency graph-conditioned linear operators then propagate state along that graph. A structured cache invalidates dependents the way a reactive UI invalidates a component subtree.

### How is SOGR different from a KV cache?

A conventional KV cache stores one hidden state per token per layer in a **model-private** basis. It is large and usually cannot move to another model. SOGR splits cache into **S** (portable structure), **C** (canonical semantics in a shared encoder space), and **H** (model-bound activations). Cross-model reuse is **lift** \(H=\mathrm{Lift}_\phi(C,S)\), not translation of raw KV tensors.

### Is this inspired by React and Vue?

Yes. The author is a frontend engineer. The claim is not that natural language is a DOM. The transferable idea is the runtime contract: virtual DOM / reactive graph, dirty-subtree updates, skip stable regions, persist objects not pixels.

### Has SOGR been proven at GPU scale?

No. This draft is a proposal plus a laptop CPU baseline. On DistilGPT-2, a 156-token paragraph becomes 54 nodes and 115 edges; serialized \(S\cup C\) is about 184× smaller than a KV snapshot on that toy. Two 0.5B models score 6/6 on full text and 3/6 when given only serialized \(S\). That is a floor, and the graph-as-text test is weak support for portability.

### Does SOGR claim Transformers become \(O(L)\)?

No. Amortized cost still contains a global \(L^2\) term every \(K\) steps. Benefit exists only if the graph stays sparse, repairs are infrequent, and the dirty set stays small.

---

## 1. The problem

Transformer language models represent context as token states and apply attention to infer interactions. That design is powerful because it assumes little about which relationships matter. Its cost is that **relationship discovery stays entangled with information propagation**. Even when dependencies are sparse, dense attention does not expose that sparsity as a first-class object.

Linear attention, state-space models, sparse or chunk attention, hybrid global layers, and paged/prefix KV caches can make attention cheaper. They still treat the **token sequence** or a **model-private tensor** as persistent state. They do not, by themselves, turn language into a **versioned dependency graph** that a runtime can invalidate like a UI component tree.

SOGR’s questions:

1. Can language be compiled into a stable SOG so later layers mostly **execute** the graph?
2. When one entity or fact changes, can the model recompute only the dirty subgraph?
3. If the graph is small, can the user hold the portable part while the server holds weights?
4. Can a new model lift \((S,C)\) instead of inheriting a KV tensor?

---

## 2. Origin: React, Vue, and the runtime contract

The analogy is not that natural language is a DOM. The transferable idea is the **runtime contract** of reactive UI systems.

| Frontend runtime (React / Vue) | SOGR runtime |
|---|---|
| Source / JSX / template | Token sequence / document |
| Compile to virtual DOM or reactive graph | Global Graph Constructor builds SOG |
| Component tree + typed props / edges | Nodes \(V\) + typed relations \(E\) with weights \(A\) |
| `setState` / reactive write dirties dependents | Version bump + \(\mathrm{Desc}_G(\Delta)\) invalidation |
| Skip unchanged subtrees | Hierarchical Block Router skips stable blocks |
| Re-render only the dirty region | Graph-conditioned Linear Executor on the dirty set |
| Persist component state, not pixels | Persist structure \(S\) and canonical vectors \(C\), not raw KV |
| Renderer is replaceable; the object stays | Model \(\phi\) is replaceable; lift \(H = \mathrm{Lift}_\phi(C,S)\) |

A compiler IR / AST is the same split: expensive analysis is infrequent; execution reuses compiled structure. Source and IR can live with the user. Token-level KV cannot: it is large and coupled to one set of weights.

---

## 3. Semantic Object Graph (the IR)

Let \(X = \{x_1,\ldots,x_L\}\) be token embeddings. SOGR introduces

\[
G = (V, E, A)
\]

- \(V\): token, phrase, entity, event, or latent semantic objects
- \(E\): typed relations (coreference, causal, locative, argument, parent/child, …)
- \(A\): confidence or routing weights

The graph may be hierarchical. A minimal node is

\[
N_i = (h_i,\, c_i,\, \mathrm{type}_i,\, \mathrm{span}_i,\, \mathrm{parent}_i,\, \mathrm{version}_i)
\]

\(h_i\) is model-bound; \(c_i\) is an optional canonical vector; type/span/parent are the model-agnostic skeleton. An edge is \(E_{ij} = (r_{ij}, w_{ij}, \mathrm{version}_{ij})\). Later global passes may revise the graph.

A sparse attention mask is usually a per-layer routing decision. In SOGR the graph is **persistent computational state**. Persistence across sessions and models is a stronger, separate hypothesis (Section 5).

**Sketch.** *Alice bought a laptop in Hangzhou because her old computer failed.* A token-only model must re-infer that *her* is Alice and that *old computer* ≠ *laptop*. An SOG can expose objects (`person:Alice`, `event:purchase`, `object:laptop`, `location:Hangzhou`, `event:failure`) and typed edges (`agent`, `theme`, `loc`, `cause`). The graph may be weighted and wrong. The engineering targets are **invalidation locality** and **amortized reuse**, not a perfect parse.

---

## 4. Core runtime (four operators + cache)

The architecture is a loop, not a one-shot parse.

```
tokens X
   │
   ▼
GGC  Global Graph Constructor     expensive, low frequency
   │
   ▼
G = (V,E,A)
   │
   ▼
GLE  Graph-conditioned Linear     cheap, high frequency × K
     Executor                     propagate only along G
   │
   ▼
HBR  Hierarchical Block Router    skip stable contiguous blocks
   │
   ▼
cache M = (S, C, H)               versions; dirty set I
   │  every K steps, or if |I| too large
   ▼
PGR  Periodic Graph Repair        add / drop / split / merge / confirm
```

**GGC** proposes nodes and weighted edges with a high-capacity (often full-attention) module. It must be infrequent or the hypothesis fails.

**GLE** executes the graph, for example

\[
h'_i = U h_i + \sum_{j \in N(i)} w_{ij}\, V_{r_{ij}} h_j
\]

Implementation may be gated linear attention, delta-rule memory, or an SSM. SOGR does not depend on one formula. High-frequency work **executes** \(G\); it does not rediscover \(G\).

**PGR** after \(K\) steps adds, drops, splits, merges, or confirms edges. This is the intended answer to finite-state limits of purely linear models.

**HBR** groups tokens into blocks with summary \(z_b\). Skip when stability \(s_b < \tau\) **and** the dependency version is unchanged. Blocks exist because GPUs prefer contiguous work over arbitrary token sparsity.

**Dirty set** is structural:

\[
I_{t+1} = \mathrm{Desc}_G(\Delta_t) \cup \Delta_t
\]

If \(|I|\) is too large, force PGR.

---

## 5. Three-layer cache \(M = (S, C, H)\)

This is a **deployment hypothesis**, not required for the four-stage runtime.

| Layer | Meaning | Lives where | Portable? |
|---|---|---|---|
| **S** structure | ids, types, spans, parents, typed edges, versions | client / disk | yes (AST-like skeleton) |
| **C** canonical semantics | compact \(c_i\) in a shared encoder, not one LLM | client with S | yes |
| **H** model state | hidden \(h_i\), model routing, block summaries | current serving model | no |

Token KV cannot move: layout is layers, heads, positions, and that model’s weights. Size is not enough: a compressed copy of \(H\) is still model-bound.

Cross-model reuse is lift, not KV translation:

\[
H = \mathrm{Lift}_\phi(C, S)
\]

Full token prefill is the fallback.

A request is \((q, S, C, \Delta)\): lift, extend dirty set, run GLE/PGR on \(I\), return tokens and a patch \(\Delta S,\Delta C\). The client merges. A new model \(\phi'\) inherits \(S\) and \(C\) only: \(H'=\mathrm{Lift}_{\phi'}(C,T(S))\).

Not claimed: that raw KV is linearly alignable across models; that schemas always match; that client edges are trusted.

---

## 6. Conditional complexity

\[
C_{\mathrm{avg}} \approx \frac{c_g L^2}{K} + c_e E + \frac{c_m I}{K}
\]

Useful only if quality holds **and** \(\rho=E/L^2 \ll 1\), \(K\) is large, and average \(I\) is small. Split execution adds communication for \(S\cup C\). Lift must beat prefill at matched quality. These are regimes, not identities.

---

## 7. CPU-laptop baseline

Hardware: non-gaming notebook, **CPU, no CUDA**.

### Token sequence → object graph (`experiments/llm_to_object`)

DistilGPT-2 (~82M) contracts a 156-token paragraph into 54 nodes and 115 edges.

| Quantity | Value |
|---|---|
| Tokens \(L\) | 156 |
| Nodes / edges | 54 / 115 |
| Serialized \(S \cup C\) | ~31 KB |
| KV snapshot | ~5.6 MB |
| Size ratio on this toy | ~184× smaller |
| Edit `Shanghai → Beijing`, 1-hop dirty | 3.7% of nodes |
| Same edit, transitive dirty | 40.7% of nodes |
| Full prefill dirty | 100% of tokens |

Objectification and size hold at this floor. One-hop locality holds. Transitive locality is only partial: an attention graph is denser than a linguistic dependency graph. Extraction has false positives (e.g. “Meanwhile”, “When Bob”).

### Same \(S\) as readable context (`experiments/compiler_lite`)

| Model | Full text | Graph \(S\) only |
|---|---|---|
| Qwen2-0.5B-Instruct | 6/6 | 3/6 |
| Qwen2.5-0.5B-Instruct | 6/6 | 3/6 |

This is compiler-lite (S as text), not a KV compiler. Weak support for portability; serialization currently loses answer-critical structure.

```bash
pip install -r experiments/llm_to_object/requirements.txt
python experiments/llm_to_object/run.py

pip install -r experiments/compiler_lite/requirements.txt
python experiments/compiler_lite/run.py
```

---

## 8. How this would be falsified

In-model efficiency fails if graphs are dense, repairs are too frequent, invalidation is almost global, or sparse kernels lose to dense GEMM.

Portable cache fails if lift cannot approach prefill without rereading most tokens, or if serialized size is not much smaller than KV.

Natural language is not a deterministic AST. Pronouns, discourse, negation, irony, and world knowledge can break local graphs. Safe systems need uncertain edges and forced global repair.

---

## 9. Manuscript and citation

| Path | Role |
|---|---|
| [`README.md`](README.md) / [`README.zh.md`](README.zh.md) | This landing page (English / 简体中文) |
| [`paper/SOGR.md`](paper/SOGR.md) / [`paper/SOGR.zh.md`](paper/SOGR.zh.md) | Full draft |
| [`paper/SOGR.pdf`](paper/SOGR.pdf) | Built PDF |
| [`paper/build_pdf.py`](paper/build_pdf.py) | `python paper/build_pdf.py` |

```bibtex
@misc{tu2026sogr,
  title        = {SOGR: Semantic Object Graph Runtime — A Structured Incremental Architecture for Long-Context Language Modeling},
  author       = {Tu, Zeping},
  year         = {2026},
  note         = {Independent Research Draft, Version 0.3},
  howpublished = {\url{https://github.com/heiyantutu/sogr}}
}
```

This draft claims an architectural hypothesis, a falsifiable experimental program, and a laptop floor. It does not claim that linear attention, sparse attention, graphs, caches, or representation alignment are novel by themselves, and it does not claim proven priority without a prior-art search.
