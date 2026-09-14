# SOGR: Semantic Object Graph Runtime

**A Structured Incremental Architecture for Long-Context Language Modeling**

Independent Research Draft · Version 0.3 · September 2026  
Author: **Zeping Tu**, Frontend Engineer  
Repository: [github.com/heiyantutu/sogr](https://github.com/heiyantutu/sogr)

**SOGR** (Semantic Object Graph Runtime) is a research proposal for language models that treat **semantic structure as a cacheable intermediate representation**, instead of rediscovering relationships over a token sequence at every Transformer layer.

The research direction comes from **frontend engineering**. Reactive frameworks such as **React** and **Vue** compile a structured view (virtual DOM or a reactive dependency graph), dirty only the dependent subtree, and persist **objects** rather than pixels. SOGR asks whether a language model can be organized around the same runtime contract: compile structure once, execute incrementally, persist the object rather than the token-level key–value (KV) tensor.

This repository contains:

- the full manuscript ([`paper/SOGR.md`](paper/SOGR.md), Typst [`paper/SOGR.typ`](paper/SOGR.typ), PDF [`paper/SOGR.pdf`](paper/SOGR.pdf))
- a CPU-laptop baseline that is the **current experimental floor**, not a GPU-scale result

> DistilGPT-2 and two 0.5B instruction models on a notebook **without a discrete GPU** are the present ceiling. Results should be read as a floor, not as evidence about 7B-class models or 16K–128K context.

**Keywords:** Semantic Object Graph Runtime; SOGR; incremental language modeling; structured KV cache alternative; portable working memory; React Vue inspired LLM architecture; dirty subgraph recomputation; graph-conditioned linear attention; long-context language model; model-agnostic intermediate representation; client-side object cache; lift instead of KV translation

---

## 1. The problem SOGR is trying to solve

Transformer language models represent context as a sequence of token states and apply attention to infer interactions among those states. That design is powerful because it makes few assumptions about which relationships matter. Its cost is that **relationship discovery is repeatedly entangled with information propagation**. Even when the effective dependency pattern is sparse, dense attention does not expose that sparsity as a first-class computational object.

Existing efficiency lines (linear attention, state-space models, sparse or chunk attention, hybrid periodic global attention, paged/prefix KV caches) still treat the **token sequence** or a **model-private hidden tensor** as the persistent state. They can make attention cheaper. They do not, by themselves, turn language into a **versioned dependency graph** that a runtime can invalidate the way a UI framework invalidates a component tree.

SOGR’s design question is therefore not “how can Transformer attention be made slightly cheaper?” It is:

1. Can language be compiled into a stable **Semantic Object Graph (SOG)** so that later layers mostly **execute** that graph instead of rediscovering it?
2. When a local entity, fact, or relation changes, can the model recompute only the **dirty dependency subgraph**?
3. If that graph is small and structurally explicit, can the user hold the portable part locally while the server holds only weights and sells graph-conditioned computation?
4. Can a new model **lift** structure plus canonical semantics into its own hidden space, instead of inheriting a raw KV tensor?

---

## 2. Origin: React, Vue, and the runtime contract

The author is a **frontend engineer**. The analogy is not that natural language is a DOM. The transferable idea is the **runtime contract** of reactive UI systems.

| Frontend runtime (React / Vue) | SOGR runtime |
|---|---|
| Source / JSX / template | Token sequence / document |
| Compile to virtual DOM or reactive graph | Global Graph Constructor builds SOG |
| Component tree + typed props / edges | Nodes \(V\) + typed relations \(E\) with weights \(A\) |
| `setState` / reactive write dirties dependents | Version bump + `Desc_G(Δ)` invalidation |
| Skip unchanged subtrees (block skip / memo) | Hierarchical Block Router skips stable blocks |
| Re-render only the dirty region | Graph-conditioned Linear Executor on the dirty set |
| Persist component state, not pixels | Persist structure \(S\) and canonical vectors \(C\), not pixels and not raw KV |
| Framework renderer is replaceable; the object stays | Model \(\phi\) is replaceable; lift \(H = \mathrm{Lift}_\phi(C,S)\) |

A second relative is a **compiler IR / AST**: expensive parse and analysis happen infrequently; execution reuses the compiled structure. Source and IR can live with the user; the compiler can be remote. Token-level KV cannot play that role: it is large and coupled to one set of weights.

---

## 3. Semantic Object Graph (the IR)

Let a sequence of \(L\) token embeddings be \(X = \{x_1,\ldots,x_L\}\). SOGR introduces a graph

\[
G = (V, E, A)
\]

- \(V\): token, phrase, entity, event, or latent semantic **objects**
- \(E\): typed dependency relations (coreference, causal, locative, argument, parent/child, …)
- \(A\): relation confidence or routing weights

The graph may be hierarchical: token nodes belong to phrase nodes, which belong to sentence or block nodes.

A minimal node is

\[
N_i = (h_i,\, c_i,\, \mathrm{type}_i,\, \mathrm{span}_i,\, \mathrm{parent}_i,\, \mathrm{version}_i)
\]

- \(h_i\): current **model-bound** hidden state
- \(c_i\): optional **canonical** semantic vector (not identified with one LLM)
- \(\mathrm{type}, \mathrm{span}, \mathrm{parent}\): model-agnostic skeleton
- \(\mathrm{version}\): cache invalidation

A minimal edge is \(E_{ij} = (r_{ij}, w_{ij}, \mathrm{version}_{ij})\). Unlike a frozen syntactic parse, later global passes may add, drop, split, merge, or reweight edges.

**Distinction from sparse attention.** A sparse mask is usually a per-layer routing decision. In SOGR the graph is **persistent computational state**: an index for incremental recomputation across many linear steps, and (as a stronger, separate hypothesis) a candidate for storage across sessions and models.

### Worked sketch

Sentence: *Alice bought a laptop in Hangzhou because her old computer failed.*

A token-only model must re-infer that *her* is Alice, that *old computer* is distinct from *laptop*, and that the *because* clause explains the purchase. An SOG can expose candidate objects (`person:Alice`, `event:purchase`, `object:laptop`, `location:Hangzhou`, `object:old_computer`, `event:failure`) and typed edges (`agent`, `theme`, `loc`, `cause`). The graph is allowed to be weighted, uncertain, and revisable. Perfection is not required; **invalidation locality** and **amortized reuse** are the engineering targets.

---

## 4. Core runtime logic (four operators + cache)

The architecture is a loop, not a one-shot parse.

```
tokens X
   │
   ▼
┌─────────────────────────────────┐
│ GGC  Global Graph Constructor   │  expensive, low frequency
│      full / high-capacity attn  │  proposes nodes + weighted edges
└──────────────┬──────────────────┘
               ▼
            G = (V,E,A)
               │
               ▼
┌─────────────────────────────────┐
│ GLE  Graph-conditioned Linear   │  cheap, high frequency  × K
│      Executor                   │  propagate only along G
└──────────────┬──────────────────┘
               ▼
┌─────────────────────────────────┐
│ HBR  Hierarchical Block Router  │  skip stable contiguous blocks
│      summaries z_b, score s_b   │
└──────────────┬──────────────────┘
               ▼
┌─────────────────────────────────┐
│ Structured cache M = (S, C, H)  │  versions; dirty set I
└──────────────┬──────────────────┘
               │  every K steps, or if |I| too large
               ▼
┌─────────────────────────────────┐
│ PGR  Periodic Graph Repair      │  add/drop/split/merge/confirm
└─────────────────────────────────┘
```

### 4.1 Global Graph Constructor (GGC)

A full-attention or otherwise high-capacity module examines current state and proposes nodes and weighted relations. The expensive operation is **intentionally infrequent**. One graph is not assumed to remain correct indefinitely.

### 4.2 Graph-conditioned Linear Executor (GLE)

Given \(G\), a linear operator propagates state only along graph-supported relations. A generic form is

\[
h'_i = U h_i + \sum_{j \in N(i)} w_{ij}\, V_{r_{ij}} h_j
\]

followed by a gated update. Implementation may use gated linear attention, delta-rule memory, SSM-like transitions, or another linear-time mechanism. SOGR does **not** depend on one linear-attention formula. The requirement is: **execute the graph**, do not rediscover it at every layer.

### 4.3 Periodic Graph Repair (PGR)

After \(K\) linear steps, a global module re-evaluates \(G\): add edges, remove stale edges, alter weights, split or merge nodes, or confirm validity. This is the intended answer to finite-state and retrieval limits of purely linear models.

### 4.4 Hierarchical Block Router (HBR)

Tokens are grouped into blocks. Block \(b\) stores a compact summary \(z_b\) and a change indicator. Stability score \(s_b = g(z_b, q)\). Skip fine-grained work when \(s_b < \tau\) **and** the dependency version is unchanged. The routing unit is **block-sized** because contiguous execution is more GPU-friendly than arbitrary token-level sparsity.

### 4.5 Structured cache and dirty-set invalidation

The cache stores node states, edge states, block summaries, and versions. A change to node \(i\) increments its version. The dirty set is structural, not merely positional:

\[
I_{t+1} = \mathrm{Desc}_G(\Delta_t) \cup \Delta_t
\]

Recompute \(I_{t+1}\). If invalidation grows past a threshold, force PGR (global repair) instead of pretending locality still holds.

Internally this is the incremental-computation store. Externally only a subset is a candidate for client-side persistence (next section).

---

## 5. Three-layer cache: \(M = (S, C, H)\)

This is a **deployment hypothesis**, not a required property of the four-stage runtime. It asks whether the same IR can become a portable working-memory object.

| Layer | What it is | Where it lives | Portable? |
|---|---|---|---|
| **\(S\)** structure | node ids, types, spans, parents, typed edges, versions | client / disk / across sessions | yes (skeleton, like an AST) |
| **\(C\)** canonical semantics | compact \(c_i\) in a **shared** encoder space, not one LLM | client with \(S\) | yes (payload that still carries meaning) |
| **\(H\)** model state | hidden \(h_i\), model routing weights on edges, block summaries \(z_b\) | scratch on the serving model | no (discard when weights change) |

**Why token KV cannot move.** A conventional KV cache is a stack of tensors whose layout is layer count, head geometry, positional encoding, and the weight matrices that produced it. It stores one state per token per layer. It is large and lives in a **private basis**. Shipping it to the client is usually pointless; a different model cannot consume it.

**Size is not enough.** If the locally stored object is merely a compressed copy of \(H\), it remains model-bound. Portability requires the split above.

Cross-model reuse is **not KV translation**. It is **lift**:

\[
H = \mathrm{Lift}_\phi(C, S)
\]

\(\mathrm{Lift}_\phi\) is a model-specific projection, optionally plus cheap local repair on uncertain edges. Full token prefill is the **fallback** when lift is insufficient, not the default path.

### Client object, server compute

A request can be written \((q, S, C, \Delta)\):

1. Server lifts \(C\) into \(H\) for current weights \(\phi\)
2. Extends invalidation \(I = \mathrm{Desc}_S(\Delta) \cup \Delta\)
3. Runs GLE and, if needed, PGR only on \(I\)
4. Returns tokens plus a versioned patch \(\Delta S, \Delta C\)
5. Client merges the patch

Long-lived context need not occupy HBM between calls. This **inverts** prefix-cache design: prefix caches keep tensors near the weights because tensors are huge; SOGR keeps objects near the user because objects are structured and, in the sparsity regime of the complexity section, small.

A new model \(\phi'\) does **not** inherit \(H\). It inherits \(S\) and \(C\), then \(H' = \mathrm{Lift}_{\phi'}(C, T(S))\), where \(T\) is a schema map if relation vocabularies differ. Residual uncertainty is PGR’s job.

**What is not claimed.** Raw per-layer keys and values are not assumed linearly alignable across models. A code model and a dialogue model may induce different node granularities; then \(S\) must be rebuilt or reconciled. Client-supplied edges are not trusted by default.

---

## 6. Conditional complexity (not “Transformer becomes \(O(L)\)”)

If global construction occurs every \(K\) steps, amortized cost is roughly

\[
C_{\mathrm{avg}} \approx \frac{c_g L^2}{K} + c_e E + \frac{c_m I}{K}
\]

Advantage versus dense attention exists only when quality is preserved **and** \(\rho = E/L^2 \ll 1\), \(K\) is large enough, and average invalidation \(I\) stays small. If \(E = \rho L^2\) with \(\rho\) not small, the graph term is still quadratic.

Split execution adds communication \(c_s s\) for serialized \(S \cup C\) or a patch. Cross-model reuse replaces a prefill on the order of \(c_g L^2\) with lift on the order of \(c_\ell |V|\) plus repair, only when that is cheaper **at matched quality**.

These are **regimes**, not identities.

---

## 7. CPU-laptop baseline (experimental floor)

Hardware: non-gaming notebook, **CPU, no CUDA**. This is the maximum the present machine can honestly support.

### 7.1 Token sequence → object graph (`experiments/llm_to_object`)

DistilGPT-2 (~82M) on CPU contracts a 156-token Alice/Hangzhou paragraph into **54 nodes** and **115 edges**.

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

**Reading.** Objectification and size reduction hold at this floor. One-hop locality holds. Transitive locality is only partial, because an **attention graph is denser than a linguistic dependency graph**. Entity extraction also produces false positives (e.g. “Meanwhile”, “When Bob”). This is a representation toy, not a quality evaluation of DistilGPT-2 as an SOGR model.

### 7.2 Same \(S\) as readable context (`experiments/compiler_lite`)

The serialized graph (not a compiled KV) is fed to two 0.5B instruct models versus the raw paragraph.

| Model | Full text | Graph \(S\) only |
|---|---|---|
| Qwen2-0.5B-Instruct | 6/6 | 3/6 |
| Qwen2.5-0.5B-Instruct | 6/6 | 3/6 |

This is **compiler-lite**, not a KV compiler. It is **weak support** for portability and a warning that current extraction/serialization **loses answer-critical structure**. It does not replace GPU-scale perplexity, long-context retrieval, or true \(S{+}C \to H\) lift.

Reproduce:

```bash
pip install -r experiments/llm_to_object/requirements.txt
python experiments/llm_to_object/run.py

pip install -r experiments/compiler_lite/requirements.txt
python experiments/compiler_lite/run.py
```

---

## 8. How this would be falsified

In-model efficiency fails if graphs are dense, repairs are too frequent, invalidation is almost global, or sparse graph kernels lose to dense GEMM.

Portable cache fails if lift cannot approach prefill quality without rereading most tokens, or if serialized \(s\) is not substantially smaller than KV.

Natural language is not a deterministic AST: pronouns, discourse, negation, irony, and world knowledge can invalidate an apparently local graph. Safe systems need uncertainty-aware edges and forced global repair.

---

## 9. Manuscript and citation

| Path | Role |
|---|---|
| [`paper/SOGR.md`](paper/SOGR.md) | Full draft (Markdown, intended for indexing and reading) |
| [`paper/SOGR.typ`](paper/SOGR.typ) | Typst source |
| [`paper/SOGR.pdf`](paper/SOGR.pdf) | Built PDF |
| [`paper/build_pdf.py`](paper/build_pdf.py) | Rebuild: `python paper/build_pdf.py` |

```bibtex
@misc{tu2026sogr,
  title        = {SOGR: Semantic Object Graph Runtime — A Structured Incremental Architecture for Long-Context Language Modeling},
  author       = {Tu, Zeping},
  year         = {2026},
  note         = {Independent Research Draft, Version 0.3},
  howpublished = {\url{https://github.com/heiyantutu/sogr}}
}
```

This draft claims an **architectural hypothesis** and a **falsifiable experimental program**, plus the current laptop floor. It does not claim that every constituent technique (linear attention, sparse attention, graphs, caches, hybrid layers, representation alignment) is novel, and it does not claim proven priority without a comprehensive prior-art search.
