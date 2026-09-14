# DistilGPT-2 → Semantic Object Graph (CPU representation toy)

Compared with SOGR draft 0.2: §3 object graph, §5 three-layer cache / portable objects, §6–7 subgraph invalidation, and §10 experiments 8 plus edit locality.
This is **not** a model-quality evaluation. It only checks whether LLM state can be contracted into objects, and whether those objects are small enough to store locally.

## 1. Input

- Model: `distilgpt2` (CPU, 0.104 s)
- Edit: `Shanghai -> Beijing`

Source:

```
Alice bought a laptop in Hangzhou because her old computer failed.
The laptop was delivered to her office near West Lake two days later.
Bob, who works in Shanghai, called Alice to ask whether the new machine
could run the same design software. Alice said the Hangzhou purchase was
only a replacement, not an upgrade, and that the failed computer would be
recycled. Meanwhile the office network still listed the old computer as the
primary workstation. When Bob visited Hangzhou the next week, Alice showed
him the laptop and the recycling receipt. The causal chain is local: the
failure in Hangzhou caused the purchase, the purchase caused the delivery,
and the delivery caused the office inventory to become stale until Alice
updated it.
```

After a one-place name edit:

```
Alice bought a laptop in Hangzhou because her old computer failed.
The laptop was delivered to her office near West Lake two days later.
Bob, who works in Beijing, called Alice to ask whether the new machine
could run the same design software. Alice said the Hangzhou purchase was
only a replacement, not an upgrade, and that the failed computer would be
recycled. Meanwhile the office network still listed the old computer as the
primary workstation. When Bob visited Hangzhou the next week, Alice showed
him the laptop and the recycling receipt. The causal chain is local: the
failure in Hangzhou caused the purchase, the purchase caused the delivery,
and the delivery caused the office inventory to become stale until Alice
updated it.
```

## 2. Output: structured object (S)

**156** tokens → **54** nodes, **115** edges.

Entity nodes (typed objects in the paper):

- `n0`  [entity]  Alice
- `n3`  [entity]  Hangzhou
- `n8`  [entity]  West Lake
- `n10`  [entity]  Bob
- `n12`  [entity]  Shanghai
- `n14`  [entity]  Alice
- `n19`  [entity]  Alice
- `n21`  [entity]  Hangzhou
- `n28`  [entity]  Meanwhile
- `n33`  [entity]  When Bob
- `n35`  [entity]  Hangzhou
- `n37`  [entity]  Alice
- `n44`  [entity]  Hangzhou
- `n52`  [entity]  Alice

Phrase-node sample (first 12):

- `n1`  bought
- `n2`  laptop
- `n4`  old computer failed
- `n5`  laptop
- `n6`  delivered
- `n7`  office near
- `n9`  two days later
- `n11`  who works
- `n13`  called
- `n15`  ask whether
- `n16`  new machine
could
- `n17`  run

Dependency-edge sample (after attention contraction; src is depended on → dst must recompute):

- n0='Alice'  --depends_on w=0.84412→  n1='bought'
- n0='Alice'  --depends_on w=0.63884→  n2='laptop'
- n1='bought'  --depends_on w=0.12724→  n2='laptop'
- n0='Alice'  --depends_on w=0.6088→  n3='Hangzhou'
- n1='bought'  --depends_on w=0.04632→  n3='Hangzhou'
- n2='laptop'  --depends_on w=0.04074→  n3='Hangzhou'
- n0='Alice'  --depends_on w=0.52822→  n4='old computer failed'
- n2='laptop'  --depends_on w=0.03309→  n4='old computer failed'

The full readable object is in `sog_readable.json` (no C/H vectors).

## 3. Output: size (paper §5.1 / §7)

| Object | Size | vs KV |
|---|---|---|
| token KV cache (near-relative of model state H) | 5,750,784 bytes (5616.0 KB) | 1× |
| S (structure: nodes + edges) | 12,952 bytes (12.6 KB) | 444.01× smaller |
| S+C (portable: structure + canonical vectors) | 31,206 bytes (30.5 KB) | 184.28× smaller |

Paper hypothesis: a structured cache must be far smaller than per-token KV before it can live on the client while the server only computes.

Verdict: **holds**. S+C is more than two orders of magnitude smaller than KV, which supports storing objects locally on size grounds.

## 4. Output: local-edit invalidation (paper §4.5 / §6 / experiment 4)

| Scope | Invalidated set | Fraction |
|---|---|---|
| Seed Δ (surface contains Shanghai) | n12='Shanghai' | 1/54 |
| One-hop Desc(Δ) | n12='Shanghai', n13='called' | 0.037 |
| Transitive closure | n12='Shanghai', n13='called', n14='Alice', n15='ask whether', n16='new machine could', n17='run', n18='same design software', n19='Alice', n20='said', n21='Hangzhou', n22='purchase', n23='only', n24='replacement', n25='upgrade', n46='purchase', n47='purchase caused', n48='delivery', n49='delivery caused', n50='office inventory', n51='become stale until', n52='Alice', n53='updated' | 0.407 |
| Full token prefill | all 156 tokens | 1.0 |

Verdict:

- One-hop invalidation **0.037**, versus full prefill: **holds** (a local place-name edit does not force recomputing the whole sequence).
- Transitive closure **0.407**: **partial**. Still below 100% of tokens, but the attention graph is denser than linguistic dependencies, so the dirty region grows. This matches §11: natural-language edges are uncertain and the graph is not an AST.

## 5. Fit to the paper

| Claim | Observation | Verdict |
|---|---|---|
| §3 / §5.2: context should be a typed object graph, not only token tensors | Extracted entity nodes such as Alice / Hangzhou / Shanghai / Bob, plus phrase nodes and dependency edges | holds (representation) |
| §5.1: S+C must be clearly smaller than KV before client-side storage is plausible | KV 5,750,784 bytes (5616.0 KB) vs S+C 31,206 bytes (30.5 KB) | **holds** |
| §5.2: what can move is structure S and canonical C, not layer-wise KV | Readable objects are text spans + types + edges; H is preview-only | holds |
| §6: a local change recomputes only I = Desc(Δ) ∪ Δ | One-hop dirty nodes for Shanghai→Beijing: n12='Shanghai', n13='called' | **holds** |
| §7 / §11: if the graph is not sparse, transitive invalidation grows | Transitive fraction 0.407 | **partial** |
| §10: this experiment is not perplexity / cross-model lift quality | No generation quality, no cross-model Lift | holds (scope) |

Overall: **size and objectification match the paper; one-hop locality matches; transitive locality only partially matches.** Treating SOG as a real runtime still needs sparser edges or the paper's PGR / invalidation threshold, or the dirty region will crawl along attention edges.
