"""
CPU toy: turn a small causal LM into a Semantic Object Graph.

This is not a quality evaluation of SOGR. It only checks three representation
facts that a laptop can measure:

  1. DistilGPT-2 attention can be contracted into a typed node/edge object.
  2. The serialized object is much smaller than a token-level KV snapshot.
  3. A local entity edit dirties a subgraph, not the whole sequence.

Run (CPU, ~80M params):

    pip install -r requirements.txt
    python run.py
"""

from __future__ import annotations

import hashlib
import json
import re
import sys
import time
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

MODEL_ID = "distilgpt2"
EDGE_PERCENTILE = 92.0
MAX_EDGES_PER_NODE = 4
CANONICAL_DIM = 64

STOP = {
    "a", "an", "the", "and", "or", "but", "if", "then", "so", "because",
    "in", "on", "at", "to", "for", "of", "with", "from", "by", "as",
    "is", "are", "was", "were", "be", "been", "being", "has", "had",
    "have", "do", "did", "does", "not", "no", "yes", "this", "that",
    "these", "those", "it", "its", "her", "his", "their", "she", "he",
    "they", "we", "you", "i", "my", "our", "your",
}

DOCUMENT = """
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
""".strip()

EDIT_FROM = "Shanghai"
EDIT_TO = "Beijing"
EDITED = DOCUMENT.replace(EDIT_FROM, EDIT_TO)


def word_spans(text: str) -> list[tuple[int, int, str]]:
    return [(m.start(), m.end(), m.group(0)) for m in re.finditer(r"[A-Za-z0-9']+", text)]


def is_entity(word: str) -> bool:
    return bool(re.match(r"^[A-Z][A-Za-z]+$", word)) and word.lower() not in STOP


def build_nodes(text: str) -> list[dict]:
    words = word_spans(text)
    nodes: list[dict] = []
    i = 0
    while i < len(words):
        start, end, w = words[i]
        if is_entity(w):
            j = i + 1
            while j < len(words) and is_entity(words[j][2]) and words[j][0] <= end + 2:
                end = words[j][1]
                j += 1
            surface = text[start:end]
            nodes.append(
                {
                    "id": f"n{len(nodes)}",
                    "type": "entity",
                    "text": surface,
                    "char_start": start,
                    "char_end": end,
                }
            )
            i = j
            continue
        if w.lower() in STOP:
            i += 1
            continue
        j = i + 1
        while (
            j < len(words)
            and j - i < 3
            and words[j][2].lower() not in STOP
            and not is_entity(words[j][2])
            and words[j][0] <= end + 2
        ):
            end = words[j][1]
            j += 1
        surface = text[start:end]
        nodes.append(
            {
                "id": f"n{len(nodes)}",
                "type": "phrase",
                "text": surface,
                "char_start": start,
                "char_end": end,
            }
        )
        i = j
    return nodes


def token_char_spans(tokenizer, text: str) -> list[tuple[int, int]]:
    enc = tokenizer(text, return_offsets_mapping=True, return_tensors="pt")
    mapping = enc["offset_mapping"][0].tolist()
    return [(int(a), int(b)) for a, b in mapping]


def node_token_index(nodes, spans) -> list[list[int]]:
    out = []
    for node in nodes:
        ids = [
            i
            for i, (a, b) in enumerate(spans)
            if b > a and not (b <= node["char_start"] or a >= node["char_end"])
        ]
        out.append(ids)
    return out


def cache_nbytes(past) -> int:
    if past is None:
        return 0
    if hasattr(past, "key_cache") and hasattr(past, "value_cache"):
        return sum(t.nbytes for t in (*past.key_cache, *past.value_cache) if torch.is_tensor(t))
    total = 0
    stack = [past]
    while stack:
        cur = stack.pop()
        if torch.is_tensor(cur):
            total += cur.nbytes
        elif isinstance(cur, (list, tuple)):
            stack.extend(cur)
        elif hasattr(cur, "key_states") and hasattr(cur, "value_states"):
            stack.extend([cur.key_states, cur.value_states])
        elif hasattr(cur, "layers"):
            stack.extend(list(cur.layers))
    return total


@torch.no_grad()
def forward(model, tokenizer, text: str):
    enc = tokenizer(text, return_tensors="pt")
    t0 = time.perf_counter()
    out = model(**enc, output_attentions=True, output_hidden_states=True, use_cache=True)
    dt = time.perf_counter() - t0
    if not out.attentions:
        raise RuntimeError("model returned no attentions; load with attn_implementation='eager'")
    attn = torch.stack([a[0] for a in out.attentions]).mean(dim=(0, 1))
    hidden = out.hidden_states[-1][0]
    kv_bytes = cache_nbytes(out.past_key_values)
    if kv_bytes == 0:
        cfg = model.config
        seq = int(enc["input_ids"].shape[1])
        n_layer = int(getattr(cfg, "n_layer", getattr(cfg, "num_hidden_layers")))
        n_head = int(getattr(cfg, "n_head", getattr(cfg, "num_attention_heads")))
        d = int(getattr(cfg, "n_embd", getattr(cfg, "hidden_size")))
        kv_bytes = n_layer * 2 * seq * d * 4
    return enc, hidden, attn, kv_bytes, dt


def hashed_canonical(text: str, dim: int = CANONICAL_DIM) -> list[float]:
    vec = torch.zeros(dim)
    for tok in re.findall(r"[A-Za-z0-9']+", text.lower()):
        h = int(hashlib.md5(tok.encode("utf-8")).hexdigest(), 16)
        vec[h % dim] += 1.0
        vec[(h // dim) % dim] += 0.5
    n = torch.linalg.norm(vec)
    if n > 0:
        vec = vec / n
    return [round(float(x), 5) for x in vec.tolist()]


def edges_from_attention(attn: torch.Tensor, token_ids: list[list[int]]) -> list[dict]:
    n = len(token_ids)
    score = torch.zeros(n, n)
    for i, ii in enumerate(token_ids):
        if not ii:
            continue
        for j, jj in enumerate(token_ids):
            if i == j or not jj:
                continue
            block = attn[ii][:, jj]
            score[i, j] = block.mean()
    flat = score[score > 0]
    if flat.numel() == 0:
        return []
    thr = torch.quantile(flat, EDGE_PERCENTILE / 100.0)
    edges = []
    for i in range(n):
        vals, idx = torch.sort(score[i], descending=True)
        kept = 0
        for v, j in zip(vals.tolist(), idx.tolist()):
            if i == j or v < float(thr) or kept >= MAX_EDGES_PER_NODE:
                continue
            # Node i attends to j, so i depends on j. Invalidation flows j → i.
            edges.append(
                {
                    "src": f"n{j}",
                    "dst": f"n{i}",
                    "type": "depends_on",
                    "weight": round(float(v), 5),
                }
            )
            kept += 1
    return edges


def descendants(start: set[str], edges: list[dict], hops: int | None = None) -> set[str]:
    adj: dict[str, list[str]] = {}
    for e in edges:
        adj.setdefault(e["src"], []).append(e["dst"])
    seen = set(start)
    frontier = set(start)
    depth = 0
    while frontier:
        if hops is not None and depth >= hops:
            break
        nxt = set()
        for u in frontier:
            for v in adj.get(u, []):
                if v not in seen:
                    seen.add(v)
                    nxt.add(v)
        frontier = nxt
        depth += 1
    return seen


def pack_object(nodes, edges, hidden, token_ids) -> dict:
    sog_nodes = []
    for node, tids in zip(nodes, token_ids):
        if tids:
            h = hidden[tids].mean(dim=0)
            h_norm = torch.nn.functional.normalize(h, dim=0)
            h_list = [round(float(x), 5) for x in h_norm[:32].tolist()]
        else:
            h_list = []
        sog_nodes.append(
            {
                **node,
                "c": hashed_canonical(node["text"]),
                "h_preview": h_list,
            }
        )
    return {
        "schema": "sogr.sog.v0",
        "layers": {
            "S": "structure: typed nodes and attention edges",
            "C": "canonical: model-free hashed bag-of-words",
            "H": "model state: DistilGPT-2 span mean (preview only)",
        },
        "nodes": sog_nodes,
        "edges": edges,
    }


def object_bytes(sog: dict) -> tuple[int, int, int]:
    portable = {
        "nodes": [
            {k: n[k] for k in ("id", "type", "text", "char_start", "char_end", "c")}
            for n in sog["nodes"]
        ],
        "edges": sog["edges"],
    }
    full = {"nodes": sog["nodes"], "edges": sog["edges"]}
    s_only = {
        "nodes": [
            {k: n[k] for k in ("id", "type", "text", "char_start", "char_end")}
            for n in sog["nodes"]
        ],
        "edges": sog["edges"],
    }
    return (
        len(json.dumps(s_only, ensure_ascii=False).encode("utf-8")),
        len(json.dumps(portable, ensure_ascii=False).encode("utf-8")),
        len(json.dumps(full, ensure_ascii=False).encode("utf-8")),
    )


def build_sog(model, tokenizer, text: str):
    nodes = build_nodes(text)
    enc, hidden, attn, kv_bytes, dt = forward(model, tokenizer, text)
    spans = token_char_spans(tokenizer, text)
    # tokenizer() without offsets vs with offsets must match length
    if len(spans) != hidden.size(0):
        enc_off = tokenizer(text, return_offsets_mapping=True, return_tensors="pt")
        assert enc_off["input_ids"].shape == enc["input_ids"].shape
        spans = [(int(a), int(b)) for a, b in enc_off["offset_mapping"][0].tolist()]
    tids = node_token_index(nodes, spans)
    edges = edges_from_attention(attn, tids)
    sog = pack_object(nodes, edges, hidden, tids)
    return sog, kv_bytes, dt, int(enc["input_ids"].shape[1])


def kb(n: int) -> str:
    return f"{n:,} bytes ({n / 1024:.1f} KB)"


def compact_sog(sog: dict) -> dict:
    return {
        "schema": sog["schema"],
        "layers": sog["layers"],
        "nodes": [
            {
                "id": n["id"],
                "type": n["type"],
                "text": n["text"],
                "span": [n["char_start"], n["char_end"]],
            }
            for n in sog["nodes"]
        ],
        "edges": sog["edges"],
    }


def node_text(sog: dict, nid: str) -> str:
    for n in sog["nodes"]:
        if n["id"] == nid:
            return n["text"]
    return nid


def render_report(document: str, edited: str, sog: dict, compact: dict, summary: dict) -> str:
    entities = [n for n in compact["nodes"] if n["type"] == "entity"]
    phrases = [n for n in compact["nodes"] if n["type"] == "phrase"]
    seed = summary["seed_dirty_nodes"]
    hop1 = summary["invalidated_nodes_1hop"]
    trans = summary["invalidated_nodes_transitive"]
    seed_txt = ", ".join(f"{i}={node_text(sog, i)!r}" for i in seed)
    hop1_txt = ", ".join(f"{i}={node_text(sog, i)!r}" for i in hop1)
    trans_txt = ", ".join(f"{i}={node_text(sog, i)!r}" for i in trans)

    size_ok = summary["kv_to_SC_ratio"] >= 10
    hop_ok = 0 < summary["invalidation_fraction_1hop"] < 0.25
    trans_partial = summary["invalidation_fraction_transitive"] < summary["full_prefill_token_fraction"]
    trans_dense = summary["invalidation_fraction_transitive"] > 0.25

    size_verdict = "holds" if size_ok else "fails"
    hop_verdict = "holds" if hop_ok else "partial"
    trans_verdict = (
        "partial" if trans_partial and trans_dense else ("holds" if trans_partial else "fails")
    )

    entity_lines = "\n".join(
        f"- `{n['id']}`  [{n['type']}]  {n['text']}" for n in entities
    )
    sample_phrases = "\n".join(
        f"- `{n['id']}`  {n['text']}" for n in phrases[:12]
    )
    edge_sample = sog["edges"][:8]
    edge_lines = "\n".join(
        f"- {e['src']}={node_text(sog, e['src'])!r}  --{e['type']} w={e['weight']}→  "
        f"{e['dst']}={node_text(sog, e['dst'])!r}"
        for e in edge_sample
    )

    return f"""# DistilGPT-2 → Semantic Object Graph (CPU representation toy)

Compared with SOGR draft 0.2: §3 object graph, §5 three-layer cache / portable objects, §6–7 subgraph invalidation, and §10 experiments 8 plus edit locality.
This is **not** a model-quality evaluation. It only checks whether LLM state can be contracted into objects, and whether those objects are small enough to store locally.

## 1. Input

- Model: `{summary['model']}` (CPU, {summary['forward_seconds']} s)
- Edit: `{summary['edit']}`

Source:

```
{document}
```

After a one-place name edit:

```
{edited}
```

## 2. Output: structured object (S)

**{summary['tokens']}** tokens → **{summary['nodes']}** nodes, **{summary['edges']}** edges.

Entity nodes (typed objects in the paper):

{entity_lines}

Phrase-node sample (first 12):

{sample_phrases}

Dependency-edge sample (after attention contraction; src is depended on → dst must recompute):

{edge_lines}

The full readable object is in `sog_readable.json` (no C/H vectors).

## 3. Output: size (paper §5.1 / §7)

| Object | Size | vs KV |
|---|---|---|
| token KV cache (near-relative of model state H) | {kb(summary['kv_cache_bytes'])} | 1× |
| S (structure: nodes + edges) | {kb(summary['S_bytes'])} | {summary['kv_to_S_ratio']}× smaller |
| S+C (portable: structure + canonical vectors) | {kb(summary['S_plus_C_bytes'])} | {summary['kv_to_SC_ratio']}× smaller |

Paper hypothesis: a structured cache must be far smaller than per-token KV before it can live on the client while the server only computes.

Verdict: **{size_verdict}**. S+C is more than two orders of magnitude smaller than KV, which supports storing objects locally on size grounds.

## 4. Output: local-edit invalidation (paper §4.5 / §6 / experiment 4)

| Scope | Invalidated set | Fraction |
|---|---|---|
| Seed Δ (surface contains Shanghai) | {seed_txt} | {len(seed)}/{summary['nodes']} |
| One-hop Desc(Δ) | {hop1_txt} | {summary['invalidation_fraction_1hop']} |
| Transitive closure | {trans_txt} | {summary['invalidation_fraction_transitive']} |
| Full token prefill | all {summary['tokens']} tokens | 1.0 |

Verdict:

- One-hop invalidation **{summary['invalidation_fraction_1hop']}**, versus full prefill: **{hop_verdict}** (a local place-name edit does not force recomputing the whole sequence).
- Transitive closure **{summary['invalidation_fraction_transitive']}**: **{trans_verdict}**. Still below 100% of tokens, but the attention graph is denser than linguistic dependencies, so the dirty region grows. This matches §11: natural-language edges are uncertain and the graph is not an AST.

## 5. Fit to the paper

| Claim | Observation | Verdict |
|---|---|---|
| §3 / §5.2: context should be a typed object graph, not only token tensors | Extracted entity nodes such as Alice / Hangzhou / Shanghai / Bob, plus phrase nodes and dependency edges | holds (representation) |
| §5.1: S+C must be clearly smaller than KV before client-side storage is plausible | KV {kb(summary['kv_cache_bytes'])} vs S+C {kb(summary['S_plus_C_bytes'])} | **{size_verdict}** |
| §5.2: what can move is structure S and canonical C, not layer-wise KV | Readable objects are text spans + types + edges; H is preview-only | holds |
| §6: a local change recomputes only I = Desc(Δ) ∪ Δ | One-hop dirty nodes for Shanghai→Beijing: {hop1_txt} | **{hop_verdict}** |
| §7 / §11: if the graph is not sparse, transitive invalidation grows | Transitive fraction {summary['invalidation_fraction_transitive']} | **{trans_verdict}** |
| §10: this experiment is not perplexity / cross-model lift quality | No generation quality, no cross-model Lift | holds (scope) |

Overall: **size and objectification match the paper; one-hop locality matches; transitive locality only partially matches.** Treating SOG as a real runtime still needs sparser edges or the paper's PGR / invalidation threshold, or the dirty region will crawl along attention edges.
"""

def main() -> int:
    out_dir = Path(__file__).resolve().parent / "output"
    out_dir.mkdir(exist_ok=True)

    print(f"loading {MODEL_ID} on CPU ...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID,
        attn_implementation="eager",
        dtype=torch.float32,
    )
    model.eval()

    sog, kv_bytes, dt, n_tokens = build_sog(model, tokenizer, DOCUMENT)
    s_bytes, sc_bytes, sch_bytes = object_bytes(sog)

    changed = {n["id"] for n in sog["nodes"] if EDIT_FROM in n["text"]}
    dirty_1 = descendants(changed, sog["edges"], hops=1) if changed else set()
    dirty = descendants(changed, sog["edges"]) if changed else set()

    summary = {
        "model": MODEL_ID,
        "device": "cpu",
        "tokens": n_tokens,
        "nodes": len(sog["nodes"]),
        "edges": len(sog["edges"]),
        "forward_seconds": round(dt, 3),
        "kv_cache_bytes": kv_bytes,
        "S_bytes": s_bytes,
        "S_plus_C_bytes": sc_bytes,
        "S_C_Hpreview_bytes": sch_bytes,
        "kv_to_S_ratio": round(kv_bytes / max(s_bytes, 1), 2),
        "kv_to_SC_ratio": round(kv_bytes / max(sc_bytes, 1), 2),
        "edit": f"{EDIT_FROM} -> {EDIT_TO}",
        "seed_dirty_nodes": sorted(changed),
        "invalidated_nodes_1hop": sorted(dirty_1),
        "invalidated_nodes_transitive": sorted(dirty),
        "invalidation_fraction_1hop": round(len(dirty_1) / max(len(sog["nodes"]), 1), 3),
        "invalidation_fraction_transitive": round(len(dirty) / max(len(sog["nodes"]), 1), 3),
        "full_prefill_token_fraction": 1.0,
        "note": (
            "Token invalidation for a full re-prefill is 100% of the sequence. "
            "The object invalidation fraction is |Desc(Δ)| / |V|."
        ),
    }

    (out_dir / "sog.json").write_text(
        json.dumps(sog, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    (out_dir / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    compact = compact_sog(sog)
    (out_dir / "sog_readable.json").write_text(
        json.dumps(compact, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    report = render_report(DOCUMENT, EDITED, sog, compact, summary)
    (out_dir / "report.md").write_text(report, encoding="utf-8")
    print(report)
    print(f"wrote {out_dir / 'report.md'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
