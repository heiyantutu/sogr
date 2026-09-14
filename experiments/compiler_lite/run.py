"""
Compiler-lite: same Semantic Object Graph S as context for two 0.5B instruct models.

Conditions per model:
  text  — full source paragraph
  sog   — serialized nodes + edges only (no raw paragraph)

This tests whether S is an adequate portable context, not a KV compiler.
"""

from __future__ import annotations

import gc
import json
import re
import sys
import time
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

ROOT = Path(__file__).resolve().parent
SOG_PATH = ROOT.parent / "llm_to_object" / "output" / "sog_readable.json"
OUT_DIR = ROOT / "output"
QWEN2_LOCAL = (
    Path.home()
    / ".cache"
    / "modelscope"
    / "models"
    / "Qwen--Qwen2-0.5B-Instruct"
    / "snapshots"
    / "master"
)

MODELS = [
    "Qwen/Qwen2.5-0.5B-Instruct",
    str(QWEN2_LOCAL) if QWEN2_LOCAL.exists() else "Qwen/Qwen2-0.5B-Instruct",
]

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

QUESTIONS = [
    {
        "id": "q1",
        "question": "Who bought the laptop?",
        "gold": ["alice"],
    },
    {
        "id": "q2",
        "question": "In which city was the laptop purchased?",
        "gold": ["hangzhou"],
    },
    {
        "id": "q3",
        "question": "Why did Alice buy a laptop?",
        "gold": ["failed", "old computer"],
    },
    {
        "id": "q4",
        "question": "Who works in Shanghai?",
        "gold": ["bob"],
    },
    {
        "id": "q5",
        "question": "Near which landmark was the laptop delivered?",
        "gold": ["west lake"],
    },
    {
        "id": "q6",
        "question": "What would happen to the failed computer?",
        "gold": ["recycl"],
    },
]

SYSTEM = (
    "Answer with a short phrase using only the provided context. "
    "If the context does not contain the answer, say UNKNOWN."
)


def serialize_sog(sog: dict) -> str:
    by_id = {n["id"]: n for n in sog["nodes"]}
    lines = ["# Semantic Object Graph", "", "## Nodes"]
    for n in sog["nodes"]:
        text = " ".join(n["text"].split())
        lines.append(f"- {n['id']} [{n['type']}] {text}")
    lines += ["", "## Edges (src is depended on, dst needs update)"]
    for e in sog["edges"]:
        s, d = by_id[e["src"]], by_id[e["dst"]]
        st = " ".join(s["text"].split())
        dt = " ".join(d["text"].split())
        lines.append(f"- {s['id']}:{st} -> {d['id']}:{dt} ({e['weight']})")
    return "\n".join(lines)


def hit(answer: str, gold: list[str]) -> bool:
    a = answer.lower()
    return any(g.lower() in a for g in gold)


def build_messages(context: str, question: str) -> list[dict]:
    user = f"Context:\n{context}\n\nQuestion: {question}\nAnswer:"
    return [
        {"role": "system", "content": SYSTEM},
        {"role": "user", "content": user},
    ]


@torch.no_grad()
def generate(model, tokenizer, messages: list[dict], max_new: int = 48) -> tuple[str, float]:
    prompt = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
    )
    enc = tokenizer(prompt, return_tensors="pt")
    t0 = time.perf_counter()
    out = model.generate(
        **enc,
        max_new_tokens=max_new,
        do_sample=False,
        pad_token_id=tokenizer.eos_token_id,
    )
    dt = time.perf_counter() - t0
    gen = out[0][enc["input_ids"].shape[1] :]
    text = tokenizer.decode(gen, skip_special_tokens=True).strip()
    text = re.sub(r"\s+", " ", text)
    return text, dt


def run_model(model_id: str, contexts: dict[str, str]) -> list[dict]:
    print(f"\nloading {model_id} on CPU ...")
    tokenizer = AutoTokenizer.from_pretrained(
        model_id, trust_remote_code=True, local_files_only=Path(model_id).exists()
    )
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        dtype=torch.float32,
        trust_remote_code=True,
        attn_implementation="eager",
        local_files_only=Path(model_id).exists(),
    )
    model.eval()
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token

    rows = []
    for q in QUESTIONS:
        for cond, ctx in contexts.items():
            messages = build_messages(ctx, q["question"])
            answer, dt = generate(model, tokenizer, messages)
            ok = hit(answer, q["gold"])
            print(f"  {q['id']} {cond:4s} hit={int(ok)} {dt:.1f}s | {answer[:120]}")
            rows.append(
                {
                    "model": model_id,
                    "qid": q["id"],
                    "question": q["question"],
                    "gold": q["gold"],
                    "condition": cond,
                    "answer": answer,
                    "hit": ok,
                    "seconds": round(dt, 2),
                }
            )

    del model
    gc.collect()
    return rows


def summarize(rows: list[dict]) -> dict:
    models = sorted({r["model"] for r in rows})
    table = {}
    for m in models:
        table[m] = {}
        for cond in ("text", "sog"):
            sub = [r for r in rows if r["model"] == m and r["condition"] == cond]
            n = max(len(sub), 1)
            table[m][cond] = {
                "n": len(sub),
                "hits": sum(r["hit"] for r in sub),
                "acc": round(sum(r["hit"] for r in sub) / n, 3),
                "seconds": round(sum(r["seconds"] for r in sub), 1),
            }
        table[m]["gap_sog_minus_text"] = round(
            table[m]["sog"]["acc"] - table[m]["text"]["acc"], 3
        )
    both_ok = 0
    text_only = 0
    sog_only = 0
    neither = 0
    qids = sorted({r["qid"] for r in rows})
    for m in models:
        for qid in qids:
            t = next(r for r in rows if r["model"] == m and r["qid"] == qid and r["condition"] == "text")
            s = next(r for r in rows if r["model"] == m and r["qid"] == qid and r["condition"] == "sog")
            if t["hit"] and s["hit"]:
                both_ok += 1
            elif t["hit"] and not s["hit"]:
                text_only += 1
            elif s["hit"] and not t["hit"]:
                sog_only += 1
            else:
                neither += 1
    return {
        "by_model": table,
        "paired": {
            "both_hit": both_ok,
            "text_only": text_only,
            "sog_only": sog_only,
            "neither": neither,
        },
    }


def verdict(summary: dict) -> str:
    gaps = [v["gap_sog_minus_text"] for v in summary["by_model"].values()]
    sog_acc = [v["sog"]["acc"] for v in summary["by_model"].values()]
    text_acc = [v["text"]["acc"] for v in summary["by_model"].values()]
    mean_gap = sum(gaps) / len(gaps)
    lines = []
    if min(sog_acc) >= 0.5 and mean_gap >= -0.17:
        lines.append(
            "Verdict: **partial support for portability**. With S as the only context, "
            "both 0.5B models answer at least half the items, and the drop versus full "
            "text is not a collapse. This does not show that S replaces KV."
        )
    elif max(sog_acc) < 0.34:
        lines.append(
            "Verdict: **current S is not yet an adequate cross-model context**. "
            "The serialized objects did not preserve enough answer information, "
            "or the 0.5B models cannot read this IR. Fix serialization/extraction "
            "before scaling the backbone."
        )
    else:
        lines.append(
            "Verdict: **weak support with a warning**. S is sometimes enough, but "
            "loss versus full text is clear. This matches 'objectification holds; "
            "compile/recovery is unproven'."
        )
    if min(text_acc) < 0.5:
        lines.append(
            "Note: the full-text condition is itself unstable, so a 0.5B reader may "
            "be weak; do not attribute the entire gap to S."
        )
    return "\n".join(lines)


def render_report(sog_text: str, rows: list[dict], summary: dict) -> str:
    by_m = summary["by_model"]
    table_lines = [
        "| Model | Text acc | S acc | S-text | Text time | S time |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for m, v in by_m.items():
        short = m.split("/")[-1]
        table_lines.append(
            f"| {short} | {v['text']['acc']} ({v['text']['hits']}/{v['text']['n']}) "
            f"| {v['sog']['acc']} ({v['sog']['hits']}/{v['sog']['n']}) "
            f"| {v['gap_sog_minus_text']:+.3f} "
            f"| {v['text']['seconds']}s | {v['sog']['seconds']}s |"
        )

    detail = []
    for q in QUESTIONS:
        detail.append(f"### {q['id']}. {q['question']}")
        detail.append("")
        detail.append(f"Gold substrings: `{', '.join(q['gold'])}`")
        detail.append("")
        for r in rows:
            if r["qid"] != q["id"]:
                continue
            mark = "HIT" if r["hit"] else "MISS"
            short = r["model"].split("/")[-1]
            detail.append(f"- **{short} / {r['condition']}** [{mark}] {r['answer']}")
        detail.append("")

    return f"""# Compiler-lite: S → two 0.5B models

Matches paper §5.4 / experiments 9–10. Condition `text` feeds the source paragraph; `sog` feeds only the object graph.
This is **not** KV compilation. It tests whether the same S can serve as cross-model context.

## Input

- Models: {', '.join(f'`{m}`' for m in MODELS)}
- Device: CPU
- Graph: `llm_to_object/output/sog_readable.json`
- Items: {len(QUESTIONS)}

The source is the Alice/Hangzhou paragraph from the previous experiment. Serialized S is about {len(sog_text)} characters (source {len(DOCUMENT)} characters).

## Summary

{chr(10).join(table_lines)}

Paired (one count per model × question): both={summary['paired']['both_hit']}, text-only={summary['paired']['text_only']}, S-only={summary['paired']['sog_only']}, neither={summary['paired']['neither']}

## Per question

{chr(10).join(detail)}
## Fit to the paper

{verdict(summary)}

Still untested: compiling S into KV, 7B-class models, and 16K context.
"""


def main() -> int:
    if not SOG_PATH.exists():
        raise SystemExit(f"missing {SOG_PATH}; run llm_to_object first")
    sog = json.loads(SOG_PATH.read_text(encoding="utf-8"))
    sog_text = serialize_sog(sog)
    contexts = {"text": DOCUMENT, "sog": sog_text}

    OUT_DIR.mkdir(exist_ok=True)
    (OUT_DIR / "sog_context.txt").write_text(sog_text, encoding="utf-8")

    rows: list[dict] = []
    for model_id in MODELS:
        rows.extend(run_model(model_id, contexts))

    summary = summarize(rows)
    payload = {"models": MODELS, "summary": summary, "rows": rows}
    (OUT_DIR / "results.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    report = render_report(sog_text, rows, summary)
    (OUT_DIR / "report.md").write_text(report, encoding="utf-8")
    try:
        print("\n" + report)
    except UnicodeEncodeError:
        sys.stdout.buffer.write(("\n" + report + "\n").encode("utf-8", errors="replace"))
    print(f"wrote {OUT_DIR / 'report.md'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
