# SOGR: Semantic Object Graph Runtime

**A Structured Incremental Architecture for Long-Context Language Modeling**

Independent Research Draft · Version 0.3 · September 2026  
Author: **Zeping Tu** (Frontend Engineer)

This repository hosts the research proposal for **SOGR** (Semantic Object Graph Runtime). The direction comes from frontend practice: React and Vue compile a structured view, dirty only the dependent subtree, and persist objects rather than pixels. SOGR asks whether a language model can use a Semantic Object Graph in the same way.

> This manuscript is a **research proposal** plus a **CPU-laptop baseline**. It is not a GPU-scale result. DistilGPT-2 and two 0.5B models are the present experimental floor.

## What's here

| Path | Description |
|------|-------------|
| [`paper/SOGR.md`](paper/SOGR.md) | Full draft in Markdown |
| [`paper/SOGR.typ`](paper/SOGR.typ) | Typst source (DejaVu layout) |
| [`paper/SOGR.pdf`](paper/SOGR.pdf) | Built PDF |
| [`experiments/llm_to_object`](experiments/llm_to_object) | CPU toy: DistilGPT-2 → object graph, size and edit locality |
| [`experiments/compiler_lite`](experiments/compiler_lite) | Same S as context for two 0.5B instruct models vs full text |

Rebuild the PDF:

```bash
python paper/build_pdf.py
```

## Core idea

Instead of rediscovering relationships over a token sequence at every layer, SOGR:

1. Uses **low-frequency global attention** to construct / repair a Semantic Object Graph (SOG)
2. Runs **high-frequency graph-conditioned linear** state propagation on that graph
3. Applies **periodic global graph repair**
4. Uses **hierarchical block-level skipping** with a structured cache for incremental recomputation
5. Splits the cache into portable structure S, canonical semantics C, and model-bound state H

## CPU baseline (this machine)

```bash
pip install -r experiments/llm_to_object/requirements.txt
python experiments/llm_to_object/run.py

pip install -r experiments/compiler_lite/requirements.txt
python experiments/compiler_lite/run.py
```

No discrete GPU is required. Results are a floor, not a quality claim for production LLMs.

## Citation

```bibtex
@misc{tu2026sogr,
  title        = {SOGR: Semantic Object Graph Runtime — A Structured Incremental Architecture for Long-Context Language Modeling},
  author       = {Tu, Zeping},
  year         = {2026},
  note         = {Independent Research Draft, Version 0.3},
  howpublished = {\url{https://github.com/heiyantutu/sogr}}
}
```

## Status

Draft 0.3 — architectural hypothesis, portable-cache deployment hypothesis, and a laptop-scale representation / compiler-lite floor.
