# SOGR: Semantic Object Graph Runtime

**A Structured Incremental Architecture for Long-Context Language Modeling**

Independent Research Draft · Version 0.1 · September 2026  
Author: **Zeping Tu**

This repository hosts the research proposal for **SOGR** (Semantic Object Graph Runtime): an architecture that treats semantic dependency structure as an explicit, cacheable intermediate representation for long-context language modeling.

> This manuscript is a **research proposal**, not a report of completed experiments. No empirical performance claim is made.

## What's here

| Path | Description |
|------|-------------|
| [`paper/SOGR.md`](paper/SOGR.md) | Full draft in Markdown |
| [`paper/SOGR.pdf`](paper/SOGR.pdf) | Original PDF draft |

## Core idea

Instead of rediscovering relationships over a token sequence at every layer, SOGR:

1. Uses **low-frequency global attention** to construct / repair a Semantic Object Graph (SOG)
2. Runs **high-frequency graph-conditioned linear** state propagation on that graph
3. Applies **periodic global graph repair**
4. Uses **hierarchical block-level skipping** with a structured cache for incremental recomputation

The intended benefit is amortized and conditional: expensive relationship discovery is reused, while stable or unrelated regions are not repeatedly recomputed.

## Citation

```bibtex
@misc{tu2026sogr,
  title        = {SOGR: Semantic Object Graph Runtime — A Structured Incremental Architecture for Long-Context Language Modeling},
  author       = {Tu, Zeping},
  year         = {2026},
  note         = {Independent Research Draft, Version 0.1},
  howpublished = {\url{https://github.com/heiyantutu/sogr}}
}
```

## Status

Draft 0.1 — architectural hypothesis and falsifiable experimental program.
