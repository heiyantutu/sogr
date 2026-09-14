# LLM → object (CPU toy)

Laptop experiment for the portable-cache claim in the SOGR draft: turn DistilGPT-2 (~82M, CPU) into a Semantic Object Graph, then compare object size with a KV snapshot and measure edit invalidation.

This is a representation demo, not a quality evaluation of SOGR.

```bash
pip install -r requirements.txt
python run.py
```

First run downloads `distilgpt2`. No GPU is used.

Outputs:

- `output/sog.json` — nodes (S), hashed canonical vectors (C), attention edges
- `output/summary.json` — sizes and 1-hop / transitive `|Desc(Δ)| / |V|` after `Shanghai → Beijing`
