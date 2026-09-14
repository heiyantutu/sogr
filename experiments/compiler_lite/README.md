# Compiler-lite (two 0.5B models)

The same Semantic Object Graph \(S\) is given as context to two 0.5B instruction-tuned models, compared against feeding the raw paragraph.

This tests whether **the object graph is an adequate portable context**, not whether vectors can be compiled into KV.

```bash
pip install -r requirements.txt
python run.py
```

Models (loaded one at a time, CPU):

- `Qwen/Qwen2.5-0.5B-Instruct`
- `Qwen/Qwen2-0.5B-Instruct`

The first run downloads the weights. The graph is reused from `../llm_to_object/output/sog_readable.json`.
