# Compiler-lite: S → two 0.5B models

Matches paper §5.4 / experiments 9–10. Condition `text` feeds the source paragraph; `sog` feeds only the object graph.
This is **not** KV compilation. It tests whether the same S can serve as cross-model context.

## Input

- Models: `Qwen/Qwen2.5-0.5B-Instruct`, `Qwen/Qwen2-0.5B-Instruct`
- Device: CPU
- Graph: `llm_to_object/output/sog_readable.json`
- Items: 6

The source is the Alice/Hangzhou paragraph from the previous experiment. Serialized S is about 6114 characters (source 739 characters).

## Summary

| Model | Text acc | S acc | S−text | Text time | S time |
|---|---:|---:|---:|---:|---:|
| Qwen/Qwen2-0.5B-Instruct | 1.0 (6/6) | 0.5 (3/6) | -0.500 | 5.4s | 82.9s |
| Qwen2.5-0.5B-Instruct | 1.0 (6/6) | 0.5 (3/6) | -0.500 | 4.7s | 86.2s |

Paired (one count per model × question): both=6, text-only=6, S-only=0, neither=0

## Per question

### q1. Who bought the laptop?

Gold substrings: `alice`

- **Qwen2.5-0.5B-Instruct / text** [HIT] Alice
- **Qwen2.5-0.5B-Instruct / sog** [HIT] Alice
- **Qwen/Qwen2-0.5B-Instruct / text** [HIT] Alice
- **Qwen/Qwen2-0.5B-Instruct / sog** [MISS] n1

### q2. In which city was the laptop purchased?

Gold substrings: `hangzhou`

- **Qwen2.5-0.5B-Instruct / text** [HIT] Hangzhou
- **Qwen2.5-0.5B-Instruct / sog** [HIT] Hangzhou
- **Qwen/Qwen2-0.5B-Instruct / text** [HIT] Hangzhou
- **Qwen/Qwen2-0.5B-Instruct / sog** [HIT] Hangzhou

### q3. Why did Alice buy a laptop?

Gold substrings: `failed, old computer`

- **Qwen2.5-0.5B-Instruct / text** [HIT] because her old computer failed
- **Qwen2.5-0.5B-Instruct / sog** [MISS] Because she asked if there was a new machine that could be replaced and upgraded to the same design software.
- **Qwen/Qwen2-0.5B-Instruct / text** [HIT] Because her old computer failed.
- **Qwen/Qwen2-0.5B-Instruct / sog** [HIT] Alice bought a laptop because she wanted to replace an old computer that had failed.

### q4. Who works in Shanghai?

Gold substrings: `bob`

- **Qwen2.5-0.5B-Instruct / text** [HIT] Bob
- **Qwen2.5-0.5B-Instruct / sog** [MISS] Alice
- **Qwen/Qwen2-0.5B-Instruct / text** [HIT] Bob
- **Qwen/Qwen2-0.5B-Instruct / sog** [MISS] n12

### q5. Near which landmark was the laptop delivered?

Gold substrings: `west lake`

- **Qwen2.5-0.5B-Instruct / text** [HIT] West Lake
- **Qwen2.5-0.5B-Instruct / sog** [MISS] n39:office near
- **Qwen/Qwen2-0.5B-Instruct / text** [HIT] West Lake
- **Qwen/Qwen2-0.5B-Instruct / sog** [MISS] n3

### q6. What would happen to the failed computer?

Gold substrings: `recycl`

- **Qwen2.5-0.5B-Instruct / text** [HIT] recycled
- **Qwen2.5-0.5B-Instruct / sog** [HIT] The failed computer would be recycled.
- **Qwen/Qwen2-0.5B-Instruct / text** [HIT] It would be recycled.
- **Qwen/Qwen2-0.5B-Instruct / sog** [HIT] The failed computer would be recycled.

## Fit to the paper

Verdict: **weak support with a warning**. S is sometimes enough, but loss versus full text is clear. This matches 'objectification holds; compile/recovery is unproven'.

Still untested: compiling S into KV, 7B-class models, and 16K context.
