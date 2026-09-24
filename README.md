Hugging Face Link : https://huggingface.co/BharatJitendra/intentgrasp-qlora-adapter

# IntentGrasp QLoRA Finetuning — Structured Intent Classification

Finetuned **Qwen2.5-1.5B-Instruct** with **QLoRA** to classify speaker intent and emit reliable structured JSON — raising accuracy from **41.5% → 90.5%** and JSON validity from **0.5% → 100%**.

---

## Results

| Model | JSON validity | Accuracy |
| :--- | :---: | :---: |
| Base Qwen2.5-1.5B — strict schema | 0.5% | — |
| Base Qwen2.5-1.5B — lenient scoring | — | 41.5% |
| **Finetuned (QLoRA)** — validation | **100%** | **90.5%** |
| **Finetuned (QLoRA)** — gem (harder split) | **100%** | **37.9%** |

Finetuning delivered two distinct wins:

- **Format reliability:** the required JSON schema went from **0.5% → 100%** conformance.
- **Task accuracy:** more than doubled, **41.5% → 90.5%**, measured against the same base model.

---

## The Task

[IntentGrasp](https://huggingface.co/datasets/yuweiyin/IntentGrasp) is an intent-classification benchmark. Each example provides a **context** (a query, dialogue, or monologue), a **question**, and a list of candidate **intents**; the goal is to select the correct intent(s).

This project reframes it as a **structured-generation** task — the model must return its answer as a strict JSON object:

```json
{"answer": ["7"], "intent": ["To book flights or ask for general flight information."]}
```

The dataset is large (~262k training rows), aggregated from 49 sources, with both single-answer and multi-answer examples.

---

## Approach

- **Task contract** — one fixed prompt layout and JSON schema, applied identically at train, inference, and eval time.
- **Formatting** — each row is serialized into a chat-format `messages` conversation plus a JSON completion. Option labels are 0-based, so the answer label equals the dataset's `answer_index` with no arithmetic.
- **Data cleaning** — filtered **[N]** rows whose `answer_index` pointed past the available options (a construction artifact in the `dyndst` source, leaving unrecoverable labels).
- **Length analysis** — measured the token-length distribution (median ~194, 99th percentile ~953 tokens) and set `max_length=1024` to cover ~99% of examples.
- **QLoRA** — 4-bit quantized base (nf4) + a rank-16 LoRA adapter on all linear layers; only **~1.2%** of parameters are trainable (18.4M / 1.56B).
- **Training** — 1 epoch, learning rate 2e-4 (cosine schedule), effective batch **[BATCH]**, with assistant-only loss masking (loss computed only on the JSON answer).

---

## Evaluation

Both models were scored on held-out data with two independent metrics:

- **JSON validity** — did the output parse and match the schema?
- **Accuracy** — did the predicted answer *set* equal the true set? (Set comparison handles multi-answer cases.)

The base model was also scored **leniently** — extracting any integer from its output — to separate *"can't follow the format"* from *"can't do the task."* It scored **41.5% leniently** but only **0.5%** on strict schema conformance.

---

## Base vs Finetuned — Sample Outputs

The base model can often reason about intent, but its output is inconsistent — wrong key (`intent` instead of `answer`), integers instead of strings, markdown fences, or invalid syntax like `(3)`. The finetuned model emits clean, schema-correct JSON every time.

| Context (truncated) | True | Finetuned output | Base output |
| :--- | :---: | :--- | :--- |
| The exchange rate for foreign ATM currency is wrong | `["6"]` | `{"answer": ["6"], ...}` ✅ | `{"intent": 4}` ❌ |
| The card PIN is not visible anywhere? | `["8"]` | `{"answer": ["8"], ...}` ✅ | `{"intent": 8}` ⚠️ right index, wrong schema |
| remind me to shop for running shoes | `["4"]` | `{"answer": ["4"], ...}` ✅ | `{"intent": "To set the reminder."}` ❌ text, not index |
| Add an artist to my playlist | `["1"]` | `{"answer": ["1"], ...}` ✅ | ` ```json {"intent": (1)} ``` ` ❌ fenced, invalid |
| add an artist to my playlist (multi-intent) | `["0","1"]` | `{"answer": ["0","1"], ...}` ✅ | `{"intent": "To add music..."}` ❌ misses one |
| What is the traffic condition in Heliopolis now? | `["7"]` | `{"answer": ["7"], ...}` ✅ | `{"intent": (7)}` ⚠️ right index, invalid syntax |
| book me a cab | `["6"]` | `{"answer": ["6"], ...}` ✅ | `{"intent": (6)}` ⚠️ right index, invalid syntax |

*(Full 20-row comparison in the notebook.)*

---

## Key Findings

- **Format reliability was the biggest single win.** The base model almost never emits the required schema; the finetuned model does so 100% of the time.
- **The model generalizes across diverse input styles** — queries, dialogues, articles, chat histories — with consistent output.
- **Errors are dominated by over-prediction** on ambiguous multi-intent cases (e.g. answering `["7","8"]` where only `["7"]` was labeled), not systematic failures.

---

## Reproduce

```bash
pip install transformers datasets peft trl bitsandbytes accelerate
```

- **Dataset:** `yuweiyin/IntentGrasp` (Hugging Face; CC-BY-NC-SA, non-commercial)
- **Base model:** `Qwen/Qwen2.5-1.5B-Instruct`
- Run the notebook top to bottom.

---

## Notes

- Trained and evaluated on a single GPU using QLoRA (4-bit) to fit in memory.
- Dataset is non-commercial (CC-BY-NC-SA 4.0); this project is for research and learning.
