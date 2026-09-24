# IntentGrasp QLoRA Finetuning — Structured Intent Classification

Finetuned Qwen2.5-1.5B-Instruct with QLoRA to classify speaker intent and emit reliable structured JSON — raising accuracy from 41.5% to 90.5% and JSON validity from 0.5% to 100%.

## Result

| Model | JSON validity | Accuracy |
|-------|--------------|----------|
| Base Qwen2.5-1.5B (strict schema) | 0.5% | — |
| Base Qwen2.5-1.5B (lenient scoring) | — | 41.5% |
| **Finetuned (QLoRA)** — validation | **100%** | **90.5%** |
| **Finetuned (QLoRA)** — gem (harder split) | [GEM_VALIDITY]% | [GEM_ACC]% |

Finetuning delivered two distinct wins: it made output **format reliably conform** to the required JSON schema (0.5% → 100%), and it **more than doubled task accuracy** (41.5% → 90.5%), evaluated against the same base model.

## The Task

[IntentGrasp](https://huggingface.co/datasets/yuweiyin/IntentGrasp) is an intent-classification benchmark. Each example gives a **context** (a query, dialogue, or monologue), a **question**, and a list of candidate **intents**; the goal is to select the correct intent(s). This project reframes it as a **structured-generation** task: the model must output its answer as a strict JSON object, e.g. `{"answer": ["7"], "intent": ["To book flights..."]}`.

The dataset is large (~262k train rows) and aggregated from 49 sources, with intents ranging from single-answer to multi-answer.

## Approach

- **Task contract** — a fixed prompt layout and JSON output schema, applied identically at train, inference, and eval time.
- **Formatting** — each row serialized into a chat-format `messages` conversation plus a JSON completion; option labels are 0-based so the answer label equals the dataset's `answer_index` with no arithmetic.
- **Data cleaning** — filtered [N] rows whose `answer_index` pointed past the available options (a construction artifact in the `dyndst` source); these have unrecoverable labels.
- **Length analysis** — measured the token-length distribution (median ~194, 99th percentile ~953 tokens) and set `max_length=1024` to cover ~99% of examples.
- **QLoRA** — 4-bit quantized base (nf4) + rank-16 LoRA adapter on all linear layers; ~1.2% of parameters trainable (18.4M / 1.56B).
- **Training** — 1 epoch, learning rate 2e-4 with cosine schedule, effective batch [BATCH], assistant-only loss masking (loss computed only on the JSON answer).

## Evaluation

Both models were scored on held-out data with two metrics:
- **JSON validity** — did the output parse and match the schema?
- **Accuracy** — did the predicted answer *set* equal the true set (set comparison handles multi-answer cases)?

The base model was also scored **leniently** (extracting any integer from its output) to separate "can't follow the format" from "can't do the task" — it scored 41.5% leniently but 0.5% on strict schema conformance.

## Base vs Finetuned (hard cases)

| context                                                      | true_answer   | finetuned_output                                                       | base_output                                                            |
|:-------------------------------------------------------------|:--------------|:-----------------------------------------------------------------------|:-----------------------------------------------------------------------|
| costumer: The exchange rate for foreign ATM currency is wron | ['6']         | {"answer": ["6"], "intent": ["To inquire about the wrong exchange rate | {"intent": 4}                                                          |
| user: thank you please contact me date1 date2 via text works | ['7']         | {"answer": ["7", "8"], "intent": ["To call now or schedule a call.", " | {"intent": (3)}                                                        |
| costumer: The card PIN is not visible anywhere?              | ['8']         | {"answer": ["8"], "intent": ["To inquire about getting the PIN."]}     | {"intent": 8}                                                          |
| ### Situation: Ted is walking down a street and sees a boy t | ['5']         | {"answer": ["5"], "intent": ["To encourage the boy."]}                 | ```json                                                                |
|                                                              |               |                                                                        | {"intent": (5)}                                                        |
|                                                              |               |                                                                        | ```                                                                    |
| user: remind me to shop for running shoes                    | ['4']         | {"answer": ["4"], "intent": ["To set the reminder."]}                  | {"intent": "To set the reminder."}                                     |
| user: Add an artist to my playlist Domingo Indie             | ['1']         | {"answer": ["1"], "intent": ["To add music to the playlist."]}         | ```json                                                                |
|                                                              |               |                                                                        | {                                                                      |
|                                                              |               |                                                                        |   "intent": (1)                                                        |
|                                                              |               |                                                                        | }                                                                      |
|                                                              |               |                                                                        | ```                                                                    |
| costumer: My card payment wasn't declined                    | ['5']         | {"answer": ["5"], "intent": ["To inquire about the declined card payme | ```json                                                                |
|                                                              |               |                                                                        | {"intent": (5)}                                                        |
|                                                              |               |                                                                        | ```                                                                    |
| user: Add circus to my  Post Garage Wave Revival list        | ['0']         | {"answer": ["0"], "intent": ["To add music to the playlist."]}         | {"intent": (0)}                                                        |
| user: Can you help me to find an occupation                  | ['0']         | {"answer": ["0"], "intent": ["To ask for career advice."]}             | {"intent": "To ask for career advice."}                                |
| ### Dialogue:                                                | ['5']         | {"answer": ["5"], "intent": ["To find the taxi."]}                     | {"intent": "To find tourist attraction."}                              |
| user: Hello! I'm looking for a museum to go to               |               |                                                                        |                                                                        |
| costumer: So, I was under the impression that you have an of | ['0']         | {"answer": ["0"], "intent": ["To inquire about the reverted top-up."]} | {"intent": (1)}                                                        |
| user: add an artist to my playlist this is verdi ; is the sl | ['0', '1']    | {"answer": ["0", "1"], "intent": ["To search for screening events.", " | ```json                                                                |
|                                                              |               |                                                                        | {"intent": "To add music to the playlist."}                            |
|                                                              |               |                                                                        | ```                                                                    |
| ### Article:                                                 | ['1']         | {"answer": ["1"], "intent": ["To state, mention, or point to the backg | {"intent": "To state, mention, or point to the background information. |
| Paper Section: Discussion                                    |               |                                                                        |                                                                        |
| Paper Content: Anothe                                        |               |                                                                        |                                                                        |
| user: What is the traffic condition in Heliopolis now?       | ['7']         | {"answer": ["7"], "intent": ["To get the traffic information."]}       | {"intent": (7)}                                                        |
| user: book me a cab                                          | ['6']         | {"answer": ["6"], "intent": ["To book a taxi."]}                       | {"intent": (6)}                                                        |
| ### Chat History:                                            | ['6']         | {"answer": ["6"], "intent": ["To provide further details."]}           | {"intent": "to ask a follow-up question"}                              |
| Title: How to force fix file systems durin                   |               |                                                                        |                                                                        |
| user: Will you let me know how much I owe you                | ['6']         | {"answer": ["6"], "intent": ["To inquire about the amount due."]}      | {"intent": 6}                                                          |
| ### Text Before Edit:                                        | ['2']         | {"answer": ["3"], "intent": ["To improve the clarity."]}               | ```json                                                                |
| There are situations where the total a                       |               |                                                                        | {"intent": "To improve the clarity"}                                   |
|                                                              |               |                                                                        | ```                                                                    |
| user: i want to hear something funny about history           | ['0']         | {"answer": ["0"], "intent": ["To tell jokes."]}                        | {"intent": (0)}                                                        |
| user: remind me to buy a gift for mothers day                | ['8']         | {"answer": ["8"], "intent": ["To set the reminder."]}                  | {"intent": 6}                                                          |
The base model often reasons about intent correctly but produces inconsistent, non-conforming output — wrong keys (`intent` instead of `answer`), markdown fences, integers instead of strings, or invalid syntax like `(3)`. The finetuned model emits clean, schema-correct JSON on every example.

## Key Findings

- Finetuning's **biggest single win was format reliability** — the base model almost never emits the required schema, while the finetuned model does so 100% of the time.
- The model **generalizes across diverse input styles** (queries, dialogues, articles, chat histories) with consistent output.
- Errors are dominated by **over-prediction on ambiguous multi-intent cases** — e.g. answering `["7", "8"]` where only `["7"]` was labeled — rather than systematic failures.

## Reproduce

1. Install: `pip install transformers datasets peft trl bitsandbytes accelerate`
2. Dataset: `yuweiyin/IntentGrasp` (loaded from Hugging Face; CC-BY-NC-SA, non-commercial).
3. Base model: `Qwen/Qwen2.5-1.5B-Instruct`.
4. Run the notebook top to bottom.

## Notes

- Trained and evaluated on a single GPU using QLoRA (4-bit) to fit in memory.
- The dataset is non-commercial (CC-BY-NC-SA 4.0); this project is for research/learning.
