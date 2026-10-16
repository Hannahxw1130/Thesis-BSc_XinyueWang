# Model Generation Parameters
## These are all the explicitly configured generation parameters

| Model | Model provider | API / inference provider | Model ID |
|---|---|---|---|
| GPT-5.4 | OpenAI | OpenAI Responses API | `gpt-5.4-2026-03-05` |
| Claude Sonnet 4.6 | Anthropic | Anthropic Messages API | `claude-sonnet-4-6` |
| Qwen-3.6-27B | Qwen | Hugging Face Inference Providers / Featherless AI | `Qwen/Qwen3.6-27B:featherless-ai` |
| Gemma-4-31B-it | Google | Hugging Face Inference Providers / Novita | `google/gemma-4-31B-it:novita` |

## GPT-5.4

```python
MODEL_NAME = "gpt-5.4-2026-03-05"
TEXT_COLUMN = "raw_text"
post_text = str(row[TEXT_COLUMN])

# client.responses.create(...) parameters
model=MODEL_NAME,
input=[{"role": "user", "content": post_text}],
reasoning={"effort": "none"},
temperature=0,
store=False
```

**Reasoning:** Explicitly disabled (`effort="none"`).

## Claude Sonnet 4.6

```python
MODEL_NAME = "claude-sonnet-4-6"
TEXT_COLUMN = "raw_text"
post_text = str(row[TEXT_COLUMN])

# client.messages.create(...) parameters
model=MODEL_NAME,
temperature=0,
messages=[{"role": "user", "content": post_text}]
```

**Thinking:** Not enabled in the request (disabled by default for this model/API configuration).

## Qwen-3.6-27B

```python
MODEL_NAME = "Qwen/Qwen3.6-27B:featherless-ai"
TEXT_COLUMN = "raw_text"
post_text = str(row[TEXT_COLUMN])
base_url="https://router.huggingface.co/v1"

# client.chat.completions.create(...) parameters
model=MODEL_NAME,
temperature=0,
messages=[{"role": "user", "content": post_text}],
extra_body={"chat_template_kwargs": {"enable_thinking": False}}
```

**Thinking:** Disabling requested through the inference provider (`enable_thinking=False`).

## Gemma-4-31B-it

```python
MODEL_NAME = "google/gemma-4-31B-it:novita"
TEXT_COLUMN = "raw_text"
post_text = str(row[TEXT_COLUMN])
base_url="https://router.huggingface.co/v1"

# client.chat.completions.create(...) parameters
model=MODEL_NAME,
temperature=0,
messages=[{"role": "user", "content": post_text}],
extra_body={"chat_template_kwargs": {"enable_thinking": False}}
```

**Thinking:** Disabling requested through the inference provider (`enable_thinking=False`).



**Top-k, top-p, and random seed were not explicitly set** in any of the four models; any applicable provider defaults were used.
