from huggingface_hub import login
#login()
from transformers import pipeline;
# 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
# ollama pull llama3.2:1b

messages = [
    {"role": "user", "content": "你知道什麼，不知道什麼"},
]
max_new_tokens = '40'
pipe = pipeline("text-generation", model="meta-llama/Llama-3.2-1B-Instruct", max_new_tokens=500)
a = pipe(messages)
print(a)
messages = [
    {"role": "user", "content": "你是什​​麼意思？"},
]
a = pipe(messages)
print(a)