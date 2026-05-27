from openai import OpenAI

# Point to your local vLLM server
client = OpenAI(
    api_key="EMPTY",  # vLLM ignores this
    base_url="http://localhost:8000/v1",
)

response = client.chat.completions.create(
    model="Qwen/Qwen3-4B",
    messages=[
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": "Explain transformers in simple terms."},
    ],
    temperature=0.0,
    max_tokens=512,
    extra_body={
        "chat_template_kwargs": {
            "enable_thinking": False
        }
    }
)

print(response.choices[0].message.content)