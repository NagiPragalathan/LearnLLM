from g4f.client import Client

client = Client()
response = client.chat.completions.create(
    model="gpt-3.5-turbo",
    messages=[
        {"role": "system", "content": """You are a Ai to understand tamil+english and reply to user in tamil+english
    Note: Dont give me explanation in English
    """},
        {"role": "user", "content": "enna pandra"}
    ],
)
print(response.choices[0].message.content)
