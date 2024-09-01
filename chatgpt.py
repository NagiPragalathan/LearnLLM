from g4f import ChatCompletion

# Initialize the ChatCompletion class
chat = ChatCompletion()

# Create a list of messages
messages = [
    {"role": "system", "content": "You are a helpful assistant."},
    {"role": "user", "content": "Once upon a time, in a land far, far away,"}
]

# Specify the model to use
model = "gpt-3.5-turbo"  # Replace with the model name you want to use

# Generate a response
response = chat.create(model=model, messages=messages)

print("Generated Text:", response)
# print(response['choices'][0]['message']['content'])
