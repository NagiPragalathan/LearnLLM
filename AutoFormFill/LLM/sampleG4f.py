from g4f.client import Client
client = Client()

chat_completion = client.chat.completions.create(
            # model="gpt-3.5-turbo",
            model="gpt-4",
            messages=[
                {"role": "System", "content": "you are a useful ai to analyze the risk, possibility of managements and loan based on given json data if it good Approve or decline."},
                {"role": "user", "content": "give me the output in the json format like {'LoneStatus':Approve or not based on given data, 'generated report':'give me the report why the approve approved or declined. what the advantages or disadvantages''}.Note: dont give any other instructions and ack i just need the json structure output."}
                ]
        )
ai_response = chat_completion.choices[0].message.content or ""
print(ai_response)