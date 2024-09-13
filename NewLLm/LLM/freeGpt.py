from freeGPT import AsyncClient
from PIL import Image
from io import BytesIO
from asyncio import run


async def main():
    while True:
        prompt = input("👦: ")
        try:
            resp = await AsyncClient.create_generation("pollinations", prompt)
            Image.open(BytesIO(resp)).show()
            print(f"🤖: Image shown.")
        except Exception as e:
            print(f"🤖: {e}")

run(main())