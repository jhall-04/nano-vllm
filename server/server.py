from fastapi import FastAPI
import asyncio

app = FastAPI()

@app.get("/")
def read_root():
    return {"Hello": "World"}

@app.get("/generate")
async def generate(prompt: str = "Hello, world!"):
    await asyncio.sleep(1)  # Simulate some processing time
    return {"message": f"This is a generated response to: {prompt}."}