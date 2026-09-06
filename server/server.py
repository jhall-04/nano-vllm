from fastapi import FastAPI
from pydantic import BaseModel
import asyncio
from engine.engine import Engine

app = FastAPI()

engine = Engine()  # Initialize the engine instance

class GenerateRequest(BaseModel):
    prompt: str = "Hello, world!"
    decode_params: dict = {}

@app.get("/")
def read_root():
    return {"Hello": "World"}

@app.post("/generate")
async def generate(request: GenerateRequest):
    engine_response = await engine.generate(request.prompt, **request.decode_params)
    return {"message": engine_response}