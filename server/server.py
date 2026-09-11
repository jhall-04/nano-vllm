from fastapi import FastAPI
from pydantic import BaseModel
import asyncio
from engine.engine import ModelRunner, Engine

app = FastAPI()

runner = ModelRunner()
engine = Engine(max_batch_size=16, Runner=runner)  # Initialize the engine instance
asyncio.create_task(engine.run_forever())  # Start the engine's run_forever loop in the background

class GenerateRequest(BaseModel):
    prompt: str = "Hello, world!"
    decode_params: dict = {}

@app.get("/")
def read_root():
    return {"Hello": "World"}

@app.post("/generate")
async def generate(request: GenerateRequest):
    engine_response = await engine.submit(request.prompt, **request.decode_params)
    return {"message": engine_response}