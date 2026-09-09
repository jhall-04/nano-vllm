from dataclasses import dataclass
import torch
import time
import asyncio
from transformers import AutoModelForCausalLM, AutoTokenizer, set_seed

  # Set a seed for reproducibility

model_name = "Qwen/Qwen2.5-0.5B-Instruct"

model = AutoModelForCausalLM.from_pretrained(
    model_name,
    dtype=torch.bfloat16,
    device_map="auto"
)
tokenizer = AutoTokenizer.from_pretrained(model_name)
"""
Sample parameters for the model:
{
  "model": "Qwen/Qwen2.5-0.5B-Instruct",
  "block_size": 16,
  "decoding": {
    "mode": "greedy",
    "temperature": 0.0,
    "top_p": 1.0,
    "seed": 1234,
    "ignore_eos": false
    }
}
"""

@dataclass
class Request:
    prompt: str
    max_new_tokens: int
    mode: str
    temperature: float
    top_p: float
    seed: int
    ignore_eos: bool
    future: asyncio.Future

class Engine:
    def __init__(self, max_batch=8, max_wait=1):
        self.queue = asyncio.Queue()
        self.max_batch = max_batch
        self.max_wait = max_wait
        self.semaphore = asyncio.Semaphore(1)

    async def submit(self, prompt, max_new_tokens=512, mode="greedy", temperature=0.7, top_p=5.0, seed=None, ignore_eos=False):
        future = asyncio.get_event_loop().create_future()
        request = Request(prompt, max_new_tokens, mode, temperature, top_p, seed, ignore_eos, future)
        await self.queue.put(request)
        return await future

    async def run_forever(self):
        while True:
            batch = await self._collect()
            await self._run_batch(batch)

    async def _collect(self):
        first = await self.queue.get()
        batch = [first]
        deadline = time.perf_counter() + self.max_wait
        while len(batch) < self.max_batch:
            remaining_time = deadline - time.perf_counter()
            if remaining_time <= 0:
                break
            try:
                request = await asyncio.wait_for(self.queue.get(), timeout=remaining_time)
                batch.append(request)
            except asyncio.TimeoutError:
                break
        return batch

    async def _run_batch(self, batch):
        prompts = [request.prompt for request in batch]
        max_new_tokens = max(request.max_new_tokens for request in batch)
        mode = batch[0].mode
        temperature = batch[0].temperature
        top_p = batch[0].top_p
        seed = batch[0].seed
        ignore_eos = batch[0].ignore_eos

        responses = await self.batch_generate(prompts, max_new_tokens, mode, temperature, top_p, seed, ignore_eos)

        for request, response in zip(batch, responses):
            request.future.set_result(response)

    def _batch_generate_sync(self, prompts, max_new_tokens, mode, temperature, top_p, seed, ignore_eos):
        if seed is not None:
            set_seed(seed)

        tokenizer.padding_side = "left"
        tokenizer.pad_token = tokenizer.eos_token

        batch = []
        sys_prompt = {"role": "system", "content": "You are a helpful assistant."}
        for prompt in prompts:
            batch.append([sys_prompt, {"role": "user", "content": prompt}])

        texts = [tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True) for messages in batch]

        model_inputs = tokenizer(texts, return_tensors="pt", padding=True).to(model.device)

        generated_ids = model.generate(
            **model_inputs,
            max_new_tokens=max_new_tokens,
            do_sample=(mode != "greedy"),
            temperature=temperature,
            top_p=top_p,
            eos_token_id=tokenizer.eos_token_id if not ignore_eos else None
        )
        generated_ids = [
            output_ids[len(input_ids):] for input_ids, output_ids in zip(model_inputs.input_ids, generated_ids)
        ]
        return tokenizer.batch_decode(generated_ids, skip_special_tokens=True)

    async def batch_generate(self, prompts, max_new_tokens, mode, temperature, top_p, seed, ignore_eos):
        # Run blocking tokenization/inference in a worker thread so the event loop
        # stays free to accept/reject requests instead of stalling under load.
        return await asyncio.to_thread(
            self._batch_generate_sync, prompts, max_new_tokens, mode, temperature, top_p, seed, ignore_eos
        )

    def _generate_sync(self, prompt, max_new_tokens, mode, temperature, top_p, seed, ignore_eos):
        messages = [
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": prompt}
        ]

        if seed is not None:
            set_seed(seed)

        text = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True
        )
        model_inputs = tokenizer([text], return_tensors="pt").to(model.device)

        generated_ids = model.generate(
            **model_inputs,
            max_new_tokens=max_new_tokens,
            do_sample=(mode != "greedy"),
            temperature=temperature,
            top_p=top_p,
            eos_token_id=tokenizer.eos_token_id if not ignore_eos else None
        )
        generated_ids = [
            output_ids[len(input_ids):] for input_ids, output_ids in zip(model_inputs.input_ids, generated_ids)
        ]
        return tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]

    async def generate(self, prompt, max_new_tokens=512, mode="greedy", temperature=0.7, top_p=5.0, seed=None, ignore_eos=False):
        # Run blocking tokenization/inference in a worker thread so the event loop
        # stays free to accept/reject requests instead of stalling under load.
        async with self.semaphore:
            return await asyncio.to_thread(
                self._generate_sync, prompt, max_new_tokens, mode, temperature, top_p, seed, ignore_eos
            )


class EngineContinuous:
        def __init__(self, max_batch=8, max_wait=1):
            self.queue = asyncio.Queue()
            self.max_batch = max_batch
            self.max_wait = max_wait
            self.semaphore = asyncio.Semaphore(1)