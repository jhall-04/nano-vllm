from collections import deque
import asyncio
from transformers import AutoModelForCausalLM, AutoTokenizer, set_seed

  # Set a seed for reproducibility

model_name = "Qwen/Qwen2.5-0.5B-Instruct"

model = AutoModelForCausalLM.from_pretrained(
    model_name,
    torch_dtype="auto",
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

class Engine:
    def __init__(self):
        self.queue = deque()
        # Model is single-instance/single-GPU, so only run one generation at a time.
        self.semaphore = asyncio.Semaphore(1)

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
