from dataclasses import dataclass, field
import asyncio
import itertools
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from collections import deque
from typing import Dict, List, Optional

class ModelRunner:
    def __init__(self, model_name: str = "Qwen/Qwen2.5-0.5B-Instruct"):
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,
            dtype=torch.bfloat16,
            device_map="auto"
        )
        self.model.eval()
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        if self.tokenizer.pad_token_id is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        # set once here rather than per-call, since these are shared mutable state on the tokenizer
        self.tokenizer.padding_side = "left"

    def use_model(self):
        return self.model, self.tokenizer


@dataclass
class ContinuousRequest:
    """Request with tracking for continuous batching."""

    id: int
    prompt: str
    output_length: int  # Max tokens to generate
    input_length: int = 0
    # Running sequence (prompt + generated so far). Populated when the request enters the batch.
    input_ids: Optional[torch.Tensor] = None
    tokens_generated: int = 0
    start_iteration: int = 0
    end_iteration: int = 0
    hit_eos: bool = False

    mode: str = "greedy"
    temperature: float = 1.0
    top_p: float = 1.0
    seed: int = 1234
    ignore_eos: bool = False

    # Per-request RNG so sampling is reproducible independent of what else is in the batch.
    generator: Optional[torch.Generator] = field(default=None, repr=False)

    @property
    def is_complete(self) -> bool:
        if self.tokens_generated >= self.output_length:
            return True
        return self.hit_eos and not self.ignore_eos

## Modified from https://mbrenndoerfer.com/writing/continuous-batching
class Engine:
    """Continuous batching engine."""

    def __init__(self, max_batch_size: int = 8, Runner: ModelRunner = None):
        self.max_batch_size = max_batch_size
        self.Runner = Runner
        self.active_batch: List[ContinuousRequest] = []
        self.waiting_queue: deque[ContinuousRequest] = deque()
        self.completed: List[ContinuousRequest] = []
        self.iteration = 0
        self.utilization_history = []
        self._next_id = itertools.count(1)
        self._pending: Dict[int, asyncio.Future] = {}

        # KV cache from the last step, reused as long as the batch's membership/order
        # doesn't change. Whenever a request joins or leaves, the cache is invalidated
        # and rebuilt with one full recompute - every other step is an O(1) incremental decode.
        self._kv_cache = None
        self._cache_batch_ids: Optional[List[int]] = None
        self._attention_mask: Optional[torch.Tensor] = None

    def submit_request(self, request: ContinuousRequest):
        """Add a request to the waiting queue."""
        self.waiting_queue.append(request)

    async def submit(self, prompt: str, **decode_params) -> str:
        """Old server schema: decode_params uses max_new_tokens instead of output_length."""
        output_length = decode_params.pop("max_new_tokens", 512)
        request = ContinuousRequest(
            id=next(self._next_id),
            prompt=prompt,
            output_length=output_length,
            **decode_params,
        )
        future = asyncio.get_running_loop().create_future()
        self._pending[request.id] = future
        self.submit_request(request)
        return await future

    async def run_forever(self, poll_interval: float = 0.01):
        """Drive the engine loop and resolve pending futures as requests complete."""
        while True:
            if self.active_batch or self.waiting_queue:
                try:
                    await asyncio.to_thread(self.step)
                    self._resolve_completed()
                except Exception as exc:
                    # A bad step must not permanently wedge the loop - fail every
                    # in-flight request and reset so future submissions still work.
                    print(f"Engine step failed, resetting batch: {exc!r}")
                    for future in self._pending.values():
                        if not future.done():
                            future.set_exception(exc)
                    self._pending.clear()
                    self.active_batch = []
                    self._kv_cache = None
                    self._cache_batch_ids = None
                    self._attention_mask = None
            else:
                await asyncio.sleep(poll_interval)

    def _resolve_completed(self):
        while self.completed:
            request = self.completed.pop(0)
            future = self._pending.pop(request.id, None)
            if future and not future.done():
                future.set_result(self.decode_output(request))

    def decode_output(self, request: ContinuousRequest) -> str:
        """Decode only the tokens generated after the prompt."""
        _, tokenizer = self.Runner.use_model()
        new_tokens = request.input_ids[request.input_length:]
        return tokenizer.decode(new_tokens, skip_special_tokens=True)

    def _fill_batch(self):
        """Fill empty slots with waiting requests."""
        while (
            len(self.active_batch) < self.max_batch_size and self.waiting_queue
        ):
            request = self.waiting_queue.popleft()
            request.start_iteration = self.iteration
            if request.input_ids is None:
                _, tokenizer = self.Runner.use_model()
                # Instruct models expect the chat template; raw prompt text reads as a
                # completion continuation instead of an instruction to respond to.
                text = tokenizer.apply_chat_template(
                    [{"role": "user", "content": request.prompt}],
                    tokenize=False,
                    add_generation_prompt=True,
                )
                ids = tokenizer(text, return_tensors="pt", add_special_tokens=False).input_ids[0]
                request.input_ids = ids
                request.input_length = ids.shape[-1]
            self.active_batch.append(request)

    def _sample_next_token(self, logits: torch.Tensor, request: ContinuousRequest) -> int:
        """Pick the next token for a single request using that request's own decode params."""
        if request.mode == "greedy":
            return int(torch.argmax(logits).item())

        if request.generator is None:
            request.generator = torch.Generator(device=logits.device).manual_seed(request.seed)

        temperature = max(request.temperature, 1e-5)
        probs = torch.softmax(logits / temperature, dim=-1)

        if request.top_p < 1.0:
            sorted_probs, sorted_idx = torch.sort(probs, descending=True)
            cumulative = torch.cumsum(sorted_probs, dim=-1)
            cutoff = cumulative - sorted_probs > request.top_p
            sorted_probs[cutoff] = 0.0
            sorted_probs /= sorted_probs.sum()
            probs = torch.zeros_like(probs).scatter(-1, sorted_idx, sorted_probs)

        next_token = torch.multinomial(probs, num_samples=1, generator=request.generator)
        return int(next_token.item())

    def _process_completions(self):
        """Remove completed requests from the batch."""
        still_active = []
        for request in self.active_batch:
            if request.is_complete:
                request.end_iteration = self.iteration
                self.completed.append(request)
            else:
                still_active.append(request)
        self.active_batch = still_active

    def _pad_rows(self, requests: List[ContinuousRequest], target_len: int, pad_id: int):
        """Left-pad each request's sequence to target_len; returns (input_ids, attention_mask)."""
        input_rows = []
        mask_rows = []
        for request in requests:
            seq = request.input_ids
            pad_len = target_len - seq.shape[-1]
            if pad_len:
                seq = torch.cat([torch.full((pad_len,), pad_id, dtype=seq.dtype), seq])
            input_rows.append(seq)
            mask_rows.append(torch.cat([
                torch.zeros(pad_len, dtype=torch.long),
                torch.ones(target_len - pad_len, dtype=torch.long),
            ]))
        return torch.stack(input_rows), torch.stack(mask_rows)

    def _prefill(self, requests: List[ContinuousRequest], model, pad_id: int):
        """Full forward pass over requests' sequences from scratch, returning (outputs, attention_mask)."""
        target_len = max(request.input_ids.shape[-1] for request in requests)
        input_ids, attention_mask = self._pad_rows(requests, target_len, pad_id)
        input_ids = input_ids.to(model.device)
        attention_mask = attention_mask.to(model.device)
        position_ids = (attention_mask.cumsum(-1) - 1).clamp(min=0)
        with torch.no_grad():
            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                position_ids=position_ids,
                use_cache=True,
            )
        return outputs, attention_mask

    def _admit_new_requests(self, new_requests: List[ContinuousRequest], model, pad_id: int):
        """Prefill only the newly admitted requests and splice their KV rows onto the existing cache."""
        target_len = self._attention_mask.shape[-1]
        prompt_len = max(request.input_ids.shape[-1] for request in new_requests)
        if prompt_len > target_len:
            # A new prompt is longer than anything seen so far - extend everyone's
            # cache/mask on the left; the attention mask keeps the padding invisible.
            extra = prompt_len - target_len
            device = self._attention_mask.device
            pad_mask = torch.zeros(self._attention_mask.shape[0], extra, dtype=torch.long, device=device)
            self._attention_mask = torch.cat([pad_mask, self._attention_mask], dim=-1)
            for layer in self._kv_cache.layers:
                pad_shape = (layer.keys.shape[0], layer.keys.shape[1], extra, layer.keys.shape[3])
                zeros = torch.zeros(pad_shape, dtype=layer.keys.dtype, device=layer.keys.device)
                layer.keys = torch.cat([zeros, layer.keys], dim=2)
                layer.values = torch.cat([zeros, layer.values], dim=2)
            target_len = prompt_len

        input_ids, attention_mask = self._pad_rows(new_requests, target_len, pad_id)
        input_ids = input_ids.to(model.device)
        attention_mask = attention_mask.to(model.device)
        position_ids = (attention_mask.cumsum(-1) - 1).clamp(min=0)

        with torch.no_grad():
            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                position_ids=position_ids,
                use_cache=True,
            )

        new_cache = outputs.past_key_values
        for layer, new_layer in zip(self._kv_cache.layers, new_cache.layers):
            layer.keys = torch.cat([layer.keys, new_layer.keys], dim=0)
            layer.values = torch.cat([layer.values, new_layer.values], dim=0)
        self._attention_mask = torch.cat([self._attention_mask, attention_mask], dim=0)

    def step(self):
        """Execute one iteration of the continuous batch."""
        # First, fill any empty slots
        self._fill_batch()

        if not self.active_batch:
            return False  # Nothing to process

        # Record utilization
        self.utilization_history.append(
            len(self.active_batch) / self.max_batch_size
        )

        model, tokenizer = self.Runner.use_model()
        pad_id = tokenizer.pad_token_id
        current_ids = [request.id for request in self.active_batch]

        if self._kv_cache is None:
            # Very first batch: nothing to reuse yet.
            outputs, self._attention_mask = self._prefill(self.active_batch, model, pad_id)
        else:
            # Requests that survived from the previous step keep their cache rows
            # (in the same relative order _process_completions/_fill_batch preserve);
            # anything appended after them this step is a brand new arrival.
            old_ids = self._cache_batch_ids
            surviving_positions = [i for i, rid in enumerate(old_ids) if rid in current_ids]
            num_survivors = len(surviving_positions)
            new_requests = self.active_batch[num_survivors:]

            if num_survivors != len(old_ids):
                # Some requests completed and left - just drop their rows, no recompute.
                indices = torch.tensor(surviving_positions, dtype=torch.long)
                self._kv_cache.batch_select_indices(indices)
                self._attention_mask = self._attention_mask[surviving_positions]

            if new_requests:
                # Only prefill the new arrivals, not the whole (possibly much larger) batch.
                self._admit_new_requests(new_requests, model, pad_id)

            # One incremental step for the whole (merged) batch: one new token each, reusing the cache.
            input_ids = torch.stack(
                [request.input_ids[-1:] for request in self.active_batch]
            ).to(model.device)
            self._attention_mask = torch.cat(
                [self._attention_mask, torch.ones(len(self.active_batch), 1, dtype=torch.long, device=model.device)],
                dim=-1,
            )
            position_ids = (self._attention_mask.cumsum(-1) - 1)[:, -1:]

            with torch.no_grad():
                outputs = model(
                    input_ids=input_ids,
                    attention_mask=self._attention_mask,
                    position_ids=position_ids,
                    past_key_values=self._kv_cache,
                    use_cache=True,
                )

        self._kv_cache = outputs.past_key_values
        self._cache_batch_ids = current_ids
        next_token_logits = outputs.logits[:, -1, :]

        # Sample one token per request, using that request's own decode params.
        for i, request in enumerate(self.active_batch):
            next_token_id = self._sample_next_token(next_token_logits[i], request)
            next_token = torch.tensor(
                [next_token_id], dtype=request.input_ids.dtype
            )
            request.input_ids = torch.cat([request.input_ids, next_token])
            request.tokens_generated += 1
            if next_token_id == tokenizer.eos_token_id:
                request.hit_eos = True

        self.iteration += 1

        # Check for completions
        self._process_completions()

        return True

    def run_until_complete(self):
        """Run until all requests are processed."""
        while self.active_batch or self.waiting_queue:
            self.step()