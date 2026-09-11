# Code Review: Batching (Checkpoint 2) — before continuous batching

Reviewed: `engine/engine.py`, `server/server.py`, `bench/load_gen.py`, `bench/metrics.py`,
diff `main...batching`, and the uncommitted `EngineContinuous` stub, against `logs/logs-v1` and
`logs/logs-v2`.

## Numbers first

Computed from your own logs (`response_time`, seconds):

| QPS | v1 mean | v1 p95 | v2 mean | v2 p95 |
|-----|---------|--------|---------|--------|
| 1 | 0.24 | 0.25 | **1.06** | 1.50 |
| 2 | 0.34 | 1.03 | 0.97 | 1.56 |
| 4 | 6.22 | 10.83 | **0.97** | 1.55 |
| 8 | 19.30 | 36.38 | **0.78** | 1.21 |

Batching is a huge win at 4/8 QPS (up to ~25x at the tail), but it's a **4.4x regression at 1
QPS**. That's `max_wait=1`: at low load every request sits out almost the whole window alone
before dispatch, whereas v1 just ran it immediately. work-log-2 says "much better results"
without that caveat — worth noting explicitly, because it also motivates continuous batching:
once dispatch is driven by free GPU slots instead of a fixed timer, a lone request doesn't have
to wait for a timeout.

Golden set: 20/20 on both v1 and v2. See "golden set can't see this" below for why that's less
reassuring than it looks.

## Blocking correctness issues

### 1. `_run_batch` silently discards per-request decode params — `engine/engine.py:80-83`

```python
mode = batch[0].mode
temperature = batch[0].temperature
top_p = batch[0].top_p
seed = batch[0].seed
```

Every request in the batch except the first is generated with request 1's sampling settings. A
caller who asks for `temperature=1.2` can silently get greedy output if they land behind a
greedy request in the queue. The result depends on *who else happened to be queued at that
moment* — load-dependent and non-reproducible, and there's nothing in the logs that would show
it happened. This is the most serious issue in the diff. Fix: generate per-request, or at minimum
validate all requests in a batch share compatible params and split the batch when they don't.

### 2. `max_new_tokens = max(...)` violates the request's own budget — `engine/engine.py:78`

A request asking for 8 tokens can get up to 512, because HF `generate()` only stops once *every*
sequence in the batch hits EOS or its own `max_new_tokens` — but you're passing one shared
`max_new_tokens` to the whole call. Two consequences: the short-budget caller gets more tokens
than it asked for, and everyone in the batch pays the wall-clock cost of the longest request
(head-of-line blocking, on top of the padding waste you already flagged in work-log-2).

### 3. The golden set can't catch either #1 or #2

`bench/load_gen.py:50` (`golden_test`) calls `asyncio.run(send_request(...))` once per prompt,
sequentially. Every golden request is therefore a batch of size 1 — the batching code path is
essentially untested by your only correctness check. `p01` (`max_new_tokens: 8`) would fail today
if it happened to be queued behind a 512-token request, and nothing would tell you.

**Action:** add a concurrent golden mode that fires all prompts at once with mixed decode params
and asserts each response still matches its own baseline. This single test would have caught #1
and #2 directly, and it's the test you want in place *before* continuous batching adds more ways
for requests to interleave.

### 4. Engine death → permanent hang — `engine/engine.py:56-59`

```python
while True:
    batch = await self._collect()
    await self._run_batch(batch)
```

No `try/except`. I reproduced the failure shape with a toy version of the same loop:

```
first: ok:a
'boom' request: hung (never resolved, no error returned to client)
later request 'c': hung too - engine is dead but server still accepts traffic
run_forever done? True | exception never surfaced to logs unless retrieved
```

One OOM, one bad param, one exception anywhere in `_run_batch` and the loop exits for good.
Every future already in flight hangs until the *caller's* timeout (if any); the server keeps
accepting new requests into a queue nothing is draining; and because nothing ever awaits the
`run_forever` task, the traceback is swallowed — asyncio only surfaces it via a "Task exception
was never retrieved" warning at GC time, easy to miss.

**Fix:** wrap `_run_batch` (or its body) in try/except; on failure, call
`future.set_exception(e)` for every request in that batch, log it, and `continue` the loop.

### 5. `asyncio.create_task` at import time — `server/server.py:9`

```python
engine = Engine()
asyncio.create_task(engine.run_forever())
```

This works today only by luck: uvicorn calls `config.load()` (which imports your app module)
from inside `Server._serve`, which is already running in the event loop. Outside that exact
path it breaks immediately — confirmed directly:

```
RuntimeError: no running event loop
```

So `import server.server` from a plain test file raises. Separately, the task object is
discarded immediately — asyncio holds only a weak reference to tasks, so it's a documented GC
hazard even when it does work — and there's no clean shutdown.

**Fix:** start the loop in a FastAPI `lifespan` handler, keep a reference to the task, cancel it
on shutdown.

### 6. No backpressure

`asyncio.Queue()` is unbounded and `submit()` has no timeout. v1 crashed outright under load; v2
will instead accept unlimited work and let latency grow without bound instead of rejecting
requests. For your benchmarks to mean anything under saturation, you want rejection to be a
visible, measured outcome, not silent queueing. Give the queue a `maxsize` and return e.g. HTTP
429 when full (or a queue-wait timeout in `submit`).

### 7. No client-disconnect handling

An abandoned request still occupies a batch slot for its full generation. Cheap to ignore now;
gets more expensive under continuous batching, where a dead sequence holds a slot for its entire
lifetime instead of just until the next static batch boundary.

## Things I checked and are fine

- `output_ids[len(input_ids):]` (`engine.py:115`) is only correct because left-padding makes all
  inputs in a batch equal length. It'll break the moment padding changes — worth a one-line
  comment so future-you doesn't relearn this the hard way.
- `_collect`'s "deadline from first arrival" design is the right shape for static batching.
- Checked whether `asyncio.wait_for(queue.get())` timing out can silently drop an item that was
  about to arrive — it can't; `asyncio.Queue.get()` re-wakes the next waiter on cancellation.
  Not a bug.

## Design issues that block the continuous-batching goal specifically

### 8. Model/tokenizer are import-time module globals — `engine/engine.py:9-16`

Just `import engine` loads a multi-hundred-MB model. This is the concrete reason
`tests/tests.py` is empty, and it's the main thing standing between you and the "multiple engine
versions for easier benchmarking" idea from your work-log-2 side note. Extract a `ModelRunner`
that owns the model/tokenizer, inject it into engines: `EngineV1(runner)`, `EngineBatch(runner)`,
`EngineContinuous(runner)` all share one loaded model and become independently importable/testable.

### 9. Dead code, but keep it

`generate()`, `_generate_sync()`, and `self.semaphore` are unreachable now that
`server.py:21` calls `submit()` instead. Don't delete — move to `engine/v1.py` behind the same
interface (`submit`/`generate` returning the same shape). That turns "build multiple engine
versions for benchmarking" from an aspiration into three files with a shared `ModelRunner`.

### 10. `top_p=5.0` is out of range — `engine/engine.py:50,155`

Valid range is `(0, 1]`. Checked: `transformers`' `GenerationConfig.validate()` accepts `5.0`
with no warning and no error, so today it's silently inert only because `do_sample=False`
(greedy) ignores `top_p` entirely. The moment anyone passes `mode="sample"` this becomes a live,
undiagnosed bug. Set the default to `1.0`.

### 11. Tokenizer mutated per call from a worker thread — `engine.py:94-95`

`tokenizer.padding_side` and `tokenizer.pad_token` are set on the shared global tokenizer on
every single batch call. Harmless with today's single worker thread; set once at load time
instead so it isn't a latent race once there's more than one path calling into the tokenizer.

## Benchmark methodology

You're about to claim a speedup from continuous batching — the harness needs to be able to see
the problem before you can prove you fixed it.

### 12. Load test can't see the problem continuous batching solves

Every request in `bench/load_gen.py` is the literal string `"Hello, world!"` — identical prompt,
identical length. Padding waste is therefore ~zero in your current benchmark, even though
work-log-2 correctly identifies it as the real cost of static batching. Add a **mixed-length
prompt distribution** before you start optimizing, or you'll have no measurable before/after.

### 13. New `aiohttp.ClientSession` per request — `load_gen.py:11`

480 sessions get created/torn down per 8 QPS run, and connection setup time is counted inside
`response_time`. Share one session across the whole run.

### 14. Only end-to-end latency is recorded

For continuous batching specifically you want queue-wait time separated from generation time,
plus time-to-first-token, inter-token latency, tokens/sec, and batch occupancy over time.
TTFT is the metric continuous batching moves the most, and right now there's no way to see it.
This pairs naturally with the plan (already in work-log-2) to drop `model.generate()` for a
hand-rolled decode loop — that's where this instrumentation naturally goes.

### 15. Open-loop timing drift

`await asyncio.sleep(1/qps)` accumulates event-loop scheduling overhead across the run instead of
scheduling against absolute deadlines — you got 477/480 expected requests at 8 QPS. Not a big
deal at current scale, but schedule against `start_time + i/qps` instead of chained sleeps if you
push QPS higher.

### 16. `metrics.py:5` reopens the output file on every line

Trivial, but open once and write all lines, rather than `open(filename, 'a')` per metric.

## Suggested order of attack

1. Fix per-request decode params (#1) and per-request `max_new_tokens` (#2); wrap the batch loop
   in try/except (#4). These are bugs in already-shipped behavior, independent of continuous
   batching.
2. Add the concurrent golden test (#3) — confirm it fails against *today's* code first, that's
   your proof the fix in step 1 mattered.
3. Extract `ModelRunner`; move v1 into `engine/v1.py` (#8, #9); move engine startup into a
   FastAPI lifespan (#5).
4. Mixed-length prompts + queue-wait/TTFT metrics (#12, #14).
5. Then start continuous batching, on top of a harness that can actually show the improvement.

Steps 1–4 are roughly a day of work and they turn step 5 from "vibes" into something you can put
a number on. The 1 QPS regression from static `max_wait` batching is worth writing into
work-log-3 explicitly as a known, expected cost that continuous batching removes — admission
becomes slot-driven instead of timer-driven, so a lone request dispatches immediately instead of
waiting out the window.

One thing the work log already gets right: reading about continuous batching before implementing
it is the correct call here. The hard part is the scheduler state machine (waiting / running /
preempted requests, KV-cache block admission), not the decode loop itself — that's where the real
design risk is, not in swapping out `model.generate()`.
