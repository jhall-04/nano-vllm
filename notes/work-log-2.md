# Checkpoint 2 Notes
This is going to be a short one. The pivot from individual requests to batching was relatively straight forward.

## Batching
To create batches the engine running the inference maintains a constant loop adding all incoming requests to a queue. Once the queue fills or the time since the first request in the queue exceeds a set value the batch is dispatched to the llm to compute the output. With this implementation I found much better results than the prior implementation which dispatched a job for every query.

## Downsides
Right now especially in highly varied length queries we see know for much of the forward pass the GPU will be under utilized running computations on padding that is irrelevant to the answer we want. To remedy this we can use a process known as continuous batching. In continuous batching we feed incoming requests to populate the areas where padding would have otherwise been empty, aiding in maintaining gpu utilization.

## Next Steps
For the next itteration there's a few things I'm considering. First, I'd like to have the requests stream tokens back to the requester. To handle this there are a few things I will have to keep in mind so I may add it as a toggle so certain tests are left working despite modifications. I also will have to update the generation method moving away from hugging face's generate() method. This will allow me to have much finer control over the generation loop and computing metrics to be used as more tests and bench marks are added. Since I have never implemented continuous batching I will be reading more about how it works before fully committing.

## Side Note
Due to the the large changes I want to consider building multiple engine versions for easier benchmarking as things evolve so I am able to see where I started and how far things have come.