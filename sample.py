from text_generation_server.models import get_model
from text_generation_server.models.causal_lm import CausalLMBatch
from text_generation_server.pb import generate_pb2
from typing import List
import time
import torch

template = "Below is an instruction that describes a task. Write a response that appropriately completes the request.\n\n### Instruction:\n{}\n\n### Response:"

text = template.format(
    "Provide a list of instructions for preparing chicken soup."
)
model = get_model(
    model_id="/data/jmrosenk/hub/hf/7B-F", # models--ibm-granite--granite-7b-instruct/snapshots/024256d38037925b95019ae23671ce0b83adf1a5",
    revision=None,
    sharded=False,
    quantize=None,
    speculate=None,
    dtype="float16",
    trust_remote_code=False
)
max_new_tokens = 100
def __generate_prefill_request():

    out = CausalLMBatch.from_pb(
        generate_pb2.Batch(
            id=0,
            requests=[
                generate_pb2.Request(
                    id=0, inputs=text, prefill_logprobs=True, truncate=48,
                    parameters=generate_pb2.NextTokenChooserParameters(
                        temperature=1.0,
                        repetition_penalty=1.0,
                        top_k=0,
                        top_p=1.0,
                        typical_p=1.0,
                        do_sample=False,
                    ),
                    stopping_parameters=generate_pb2.StoppingCriteriaParameters(max_new_tokens=max_new_tokens)
                )
            ],
            size=1
        ),
        model.tokenizer,
        model.dtype,
        model.device
    )
    return out

batch1 = __generate_prefill_request()

output = []
res = model.generate_token(batch1)
output.extend(res[0][0].tokens.texts)
for i in range(max_new_tokens):
    t0 = time.time_ns()
    res = model.generate_token(batch1)
    output.extend(res[0][0].tokens.texts)
    torch.cuda.synchronize(device=model.device)
    t_tok = time.time_ns()-t0
    print("t_tok: %.2f ms" % (t_tok/1000.0/1000.0))
print("".join(output))