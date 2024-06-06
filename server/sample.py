from text_generation_server.models import get_model
from text_generation_server.models.causal_lm import CausalLMBatch
from text_generation_server.pb import generate_pb2
from typing import List
import time
import torch
import os

torch._dynamo.config.cache_size_limit = 256
torch._dynamo.config.assume_static_by_default= True
torch._dynamo.config.dynamic_shapes=False
torch._dynamo.config.automatic_dynamic_shapes=False

torch._dynamo.config.dynamic_shapes = False
torch._dynamo.config.automatic_dynamic_shapes = False

# Bug in PT 2.1.2
torch._inductor.config.split_cat_fx_passes = False
torch._inductor.config.size_asserts = False
torch._inductor.config.joint_graph_constant_folding = False

os.environ['FMS_IMPLEMENTATION']="True"
os.environ['FMS_TORCH_COMPILE']="True"
os.environ['SENCORES']='32'
os.environ['SENCORELETS']='2'
os.environ['DATA_PREC']='fp16'
# os.environ['FLEX_COMPUTE']="SENULATOR"
# os.environ['FLEX_DEVICE']="MOCK"
os.environ['FLEX_COMPUTE']='SENTIENT'
os.environ['FLEX_DEVICE']='VFIO'
os.environ['FLEX_OVERWRITE_NMB_FRAME']='1'
# export DT_OPT=varsub=1,lxopt=1,opfusion=1,arithfold=1,dataopt=1,patchinit=1,patchprog=1,autopilot=1,weipreload=0,kvcacheopt=1,progshareopt=1
os.environ['DTCOMPILER_KEEP_EXPORT']='true'
os.environ['DEE_DUMP_GRAPHS']='llama7b512'
# SENLIB_DEVEL_CONFIG_FILE and FLEX_RDMA_PCI_BUS_ADDR_0 need to point to same device
os.environ['SENLIB_DEVEL_CONFIG_FILE']='/opt/app-root/src2/senliba0.json'


os.environ['COMPILATION_MODE']='offline_decoder'


os.environ['FLEX_RDMA_PCI_BUS_ADDR_0']="0000:a0:00.0"
os.environ['AIU_WORLD_RANK_0']="0000:bt:00.0"

from torch_sendnn import torch_sendnn
template = "Below is an instruction that describes a task. Write a response that appropriately completes the request. Be polite in your response to the user.\n\n### Instruction:\n{}\n\n### Response:"

text = template.format(
    "Provide a list of instructions for preparing chicken soup for a family of four."
)
model = get_model(
    model_id="/opt/app-root/src2/hf_llama_7b", # models--ibm-granite--granite-7b-instruct/snapshots/024256d38037925b95019ae23671ce0b83adf1a5",
    revision=None,
    sharded=False,
    quantize=None,
    speculate=None,
    dtype="float16",
    trust_remote_code=False
)
max_new_tokens = 20
def __generate_prefill_request():

    out = CausalLMBatch.from_pb(
        generate_pb2.Batch(
            id=0,
            requests=[
                generate_pb2.Request(
                    id=0, inputs=text, prefill_logprobs=True, truncate=64,
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
    t_tok = time.time_ns()-t0
    print("t_tok: %.2f ms" % (t_tok/1000.0/1000.0))
print("".join(output))

if os.environ['FMS_TORCH_COMPILE'] == "True":
    torch_sendnn.update_lazyhandle()

batch1 = __generate_prefill_request()

output = []
res = model.generate_token(batch1)
output.extend(res[0][0].tokens.texts)
for i in range(max_new_tokens):
    t0 = time.time_ns()
    res = model.generate_token(batch1)
    output.extend(res[0][0].tokens.texts)
    t_tok = time.time_ns()-t0
    print("t_tok: %.2f ms" % (t_tok/1000.0/1000.0))
print("".join(output))

batch1 = __generate_prefill_request()

output = []
res = model.generate_token(batch1)
output.extend(res[0][0].tokens.texts)
for i in range(max_new_tokens):
    t0 = time.time_ns()
    res = model.generate_token(batch1)
    output.extend(res[0][0].tokens.texts)
    t_tok = time.time_ns()-t0
    print("t_tok: %.2f ms" % (t_tok/1000.0/1000.0))
print("".join(output))
