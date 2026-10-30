"""Isolate original and explicit-int64 per-token quantization indexing."""
import sys, torch
from vllm.triton_utils import triton,tl
from vllm.model_executor.layers.quantization.utils.int8_utils import per_token_quant_int8,round_int8
@triton.jit
def quant_u32(x_ptr,xq_ptr,scale_ptr,stride_x,stride_xq,N,BLOCK:tl.constexpr):
 row=tl.program_id(0).to(tl.uint32)
 stride_x=stride_x.to(tl.uint32);stride_xq=stride_xq.to(tl.uint32)
 col=tl.arange(0,BLOCK); mask=col<N
 x=tl.load(x_ptr+row*stride_x+col,mask=mask,other=0.).to(tl.float32)
 a=tl.maximum(tl.max(tl.abs(x)),1.e-10);q=round_int8(x*(127/a))
 tl.store(xq_ptr+row*stride_xq+col,q,mask=mask)
 tl.store(scale_ptr+row,a/127)
mode=sys.argv[1];M=int(sys.argv[2]);N=2304
x=torch.ones((M,N),device='cuda',dtype=torch.bfloat16);torch.cuda.synchronize()
print('START',mode,M,N,'max_element_offset',M*N-1,flush=True)
if mode=='original':q,s=per_token_quant_int8(x)
else:
 q=torch.empty((M,N),device='cuda',dtype=torch.int8);s=torch.empty((M,1),device='cuda',dtype=torch.float32)
 quant_u32[(M,)](x,q,s,x.stride(0),q.stride(0),N,BLOCK=4096,num_warps=8,num_stages=1)
torch.cuda.synchronize()
for r in sorted(set([0,M//2,M-1])):
 assert (q[r]==127).all().item(),f'bad q row {r}'
 assert abs(s[r].item()-1/127)<1e-6,f'bad scale row {r}'
print('PASS',mode,M,flush=True)
