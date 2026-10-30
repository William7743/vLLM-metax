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

import vllm.model_executor.layers.quantization.utils.int8_utils as int8_utils
int8_utils._per_token_quant_int8=quant_u32
print('EXPERIMENTAL_U32_KERNEL',flush=True)
import vllm_metax.model_executor.layers.fused_moe.fused_moe as moe
_real_quant=moe.moe_kernel_quantize_input
def guarded_quant(*args,**kwargs):
 a=args[0] if args else kwargs['A']
 if a.numel()-1 > 2**32-1:
  raise RuntimeError('uint32 offset range exceeded; refusing unsafe kernel launch')
 return _real_quant(*args,**kwargs)
moe.moe_kernel_quantize_input=guarded_quant
exec(compile(open('benchmark_stage1.py').read(), 'benchmark_stage1.py','exec'))
