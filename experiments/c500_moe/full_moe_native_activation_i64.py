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
import ctypes as C
lib=C.CDLL('./libexperimental_moe.so')
func=lib.experimental_moe_gemm
func.argtypes=[C.c_int]*6+[C.c_void_p]*9+[C.c_int,C.c_void_p]
func.restype=C.c_int
original_gemm=moe.mctlass_ops.cutlass_moe_mm_w8a8
def patched_gemm(a,b,c,asc,bsc,bias,weights,ids,experts,padded,em,topk,mul):
 assert bias is None and c.is_contiguous() and a.is_contiguous() and b.is_contiguous()
 ptr=lambda x:C.c_void_p(x.data_ptr())
 stream=C.c_void_p(torch.cuda.current_stream().cuda_stream)
 status=func(a.size(0),b.size(1),a.size(1),b.size(0),em,topk,ptr(a),ptr(b),ptr(c),ptr(asc),ptr(bsc),ptr(weights),ptr(ids),ptr(experts),ptr(padded),mul,stream)
 if status: raise RuntimeError('experimental GEMM status '+str(status))
moe.mctlass_ops.cutlass_moe_mm_w8a8=patched_gemm
_nativelib=C.CDLL('/tmp/native_activation_i64/libnative_silu_i64.so')
_nativefn=_nativelib.native_silu_i64
_nativefn.argtypes=[C.c_void_p,C.c_void_p,C.c_int,C.c_int,C.c_void_p];_nativefn.restype=C.c_int
def silu_mul_i64(out,inp):
 assert out.dtype==inp.dtype==torch.bfloat16 and out.is_contiguous() and inp.is_contiguous() and out.shape[1]==2304
 status=_nativefn(out.data_ptr(),inp.data_ptr(),out.shape[0],out.shape[1],torch.cuda.current_stream().cuda_stream)
 if status:raise RuntimeError('native silu launch error '+str(status))
from vllm.model_executor.layers.fused_moe.activation import MoEActivation
def activation_i64(activation,out,inp):
 assert activation==MoEActivation.SILU
 silu_mul_i64(out,inp)
moe.apply_moe_activation=activation_i64
exec(compile(open('benchmark_stage1.py').read(), 'benchmark_stage1.py','exec'))
