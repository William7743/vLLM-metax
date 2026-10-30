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

import numpy as np, json
T=int(sys.argv[1]);E,H,F,K=384,5120,2304,6
torch.set_grad_enabled(False);torch.manual_seed(20261027);torch.cuda.manual_seed_all(20261027);torch.set_num_threads(4)
rng=np.random.default_rng(771000+T);ids_np=np.empty((T,K),np.int32)
for i in range(T):ids_np[i]=rng.choice(E,K,replace=False)
ids=torch.as_tensor(ids_np,device='cuda');weights=torch.full((T,K),1/K,dtype=torch.float32,device='cuda')
x=(torch.randn((T,H),device='cuda',dtype=torch.bfloat16)*.02).contiguous();w1=torch.randint(-127,128,(E,2*F,H),device='cuda',dtype=torch.int8);w2=torch.randint(-127,128,(E,H,F),device='cuda',dtype=torch.int8)
sw1=torch.full((E,2*F,1),1/128,device='cuda',dtype=torch.float32);sw2=torch.full((E,H,1),1/128,device='cuda',dtype=torch.float32)
orig_q=moe.moe_kernel_quantize_input;orig_get=moe.mctlass_ops.cutlass_moe_mm_w8a8_get_kernel_m;state={}
def quant(*a,**kw):
 result=orig_q(*a,**kw);state['q']=result[0];return result
def get(a,b,c,topk):
 q=state.pop('q');assert q.dtype==torch.int8 and q.shape==a.shape and q.stride()==a.stride();return orig_get(q,b,c,topk)
moe.moe_kernel_quantize_input=quant;moe.mctlass_ops.cutlass_moe_mm_w8a8_get_kernel_m=get
def call(a,wt,idx):return moe.fused_experts_impl(a,w1,w2,wt,idx,inplace=False,activation='silu',use_int8_w8a8=True,per_channel_quant=True,w1_scale=sw1,w2_scale=sw2)
out=call(x,weights,ids);torch.cuda.synchronize();moe.mctlass_ops.cutlass_moe_mm_w8a8=original_gemm
checks=[]
for lo in range(0,T,16384):
 hi=min(T,lo+16384);ref=call(x[lo:hi],weights[lo:hi],ids[lo:hi]);torch.cuda.synchronize();eq=bool(torch.equal(out[lo:hi],ref));diff=(out[lo:hi].float()-ref.float()).abs();maximum=float(diff.max().item());checks.append({'lo':lo,'hi':hi,'exact':eq,'max_abs':maximum});del ref,diff
print('RESULT '+json.dumps({'T':T,'all_exact':all(z['exact'] for z in checks),'checks':checks}),flush=True)
assert all(z['exact'] for z in checks)
