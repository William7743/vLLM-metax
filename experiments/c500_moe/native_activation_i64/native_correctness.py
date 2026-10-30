import sys,json,torch
import ctypes as C
from pathlib import Path
lib=C.CDLL(str(Path(__file__).resolve().parent / 'libnative_silu_i64.so'));fn=lib.native_silu_i64
fn.argtypes=[C.c_void_p,C.c_void_p,C.c_int,C.c_int,C.c_void_p];fn.restype=C.c_int
def silu_mul_i64(out,inp):
 status=fn(out.data_ptr(),inp.data_ptr(),out.shape[0],out.shape[1],torch.cuda.current_stream().cuda_stream)
 if status: raise RuntimeError('native launch error '+str(status))
from vllm.model_executor.layers.fused_moe.activation import apply_moe_activation, MoEActivation
M=int(sys.argv[1]);D=int(sys.argv[2]);torch.manual_seed(42)
x=torch.randn((M,2*D),device='cuda',dtype=torch.bfloat16);out=torch.empty((M,D),device='cuda',dtype=torch.bfloat16)
silu_mul_i64(out,x);torch.cuda.synchronize();bad=0;maxerr=0.;finite=True
for lo in range(0,M,98304):
 hi=min(M,lo+98304);ref=torch.empty((hi-lo,D),device='cuda',dtype=torch.bfloat16);apply_moe_activation(MoEActivation.SILU,ref,x[lo:hi]);torch.cuda.synchronize()
 a=out[lo:hi];bad+=int((a!=ref).sum().item());maxerr=max(maxerr,float((a.float()-ref.float()).abs().max().item()));finite &= bool(torch.isfinite(a).all().item());del ref
print('RESULT '+json.dumps({'M':M,'D':D,'different_elements':bad,'max_abs':maxerr,'finite':finite}),flush=True)
