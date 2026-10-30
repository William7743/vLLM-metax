"""Run the original long benchmark with the stage-one INT8 query reuse."""
import sys, json, traceback
import benchmark_long as bench
import torch
moe=bench.moe
original_quant=moe.moe_kernel_quantize_input
ops=moe.mctlass_ops
original_get=ops.cutlass_moe_mm_w8a8_get_kernel_m
state={}
def quant(*args,**kwargs):
 result=original_quant(*args,**kwargs);state['q']=result[0];return result
def query(a,b,c,topk):
 q=state.pop('q')
 if q.dtype!=torch.int8 or q.shape!=a.shape or q.stride()!=a.stride() or q.device!=a.device:
  raise RuntimeError('Quantized input metadata does not match query input')
 return original_get(q,b,c,topk)
moe.moe_kernel_quantize_input=quant;ops.cutlass_moe_mm_w8a8_get_kernel_m=query
T=int(sys.argv[1]);torch.cuda.reset_peak_memory_stats()
try:bench.run(T)
except Exception as exc:
 traceback.print_exc();print('FAILURE',json.dumps({'T':T,'type':type(exc).__name__,'message':str(exc)}),flush=True);raise
finally:
 moe.moe_kernel_quantize_input=original_quant;ops.cutlass_moe_mm_w8a8_get_kernel_m=original_get
