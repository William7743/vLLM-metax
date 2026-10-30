"""Five complete local-expert MoE cases, vLLM-MetaX W8A8 mcTlass backend.

`smoke` uses independent CPU reference. `case N` executes and validates one
shape in a fresh process, preserving failures. Initialization is not timed.
"""
import os
os.environ['MACA_VLLM_ENABLE_MCTLASS_FUSED_MOE']='1'
os.environ['MACA_VLLM_ENABLE_MCTLASS_PYTHON_API']='1'
os.environ['VLLM_FUSED_MOE_CHUNK_SIZE']=os.environ['TEST_MOE_CHUNK_SIZE']
from pathlib import Path
import sys, json, time, statistics, hashlib, gc
import numpy as np
import torch
import vllm
import vllm_metax.model_executor.layers.fused_moe.fused_moe as moe

P=Path(__file__).resolve().parent
torch.set_grad_enabled(False)
torch.manual_seed(20261027)
torch.cuda.manual_seed_all(20261027)
torch.set_num_threads(4)

def cpu_ref(x, w1, w2, ids, weights, sw1, sw2):
    """CPU reference: quantize per token, two int64 products, BF16 boundaries."""
    xf=x.float().cpu().numpy(); ids=ids.cpu().numpy(); weights=weights.cpu().numpy()
    w1=w1.cpu().numpy().astype(np.int64);w2=w2.cpu().numpy().astype(np.int64)
    sw1=sw1.cpu().numpy().reshape(w1.shape[:2]);sw2=sw2.cpu().numpy().reshape(w2.shape[:2])
    def bf(v):return torch.from_numpy(np.asarray(v,np.float32)).to(torch.bfloat16).float().numpy()
    def quant(v):
        vmax=max(float(np.max(np.abs(v))),1e-10);scale=np.float32(vmax/127.)
        z=v*np.float32(127./vmax);q=np.copysign(np.floor(np.abs(z)+0.5),z).clip(-127,127).astype(np.int64)
        return q,scale
    out=np.zeros((len(xf),w2.shape[1]),np.float32)
    for t in range(len(xf)):
        q1,s1=quant(xf[t]);values=[]
        for slot,e0 in enumerate(ids[t]):
            e=int(e0);z=bf((w1[e]@q1).astype(np.float32)*s1*sw1[e])
            gate,up=np.split(z,2);h=bf(gate/(1.+np.exp(-gate))*up)
            q2,s2=quant(h);y=bf((w2[e]@q2).astype(np.float32)*s2*sw2[e]*weights[t,slot]);values.append(y)
        out[t]=np.sum(values,axis=0)
    return bf(out)

def run(T,smoke=False):
    if smoke:E,H,F,K=4,128,64,2
    else:E,H,F,K=384,5120,2304,6
    global PHASE
    PHASE='route_generation'
    if smoke:
        ids=torch.tensor([[0,1],[1,3],[2,0],[3,2]],dtype=torch.int32,device='cuda');T=4
        route_hash=None;route_capacity=None
    else:
        rng=np.random.default_rng(771000+T);route_np=np.empty((T,K),dtype=np.int32)
        for i in range(T):route_np[i]=rng.choice(E,size=K,replace=False)
        route_hash=hashlib.sha256(route_np.tobytes()).hexdigest()
        route_capacity=int(np.bincount(route_np.reshape(-1),minlength=E).max())
        PHASE='upload_routes';ids=torch.from_numpy(route_np).to('cuda');del route_np
    PHASE='input_allocation'

    weights=torch.full((T,K),1/K,dtype=torch.float32,device='cuda')
    # Same expert shapes, deterministic synthetic data; no model weights downloaded.
    x=torch.randn((T,H),dtype=torch.bfloat16,device='cuda');x.mul_(.02)
    PHASE='weight_allocation'
    w1=torch.randint(-127,128,(E,2*F,H),dtype=torch.int8,device='cuda')
    w2=torch.randint(-127,128,(E,H,F),dtype=torch.int8,device='cuda')
    sw1=torch.full((E,2*F,1),1/128,dtype=torch.float32,device='cuda')
    sw2=torch.full((E,H,1),1/128,dtype=torch.float32,device='cuda')
    def call():return moe.fused_experts_impl(x,w1,w2,weights,ids,inplace=False,
        activation='silu',use_int8_w8a8=True,per_channel_quant=True,w1_scale=sw1,w2_scale=sw2)
    # Instrument only an untimed invocation to prove both vendor GEMMs run.
    traces=[];original=moe.mctlass_ops.cutlass_moe_mm_w8a8
    def wrapped(*args,**kw):
        traces.append({'a_shape':list(args[0].shape),'b_shape':list(args[1].shape),
                       'output_dtype':str(args[2].dtype),'topk':args[11]})
        return original(*args,**kw)
    moe.mctlass_ops.cutlass_moe_mm_w8a8=wrapped
    PHASE='first_full_moe_call'
    try:y=call();torch.cuda.synchronize()
    finally:moe.mctlass_ops.cutlass_moe_mm_w8a8=original
    chunk_size=int(os.environ['VLLM_FUSED_MOE_CHUNK_SIZE']); expected_calls=2*((T+chunk_size-1)//chunk_size)
    assert len(traces)==expected_calls, f'Expected {expected_calls} mcTlass calls, got {traces}'
    assert y.shape==(T,H)
    PHASE='output_finite_check'
    for lo in range(0,T,16384):
        assert torch.isfinite(y[lo:lo+16384]).all().item()
    # Validate both the first and last token in independent expert references.
    validations=[]
    offsets=list(range(T)) if smoke else sorted(set([0,T//2,T-1]))
    PHASE='cpu_validation'
    for offset in offsets:
        selected=ids[offset:offset+1].cpu().numpy()
        unique=sorted(set(int(v) for v in selected.flatten()))
        mapping={e:i for i,e in enumerate(unique)}
        localids=torch.tensor([[mapping[int(e)] for e in row] for row in selected],dtype=torch.int32)
        ref=cpu_ref(x[offset:offset+1],w1[unique],w2[unique],localids,weights[offset:offset+1],sw1[unique],sw2[unique])
        got=y[offset:offset+1].float().cpu().numpy();absdiff=np.abs(got-ref)
        maxerr=float(absdiff.max());rmse=float(np.sqrt(np.mean((got-ref)**2)))
        rmax=float(np.max(np.abs(ref)));relative_rmse=rmse/max(float(np.sqrt(np.mean(ref**2))),1e-12)
        atol=max(.002,rmax*.01)
        cpu_pass=bool(np.allclose(got,ref,rtol=.04,atol=atol))
        if not cpu_pass: print(f'CPU_CHECK_FAILED token={offset} max_abs={maxerr} relative_rmse={relative_rmse}',flush=True)
        validations.append(dict(cpu_pass=cpu_pass,token_index=offset,cpu_max_abs_error=maxerr,cpu_relative_rmse=relative_rmse,atol=atol,rtol=.04))
    count=len(offsets)
    result_dtype=str(y.dtype)
    del y;gc.collect()
    PHASE='warmup'
    for _ in range(2):
        z=call();del z
    torch.cuda.synchronize()
    times=[]
    PHASE='timing'
    for iteration in range(7):
        start=torch.cuda.Event(enable_timing=True);end=torch.cuda.Event(enable_timing=True)
        start.record();z=call();end.record();end.synchronize();times.append(start.elapsed_time(end))
        if iteration==6:
            for lo in range(0,T,16384):
                assert torch.isfinite(z[lo:lo+16384]).all().item()
        del z
    med=statistics.median(times)
    result=dict(T=T,experts=E,hidden=H,intermediate=F,topk=K,gate_up_N=2*F,
      input_dtype=str(x.dtype),weight_dtype=str(w1.dtype),output_dtype=result_dtype,
      scale='per-token activation dynamic; per-output-channel weights=1/128',
      seed=20261027,synthetic=True,precomputed_routes=True,route_seed=771000+T,route_sha256=route_hash,route_capacity=route_capacity,numpy_version=np.__version__,gpu_peak_allocated=torch.cuda.max_memory_allocated(),gpu_peak_reserved=torch.cuda.max_memory_reserved(),backend_calls=traces,
      check_tokens=count,validations=validations,cpu_max_abs_error=max(v["cpu_max_abs_error"] for v in validations),cpu_relative_rmse=max(v["cpu_relative_rmse"] for v in validations),
      correctness_allclose_rtol=.04,correctness_atol=max(v["atol"] for v in validations),
      timing='event around fused_experts_impl: input quantize, alignment, gate/up, activation, second quantize, down, combine; excludes router topk, allocation of inputs/weights, host reference, communication',
      chunk_size=chunk_size,moe_chunks=(T+chunk_size-1)//chunk_size,warmups=2,raw_ms=times,median_ms=med,mean_ms=statistics.mean(times),
      range_over_median=(max(times)-min(times))/med,
      device=torch.cuda.get_device_name(0),torch=torch.__version__,vllm=vllm.__version__)
    outfile=P/('smoke_result.json' if smoke else f'case_{T}.json')
    outfile.write_text(json.dumps(result,indent=2),encoding='utf-8')
    print('RESULT',json.dumps(result),flush=True)

PHASE='initializing'
if __name__=='__main__':
    requested=int(sys.argv[2]) if len(sys.argv)>2 else 0
    try:
        torch.cuda.reset_peak_memory_stats()
        if sys.argv[1]=='smoke':run(4,True)
        elif sys.argv[1]=='case':run(requested)
        else:raise ValueError('Usage: benchmark_long.py smoke | case T')
    except Exception as exc:
        import traceback
        msg=str(exc);is_oom=isinstance(exc,torch.cuda.OutOfMemoryError) or ('out of memory' in msg.lower() and ('cuda' in msg.lower() or 'gpu' in msg.lower() or 'maca' in msg.lower()))
        try:
            free,total=torch.cuda.mem_get_info(); mem=dict(free=free,total=total,allocated=torch.cuda.memory_allocated(),reserved=torch.cuda.memory_reserved(),peak_allocated=torch.cuda.max_memory_allocated(),peak_reserved=torch.cuda.max_memory_reserved())
        except Exception as me:mem=dict(error=repr(me))
        err=dict(T=requested,status='gpu_oom' if is_oom else 'unexpected_error',phase=PHASE,exception_type=type(exc).__name__,message=msg,device_memory=mem,traceback=traceback.format_exc())
        (P/f'failure_{requested}.json').write_text(json.dumps(err,indent=2),encoding='utf-8')
        print('FAILURE '+json.dumps(err),flush=True)
        raise SystemExit(42 if is_oom else 1)
