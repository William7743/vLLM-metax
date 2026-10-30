#include <cuda_runtime.h>
#include <cuda_bf16.h>
#include <cmath>
#include "cuda_compat.h"
#include "cuda_vec_utils.cuh"
namespace vllm {
template <typename scalar_t,
          scalar_t (*ACT_FN)(const scalar_t&, const float),
          bool act_first,
          bool HAS_CLAMP>
__device__ __forceinline__ scalar_t compute(const scalar_t& x,
                                            const scalar_t& y,
                                            const float limit,
                                            const float alpha,
                                            const float beta,
                                            const bool step4) {
  if constexpr (act_first) {
    scalar_t gate = x;
    scalar_t up = y;
    if constexpr (HAS_CLAMP) {
      if (step4) {
        up = (scalar_t)fmaxf(fminf((float)up, limit), -limit);
      } else {
        gate = (scalar_t)fminf((float)gate, limit);
        up = (scalar_t)fmaxf(fminf((float)up, limit), -limit);
      }
    }

    const float activated = (float)ACT_FN(gate, alpha);
    return (scalar_t)((step4 ? fminf(activated, limit) : activated) *
                      ((float)up + beta));
  } else {
    scalar_t gate = x;
    scalar_t up = y;
    if constexpr (HAS_CLAMP) {
      gate = (scalar_t)fmaxf(fminf((float)gate, limit), -limit);
      up = (scalar_t)fminf((float)up, limit);
    }
    
    return (scalar_t)(((float)gate + beta) * ACT_FN(up, alpha));
  }
}

template <typename packed_t,
          packed_t (*PACKED_ACT_FN)(const packed_t&, const float),
          bool act_first,
          bool HAS_CLAMP>
__device__ __forceinline__ packed_t packed_compute(
    const packed_t& x,
    const packed_t& y,
    const float limit,
    const float alpha,
    const float beta,
    const bool step4) {
  if constexpr (act_first) {
    packed_t gate = x;
    packed_t up = y;
    float2 u = cast_to_float2(up);

    if constexpr (HAS_CLAMP) {
      if (step4) {
        u.x = fmaxf(fminf(u.x, limit), -limit);
        u.y = fmaxf(fminf(u.y, limit), -limit);
      } else {
        float2 g = cast_to_float2(gate);
        g.x = fminf(g.x, limit);
        g.y = fminf(g.y, limit);
        gate = cast_to_packed<packed_t>(g);
        u.x = fmaxf(fminf(u.x, limit), -limit);
        u.y = fmaxf(fminf(u.y, limit), -limit);
      }
    }

    float2 activated = cast_to_float2(PACKED_ACT_FN(gate, alpha));
    if (step4) {
      activated.x = fminf(activated.x, limit);
      activated.y = fminf(activated.y, limit);
    }
    activated.x *= (u.x + beta);
    activated.y *= (u.y + beta);

    return cast_to_packed<packed_t>(activated);
  } else {
    packed_t gate = x;
    packed_t up = y;
    float2 g = cast_to_float2(gate);

    if constexpr (HAS_CLAMP) {
      float2 u = cast_to_float2(up);
      g.x = fmaxf(fminf(g.x, limit), -limit);
      g.y = fmaxf(fminf(g.y, limit), -limit);
      u.x = fminf(u.x, limit);
      u.y = fminf(u.y, limit);
      up = cast_to_packed<packed_t>(u);
    }

    float2 activated = cast_to_float2(PACKED_ACT_FN(up, alpha));
    activated.x *= (g.x + beta);
    activated.y *= (g.y + beta);

    return cast_to_packed<packed_t>(activated);
  }
}

// Activation and gating kernel template.
template <typename scalar_t,
          typename packed_t,
          scalar_t (*ACT_FN)(const scalar_t&, const float),
          packed_t (*PACKED_ACT_FN)(const packed_t&, const float),
          bool act_first,
          bool use_vec,
          bool HAS_CLAMP,
          bool use_256b = false>
__global__ void act_and_mul_kernel(
    scalar_t* __restrict__ out,         // [..., d]
    const scalar_t* __restrict__ input, // [..., 2, d]
    const int d,
    const float limit,
    const float alpha,
    const float beta,
    const bool step4) {
  const int64_t row = static_cast<int64_t>(blockIdx.x);
  const scalar_t* x_ptr = input + row * 2 * d;
  const scalar_t* y_ptr = x_ptr + d;
  scalar_t* out_ptr = out + row * d;

  if constexpr (use_vec) {
    using cuda_t = typename CUDATypeConverter<scalar_t>::Type;
    using pvec_t = PackedVec<cuda_t, use_256b>;

    const pvec_t* x_vec = reinterpret_cast<const pvec_t*>(x_ptr);
    const pvec_t* y_vec = reinterpret_cast<const pvec_t*>(y_ptr);
    pvec_t* out_vec = reinterpret_cast<pvec_t*>(out_ptr);

    const int num_vecs = d / 2 / pvec_t::NUM_ELTS;

    for (int i = threadIdx.x; i < num_vecs; i += blockDim.x) {
      pvec_t x, y;

      if constexpr (use_256b) {
        ld256(x, &x_vec[i]);
        ld256(y, &y_vec[i]);
      } else {
        ld128(x, &x_vec[i]);
        ld128(y, &y_vec[i]);
      }

#pragma unroll
      for (int j = 0; j < pvec_t::NUM_ELTS; ++j) {
        x.elts[j] =
            packed_compute<packed_t,
                           PACKED_ACT_FN,
                           act_first,
                           HAS_CLAMP>(
                x.elts[j],
                y.elts[j],
                limit,
                alpha,
                beta,
                step4);
      }

      if constexpr (use_256b) {
        st256(x, &out_vec[i]);
      } else {
        st128(x, &out_vec[i]);
      }
    }
  } else {
    for (int64_t idx = threadIdx.x; idx < d; idx += blockDim.x) {
      const scalar_t x = VLLM_LDG(&x_ptr[idx]);
      const scalar_t y = VLLM_LDG(&y_ptr[idx]);

      out_ptr[idx] =
          compute<scalar_t,
                  ACT_FN,
                  act_first,
                  HAS_CLAMP>(
              x,
              y,
              limit,
              alpha,
              beta,
              step4);
    }
  }
}

template <typename T>
__device__ __forceinline__ T silu_kernel(const T& x, const float alpha) {
  // x * sigmoid(alpha * x)
  return (T)(((float)x) / (1.0f + expf((float)-x * alpha)));
}

template <typename packed_t>
__device__ __forceinline__ packed_t packed_silu_kernel(const packed_t& val,
                                                       const float alpha) {
  // x * sigmoid(alpha * x)
  float2 fval = cast_to_float2(val);
  fval.x = fval.x / (1.0f + expf(-fval.x * alpha));
  fval.y = fval.y / (1.0f + expf(-fval.y * alpha));
  return cast_to_packed<packed_t>(fval);
}


}
extern "C" int native_silu_i64(void* out,const void* in,int rows,int d,void* stream){
 if(rows<=0 || d<=0 || d%8) return 1;
 int threads=d/8 < 1024 ? d/8 : 1024;
 vllm::act_and_mul_kernel<__nv_bfloat16,__nv_bfloat162,vllm::silu_kernel<__nv_bfloat16>,vllm::packed_silu_kernel<__nv_bfloat162>,true,true,false,false><<<rows,threads,0,(cudaStream_t)stream>>>((__nv_bfloat16*)out,(const __nv_bfloat16*)in,d,0.f,1.f,0.f,false);
 return (int)cudaGetLastError();
}
