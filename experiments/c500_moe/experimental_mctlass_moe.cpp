#include <mcr/mc_runtime.h>
#include <mctlass/frontend_op/gemm_config.h>
#include <mctlass/frontend_op/mctlass_moe_gemm.h>
#include <cstdint>
using Op=mctlassMoeGemm<int8_t,mctlass::layout::RowMajor,int8_t,mctlass::layout::ColumnMajor,maca_bfloat16,mctlass::layout::RowMajor,float>;
extern "C" int experimental_moe_gemm(int batch,int n,int k,int experts,int em,int topk,const void*a,const void*b,void*c,const void*sa,const void*sb,const void*weights,void*ids,void*experts_ids,void*padded,int mul,void*stream) {
 Op op;
 Op::epilogueParams ep((const float*)sa,(const float*)sb,(const float*)weights);
 Op::moeParams mp((int*)ids,(int*)experts_ids,(int32_t*)padded,em,topk,mul!=0);
 Op::Arguments args(mctlass::gemm::GemmUniversalMode::kGemm,mctlass::gemm::BatchedGemmCoord(batch*topk,n,k,experts),ep,a,b,c,mp);
 return (int)op(args,nullptr,(mcStream_t)stream);
}
