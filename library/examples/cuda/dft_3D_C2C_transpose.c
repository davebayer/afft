#include <stdio.h>
#include <stdlib.h>

#include <cuda_runtime.h>
#include <cuComplex.h>

#include <afft/afft.h>

#include <helpers/afft.h>
#include <helpers/cuda.h>

int main(void)
{
  const afft_Size shape[]          = {500, 250, 1020};
  const afft_Size srcPaddedShape[] = {500, 250, 1024};
  const afft_Size dstPaddedShape[] = {500, 1020, 256};

  const afft_Size srcElemCount = srcPaddedShape[0] * srcPaddedShape[1] * srcPaddedShape[2];
  const afft_Size dstElemCount = dstPaddedShape[0] * dstPaddedShape[1] * dstPaddedShape[2];

  const afft_Alignment alignment = afft_Alignment_avx2;

  CALL_AFFT(afft_init(&errDetails)); // initialize afft library

  cuComplex* src;
  cuComplex* dst;

  CALL_CUDART(cudaMallocManaged((void**)&src, srcElemCount * sizeof(cuComplex), cudaMemAttachGlobal));
  CALL_CUDART(cudaMallocManaged((void**)&dst, dstElemCount * sizeof(cuComplex), cudaMemAttachGlobal));

  // check if src and dst are not NULL
  // initialize source vector

  afft_dft_Parameters dftParams =
  {
    .direction     = afft_Direction_forward,
    .precision     = {afft_Precision_float, afft_Precision_float, afft_Precision_float},
    .shapeRank     = 3,
    .shape         = shape,
    .transformRank = 1,
    .axes          = (afft_Axis[]){2},
    .normalization = afft_Normalization_none,
    .placement     = afft_Placement_outOfPlace,
    .type          = afft_dft_Type_complexToComplex,
  };

  afft_Size srcStrides[3] = {0};
  afft_Size dstStrides[3] = {0};

  CALL_AFFT(afft_makeStrides(3, srcPaddedShape, srcStrides, 1, &errDetails));
  CALL_AFFT(afft_makeTransposedStrides(3, dstPaddedShape, (afft_Axis[]){0, 2, 1}, dstStrides, 1, &errDetails));

  afft_cuda_Parameters cudaParams =
  {
    .deviceCount = 1,
    .devices     = (int[]){0},
  };

  afft_CentralizedMemoryLayout memoryLayout =
  {
    .complexFormat = afft_ComplexFormat_interleaved,
    .srcStrides    = srcStrides,
    .dstStrides    = dstStrides,
  };

  afft_cuda_BackendParameters backendParams =
  {
    .strategy  = afft_SelectStrategy_first,
    .mask      = (afft_Backend_vkfft),
    .orderSize = 1,
    .order     = (afft_Backend[]){afft_Backend_vkfft},
  };

  afft_Plan* plan = NULL;

  CALL_AFFT(afft_Plan_create((afft_PlanParameters){.transform       = afft_Transform_dft,
                                                   .target          = afft_Target_cuda,
                                                   .transformParams = &dftParams,
                                                   .targetParams    = &cudaParams,
                                                   .memoryLayout    = &memoryLayout,
                                                   .backendParams   = &backendParams},
                             &plan,
                             &errDetails)); // generate the plan of the transform

  CALL_AFFT(afft_Plan_execute(plan, (void* const*)&src, (void* const*)&dst, NULL, &errDetails)); // execute the transform

  CALL_CUDART(cudaDeviceSynchronize()); // synchronize the device

  // use results from dst vector

  afft_Plan_destroy(plan); // destroy the plan of the transform

  CALL_CUDART(cudaFree(src)); // free source vector
  CALL_CUDART(cudaFree(dst)); // free destination vector

  CALL_AFFT(afft_finalize(&errDetails)); // deinitialize afft library
}
