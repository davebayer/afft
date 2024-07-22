#include <stdio.h>
#include <stdlib.h>

#include <cuda_runtime.h>
#include <cuComplex.h>

#include <afft/afft.h>

afft_ErrorDetails errDetails = {};

#define CUDA_CALL(call) do { \
    cudaError_t _err = (call); \
    if (_err != cudaSuccess) \
    { \
      fprintf(stderr, "cuda error (%s:%d): %s\n", __FILE__, __LINE__, cudaGetErrorString(_err)); \
      exit(EXIT_FAILURE); \
    } \
  } while (0)

#define AFFT_CALL(call) do { \
    afft_Error _err = (call); \
    if (_err != afft_Error_success) \
    { \
      fprintf(stderr, "afft error (%s:%d): %d\n", __FILE__, __LINE__, errDetails.message); \
      exit(EXIT_FAILURE); \
    } \
  } while (0)

int main(void)
{
  const afft_Size shape[]          = {500, 250, 1020};
  const afft_Size srcPaddedShape[] = {500, 250, 1024};
  const afft_Size dstPaddedShape[] = {500, 1020, 256};

  const afft_Size srcElemCount = srcPaddedShape[0] * srcPaddedShape[1] * srcPaddedShape[2];
  const afft_Size dstElemCount = dstPaddedShape[0] * dstPaddedShape[1] * dstPaddedShape[2];

  const afft_Alignment alignment = afft_Alignment_avx2;

  AFFT_CALL(afft_init(&errDetails)); // initialize afft library

  cuComplex* src;
  cuComplex* dst;

  CUDA_CALL(cudaMallocManaged(&src, srcElemCount * sizeof(cuComplex), cudaMemAttachGlobal));
  CUDA_CALL(cudaMallocManaged(&dst, dstElemCount * sizeof(cuComplex), cudaMemAttachGlobal));

  // check if src and dst are not NULL
  // initialize source vector

  afft_dft_Parameters dftParams =
  {
    .direction     = afft_Direction_forward,
    .precision     = {afft_Precision_float, afft_Precision_float, afft_Precision_float},
    .shapeRank     = 3,
    .shape         = shape,
    .axesRank      = 1,
    .axes          = (afft_Axis[]){2},
    .normalization = afft_Normalization_none,
    .placement     = afft_Placement_outOfPlace,
    .type          = afft_dft_Type_complexToComplex,
  };

  afft_Size srcStrides[3] = {0};
  afft_Size dstStrides[3] = {0};

  AFFT_CALL(afft_makeStrides(3, srcPaddedShape, 1, srcStrides, &errDetails));
  AFFT_CALL(afft_makeTransposedStrides(3, dstPaddedShape, (afft_Axis[]){0, 2, 1}, 1, dstStrides, &errDetails));

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

  AFFT_CALL(afft_Plan_create((afft_Plan_Parameters){.transform       = afft_Transform_dft,
                                                    .target          = afft_Target_cuda,
                                                    .transformParams = &dftParams,
                                                    .targetParams    = &cudaParams,
                                                    .memoryLayout    = &memoryLayout,
                                                    .backendParams   = &backendParams},
                             &plan,
                             &errDetails)); // generate the plan of the transform

  AFFT_CALL(afft_Plan_execute(plan, (void* const*)&src, (void* const*)&dst, NULL, &errDetails)); // execute the transform

  // use results from dst vector

  afft_Plan_destroy(plan); // destroy the plan of the transform

  CUDA_CALL(cudaFree(src)); // free source vector
  CUDA_CALL(cudaFree(dst)); // free destination vector

  AFFT_CALL(afft_finalize(&errDetails)); // deinitialize afft library
}
