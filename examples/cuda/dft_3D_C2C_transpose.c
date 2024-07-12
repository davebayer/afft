#include <stdio.h>
#include <stdlib.h>

#include <cuda_runtime.h>
#include <cuComplex.h>

#include <afft/afft.h>

#define AFFT_CALL(call) do { \
    afft_Error _err = (call); \
    if (_err != afft_Error_success) \
    { \
      fprintf(stderr, "afft error (%s:%d): %d\n", __FILE__, __LINE__, _err); \
      exit(EXIT_FAILURE); \
    } \
  } while (0)

int main(void)
{
  const size_t shape[]          = {500, 250, 1020};
  const size_t srcPaddedShape[] = {500, 250, 1024};
  const size_t dstPaddedShape[] = {500, 1020, 256};

  const size_t srcElemCount = srcPaddedShape[0] * srcPaddedShape[1] * srcPaddedShape[2];
  const size_t dstElemCount = dstPaddedShape[0] * dstPaddedShape[1] * dstPaddedShape[2];

  const afft_Alignment alignment = afft_Alignment_avx2;

  AFFT_CALL(afft_init()); // initialize afft library

  cuComplex* src = afft_gpu_unifiedAlloc(srcElemCount * sizeof(cuComplex)); // source vector
  cuComplex* dst = afft_gpu_unifiedAlloc(dstElemCount * sizeof(cuComplex)); // destination vector

  // check if src and dst are not NULL
  // initialize source vector

  const afft_dft_Parameters dftParams =
  {
    .direction     = afft_Direction_forward,
    .precision     = {afft_Precision_float, afft_Precision_float, afft_Precision_float},
    .shapeRank     = 3,
    .shape         = shape,
    .axesRank      = 1,
    .axes          = (size_t[]){2},
    .normalization = afft_Normalization_none,
    .placement     = afft_Placement_outOfPlace,
    .type          = afft_dft_Type_complexToComplex,
  };

  size_t srcStrides[3] = {0};
  size_t dstStrides[3] = {0};

  AFFT_CALL(afft_makeStrides(3, srcPaddedShape, 1, srcStrides));
  AFFT_CALL(afft_makeTransposedStrides(3, dstPaddedShape, (size_t[]){0, 2, 1}, 1, dstStrides));

  const afft_gpu_Parameters gpuParams =
  {
    .memoryLayout   = {.srcStrides = srcStrides,
                       .dstStrides = dstStrides},
    .complexFormat  = afft_ComplexFormat_interleaved,
    .preserveSource = true,
    .device         = 0,
  };

  const afft_gpu_BackendParameters backendParams =
  {
    .strategy  = afft_SelectStrategy_first,
    .mask      = (afft_Backend_vkfft),
    .orderSize = 1,
    .order     = (afft_Backend[]){afft_Backend_vkfft},
  };

  afft_Plan* plan = NULL;

  AFFT_CALL(afft_Plan_createWithBackendParameters(dftParams, gpuParams, backendParams, &plan)); // generate the plan of the transform

  AFFT_CALL(afft_Plan_execute(plan, (void* const*)&src, (void* const*)&dst)); // execute the transform

  // use results from dst vector

  afft_Plan_destroy(plan); // destroy the plan of the transform

  afft_gpu_unifiedFree(src); // free source vector
  afft_gpu_unifiedFree(dst); // free destination vector

  AFFT_CALL(afft_finalize()); // deinitialize afft library
}
