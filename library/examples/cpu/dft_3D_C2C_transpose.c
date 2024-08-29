#include <stdio.h>
#include <stdlib.h>

#include <afft/afft.h>

#include <helpers/afft.h>

typedef float _Complex FloatComplex;

int main(void)
{
  const size_t shape[]          = {500, 250, 1020};
  const size_t srcPaddedShape[] = {500, 250, 1024};
  const size_t dstPaddedShape[] = {500, 1020, 256};

  const size_t srcElemCount = srcPaddedShape[0] * srcPaddedShape[1] * srcPaddedShape[2];
  const size_t dstElemCount = dstPaddedShape[0] * dstPaddedShape[1] * dstPaddedShape[2];

  const afft_Alignment alignment = afft_Alignment_cpuNative;

  AFFT_CALL(afft_init(&errDetails)); // initialize afft library

  FloatComplex* src = afft_alignedAlloc(srcElemCount * sizeof(FloatComplex), alignment); // source vector
  FloatComplex* dst = afft_alignedAlloc(dstElemCount * sizeof(FloatComplex), alignment); // destination vector

  if (src == NULL || dst == NULL)
  {
    fprintf(stderr, "Memory allocation failed\n");
    exit(EXIT_FAILURE);
  }

  // check if src and dst are not NULL
  // initialize source vector

  afft_dft_Parameters dftParams =
  {
    .direction     = afft_Direction_forward,
    .precision     = {afft_Precision_float, afft_Precision_float, afft_Precision_float},
    .shapeRank     = 3,
    .shape         = shape,
    .transformRank = 1,
    .axes          = (afft_Axis[]){1},
    .normalization = afft_Normalization_unitary,
    .placement     = afft_Placement_outOfPlace,
    .type          = afft_dft_Type_complexToComplex,
  };

  afft_Size srcStrides[3] = {0};
  afft_Size dstStrides[3] = {0};

  AFFT_CALL(afft_makeStrides(3, srcPaddedShape, srcStrides, 1, &errDetails));
  AFFT_CALL(afft_makeTransposedStrides(3, dstPaddedShape, (afft_Axis[]){0, 2, 1}, dstStrides, 1, &errDetails));

  afft_cpu_Parameters cpuParams =
  {
    .threadLimit = 4
  };

  afft_CentralizedMemoryLayout memoryLayout =
  {
    .complexFormat = afft_ComplexFormat_interleaved,
    .alignment     = alignment,
    .srcStrides    = srcStrides,
    .dstStrides    = dstStrides,
  };

  // afft_cpu_BackendParameters backendParams =
  // {
  //   .strategy  = afft_SelectStrategy_first,
  //   .mask      = (afft_BackendMask_fftw3 | afft_BackendMask_mkl | afft_BackendMask_pocketfft),
  //   .orderSize = 2,
  //   .order     = (afft_Backend[]){afft_Backend_mkl, afft_Backend_fftw3},
  //   .fftw3     = {.plannerFlag = afft_fftw3_PlannerFlag_exhaustive,
  //                 .timeLimit   = 2.0},
  // };

  afft_Plan* plan = NULL;

  AFFT_CALL(afft_Plan_create((afft_PlanParameters){.transform       = afft_Transform_dft,
                                                   .target          = afft_Target_cpu,
                                                   .transformParams = &dftParams,
                                                   .targetParams    = &cpuParams,
                                                   .memoryLayout    = &memoryLayout,
                                                   .backendParams   = NULL},
                             &plan,
                             &errDetails)); // generate the plan of the transform

  AFFT_CALL(afft_Plan_execute(plan,
                              (void* const*)&src,
                              (void* const*)&dst,
                              NULL,
                              &errDetails)); // execute the transform

  // use results from dst vector

  afft_Plan_destroy(plan); // destroy the plan of the transform

  afft_alignedFree(src, alignment); // free source vector
  afft_alignedFree(dst, alignment); // free destination vector

  AFFT_CALL(afft_finalize(&errDetails)); // deinitialize afft library
}