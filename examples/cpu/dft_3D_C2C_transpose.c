#include <stdio.h>
#include <stdlib.h>

#include <afft/afft.h>

typedef float _Complex FloatComplex;

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

  FloatComplex* src = afft_cpu_alignedAlloc(srcElemCount * sizeof(FloatComplex), alignment); // source vector
  FloatComplex* dst = afft_cpu_alignedAlloc(dstElemCount * sizeof(FloatComplex), alignment); // destination vector

  // check if src and dst are not NULL
  // initialize source vector

  const afft_dft_Parameters dftParams =
  {
    .direction     = afft_Direction_forward,
    .precision     = {afft_Precision_float, afft_Precision_float, afft_Precision_float},
    .shapeRank     = 3,
    .shape         = shape,
    .axesRank      = 1,
    .axes          = (size_t[]){1},
    .normalization = afft_Normalization_unitary,
    .placement     = afft_Placement_outOfPlace,
    .type          = afft_dft_Type_complexToComplex,
  };

  size_t srcStrides[3] = {0};
  size_t dstStrides[3] = {0};

  AFFT_CALL(afft_makeStrides(3, srcPaddedShape, 1, srcStrides));
  AFFT_CALL(afft_makeTransposedStrides(3, dstPaddedShape, (size_t[]){0, 2, 1}, 1, dstStrides));

  const afft_cpu_Parameters cpuParams =
  {
    .memoryLayout   = {.srcStrides = srcStrides,
                       .dstStrides = dstStrides},
    .complexFormat  = afft_ComplexFormat_interleaved,
    .preserveSource = true,
    .alignment      = alignment,
    .threadLimit    = 4
  };

  const afft_cpu_BackendParameters backendParams =
  {
    .strategy  = afft_SelectStrategy_first,
    .mask      = (afft_Backend_fftw3 | afft_Backend_mkl | afft_Backend_pocketfft),
    .orderSize = 2,
    .order     = (afft_Backend[]){afft_Backend_mkl, afft_Backend_fftw3},
    .fftw3     = {.plannerFlag = afft_fftw3_PlannerFlag_exhaustive,
                  .timeLimit   = 2.0},
  };

  afft_Plan* plan = NULL;

  AFFT_CALL(afft_Plan_createWithBackendParameters(dftParams, cpuParams, backendParams, &plan)); // generate the plan of the transform

  AFFT_CALL(afft_Plan_execute(plan, (void* const*)&src, (void* const*)&dst)); // execute the transform

  // use results from dst vector

  afft_Plan_destroy(plan); // destroy the plan of the transform

  afft_cpu_alignedFree(src, alignment); // free source vector
  afft_cpu_alignedFree(dst, alignment); // free destination vector

  AFFT_CALL(afft_finalize()); // deinitialize afft library
}