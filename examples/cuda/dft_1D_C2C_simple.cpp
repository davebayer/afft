#include <complex>
#include <vector>

#include <afft/afft.hpp>

template<typename T>
using UnifiedMemoryVector = std::vector<T, afft::cuda::UnifiedMemoryAllocator<T>>;

int main(void)
{
  using PrecT = float;

  constexpr std::size_t size{1024}; // size of wthe transform

  afft::init(); // initialize afft library, also initializes CUDA if uninitialized

  UnifiedMemoryVector<std::complex<PrecT>> src(size); // source vector
  UnifiedMemoryVector<std::complex<PrecT>> dst(size); // destination vector

  // initialize source vector

  afft::dft::Parameters dftParams{}; // parameters for dft
  dftParams.direction     = afft::Direction::forward; // it will be a forward transform
  dftParams.precision     = afft::makePrecision<PrecT>(); // set up precision of the transform
  dftParams.shape         = {{size}}; // set up the dimensions
  dftParams.type          = afft::dft::Type::complexToComplex; // let's use complex-to-complex transform
  dftParams.destructive   = true; // allow to destroy source data

  afft::cuda::Parameters cudaParams{}; // it will run on a gpu
  cudaParams.devices = {{0}}; // use device 0

  auto plan = afft::makePlan(dftParams, cudaParams); // generate the plan of the transform, uses current device

  afft::cuda::ExecutionParameters cudaExecParams{}; // execution parameters for CUDA
  cudaExecParams.stream = cudaStream_t{0}; // use stream 0

  plan->execute(src.data(), dst.data(), cudaExecParams); // execute the transform into zero stream

  if (cudaDeviceSynchronize() != cudaSuccess)
  {
    throw std::runtime_error("CUDA error: failed to synchronize");
  }

  // use results from dst vector

  afft::finalize(); // finalize afft library
}
