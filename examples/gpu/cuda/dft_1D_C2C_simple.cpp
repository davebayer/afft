#include <complex>
#include <vector>

#include <afft/afft.hpp>

template<typename T>
using UnifiedMemoryVector = std::vector<T, afft::gpu::UnifiedMemoryAllocator<T>>;

int main(void)
{
  using PrecT = float;

  constexpr std::size_t size{1024}; // size of the transform

  afft::init(); // initialize afft library, also initializes CUDA

  UnifiedMemoryVector<std::complex<PrecT>> src(size); // source vector
  UnifiedMemoryVector<std::complex<PrecT>> dst(size); // destination vector

  // initialize source vector

  const afft::dft::Parameters dftParams // parameters for dft
  {
    .dimensions       = {.shape = {{size}}}, // set up the dimensions
    .commonParameters = {.destroySource = true}, // destroy source vector after the transform
    .direction        = afft::Direction::forward, // it will be a forward transform
    .precision        = afft::makePrecision<PrecT>(), // set up precision of the transform
    .type             = afft::dft::complexToComplex, // let's use complex-to-complex transform
  };

  // create scope just to make sure the plan is destroyed before afft::finalize() is called
  {
    auto plan = afft::makePlan(dftParams, afft::gpu::Parameters{}); // generate the plan of the transform, uses current device

    plan.execute(src.data(), dst.data()); // execute the transform into zero stream
  }

  if (cudaDeviceSynchronize() != cudaSuccess)
  {
    throw std::runtime_error("CUDA error: failed to synchronize");
  }

  // use results from dst vector

  afft::finalize(); // deinitialize afft library
}
