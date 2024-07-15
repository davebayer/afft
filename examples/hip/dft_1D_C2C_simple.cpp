#include <complex>
#include <vector>

#include <afft/afft.hpp>

template<typename T>
using ManagedVector = std::vector<T, afft::gpu::ManagedAllocator<T>>;

int main(void)
{
  using PrecT = float;

  constexpr afft::Size size{1024}; // size of the transform

  afft::init(); // initialize afft library, also initializes CUDA

  ManagedVector<std::complex<PrecT>> src(size); // source vector
  ManagedVector<std::complex<PrecT>> dst(size); // destination vector

  // initialize source vector

  afft::dft::Parameters dftParams{}; // parameters for dft
  dftParams.direction     = afft::Direction::forward; // it will be a forward transform
  dftParams.precision     = afft::makePrecision<PrecT>(); // set up precision of the transform
  dftParams.shape         = afft::makeScalarView(size); // set up the dimensions
  dftParams.type          = afft::dft::Type::complexToComplex; // let's use complex-to-complex transform

  afft::gpu::Parameters gpuParams{}; // it will run on a gpu
  gpuParams.destroySource = true; // allow to destroy source data

  auto plan = afft::makePlan(dftParams, gpuParams); // generate the plan of the transform, uses current device

  plan.execute(src.data(), dst.data()); // execute the transform into zero stream

  if (hipDeviceSynchronize() != hipSuccess)
  {
    throw std::runtime_error("HIP error: failed to synchronize");
  }

  // use results from dst vector
}
