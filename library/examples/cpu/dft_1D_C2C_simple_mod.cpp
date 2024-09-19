#include <complex>
#include <vector>

import afft;

template<typename T>
using AlignedVector = std::vector<T, afft::AlignedAllocator<T>>;

int main(void)
{
  using PrecT = float;

  constexpr afft::Size size{1024}; // size of the transform

  afft::init(); // initialize afft library

  AlignedVector<std::complex<PrecT>> src(size); // source vector
  AlignedVector<std::complex<PrecT>> dst(size); // destination vector

  // initialize source vector

  afft::dft::Parameters dftParams // parameters for dft
  {
    .direction     = afft::Direction::forward, // it will be a forward transform
    .precision     = afft::makePrecision<PrecT>(), // set up precision of the transform
    .shape         = afft::makeScalarView(size), // set up the dimensions
    .normalization = afft::Normalization::none, // do not normalize
    .type          = afft::dft::Type::complexToComplex, // let's use complex-to-complex transform
  };
  
  afft::cpu::BackendParameters cpuBackendParams // set up parameters for the CPU backend
  {
    .threadLimit = 1, // we will use 1 thread
  };

  auto plan = afft::makePlan({dftParams, afft::cpu::Parameters{}}, cpuBackendParams); // generate the plan of the transform

  plan->execute(src.data(), dst.data()); // execute the transform

  // use results from dst vector
}
