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
    .destructive   = false, // do not allow to destroy source data
    .type          = afft::dft::Type::complexToComplex, // let's use complex-to-complex transform
  };
  
  afft::cpu::Parameters cpuParams // it will run on a cpu
  {
    .threadLimit = 16, // we will use up to 16 threads
  };

  auto plan = afft::makePlan({dftParams, cpuParams}); // generate the plan of the transform

  plan->execute(src.data(), dst.data()); // execute the transform

  // use results from dst vector
}
