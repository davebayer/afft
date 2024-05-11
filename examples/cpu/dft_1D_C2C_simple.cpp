#include <complex>
#include <vector>

#include <afft/afft.hpp>

template<typename T>
using AlignedVector = std::vector<T, afft::cpu::AlignedAllocator<T>>;

int main(void)
{
  using PrecT = float;

  constexpr std::size_t size{1024}; // size of the transform

  afft::init(); // initialize afft library

  AlignedVector<std::complex<PrecT>> src(size); // source vector
  AlignedVector<std::complex<PrecT>> dst(size); // destination vector

  // initialize source vector

  afft::dft::Parameters dftParams{}; // parameters for dft
  dftParams.direction     = afft::Direction::forward; // it will be a forward transform
  dftParams.precision     = afft::makePrecision<PrecT>(); // set up precision of the transform
  dftParams.shape         = {{size}}; // set up the dimensions
  dftParams.type          = afft::dft::Type::complexToComplex; // let's use complex-to-complex transform
  dftParams.destroySource = true; // allow to destroy source data
  dftParams.normalize     = afft::Normalize::orthogonal; // use orthogonal normalization

  afft::cpu::Parameters cpuParams{}; // it will run on a cpu
  cpuParams.alignment     = afft::getAlignment(src.data(), dst.data()); // get alignment of the pointers
  cpuParams.threadLimit   = 4; // we will use up to 4 threads

  auto plan = afft::makePlan(dftParams, cpuParams); // generate the plan of the transform

  plan.execute(src.data(), dst.data()); // execute the transform

  // use results from dst vector
}
