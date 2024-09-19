#include <complex>
#include <vector>

#include <afft/afft.hpp>

template<typename T>
using AlignedVector = std::vector<T, afft::AlignedAllocator<T>>;

int main(void)
{
  using PrecT = float;

  constexpr std::size_t size{1024}; // size of the transform

  AlignedVector<std::complex<PrecT>> src(size); // source vector
  AlignedVector<std::complex<PrecT>> dst(size); // destination vector

  // initialize source vector

  afft::dft::Parameters dftParams{}; // parameters for dft
  dftParams.direction     = afft::Direction::forward; // it will be a forward transform
  dftParams.precision     = afft::makePrecision<PrecT>(); // set up precision of the transform
  dftParams.shape         = afft::makeScalarView(size); // set up the dimensions
  dftParams.axes          = {{1}}; // set up the axes
  dftParams.type          = afft::dft::Type::complexToComplex; // let's use complex-to-complex transform
  dftParams.normalization = afft::Normalization::orthogonal; // use orthogonal normalization

  afft::CentralizedMemoryLayout memoryLayout{}; // set up memory layout
  memoryLayout.alignment = afft::alignmentOf(src.data(), dst.data());

  afft::cpu::BackendParameters cpuBackendParams{}; // set up parameters for the cpu backend
  cpuBackendParams.allowDestructive = true; // allow destructive transform
  cpuBackendParams.threadLimit      = 4; // we will use up to 4 threads

  auto plan = afft::makePlan({dftParams, afft::cpu::Parameters{}, memoryLayout}, cpuBackendParams); // generate the plan of the transform

  plan->execute(src.data(), dst.data()); // execute the transform

  // use results from dst vector
}
