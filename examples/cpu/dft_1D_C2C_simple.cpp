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

  const afft::dft::Parameters dftParams // parameters for dft
  {
    .dimensions       = {.shape = {{size}}}, // set up the dimensions
    .commonParameters = {.destroySource = true, // allow to destroy source data
                         .normalize     = afft::Normalize::orthogonal}, // use orthogonal normalization
    .direction        = afft::Direction::forward, // it will be a forward transform
    .precision        = afft::makePrecision<PrecT>(), // set up precision of the transform
    .type             = afft::dft::Type::complexToComplex, // let's use complex-to-complex transform
  };

  const afft::cpu::Parameters cpuParams // it will run on a cpu
  {
    .alignment        = afft::getAlignment(src.data(), dst.data()), // get alignment of the pointers
    .threadLimit      = 4, // we will use up to 4 threads
  };

  // create scope just to make sure the plan is destroyed before afft::finalize() is called
  {
    auto plan = afft::makePlan(dftParams, cpuParams); // generate the plan of the transform

    plan.execute(src.data(), dst.data()); // execute the transform
  }

  // use results from dst vector

  afft::finalize(); // deinitialize afft library
}
