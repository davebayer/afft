# afft library
## Introduction
`afft` is a C++20 library for FFT related computations. It provides unified interface to various implementations of transforms in C and C++ on CPUs or GPUs. The main goals are:
- user friendly interface,
- support for wide range of the features offered by the backend libraries,
- low overhead,
- and being multiplatform (`Linux`, `Windows` and `MacOS`).

Currently supported transfors are:
- *Discrete Fourier Transform* (DFT) for real and complex inputs (in interleaved or plannar format) and
- *Discrete Trigonomic Transform* (DTT) of types DCT (1-4) or DST (1-4).

A transform may be executed *in-place* or *out-of-place* over multidimensional strided arrays in various precision. The created plans can be stored in a LRU plan cache.

:warning: **Take into account that not all of the afft functionality is supported by all transform backends.**

## License
This library is available under MIT license. See `LICENSE` for details.

## Examples
### Simple 1D complex-to-complex transform
```cpp
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
    .type             = afft::dft::complexToComplex, // let's use complex-to-complex transform
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
```