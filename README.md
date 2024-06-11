# afft library
## Introduction
`afft` is a C++17 library for FFT related computations. It provides unified interface to various implementations of transforms in C and C++ on CPUs or GPUs. The main goals are:
- user friendly interface,
- support for wide range of the features offered by the backend libraries,
- low overhead,
- and being multiplatform (`Linux`, `Windows` and `MacOS`).

Currently supported transfors are:
- *Discrete Fourier Transform* (DFT) for real and complex inputs (in interleaved or plannar format),
- *Discrete Hartley Transform* (DHT) of real data and
- *Discrete Trigonomic Transform* (DTT) of types DCT (1-4) or DST (1-4).

A transform may be executed *in-place* or *out-of-place* over multidimensional strided arrays in various precision. The created plans can be stored in a LRU plan cache.

Compiles with GCC (10-14), Clang (12-19).

:warning: **Take into account that not all of the afft functionality is supported by each transform backend.**

## License
This library is available under MIT license. See `LICENSE` for details.

## Examples
### Simple 1D complex-to-complex transform in Y axis of 3D padded data
```cpp
#include <complex>
#include <vector>

#include <afft/afft.hpp>

template<typename T>
using AlignedVector = std::vector<T, afft::cpu::AlignedAllocator<T>>;

int main(void)
{
  using PrecT = float;

  constexpr auto shape          = std::to_array<std::size_t>({500, 250, 1020});
  constexpr auto srcPaddedShape = std::to_array<std::size_t>({500, 250, 1024});
  constexpr auto dstPaddedShape = std::to_array<std::size_t>({500, 1020, 256});

  constexpr auto srcElemCount = std::accumulate(srcPaddedShape.begin(),
                                                srcPaddedShape.end(),
                                                std::size_t{1},
                                                std::multiplies<>{});
  constexpr auto dstElemCount = std::accumulate(dstPaddedShape.begin(),
                                                dstPaddedShape.end(),
                                                std::size_t{1},
                                                std::multiplies<>{});

  afft::init(); // initialize afft library

  AlignedVector<std::complex<PrecT>> src(srcElemCount); // source vector
  AlignedVector<std::complex<PrecT>> dst(dstElemCount); // destination vector

  // initialize source vector

  const afft::dft::Parameters<3, 1> dftParams
  {
    .direction     = afft::Direction::forward,
    .precision     = afft::makePrecision<PrecT>(),
    .shape         = shape,
    .axes          = {{1}},
    .normalization = afft::Normalization::unitary,
    .placement     = afft::Placement::outOfPlace,
    .type          = afft::dft::Type::complexToComplex,
  };

  const afft::cpu::Parameters<3> cpuParams
  {
    .memoryLayout   = {.srcStrides = afft::makeStrides(srcPaddedShape),
                       .dstStrides = afft::makeTransposedStrides(
                         dstPaddedShape, {{0, 2, 1}})},
    .complexFormat  = afft::ComplexFormat::interleaved,
    .preserveSource = true,
    .alignment      = afft::getAlignment(src.data(), dst.data()),
    .threadLimit    = 8;
  };

  const afft::cpu::BackendParameters backendParams
  {
    .strategy = afft::SelectStrategy::best,
    .mask     = (afft::Backend::fftw3 | afft::Backend::mkl),
    .order    = {{afft::Backend::mkl, afft::Backend::fftw3}},
    .fftw3    = {.plannerFlag = afft::fftw3::PlannerFlag::exhaustive,
                 .timeLimit   = std::chrono::seconds{30}},
  };

  // create scope just to make sure the plan is destroyed before afft::finalize() is called
  {
    auto plan = afft::makePlan(dftParams, cpuParams, backendParams); // generate the plan of the transform

    plan.execute(src.data(), dst.data()); // execute the transform
  }

  // use results from dst vector

  afft::finalize(); // deinitialize afft library
}
```