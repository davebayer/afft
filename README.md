# afft library
`afft` is a C/C++ library for FFT related computations. It provides unified interface to various implementations of transforms in C and C++ . The main goals are:
- user friendly interface,
- support for wide range of the features offered by the backend libraries,
- low overhead and
- being multiplatform (`Linux`, `Windows` and `MacOS`).

Currently supported transfors are:
- *Discrete Fourier Transform* (DFT) for real and complex inputs (in interleaved or plannar format),
- *Discrete Hartley Transform* (DHT) of real data and
- *Discrete Trigonomic Transform* (DTT) of types DCT (1-4) or DST (1-4).

A transform may be executed *in-place* or *out-of-place* over multidimensional strided arrays in various precision. The created plans can be stored in a LRU plan cache.

The library supports execution in any floating point precision on `CPU`, `CUDA`, `HIP` and `OpenCL` targets and may be distributed over multiple targets or processes (via e. g. MPI).

The transformations are implemented by the backend libraries. Currently, the library supports `clFFT`, `cuFFT`, `FFTW3`, `HeFFTe`, `hipFFT`, `Intel MKL`, `PocketFFT`, `rocFFT` and `VkFFT`. More backend libraries shall be added in the future.

:warning: **Take into account that not all of the afft functionality is supported by each transform backend.**

## License
This library is available under MIT license. See `LICENSE` for details.

## Examples
### Simple 1D complex-to-complex transform in Y axis of 3D padded data
```cpp
#include <array>
#include <chrono>
#include <complex>
#include <vector>

#include <afft/afft.hpp>

// PrecT is the precision type of the transform
using PrecT = float;

// alias for std::vector with aligned allocator
template<typename T>
using AlignedVector = std::vector<T, afft::AlignedAllocator<T>>;

// shape of the transform
constexpr std::array<afft::Size, 3> shape{500, 250, 1020};

// padded source shape
constexpr std::array<afft::Size, 3> srcPaddedShape{500, 250, 1024};

// padded destination shape
constexpr std::array<afft::Size, 3> dstPaddedShape{500, 1020, 256};

// order of the axes in the destination shape
constexpr std::array<afft::Axis, 3> dstAxesOrder{0, 2, 1};

// alignment of the memory
constexpr afft::Alignment alignment = afft::Alignment::cpuNative;

int main()
{
  // make DFT parameters
  afft::dft::Parameters dftParams{};
  dftParams.direction     = afft::Direction::forward;
  dftParams.precision     = afft::makePrecision<PrecT>(); // use same precision for source, destination and execution
  dftParams.shape         = shape;
  dftParams.axes          = {{1}};
  dftParams.normalization = afft::Normalization::none;
  dftParams.placement     = afft::Placement::outOfPlace;
  dftParams.type          = afft::dft::Type::complexToComplex;

  // make CPU parameters
  afft::cpu::Parameters cpuParams{};
  cpuParams.threadLimit = 4; // limit the number of threads to 4

  // make strides for the source and destination shapes
  const auto srcStrides = afft::makeStrides(afft::View<afft::Size, 3>{srcPaddedShape});
  const auto dstStrides = afft::makeTransposedStrides(afft::View<afft::Size, 3>{dstPaddedShape},
                                                      afft::View<afft::Axis, 3>{dstAxesOrder});

  // make memory layout
  afft::CentralizedMemoryLayout memoryLayout{};
  memoryLayout.alignment     = alignment;
  memoryLayout.complexFormat = afft::ComplexFormat::interleaved; // std::complex uses interleaved format
  memoryLayout.srcStrides    = srcStrides;
  memoryLayout.dstStrides    = dstStrides;

  // make backend parameters
  afft::cpu::BackendParameters backendParams{};
  backendParams.strategy          = afft::SelectStrategy::first;
  backendParams.mask              = (afft::Backend::fftw3 | afft::Backend::mkl | afft::Backend::pocketfft);
  backendParams.order             = {{afft::Backend::mkl, afft::Backend::fftw3}};
  backendParams.fftw3.plannerFlag = afft::fftw3::PlannerFlag::measure; // FFTW3 specific planner flag
  backendParams.fftw3.timeLimit   = std::chrono::seconds{2}; // limit the time for the FFTW3 planner

  // make the plan with the parameters
  std::unique_ptr<afft::Plan> plan = afft::makePlan(dftParams, cpuParams, memoryLayout, backendParams);

  // create source and destination vectors
  AlignedVector<std::complex<PrecT>> src(plan->getSrcElemCounts().front()); // source vector
  AlignedVector<std::complex<PrecT>> dst(plan->getDstElemCounts().front()); // destination vector

  // initialize source vector

  // execute the transform
  plan->execute(src.data(), dst.data());

  // use the result in the destination vector
}
```