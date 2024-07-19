#include <stdio.h>
#include <stdlib.h>

#include <afft/afft.hpp>

template<typename T>
using AlignedVector = std::vector<T, afft::AlignedAllocator<T>>;

using PrecT = float;

int main()
{
  constexpr std::array<afft::Axis, 3> dstAxesOrder{0, 2, 1};

  constexpr std::array<afft::Size, 3> shape          = {500, 250, 1020};
  constexpr std::array<afft::Size, 3> srcPaddedShape = {500, 250, 1024};
  constexpr std::array<afft::Size, 3> dstPaddedShape = {500, 1020, 256};

  constexpr auto srcElemCount = srcPaddedShape[0] * srcPaddedShape[1] * srcPaddedShape[2];
  constexpr auto dstElemCount = dstPaddedShape[0] * dstPaddedShape[1] * dstPaddedShape[2];

  constexpr afft::Alignment alignment = afft::Alignment::avx2;

  afft::dft::Parameters dftParams{};
  dftParams.direction     = afft::Direction::forward;
  dftParams.precision     = afft::makePrecision<PrecT>();
  dftParams.shape         = shape;
  dftParams.axes          = {{1}};
  dftParams.normalization = afft::Normalization::unitary;
  dftParams.placement     = afft::Placement::outOfPlace;
  dftParams.type          = afft::dft::Type::complexToComplex;

  afft::cpu::Parameters cpuParams{};
  cpuParams.threadLimit = 4;

  const auto srcStrides = afft::makeStrides(afft::View<afft::Size, 3>{srcPaddedShape});
  const auto dstStrides = afft::makeTransposedStrides(afft::View<afft::Size, 3>{dstPaddedShape}, afft::View<afft::Axis, 3>{dstAxesOrder});

  afft::CentralizedMemoryLayout memoryLayout{};
  memoryLayout.alignment     = alignment;
  memoryLayout.complexFormat = afft::ComplexFormat::interleaved;
  memoryLayout.srcStrides    = srcStrides;
  memoryLayout.dstStrides    = dstStrides;

  afft::cpu::BackendParameters backendParams{};
  backendParams.strategy          = afft::SelectStrategy::first;
  backendParams.mask              = (afft::Backend::fftw3 | afft::Backend::mkl | afft::Backend::pocketfft);
  backendParams.order             = {{afft::Backend::mkl, afft::Backend::fftw3}};
  backendParams.fftw3.plannerFlag = afft::fftw3::PlannerFlag::measure;
  backendParams.fftw3.timeLimit   = std::chrono::seconds{2};

  AlignedVector<std::complex<PrecT>> src(srcElemCount); // source vector
  AlignedVector<std::complex<PrecT>> dst(dstElemCount); // destination vector

  // check if src and dst are not NULL
  // initialize source vector

  auto plan = afft::makePlan(dftParams, cpuParams, memoryLayout, backendParams); // generate the plan of the transform

  plan->execute(src.data(), dst.data()); // execute the transform
}