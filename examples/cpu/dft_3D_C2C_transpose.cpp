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

// shape rank
constexpr std::size_t shapeRank{3};

// shape of the transform
constexpr std::array<afft::Size, shapeRank> shape{500, 250, 1020};

// transform rank
constexpr std::size_t transformRank{1};

// axes of the transform
constexpr std::array<afft::Axis, transformRank> transformAxes{1};

// padded source shape
constexpr std::array<afft::Size, shapeRank> srcPaddedShape{500, 250, 1024};

// padded destination shape
constexpr std::array<afft::Size, shapeRank> dstPaddedShape{500, 1020, 256};

// order of the axes in the destination shape
constexpr std::array<afft::Axis, shapeRank> dstAxesOrder{0, 2, 1};

// alignment of the memory
constexpr afft::Alignment alignment = afft::Alignment::cpuNative;

// order of the backends
constexpr std::array backendOrder{afft::Backend::fftw3, afft::Backend::mkl, afft::Backend::pocketfft};

int main()
{
  // make DFT parameters
  afft::dft::Parameters dftParams{};
  dftParams.direction     = afft::Direction::forward;
  dftParams.precision     = afft::makePrecision<PrecT>(); // use same precision for source, destination and execution
  dftParams.shape         = shape;
  dftParams.axes          = transformAxes;
  dftParams.normalization = afft::Normalization::none;
  dftParams.placement     = afft::Placement::outOfPlace;
  dftParams.type          = afft::dft::Type::complexToComplex;

  // make CPU parameters
  afft::cpu::Parameters cpuParams{};
  cpuParams.threadLimit = 4; // limit the number of threads to 4

  // make strides for the source and destination shapes
  std::array<afft::Size, shapeRank> srcStrides{};
  std::array<afft::Size, shapeRank> dstStrides{};

  afft::makeStrides(shapeRank, std::data(srcPaddedShape), std::data(srcStrides));
  afft::makeTransposedStrides(shapeRank, std::data(dstPaddedShape), std::data(dstAxesOrder), std::data(dstStrides));

  // make memory layout
  afft::CentralizedMemoryLayout memoryLayout{};
  memoryLayout.alignment     = alignment;
  memoryLayout.complexFormat = afft::ComplexFormat::interleaved; // std::complex uses interleaved format
  memoryLayout.srcStrides    = std::data(srcStrides);
  memoryLayout.dstStrides    = std::data(dstStrides);

  // make backend parameters
  afft::cpu::BackendParameters backendParams{};
  backendParams.strategy          = afft::SelectStrategy::first;
  backendParams.mask              = (afft::Backend::fftw3 | afft::Backend::mkl | afft::Backend::pocketfft);
  backendParams.order             = backendOrder;
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
