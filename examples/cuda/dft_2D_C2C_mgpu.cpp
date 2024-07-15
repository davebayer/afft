#include <complex>
#include <vector>

#include <cuda_runtime.h>

#include <afft/afft.hpp>

template<typename T>
using ManagedVector = std::vector<T, afft::cuda::ManagedAllocator<T>>;

int main(void)
{
  using PrecT = float;

  constexpr afft::Size size{1024}; // size of the transform

  afft::init(); // initialize afft library, also initializes CUDA if uninitialized

  afft::dft::Parameters dftParams{}; // parameters for dft
  dftParams.direction     = afft::Direction::forward; // it will be a forward transform
  dftParams.precision     = afft::makePrecision<PrecT>(); // set up precision of the transform
  dftParams.shape         = afft::makeScalarView(size); // set up the dimensions
  dftParams.type          = afft::dft::Type::complexToComplex; // let's use complex-to-complex transform
  dftParams.placement     = afft::Placement::inPlace; // it will be an in-place transform
  dftParams.destructive   = true; // allow to destroy source data

  afft::cuda::Parameters cudaParams{}; // it will run on a gpu
  cudaParams.devices = {{0, 1}}; // use devices 0 and 1

  afft::DistribMemoryLayout memoryLayout{};
  memoryLayout.dstDistribAxes = {{0}}; // distribute the transform along the first axis
  memoryLayout.srcDistribAxes = {{1}}; // distribute the transform along the second axis

  auto plan = afft::makePlan(dftParams, cudaParams); // generate the plan of the transform, uses current device

  const auto srcElemCounts = plan->getSrcElemCounts(); // get the number of elements in the source buffers
  const auto dstElemCounts = plan->getDstElemCounts(); // get the number of elements in the destination buffers

  ManagedVector<std::complex<PrecT>> src0(srcElemCounts[0]); // device 0 source vector
  ManagedVector<std::complex<PrecT>> src1(srcElemCounts[1]); // device 1 source vector
  ManagedVector<std::complex<PrecT>> dst0(dstElemCounts[0]); // device 0 destination vector
  ManagedVector<std::complex<PrecT>> dst1(dstElemCounts[1]); // device 1 destination vector

  // initialize source vectors

  afft::cuda::ExecutionParameters cudaExecParams{}; // execution parameters for CUDA
  cudaExecParams.stream = cudaStream_t{0}; // use stream 0

  plan->execute({{src0.data(), src1.data()}}, {{dst0.data(), dst1.data()}}, cudaExecParams); // execute the transform into zero stream

  if (cudaDeviceSynchronize() != cudaSuccess)
  {
    throw std::runtime_error("CUDA error: failed to synchronize");
  }

  // use results from dst vectors

  afft::finalize(); // finalize afft library
}
