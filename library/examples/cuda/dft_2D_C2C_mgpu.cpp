#include <complex>
#include <vector>

#include <cuda_runtime.h>

#include <afft/afft.hpp>

#include <helpers/cuda.hpp>

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

  afft::DistributedMemoryLayout memoryLayout{};
  memoryLayout.dstDistribAxes = {{0}}; // distribute the transform along the first axis
  memoryLayout.srcDistribAxes = {{1}}; // distribute the transform along the second axis

  auto plan = afft::makePlan({dftParams, cudaParams, memoryLayout}); // generate the plan of the transform, uses current device

  const auto srcElemCounts = plan->getSrcElemCounts(); // get the number of elements in the source buffers
  const auto dstElemCounts = plan->getDstElemCounts(); // get the number of elements in the destination buffers

  std::vector<std::complex<PrecT>> src0(srcElemCounts[0]); // device 0 source vector
  std::vector<std::complex<PrecT>> src1(srcElemCounts[1]); // device 1 source vector
  std::vector<std::complex<PrecT>> dst0(dstElemCounts[0]); // device 0 destination vector
  std::vector<std::complex<PrecT>> dst1(dstElemCounts[1]); // device 1 destination vector

  // initialize source vectors

  afft::cuda::ExecutionParameters cudaExecParams{}; // execution parameters for CUDA
  cudaExecParams.stream = cudaStream_t{0}; // use stream 0

  plan->execute(afft::View<std::complex<PrecT>*>{std::array{src0.data(), src1.data()}},
                afft::View<std::complex<PrecT>*>{std::array{dst0.data(), dst1.data()}},
                cudaExecParams); // execute the transform into zero stream

  CALL_CUDART(cudaDeviceSynchronize()); // synchronize the device

  // use results from dst vectors

  afft::finalize(); // finalize afft library
}
