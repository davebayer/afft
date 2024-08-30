#include <complex>
#include <vector>

#include <afft/afft.hpp>

#include <helpers/cuda.hpp>

int main(void)
{
  using PrecT = float;

  constexpr afft::Size size{1024}; // size of wthe transform

  afft::init(); // initialize afft library, also initializes CUDA if uninitialized

  std::vector<std::complex<PrecT>> src(size); // source vector
  std::vector<std::complex<PrecT>> dst(size); // destination vector

  // initialize source vector

  afft::dft::Parameters dftParams{}; // parameters for dft
  dftParams.direction     = afft::Direction::forward; // it will be a forward transform
  dftParams.precision     = afft::makePrecision<PrecT>(); // set up precision of the transform
  dftParams.shape         = afft::makeScalarView(size); // set up the dimensions
  dftParams.type          = afft::dft::Type::complexToComplex; // let's use complex-to-complex transform
  dftParams.destructive   = true; // allow to destroy source data

  afft::cuda::Parameters cudaParams{}; // it will run on a gpu
  cudaParams.devices = {{0}}; // use device 0

  afft::SelectParameters selectParams{}; // select parameters
  selectParams.mask = afft::BackendMask::cufft; // use cuFFT backend

  auto plan = afft::makePlan({dftParams, cudaParams}, selectParams); // generate the plan of the transform, uses current device

  afft::cuda::ExecutionParameters cudaExecParams{}; // execution parameters for CUDA
  cudaExecParams.stream = cudaStream_t{0}; // use stream 0

  plan->execute(src.data(), dst.data(), cudaExecParams); // execute the transform into zero stream

  CUDART_CALL(cudaDeviceSynchronize()); // synchronize the device

  // use results from dst vector

  afft::finalize(); // finalize afft library
}
