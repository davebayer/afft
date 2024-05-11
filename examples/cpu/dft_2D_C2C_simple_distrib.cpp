#include <complex>
#include <vector>

#include <mpi.h>

#include <afft/afft.hpp>

template<typename T>
using AlignedVector = std::vector<T, afft::cpu::AlignedAllocator<T>>;

int main(void)
{
  using PrecT = float;

  constexpr std::array<std::size_t, 2> shape{1024, 1024}; // shape of the transform

  afft::init(); // initialize afft library

  AlignedVector<std::complex<PrecT>> src{}; // source vector
  AlignedVector<std::complex<PrecT>> dst{}; // destination vector

  afft::dft::Parameters dftParams{}; // parameters for dft
  dftParams.direction        = afft::Direction::forward; // it will be a forward transform
  dftParams.precision        = afft::makePrecision<PrecT>(); // set up precision of the transform
  dftParams.shape            = shape; // set up the dimensions
  dftParams.type             = afft::dft::Type::complexToComplex; // let's use complex-to-complex transform
  dftParams.normalization    = afft::Normalization::orthogonal; // use orthogonal normalization

  afft::distrib::cpu::Parameters<afft::distrib::Implementation::mpi> mpiCpuParams{}; // it will run on multiple CPUs using MPI implementation
  mpiCpuParams.communicator  = MPI_COMM_WORLD; // set up the communicator (unnecessary, MPI_COMM_WORLD is chosen as the default communicator)
  mpiCpuParams.destroySource = true; // allow to destroy source data
  mpiCpuParams.alignment     = afft::cpu::AlignedAllocator<>{}.getAlignment(); // get alignment of the pointers
  mpiCpuParams.threadLimit   = 4; // we will use up to 4 threads per MPI process

  auto plan = afft::makePlan(dftParams, mpiCpuParams); // generate the plan of the transform, also sets up the memory layout in cpuParams

  const auto [srcElemCount, _placeHolder1] = plan.getSrcBufferElemCount(); // get the required size of the source buffer (may be greater than the volume of the source data)
  const auto [dstElemCount, _placeHolder2] = plan.getDstBufferElemCount(); // get the required size of the destination buffer (may be greater than the volume of the destination data)

  src.resize(srcElemCount); // resize the source vector
  dst.resize(dstElemCount); // resize the destination vector

  // initialize source vector according to memory layouts

  plan.execute(src.data(), dst.data()); // execute the transform

  // use results from dst vector
}
