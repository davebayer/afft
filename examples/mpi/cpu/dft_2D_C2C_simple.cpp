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

  MPI_Init(nullptr, nullptr); // initialize MPI
  afft::init(); // initialize afft library

  AlignedVector<std::complex<PrecT>> src{}; // source vector
  AlignedVector<std::complex<PrecT>> dst{}; // destination vector

  afft::dft::Parameters dftParams{}; // parameters for dft
  dftParams.direction     = afft::Direction::forward; // it will be a forward transform
  dftParams.precision     = afft::makePrecision<PrecT>(); // set up precision of the transform
  dftParams.shape         = shape; // set up the dimensions
  dftParams.type          = afft::dft::Type::complexToComplex; // let's use complex-to-complex transform
  dftParams.normalization = afft::Normalization::none; // use orthogonal normalization
  dftParams.destructive   = true; // allow to destroy source data

  afft::mpi::Parameters mpiParams{}; // parameters for MPI
  mpiParams.comm = MPI_COMM_WORLD; // set up the communicator (unnecessary, MPI_COMM_WORLD is chosen as the default communicator)

  afft::cpu::Parameters cpuParams{}; // parameters for CPU
  cpuParams.threadLimit = 2; // we will use up to 4 threads

  afft::DistribMemoryLayout memoryLayout{}; // set up memory layout
  memoryLayout.alignment = afft::cpu::defaultAlignment; // set up the alignment
  
  {
    auto plan = afft::makePlan(dftParams, mpiParams, cpuParams); // generate the plan of the transform, also sets up the memory layout in cpuParams

    const std::size_t srcElemCount = plan.getSrcElemCounts().front(); // get the number of elements in the source vector
    const std::size_t dstElemCount = plan.getDstElemCounts().front(); // get the number of elements in the destination vector

    src.resize(srcElemCount); // resize the source vector
    dst.resize(dstElemCount); // resize the destination vector

    // initialize source vector according to memory layouts

    plan.execute(src.data(), dst.data()); // execute the transform
  }

  // use results from dst vector

  afft::finalize(); // finalize afft library
  MPI_Finalize(); // finalize MPI
}
