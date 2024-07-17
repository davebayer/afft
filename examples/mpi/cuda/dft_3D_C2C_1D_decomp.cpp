#include <array>
#include <complex>
#include <cstdio>
#include <memory>
#include <type_traits>
#include <vector>

#include <cuda_runtime.h>
#include <mpi.h>

#include <afft/afft.hpp>

#include <helpers/cuda.hpp>

/// @brief PrecT is the precision of the transform
using PrecT = float;

/// @brief ManagedVector is a vector using CUDA unified memory
template<typename T>
using ManagedVector = std::vector<T, afft::cuda::ManagedAllocator<T>>;

/// @brief Complex managed vector
using ComplexManagedVector = ManagedVector<std::complex<PrecT>>;

/// @brief The shape of the transform
constexpr auto shape = std::array<afft::Size, 3>{128, 128, 128};

/// @brief The decomposition axes of the original domain
constexpr auto orgDomainDecompAxes = std::array<afft::Axis, 1>{0};

/// @brief The decomposition axes of the frequency domain
constexpr auto freqDomainDecompAxes = std::array<afft::Axis, 1>{1};

int main(void)
{
  // return value
  int retval = EXIT_SUCCESS;

  // initialize MPI library
  MPI_Init(nullptr, nullptr);

  // initialize afft library
  afft::init();

  try
  {

    // set MPI communicator
    const MPI_Comm comm = MPI_COMM_WORLD;

    // get the rank of the MPI process in MPI_COMM_WORLD communicator
    int rank{};
    MPI_Comm_rank(comm, &rank);

    // select CUDA device
    const int cudaDevice = rank % helpers::cuda::getDeviceCount();

    // create CUDA stream
    auto cudaStream = helpers::cuda::makeStream();

    // make DFT parameters
    afft::dft::Parameters dftParams{};
    dftParams.direction     = afft::Direction::forward;
    dftParams.precision     = afft::makePrecision<PrecT>();
    dftParams.shape         = shape;
    dftParams.axes          = afft::allAxes;
    dftParams.normalization = afft::Normalization::none;
    dftParams.placement     = afft::Placement::outOfPlace;
    dftParams.destructive   = true;
    dftParams.type          = afft::dft::Type::complexToComplex;

    // make MPI parameters
    afft::mpi::Parameters mpiParams{};
    mpiParams.comm = comm;

    // make CUDA parameters
    afft::cuda::Parameters cudaParams{};
    cudaParams.devices = afft::makeScalarView(cudaDevice);

    // make memory layout parameters
    afft::DistributedMemoryLayout memoryLayout{};
    memoryLayout.complexFormat  = afft::ComplexFormat::interleaved;
    memoryLayout.srcDistribAxes = orgDomainDecompAxes;
    memoryLayout.dstDistribAxes = freqDomainDecompAxes;
    
    // make forward plan
    auto fwdPlan = afft::makePlan(dftParams, mpiParams, cudaParams, memoryLayout);

    // get the number of elements in the source vector
    const std::size_t srcElemCount = fwdPlan->getSrcElemCounts().front();

    // get the number of elements in the destination vector
    const std::size_t dstElemCount = fwdPlan->getDstElemCounts().front();

    // make source vector
    ComplexManagedVector src(srcElemCount, ComplexManagedVector::allocator_type{cudaMemAttachHost});

    // make destination vector
    ComplexManagedVector dst(dstElemCount, ComplexManagedVector::allocator_type{cudaMemAttachHost});

    // initialize source vector accoring to memory layout specified in memoryLayout variable (it was filled by makePlan function)

    // make CUDA execution parameters
    afft::cuda::ExecutionParameters cudaExecParams{};
    cudaExecParams.stream = cudaStream.get();

    // execute the forward transform
    fwdPlan->execute(src.data(), dst.data(), cudaExecParams);
  }
  catch (const std::exception& e)
  {
    std::fprintf(stderr, "Error: %s\n", e.what());
    retval = EXIT_FAILURE;
  }
  catch (...)
  {
    std::fprintf(stderr, "Unknown error\n");
    retval = EXIT_FAILURE;
  }

  // finalize afft library
  afft::finalize();

  // finalize MPI library
  MPI_Finalize();

  return retval;
}
