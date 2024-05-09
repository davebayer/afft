#include <complex>
#include <vector>

#define CL_VERSION_2_0
#include <CL/cl.h>

#include <afft/afft.hpp>

template<typename T>
using UnifiedMemoryVector = std::vector<T, afft::gpu::UnifiedMemoryAllocator<T>>;

int main(void)
{
  using PrecT = float;

  constexpr std::size_t size{1024}; // size of the transform

  cl_int           error{};
  cl_platform_id   platform{};
  cl_device_id     device{};
  cl_context       context{};
  cl_command_queue queue{};

  if (clGetPlatformIDs(1, &platform, nullptr) != CL_SUCCESS)
  {
    throw std::runtime_error("OpenCL error: failed to get platform ID");
  }

  if (clGetDeviceIDs(platform, CL_DEVICE_TYPE_GPU, 1, &device, nullptr) != CL_SUCCESS)
  {
    throw std::runtime_error("OpenCL error: failed to get device ID");
  }

  context = clCreateContext(nullptr, 1, &device, nullptr, nullptr, &error);

  if (error != CL_SUCCESS)
  {
    throw std::runtime_error("OpenCL error: failed to create context");
  }

  queue = clCreateCommandQueue(context, device, 0, &error);

  if (error != CL_SUCCESS)
  {
    throw std::runtime_error("OpenCL error: failed to create command queue");
  }

  afft::init(); // initialize afft library, also initializes CUDA

  UnifiedMemoryVector<std::complex<PrecT>> src(size, {context}); // source vector
  UnifiedMemoryVector<std::complex<PrecT>> dst(size, {context}); // destination vector

  // initialize source vector

  afft::dft::Parameters dftParams{}; // parameters for dft
  dftParams.direction                      = afft::Direction::forward; // it will be a forward transform
  dftParams.precision                      = afft::makePrecision<PrecT>(); // set up precision of the transform
  dftParams.commonParameters.destroySource = true; // destroy source vector after the transform
  dftParams.shape                          = {{size}}; // set up the dimensions
  dftParams.type                           = afft::dft::Type::complexToComplex; // let's use complex-to-complex transform

  afft::gpu::Parameters gpuParams{}; // parameters for GPU
  gpuParams.context = context; // set up OpenCL context
  gpuParams.device  = device; // set up OpenCL device

  // create scope just to make sure the plan is destroyed before afft::finalize() is called
  {
    auto plan = afft::makePlan(dftParams, gpuParams); // generate the plan of the transform

    const afft::gpu::ExecutionParameters execParams
    {
      .commandQueue = queue, // set up OpenCL command queue
    };

    plan.execute(src.data(), dst.data(), execParams); // execute the transform into zero stream
  }

  // use results from dst vector

  afft::finalize(); // deinitialize afft library

  if (clReleaseCommandQueue(queue) != CL_SUCCESS)
  {
    throw std::runtime_error("OpenCL error: failed to release command queue");
  }
  
  if (clReleaseContext(context) != CL_SUCCESS)
  {
    throw std::runtime_error("OpenCL error: failed to release context");
  }
}
