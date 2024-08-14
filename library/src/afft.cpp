/*
  This file is part of afft library.

  Copyright (c) 2024 David Bayer

  Permission is hereby granted, free of charge, to any person obtaining a copy
  of this software and associated documentation files (the "Software"), to deal
  in the Software without restriction, including without limitation the rights
  to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
  copies of the Software, and to permit persons to whom the Software is
  furnished to do so, subject to the following conditions:

  The above copyright notice and this permission notice shall be included in all
  copies or substantial portions of the Software.

  THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
  IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
  FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
  AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
  LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
  OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
  SOFTWARE.
*/

// Define the header only mode
#define AFFT_HEADER_ONLY

// Disable the inline keyword for symbols that should be included in the library
#define AFFT_HEADER_ONLY_INLINE

// Include the library headers
#include <afft/afft.hpp>

/**********************************************************************************************************************/
// Backend
/**********************************************************************************************************************/
/**
 * @brief Get the name of the backend.
 * @param backend Backend.
 * @return Name of the backend.
 */
extern "C" const char*
getBackendName(afft_Backend backend)
{
  return afft::getBackendName(static_cast<afft::Backend>(backend)).data();
}

/**********************************************************************************************************************/
// Error
/**********************************************************************************************************************/
/**
 * @brief Set error details message.
 * @param errDetails Error details.
 * @param message Message.
 */
static void setErrorDetailsMessage(afft_ErrorDetails& errDetails, const char* message) noexcept
{
  std::strncpy(errDetails.message, message, AFFT_MAX_ERROR_MESSAGE_SIZE);
  errDetails.message[AFFT_MAX_ERROR_MESSAGE_SIZE - 1] = '\0';
}

/**
 * @brief Clear error details return value.
 * @param errDetails Error details.
 */
static void clearErrorDetailsRetval(afft_ErrorDetails& errDetails) noexcept
{
  std::memset(&errDetails.retval, 0, sizeof(errDetails.retval));
}

/**
 * @brief Set error details.
 * @tparam T Error return value type.
 * @param errorDetails Error details.
 * @param message Message.
 * @param retval Return value.
 */
template<typename T = std::monostate>
static void setErrorDetails(afft_ErrorDetails* errorDetails, const char* message, T&& retval = {})
{
  // Require that T fits into errorDetails->retval.
  static_assert(sizeof(T) <= sizeof(errorDetails->retval));

  if (errorDetails != nullptr)
  {
    // Set error message.
    setErrorDetailsMessage(*errorDetails, message);

    // Set return value.
    if constexpr (std::is_same_v<T, std::monostate>)
    {
      clearErrorDetailsRetval(*errorDetails);
    }
    else
    {
      std::memcpy(&errorDetails->retval, &retval, sizeof(retval));
    }
  }
}

/**
 * @brief Handle exception.
 * @param errDetails Error details.
 * @return Error code.
 */
static afft_Error handleException(afft_ErrorDetails* errDetails) noexcept
try
{
  throw;
}
catch (const afft::Exception& e)
{
  if (errDetails != nullptr)
  {
    setErrorDetailsMessage(*errDetails, e.what());

    [[maybe_unused]] auto setRetvalMember = [&](auto& member) noexcept -> void
    {
      using T = std::decay_t<decltype(member)>;

      const T* value = std::get_if<T>(&e.getErrorRetval());

      if (value != nullptr)
      {
        member = *value;
      }
      else
      {
        clearErrorDetailsRetval(*errDetails);
      }
    };

    switch (e.getError())
    {
#   ifdef AFFT_ENABLE_MPI
    case afft::Error::mpi:
      setRetvalMember(errDetails->retval.mpi);
      break;
#   endif
#   ifdef AFFT_ENABLE_CUDA
    case afft::Error::cudaDriver:
      setRetvalMember(errDetails->retval.cudaDriver);
      break;
    case afft::Error::cudaRuntime:
      setRetvalMember(errDetails->retval.cudaRuntime);
      break;
    case afft::Error::cudaRtc:
      setRetvalMember(errDetails->retval.cudaRtc);
      break;
#   endif
#   ifdef AFFT_ENABLE_HIP
    case afft::Error::hip:
      setRetvalMember(errDetails->retval.hip);
      break;
#   endif
#   ifdef AFFT_ENABLE_OPENCL
    case afft::Error::opencl:
      setRetvalMember(errDetails->retval.opencl);
      break;
#   endif
    default:
      clearErrorDetailsRetval(*errDetails);
      break;
    }
  }

  return static_cast<afft_Error>(e.getError());
}
catch (const std::exception& e)
{
  setErrorDetails(errDetails, e.what());

  return afft_Error_internal;
}
catch (...)
{
  setErrorDetails(errDetails, "Unknown error");

  return afft_Error_internal;
}

/**********************************************************************************************************************/
// Init
/**********************************************************************************************************************/
/**
 * @brief Initialize the library.
 * @param errDetails Error details.
 * @return Error code.
 */
extern "C" afft_Error
afft_init(afft_ErrorDetails* errDetails)
try
{
  afft::init();

  return afft_Error_success;
}
catch (...)
{
  return handleException(errDetails);
}

/**
 * @brief Finalize the library.
 * @param errDetails Error details.
 * @return Error code.
 */
extern "C" afft_Error
afft_finalize(afft_ErrorDetails* errDetails)
try
{
  afft::finalize();
  
  return afft_Error_success;
}
catch (...)
{
  return handleException(errDetails);
}

/**********************************************************************************************************************/
// Memory
/**********************************************************************************************************************/
/**
 * @brief Allocate aligned memory.
 * @param sizeInBytes Size of the memory block in bytes.
 * @param alignment Alignment of the memory block.
 * @return Pointer to the allocated memory block or NULL if the allocation failed.
 */
extern "C" void*
afft_alignedAlloc(size_t         sizeInBytes,
                  afft_Alignment alignment)
{
  return ::operator new[](sizeInBytes, static_cast<std::align_val_t>(alignment), std::nothrow);
}

/**
 * @brief Free aligned memory.
 * @param ptr Pointer to the memory block.
 */
extern "C" void
afft_alignedFree(void*          ptr,
                 afft_Alignment alignment)
{
  ::operator delete[](ptr, static_cast<std::align_val_t>(alignment), std::nothrow);
}


/**********************************************************************************************************************/
// Plan
/**********************************************************************************************************************/
template<typename T>
inline T& assignIfNotNull(T& dst, const T* ptr)
{
  if (ptr != nullptr)
  {
    dst = *ptr;
  }

  return dst;
}

template<afft::MpBackend mpBackend, afft::Target target>
static std::unique_ptr<afft::Plan> planCreateHelper2(const afft::detail::Desc& desc,
                                                     const void*               cBackendParams)
{
  using BackendParameters = afft::BackendParameters<mpBackend, target>;

  if (cBackendParams == nullptr)
  {
    return afft::detail::makePlan(desc, BackendParameters{});
  }
  else
  {
    // fixme
    return afft::detail::makePlan(desc, *static_cast<const BackendParameters*>(cBackendParams));
  }
}

template<afft::MpBackend mpBackend>
static std::unique_ptr<afft::Plan> planCreateHelper1(const afft::detail::Desc& desc,
                                                     const void*               cBackendParams)
{
  switch (desc.getTarget())
  {
  case afft::Target::cpu:
    return planCreateHelper2<mpBackend, afft::Target::cpu>(desc, cBackendParams);
  case afft::Target::cuda:
# ifdef AFFT_ENABLE_CUDA
    return planCreateHelper2<mpBackend, afft::Target::cuda>(desc, cBackendParams);
# else
    throw afft::Exception{afft::Error::invalidArgument, "CUDA target is not enabled"};
# endif /* AFFT_ENABLE_CUDA */
  case afft::Target::hip:
# ifdef AFFT_ENABLE_HIP
    return planCreateHelper2<mpBackend, afft::Target::hip>(desc, cBackendParams);
# else
    throw afft::Exception{afft::Error::invalidArgument, "HIP target is not enabled"};
# endif /* AFFT_ENABLE_HIP */
  case afft::Target::opencl:
# ifdef AFFT_ENABLE_OPENCL
    return planCreateHelper2<mpBackend, afft::Target::opencl>(desc, cBackendParams);
# else
    throw afft::Exception{afft::Error::invalidArgument, "OpenCL target is not enabled"};
# endif /* AFFT_ENABLE_OPENCL */
  default:
    throw afft::Exception{afft::Error::invalidArgument, "invalid target"};
  }
}

/**
 * @brief Create a plan for a given transform and architecture.
 * @param planParams Plan parameters.
 * @param planPtr Pointer to the plan.
 * @param errorDetails Error details.
 * @return Error code.
 */
extern "C" afft_Error
afft_Plan_create(afft_PlanParameters planParams,
                 afft_Plan**         planPtr,
                 afft_ErrorDetails*  errorDetails)
try
{
  if (planPtr == nullptr)
  {
    setErrorDetails(errorDetails, "invalid plan pointer");
    return afft_Error_invalidArgument;
  }

  afft::detail::Desc desc{planParams};

  std::unique_ptr<afft::Plan> plan{};

  switch (desc.getMpBackend())
  {
  case afft::MpBackend::none:
    plan = planCreateHelper1<afft::MpBackend::none>(desc, planParams.backendParams);
    break;
  case afft::MpBackend::mpi:
# ifdef AFFT_ENABLE_MPI
    plan = planCreateHelper1<afft::MpBackend::mpi>(desc, planParams.backendParams);
    break;
# else
    setErrorDetails(errorDetails, "MPI backend is not enabled");
    return afft_Error_invalidArgument;
# endif /* AFFT_ENABLE_MPI */
  default:
    setErrorDetails(errorDetails, "invalid multi-process backend");
    return afft_Error_invalidArgument;
  }

  *planPtr = reinterpret_cast<afft_Plan*>(plan.release());

  return afft_Error_success;
}
catch (...)
{
  return handleException(errorDetails);
}

/**
 * @brief Destroy a plan object.
 * @param plan Plan object.
 */
extern "C" void afft_Plan_destroy(afft_Plan* plan)
{
  ::operator delete(reinterpret_cast<afft::Plan*>(plan), std::nothrow);
}

/**
 * @brief Get the plan multi-process backend.
 * @param plan Plan object.
 * @param mpBackend Pointer to the multi-process backend variable.
 * @param errDetails Error details.
 * @return Error code.
 */
extern "C" afft_Error afft_Plan_getMpBackend(afft_Plan*         plan,
                                             afft_MpBackend*    mpBackend,
                                             afft_ErrorDetails* errDetails)
try
{
  if (plan == nullptr)
  {
    setErrorDetails(errDetails, "invalid plan");
    return afft_Error_invalidArgument;
  }

  if (mpBackend == nullptr)
  {
    setErrorDetails(errDetails, "invalid multi-process backend pointer");
    return afft_Error_invalidArgument;
  }

  *mpBackend = static_cast<afft_MpBackend>(reinterpret_cast<afft::Plan*>(plan)->getMpBackend());

  return afft_Error_success;
}
catch (...)
{
  return handleException(errDetails);
}

/**
 * @brief Get the plan multi-process backend parameters.
 * @param plan Plan object.
 * @param mpBackendParams Pointer to the multi-process backend parameters.
 * @param errDetails Error details.
 * @return Error code.
 */
extern "C" afft_Error
afft_Plan_getMpBackendParameters(afft_Plan*         plan,
                                 void*              mpBackendParams,
                                 afft_ErrorDetails* errDetails)
try
{
  if (plan == nullptr)
  {
    setErrorDetails(errDetails, "invalid plan");
    return afft_Error_invalidArgument;
  }

  if (mpBackendParams == nullptr)
  {
    setErrorDetails(errDetails, "invalid multi-process backend parameters pointer");
    return afft_Error_invalidArgument;
  }

  const afft::detail::Desc& desc = afft::detail::DescGetter::get(*reinterpret_cast<afft::Plan*>(plan));

  switch (desc.getMpBackend())
  {
  case afft::MpBackend::none:
    break;
  case afft::MpBackend::mpi:
# ifdef AFFT_ENABLE_MPI
    *static_cast<afft_mpi_Parameters*>(mpBackendParams) = desc.getCMpParameters<afft::MpBackend::mpi>();
    break;
# else
    setErrorDetails(errDetails, "MPI backend is not enabled");
    return afft_Error_invalidArgument;
# endif /* AFFT_ENABLE_MPI */
  default:
    setErrorDetails(errDetails, "invalid multi-process backend");
    return afft_Error_internal;
  }

  return afft_Error_success;
}
catch (...)
{
  return handleException(errDetails);
}

/**
 * @brief Get the plan transform.
 * @param plan Plan object.
 * @param transform Pointer to the transform variable.
 * @param errDetails Error details.
 * @return Error code.
 */
extern "C" afft_Error
afft_Plan_getTransform(afft_Plan*         plan,
                       afft_Transform*    transform,
                       afft_ErrorDetails* errDetails)
try
{
  if (plan == nullptr)
  {
    setErrorDetails(errDetails, "invalid plan");
    return afft_Error_invalidArgument;
  }

  if (transform == nullptr)
  {
    setErrorDetails(errDetails, "invalid transform pointer");
    return afft_Error_invalidArgument;
  }

  *transform = static_cast<afft_Transform>(reinterpret_cast<afft::Plan*>(plan)->getTransform());

  return afft_Error_success;
}
catch (...)
{
  return handleException(errDetails);
}

/**
 * @brief Get the plan transform parameters.
 * @param plan Plan object.
 * @param transformParams Pointer to the transform parameters.
 * @param errDetails Error details.
 * @return Error code.
 */
extern "C" afft_Error
afft_Plan_getTransformParameters(afft_Plan*         plan,
                                 void*              transformParams,
                                 afft_ErrorDetails* errDetails)
try
{
  if (plan == nullptr)
  {
    setErrorDetails(errDetails, "invalid plan");
    return afft_Error_invalidArgument;
  }

  if (transformParams == nullptr)
  {
    setErrorDetails(errDetails, "invalid transform parameters pointer");
    return afft_Error_invalidArgument;
  }

  const afft::detail::Desc& desc = afft::detail::DescGetter::get(*reinterpret_cast<afft::Plan*>(plan));

  switch (desc.getTransform())
  {
  case afft::Transform::dft:
    *static_cast<afft_dft_Parameters*>(transformParams) = desc.getCTransformParameters<afft::Transform::dft>();
    break;
  case afft::Transform::dht:
    *static_cast<afft_dht_Parameters*>(transformParams) = desc.getCTransformParameters<afft::Transform::dht>();
    break;
  case afft::Transform::dtt:
    *static_cast<afft_dtt_Parameters*>(transformParams) = desc.getCTransformParameters<afft::Transform::dtt>();
    break;
  default:
    setErrorDetails(errDetails, "invalid transform");
    return afft_Error_internal;
  }

  return afft_Error_success;
}
catch (...)
{
  return handleException(errDetails);

}

/**
 * @brief Get the plan target.
 * @param plan Plan object.
 * @param target Pointer to the target variable.
 * @param errDetails Error details.
 * @return Error code.
 */
extern "C" afft_Error
afft_Plan_getTarget(afft_Plan*         plan,
                    afft_Target*       target,
                    afft_ErrorDetails* errDetails)
try
{
  if (plan == nullptr)
  {
    setErrorDetails(errDetails, "invalid plan");
    return afft_Error_invalidArgument;
  }

  if (target == nullptr)
  {
    setErrorDetails(errDetails, "invalid target pointer");
    return afft_Error_invalidArgument;
  }

  *target = static_cast<afft_Target>(reinterpret_cast<afft::Plan*>(plan)->getTarget());

  return afft_Error_success;
}
catch (...)
{
  return handleException(errDetails);
}

/**
 * @brief Get the target count.
 * @param plan Plan object.
 * @param targetCount Pointer to the target count variable.
 * @param errDetails Error details.
 * @return Error code.
 */
extern "C" afft_Error
afft_Plan_getTargetCount(afft_Plan*         plan,
                         size_t*            targetCount,
                         afft_ErrorDetails* errDetails)
try
{
  if (plan == nullptr)
  {
    setErrorDetails(errDetails, "invalid plan");
    return afft_Error_invalidArgument;
  }

  if (targetCount == nullptr)
  {
    setErrorDetails(errDetails, "invalid target count pointer");
    return afft_Error_invalidArgument;
  }

  *targetCount = reinterpret_cast<afft::Plan*>(plan)->getTargetCount();

  return afft_Error_success;
}
catch (...)
{
  return handleException(errDetails);
}

/**
 * @brief Get the plan target parameters.
 * @param plan Plan object.
 * @param targetParams Pointer to the target parameters.
 * @param errDetails Error details.
 * @return Error code.
 */
extern "C" afft_Error
afft_Plan_getTargetParameters(afft_Plan*         plan,
                              void*              targetParams,
                              afft_ErrorDetails* errDetails)
try
{
  if (plan == nullptr)
  {
    setErrorDetails(errDetails, "invalid plan");
    return afft_Error_invalidArgument;
  }

  if (targetParams == nullptr)
  {
    setErrorDetails(errDetails, "invalid target parameters pointer");
    return afft_Error_invalidArgument;
  }

  const afft::detail::Desc& desc = afft::detail::DescGetter::get(*reinterpret_cast<afft::Plan*>(plan));

  switch (desc.getTarget())
  {
# ifdef AFFT_ENABLE_CPU
  case afft::Target::cpu:
    *static_cast<afft_cpu_Parameters*>(targetParams) = desc.getCTargetParameters<afft::Target::cpu>();
    break;
# endif /* AFFT_ENABLE_CPU */
# ifdef AFFT_ENABLE_CUDA
  case afft::Target::cuda:
    *static_cast<afft_cuda_Parameters*>(targetParams) = desc.getCTargetParameters<afft::Target::cuda>();
    break;
# endif /* AFFT_ENABLE_CUDA */
# ifdef AFFT_ENABLE_HIP
  case afft::Target::hip:
    *static_cast<afft_hip_Parameters*>(targetParams) = desc.getCTargetParameters<afft::Target::hip>();
    break;
# endif /* AFFT_ENABLE_HIP */
# ifdef AFFT_ENABLE_OPENCL
  case afft::Target::opencl:
    *static_cast<afft_opencl_Parameters*>(targetParams) = desc.getCTargetParameters<afft::Target::opencl>();
    break;
# endif /* AFFT_ENABLE_OPENCL */
  default:
    setErrorDetails(errDetails, "invalid target");
    return afft_Error_internal;
  }

  return afft_Error_success;
}
catch (...)
{
  return handleException(errDetails);
}

/**
 * @brief Get the plan backend.
 * @param plan Plan object.
 * @param backend Pointer to the backend variable.
 * @param errDetails Error details.
 * @return Error code.
 */
extern "C" afft_Error
afft_Plan_getBackend(afft_Plan*         plan,
                     afft_Backend*      backend,
                     afft_ErrorDetails* errDetails)
try
{
  if (plan == nullptr)
  {
    setErrorDetails(errDetails, "invalid plan");
    return afft_Error_invalidArgument;
  }

  if (backend == nullptr)
  {
    setErrorDetails(errDetails, "invalid backend pointer");
    return afft_Error_invalidArgument;
  }

  *backend = static_cast<afft_Backend>(reinterpret_cast<afft::Plan*>(plan)->getBackend());

  return afft_Error_success;
}
catch (...)
{
  return handleException(errDetails);
}

/**
 * @brief Get the plan workspace size.
 * @param plan Plan object.
 * @param workspaceSizes Pointer to the workspace sizes of target count size.
 * @param errDetails Error details.
 * @return Error code.
 */
extern "C" afft_Error
afft_Plan_getWorkspaceSizes(afft_Plan*         plan,
                            const size_t**     workspaceSizes,
                            afft_ErrorDetails* errDetails)
try
{
  if (plan == nullptr)
  {
    setErrorDetails(errDetails, "invalid plan");
    return afft_Error_invalidArgument;
  }

  if (workspaceSizes == nullptr)
  {
    setErrorDetails(errDetails, "invalid workspace sizes pointer");
    return afft_Error_invalidArgument;
  }

  *workspaceSizes = reinterpret_cast<afft::Plan*>(plan)->getWorkspaceSizes().data();

  return afft_Error_success;
}
catch (...)
{
  return handleException(errDetails);
}

/**
 * @brief Execute a plan.
 * @param plan Plan object.
 * @param src Source data pointer array of target count size (x2 if planar complex).
 * @param dst Destination data pointer array of target count size (x2 if planar complex).
 * @param execParams Execution parameters.
 * @param errDetails Error details.
 * @return Error code.
 */
extern "C" afft_Error
afft_Plan_execute(afft_Plan*         plan,
                  void* const*       src,
                  void* const*       dst,
                  const void*        execParams,
                  afft_ErrorDetails* errDetails)
try
{
  if (plan == nullptr)
  {
    setErrorDetails(errDetails, "invalid plan");
    return afft_Error_invalidArgument;
  }

  if (src == nullptr)
  {
    setErrorDetails(errDetails, "invalid source data pointer array");
    return afft_Error_invalidArgument;
  }

  if (dst == nullptr)
  {
    setErrorDetails(errDetails, "invalid destination data pointer array");
    return afft_Error_invalidArgument;
  }

  auto* cxxPlan = reinterpret_cast<afft::Plan*>(plan);

  const std::size_t targetCount = cxxPlan->getTargetCount();

  const afft::View<void*> srcView{src, targetCount};
  const afft::View<void*> dstView{dst, targetCount};

  if (execParams == nullptr)
  {
    cxxPlan->executeUnsafe(srcView, dstView);
  }
  else
  {
    switch (cxxPlan->getTarget())
    {
    case afft::Target::cpu:
#   ifdef AFFT_ENABLE_CPU
    {
      cxxPlan->executeUnsafe(srcView, dstView, *reinterpret_cast<const afft::cpu::ExecutionParameters*>(execParams));
      break;
    }
#   else
      setErrorDetails(errDetails, "CPU target is not enabled");
      return afft_Error_internal;
#   endif /* AFFT_ENABLE_CPU */
    case afft::Target::cuda:
#   ifdef AFFT_ENABLE_CUDA
    {
      cxxPlan->executeUnsafe(srcView, dstView, *reinterpret_cast<const afft::cuda::ExecutionParameters*>(execParams));
      break;
    }
#   else
      setErrorDetails(errDetails, "CUDA target is not enabled");
      return afft_Error_internal;
#   endif /* AFFT_ENABLE_CUDA */
    case afft::Target::hip:
#   ifdef AFFT_ENABLE_HIP
    {
      cxxPlan->executeUnsafe(srcView, dstView, *reinterpret_cast<const afft::hip::ExecutionParameters*>(execParams));
      break;
    }
#   else
      setErrorDetails(errDetails, "HIP target is not enabled");
      return afft_Error_internal;
#   endif /* AFFT_ENABLE_HIP */
    case afft::Target::opencl:
#   ifdef AFFT_ENABLE_OPENCL
    {
      cxxPlan->executeUnsafe(srcView, dstView, *reinterpret_cast<const afft::opencl::ExecutionParameters*>(execParams));
      break;
    }
#   else
      setErrorDetails(errDetails, "OpenCL target is not enabled");
      return afft_Error_internal;
#   endif /* AFFT_ENABLE_OPENCL */
    default:
      setErrorDetails(errDetails, "invalid target");
      return afft_Error_internal;
    }
  }

  return afft_Error_success;
}
catch (...)
{
  return handleException(errDetails);
}

/**********************************************************************************************************************/
// Utils
/**********************************************************************************************************************/
/**
 * @brief Make strides.
 * @param shapeRank Rank of the shape.
 * @param shape Shape of the array.
 * @param strides Strides of the array.
 * @param fastestAxisStride Stride of the fastest axis.
 * @param errorDetails Error details.
 * @return Error code.
 */
extern "C" afft_Error
afft_makeStrides(const size_t       shapeRank,
                 const afft_Size*   shape,
                 afft_Size*         strides,
                 const afft_Size    fastestAxisStride,
                 afft_ErrorDetails* errorDetails)
try
{
  afft::makeStrides(shapeRank, shape, strides, fastestAxisStride);

  return afft_Error_success;
}
catch (...)
{
  return handleException(errorDetails);
}

/**
 * @brief Make transposed strides.
 * @param shapeRank Rank of the shape.
 * @param resultShape Shape of the result array.
 * @param orgAxesOrder Original axes order.
 * @param strides Strides of the array.
 * @param fastestAxisStride Stride of the fastest axis.
 * @param errorDetails Error details.
 * @return Error code.
 */
extern "C" afft_Error
afft_makeTransposedStrides(const size_t       shapeRank,
                           const afft_Size*   resultShape,
                           const afft_Axis*   orgAxesOrder,
                           afft_Size*         strides,
                           const afft_Size    fastestAxisStride,
                           afft_ErrorDetails* errorDetails)
try
{
  afft::makeTransposedStrides(shapeRank, resultShape, orgAxesOrder, strides, fastestAxisStride);

  return afft_Error_success;
}
catch (...)
{
  return handleException(errorDetails);
}

/**********************************************************************************************************************/
// Version
/**********************************************************************************************************************/
/**
 * @brief Get the version of the clFFT library.
 * @return Version
 */
extern "C" afft_Version afft_clfft_getVersion()
{
  return afft::clfft::getVersion();
}

/**
 * @brief Get the version of the cuFFT library.
 * @return Version
 */
extern "C" afft_Version afft_cufft_getVersion()
{
  return afft::cufft::getVersion();
}

/**
 * @brief Get the version of the FFTW3 library.
 * @param precision Precision
 * @return Version
 */
extern "C" afft_Version afft_fftw3_getVersion(afft_Precision precision)
try
{
  return afft::fftw3::getVersion(static_cast<afft::Precision>(precision));
}
catch (...)
{
  return afft_Version{};
}

/**
 * @brief Get the version of the HeFFTe library.
 * @return Version
 */
extern "C" afft_Version afft_heffte_getVersion()
{
  return afft::heffte::getVersion();
}

/**
 * @brief Get the version of the hipFFT library.
 * @return Version
 */
extern "C" afft_Version afft_hipfft_getVersion()
{
  return afft::hipfft::getVersion();
}

/**
 * @brief Get the version of the MKL library.
 * @return Version
 */
extern "C" afft_Version afft_mkl_getVersion()
{
  return afft::mkl::getVersion();
}

/**
 * @brief Get the version of the PocketFFT library.
 * @return Version
 */
extern "C" afft_Version afft_pocketfft_getVersion()
{
  return afft::pocketfft::getVersion();
}

/**
 * @brief Get the version of the rocFFT library.
 * @return Version
 */
extern "C" afft_Version afft_rocfft_getVersion()
{
  return afft::rocfft::getVersion();
}

/**
 * @brief Get the version of the VkFFT library.
 * @return Version
 */
extern "C" afft_Version afft_vkfft_getVersion()
{
  return afft::vkfft::getVersion();
}
