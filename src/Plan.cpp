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

#include <afft/afft.hpp>

#include "error.hpp"

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
    *static_cast<afft_mpi::Parameters*>(mpBackendParams) = desc.getCMpBackendParameters<afft::MpBackend::mpi>();
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
  case afft::Target::cpu:
    *static_cast<afft_cpu_Parameters*>(targetParams) = desc.getCTargetParameters<afft::Target::cpu>();
    break;
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
    {
      cxxPlan->executeUnsafe(srcView, dstView, *reinterpret_cast<const afft::cpu::ExecutionParameters*>(execParams));
      break;
    }
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
