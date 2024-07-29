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

#include "architecture.hpp"
#include "backend.hpp"
#include "common.hpp"
#include "transform.hpp"

/**
 * @brief Create a plan for a given transform and architecture.
 * @param transformParams Transform parameters.
 * @param archParams Architecture parameters.
 * @param planPtr Pointer to the plan.
 * @return Error code.
 */
extern "C" afft_Error _afft_Plan_create(afft_TransformParameters    transformParams,
                                        afft_ArchitectureParameters archParams,
                                        afft_Plan**                 planPtr)
try
{
  if (planPtr == nullptr)
  {
    return afft_Error_invalidArgument;
  }

  auto create2 = [&](auto cxxTransformParams, auto cxxArchParams)
  {
    auto cxxPlan = afft::makePlan(cxxTransformParams, cxxArchParams);

    *planPtr = reinterpret_cast<afft_Plan*>(cxxPlan.release());
    
    return afft_Error_success;
  };

  auto create1 = [&](auto cxxTransformParams)
  {
    const std::size_t shapeRank = cxxTransformParams.shape.size();

    switch (archParams.target)
    {
    case afft_Target_cpu:
      switch (archParams.distribution)
      {
      case afft_Distribution_spst:
        return create2(cxxTransformParams, Convert<afft::spst::cpu::Parameters<>>::fromC(archParams.spstCpu, shapeRank));
      case afft_Distribution_mpst:
        return create2(cxxTransformParams, Convert<afft::mpst::cpu::Parameters<>>::fromC(archParams.mpstCpu, shapeRank));
      default:
        return afft_Error_invalidArchitectureParameters;
      }
    case afft_Target_gpu:
      switch (archParams.distribution)
      {
      case afft_Distribution_spst:
        return create2(cxxTransformParams, Convert<afft::spst::gpu::Parameters<>>::fromC(archParams.spstGpu, shapeRank));
      // case afft_Distribution_spmt:
      //   return create2(cxxTransformParams, Convert<afft::spmt::gpu::Parameters<>>::fromC(archParams.spmtGpu, shapeRank));
      case afft_Distribution_mpst:
        return create2(cxxTransformParams, Convert<afft::mpst::gpu::Parameters<>>::fromC(archParams.mpstGpu, shapeRank));
      default:
        return afft_Error_invalidArchitectureParameters;
      }
    default:
      return afft_Error_invalidArchitectureParameters;
    }
  };

  switch (transformParams.transform)
  {
  case afft_Transform_dft:
    return create1(Convert<afft::dft::Parameters<>>::fromC(transformParams.dft));
  case afft_Transform_dht:
    return create1(Convert<afft::dht::Parameters<>>::fromC(transformParams.dht));
  case afft_Transform_dtt:
    return create1(Convert<afft::dtt::Parameters<>>::fromC(transformParams.dtt));
  default:
    return afft_Error_invalidArgument;
  }
}
catch (...)
{
  return afft_Error_internal;
}

/**
 * @brief Create a plan for a given transform, architecture, and backend.
 * @param transformParams Transform parameters.
 * @param archParams Architecture parameters.
 * @param backendParams Backend parameters.
 * @param planPtr Pointer to the plan.
 * @return Error code.
 */
extern "C" afft_Error _afft_Plan_createWithBackendParameters(afft_TransformParameters    transformParams,
                                                             afft_ArchitectureParameters archParams,
                                                             afft_BackendParameters      backendParams,
                                                             afft_Plan**                 planPtr)
try
{
  if (planPtr == nullptr)
  {
    return afft_Error_invalidArgument;
  }

  if (archParams.target != backendParams.target || archParams.distribution != backendParams.distribution)
  {
    return afft_Error_architectureMismatch;
  }

  auto create2 = [&](auto cxxTransformParams, auto cxxArchParams, auto cxxBackendParams)
  {
    auto cxxPlan = afft::makePlan(cxxTransformParams, cxxArchParams, cxxBackendParams);

    *planPtr = reinterpret_cast<afft_Plan*>(cxxPlan.release());

    return afft_Error_success;
  };

  auto create1 = [&](auto cxxTransformParams)
  {
    const std::size_t shapeRank = cxxTransformParams.shape.size();

    switch (archParams.target)
    {
    case afft_Target_cpu:
      switch (archParams.distribution)
      {
      case afft_Distribution_spst:
        return create2(cxxTransformParams,
                       Convert<afft::spst::cpu::Parameters<>>::fromC(archParams.spstCpu, shapeRank),
                       Convert<afft::spst::cpu::BackendParameters>::fromC(backendParams.spstCpu));
      case afft_Distribution_mpst:
        return create2(cxxTransformParams,
                       Convert<afft::mpst::cpu::Parameters<>>::fromC(archParams.mpstCpu, shapeRank),
                       Convert<afft::mpst::cpu::BackendParameters>::fromC(backendParams.mpstCpu));
      default:
        return afft_Error_invalidArchitectureParameters;
      }
    case afft_Target_gpu:
      switch (archParams.distribution)
      {
      case afft_Distribution_spst:
        return create2(cxxTransformParams,
                       Convert<afft::spst::gpu::Parameters<>>::fromC(archParams.spstGpu, shapeRank),
                       Convert<afft::spst::gpu::BackendParameters>::fromC(backendParams.spstGpu));
      // case afft_Distribution_spmt:
      //   return create2(cxxTransformParams,
      //                  Convert<afft::spmt::gpu::Parameters<>>::fromC(archParams.spmtGpu, shapeRank),
      //                  Convert<afft::spmt::gpu::BackendParameters>::fromC(backendParams.spmtGpu));
      case afft_Distribution_mpst:
        return create2(cxxTransformParams,
                       Convert<afft::mpst::gpu::Parameters<>>::fromC(archParams.mpstGpu, shapeRank),
                       Convert<afft::mpst::gpu::BackendParameters>::fromC(backendParams.mpstGpu));
      default:
        return afft_Error_invalidArchitectureParameters;
      }
    default:
      return afft_Error_invalidArchitectureParameters;
    }
  };

  switch (transformParams.transform)
  {
  case afft_Transform_dft:
    return create1(Convert<afft::dft::Parameters<>>::fromC(transformParams.dft));
  case afft_Transform_dht:
    return create1(Convert<afft::dht::Parameters<>>::fromC(transformParams.dht));
  case afft_Transform_dtt:
    return create1(Convert<afft::dtt::Parameters<>>::fromC(transformParams.dtt));
  default:
    return afft_Error_invalidArgument;
  }
}
catch (...)
{
  return afft_Error_internal;
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
 * @brief Get the plan transform.
 * @param plan Plan object.
 * @param transform Pointer to the transform variable.
 * @return Error code.
 */
extern "C" afft_Error afft_Plan_getTransform(const afft_Plan* plan, afft_Transform* transform)
try
{
  if (plan == nullptr)
  {
    return afft_Error_invalidPlan;
  }

  if (transform == nullptr)
  {
    return afft_Error_invalidArgument;
  }

  *transform = Convert<afft::Transform>::toC(reinterpret_cast<const afft::Plan*>(plan)->getTransform());

  return afft_Error_success;
}
catch (afft_Error e)
{
  return e;
}
catch (...)
{
  return afft_Error_internal;
}

/**
 * @brief Get the plan target.
 * @param plan Plan object.
 * @param target Pointer to the target variable.
 * @return Error code.
 */
extern "C" afft_Error afft_Plan_getTarget(const afft_Plan* plan, afft_Target* target)
try
{
  if (plan == nullptr)
  {
    return afft_Error_invalidPlan;
  }

  if (target == nullptr)
  {
    return afft_Error_invalidArgument;
  }

  *target = Convert<afft::Target>::toC(reinterpret_cast<const afft::Plan*>(plan)->getTarget());

  return afft_Error_success;
}
catch (afft_Error e)
{
  return e;
}
catch (...)
{
  return afft_Error_internal;
}

/**
 * @brief Get the target count.
 * @param plan Plan object.
 * @param targetCount Pointer to the target count variable.
 * @return Error code.
 */
extern "C" afft_Error afft_Plan_getTargetCount(const afft_Plan* plan, size_t* targetCount)
try
{
  if (plan == nullptr)
  {
    return afft_Error_invalidPlan;
  }

  if (targetCount == nullptr)
  {
    return afft_Error_invalidArgument;
  }

  *targetCount = reinterpret_cast<const afft::Plan*>(plan)->getTargetCount();

  return afft_Error_success;
}
catch (...)
{
  return afft_Error_internal;
}

/**
 * @brief Get the plan distribution.
 * @param plan Plan object.
 * @param distribution Pointer to the distribution variable.
 * @return Error code.
 */
extern "C" afft_Error afft_Plan_getDistribution(const afft_Plan* plan, afft_Distribution* distribution)
try
{
  if (plan == nullptr)
  {
    return afft_Error_invalidPlan;
  }

  if (distribution == nullptr)
  {
    return afft_Error_invalidArgument;
  }

  *distribution = Convert<afft::Distribution>::toC(reinterpret_cast<const afft::Plan*>(plan)->getDistribution());

  return afft_Error_success;
}
catch (afft_Error e)
{
  return e;
}
catch (...)
{
  return afft_Error_internal;
}

/**
 * @brief Get the plan backend.
 * @param plan Plan object.
 * @param backend Pointer to the backend variable.
 * @return Error code.
 */
extern "C" afft_Error afft_Plan_getBackend(const afft_Plan* plan, afft_Backend* backend)
try
{
  if (plan == nullptr)
  {
    return afft_Error_invalidPlan;
  }

  if (backend == nullptr)
  {
    return afft_Error_invalidArgument;
  }

  *backend = Convert<afft::Backend>::toC(reinterpret_cast<const afft::Plan*>(plan)->getBackend());

  return afft_Error_success;
}
catch (afft_Error e)
{
  return e;
}
catch (...)
{
  return afft_Error_internal;
}

/**
 * @brief Get the plan workspace size.
 * @param plan Plan object.
 * @param count Pointer to the count variable.
 * @param workspaceSizes Pointer to the workspace sizes.
 * @return Error code.
 */
extern "C" afft_Error afft_Plan_getWorkspaceSize(const afft_Plan* plan, size_t* count, const size_t** workspaceSizes)
{
  if (plan == nullptr)
  {
    return afft_Error_invalidPlan;
  }

  if (count == nullptr)
  {
    return afft_Error_invalidArgument;
  }

  if (workspaceSizes == nullptr)
  {
    return afft_Error_invalidArgument;
  }

  const auto ws = reinterpret_cast<const afft::Plan*>(plan)->getWorkspaceSize();

  *count          = ws.size();
  *workspaceSizes = ws.data();

  return afft_Error_success;
}

/**
 * @brief Execute a plan.
 * @param plan Plan object.
 * @param src Source data pointer array of target count size (x2 if planar complex).
 * @param dst Destination data pointer array of target count size (x2 if planar complex).
 * @return Error code.
 */
extern "C" afft_Error afft_Plan_execute(afft_Plan* plan, void* const* src, void* const* dst)
try
{
  if (plan == nullptr)
  {
    return afft_Error_invalidPlan;
  }

  if (src == nullptr)
  {
    return afft_Error_invalidArgument;
  }

  if (dst == nullptr)
  {
    return afft_Error_invalidArgument;
  }

  afft::Plan* cxxPlan = reinterpret_cast<afft::Plan*>(plan);

  const auto& desc = afft::detail::DescGetter::get(*cxxPlan);

  const auto [srcBufferCount, dstBufferCount] = desc.getSrcDstBufferCount();

  cxxPlan->executeUnsafe(afft::View<void*>{src, srcBufferCount}, afft::View<void*>{dst, dstBufferCount});

  return afft_Error_success;
}
catch (...)
{
  return afft_Error_internal;
}

/**
 * @brief Execute a plan with parameters.
 * @param plan Plan object.
 * @param src Source data pointer array of target count size (x2 if planar complex).
 * @param dst Destination data pointer array of target count size (x2 if planar complex).
 * @param execParams Execution parameters.
 * @return Error code.
 */
extern "C" afft_Error _afft_Plan_executeWithParameters(afft_Plan*               plan,
                                                       void* const*             src,
                                                       void* const*             dst,
                                                       afft_ExecutionParameters execParams)
try
{
  if (plan == nullptr)
  {
    return afft_Error_invalidPlan;
  }

  if (src == nullptr)
  {
    return afft_Error_invalidArgument;
  }

  if (dst == nullptr)
  {
    return afft_Error_invalidArgument;
  }

  auto execute = [&](auto cxxExecParams)
  {
    afft::Plan* cxxPlan = reinterpret_cast<afft::Plan*>(plan);

    const auto& desc = afft::detail::DescGetter::get(*cxxPlan);

    const auto [srcBufferCount, dstBufferCount] = desc.getSrcDstBufferCount();

    cxxPlan->executeUnsafe(afft::View<void*>{src, srcBufferCount},
                           afft::View<void*>{dst, dstBufferCount},
                           cxxExecParams);

    return afft_Error_success;
  };

  switch (execParams.target)
  {
  case afft_Target_cpu:
    switch (execParams.distribution)
    {
    case afft_Distribution_spst:
      return execute(Convert<afft::spst::cpu::ExecutionParameters>::fromC(execParams.spstCpu));
    case afft_Distribution_mpst:
      return execute(Convert<afft::mpst::cpu::ExecutionParameters>::fromC(execParams.mpstCpu));
    default:
      return afft_Error_invalidExecutionParameters;
    }
  case afft_Target_gpu:
    switch (execParams.distribution)
    {
    case afft_Distribution_spst:
      return execute(Convert<afft::spst::gpu::ExecutionParameters>::fromC(execParams.spstGpu));
    // case afft_Distribution_spmt:
    //   return execute(Convert<afft::spmt::gpu::ExecutionParameters>::fromC(execParams.spmtGpu));
    case afft_Distribution_mpst:
      return execute(Convert<afft::mpst::gpu::ExecutionParameters>::fromC(execParams.mpstGpu));
    default:
      return afft_Error_invalidExecutionParameters;
    }
  default:
    return afft_Error_invalidExecutionParameters;
  }

  return afft_Error_success;
}
catch (...)
{
  return afft_Error_internal;
}
