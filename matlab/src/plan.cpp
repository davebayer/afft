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
#include <matlabw/mx/mx.hpp>

#include "parser.hpp"
#include "planCache.hpp"

using namespace matlabw;

/// @brief Matlab plan data.
struct PlanData
{
  afft::Plan*   plan;       ///< Plan pointer.
  std::uint64_t cacheEpoch; ///< Plan cache epoch.
};

/**
 * @brief Create a plan.
 * @param lhs Left-hand side array of size 1.
 *            * lhs[0] holds the plan data.
 * @param rhs Right-hand side array of size 2.
 *            * rhs[0] holds the transform parameters as a struct,
 *            * rhs[1] holds the target parameters as a struct.
 */
void planCreate(mx::Span<mx::Array> lhs, mx::View<mx::ArrayCref> rhs)
{
  if (rhs.size() != 4)
  {
    throw mx::Exception("afft:planCreate:invalidInputCount", "invalid number of input arguments, expected 2");
  }

  if (lhs.size() != 1)
  {
    throw mx::Exception("afft:planCreate:invalidOutputCount", "invalid number of output arguments, expected 1");
  }

  if (!rhs[0].isStruct())
  {
    throw mx::Exception("afft:planCreate:invalidArgument", "transform parameters must be a struct");
  }

  if (!rhs[1].isStruct())
  {
    throw mx::Exception("afft:planCreate:invalidArgument", "target parameters must be a struct");
  }

  if (!rhs[2].isStruct())
  {
    throw mx::Exception("afft:planCreate:invalidArgument", "backend parameters must be a struct");
  }

  if (!rhs[3].isStruct())
  {
    throw mx::Exception("afft:planCreate:invalidArgument", "select parameters must be a struct");
  }

  TransformParametersParser transformParamsParser{};
  TargetParametersParser    targetParamsParser{};

  const auto transformParamsVariant = transformParamsParser(rhs[0]);
  const auto targetParamsVariant    = targetParamsParser(rhs[1]);

  afft::Description desc = std::visit([&](const auto& transformParams, const auto& targetParams) -> afft::Description
  {
    using TransformParamsT = std::decay_t<decltype(transformParams)>;
    using TargetParamsT    = std::decay_t<decltype(targetParams)>;

    if constexpr (std::is_same_v<TransformParamsT, std::monostate>)
    {
      throw mx::Exception("afft:planCreate:invalidArgument", "invalid transform parameters");
    }
    else if constexpr (std::is_same_v<TargetParamsT, std::monostate>)
    {
      throw mx::Exception("afft:planCreate:invalidArgument", "invalid target parameters");
    }
    else
    {
      return afft::Description{transformParams, targetParams};
    }
  }, transformParamsVariant, targetParamsVariant);

  auto planIt = planCache.find(desc);

  if (planIt == planCache.end())
  {
    BackendParametersParser backendParamsParser{};
    SelectParametersParser  selectParamsParser{};

    const auto backendParamsVariant = backendParamsParser(rhs[2], desc.getTarget());
    const auto selectParamsVariant  = selectParamsParser(rhs[3]);

    planIt = std::visit([&](const auto& backendParams, const auto& selectParams) -> afft::PlanCache::iterator
    {
      using BackendParamsT = std::decay_t<decltype(backendParams)>;
      using SelectParamsT  = std::decay_t<decltype(selectParams)>;

      if constexpr (std::is_same_v<BackendParamsT, std::monostate>)
      {
        throw mx::Exception{"afft:planCreate:invalidArgument", "invalid backend parameters"};
      }
      else if constexpr (std::is_same_v<SelectParamsT, std::monostate>)
      {
        throw mx::Exception{"afft:planCreate:invalidArgument", "invalid select parameters"};
      }
      else
      {
        return planCache.emplace(std::cref(desc), backendParams, selectParams);
      }
    }, backendParamsVariant, selectParamsVariant);
  }

  lhs[0] = mx::makeUninitNumericArray<std::uint8_t>({{sizeof(PlanData)}});

  PlanData* planData = static_cast<PlanData*>(lhs[0].getData());
  planData->plan       = planIt->get();
  planData->cacheEpoch = planCacheEpoch;
}

/**
 * @brief Execute a plan.
 * @param lhs Left-hand side array of size 1.
 *            * lhs[0] holds the output array.
 * @param rhs Right-hand side array of size 2.
 *            * rhs[0] holds the pointer to the plan as a scalar NumericArray<std::uint64_t>,
 *            * rhs[1] holds the input array.
 */
void planExecute(mx::Span<mx::Array> lhs, mx::View<mx::ArrayCref> rhs)
{
  if (rhs.size() != 2)
  {
    throw mx::Exception{"afft:invalidInputCount", "invalid number of input arguments, expected 2"};
  }

  if (lhs.size() != 1)
  {
    throw mx::Exception{"afft:invalidOutputCount", "invalid number of output arguments, expected 1"};
  }

  if (!rhs[0].isUint8() || rhs[0].getSize() != sizeof(PlanData))
  {
    throw mx::Exception{"afft:Plan:execute:invalidArgument", "invalid plan data"};
  }

  const PlanData* planData = static_cast<const PlanData*>(rhs[0].getData());

  if (planData->plan == nullptr)
  {
    throw mx::Exception{"afft:Plan:execute:invalidPlan", "invalid plan"};
  }

  if (planData->cacheEpoch != planCacheEpoch)
  {
    throw mx::Exception{"afft:Plan:execute:invalidPlan", "the plan was created in a different plan cache epoch"};
  }

  afft::Plan* plan = planData->plan;

  const auto [srcPrec, dstPrec] = plan->getSrcDstPrecision();
  const auto [srcCmpl, dstCmpl] = plan->getSrcDstComplexity();

  const auto& desc = plan->getDescription().get(afft::detail::DescToken::make());

  auto checkSrcArray = [srcPrec, srcCmpl, desc](auto&& src)
  {
    switch (src.getClassId())
    {
    case mx::ClassId::single:
      if (srcPrec != afft::Precision::f32)
      {
        throw mx::Exception{"afft:Plan:execute:invalidArgument", "invalid input array precision"};
      }
      if (src.isComplex() && srcCmpl != afft::Complexity::complex)
      {
        throw mx::Exception{"afft:Plan:execute:invalidArgument", "invalid input array complexity"};
      }
      break;
    case mx::ClassId::_double:
      if (srcPrec != afft::Precision::f64)
      {
        throw mx::Exception{"afft:Plan:execute:invalidArgument", "invalid input array precision"};
      }
      if (src.isComplex() && srcCmpl != afft::Complexity::complex)
      {
        throw mx::Exception{"afft:Plan:execute:invalidArgument", "invalid input array complexity"};
      }
      break;
    default:
      throw mx::Exception{"afft:Plan:execute:invalidArgument", "invalid input array type"};
    }

    const auto shapeRank = desc.getShapeRank();
    const auto srcShape  = desc.getSrcShape<std::size_t>();
    const auto srcDims   = src.getDims();

    if (!std::equal(srcDims.rbegin(), srcDims.rend(), srcShape.data, srcShape.data + shapeRank))
    {
      throw mx::Exception{"afft:Plan:execute:invalidArgument", "invalid input array size"};
    }
  };

  auto makeDstArray = [dstPrec, dstCmpl, desc](auto&& makeDstArrayFn)
  {
    static_assert(std::is_invocable_v<decltype(makeDstArrayFn), mx::View<std::size_t>, mx::ClassId, mx::Complexity>);

    mx::ClassId dstClassId{};

    switch (dstPrec)
    {
    case afft::Precision::f32:
      dstClassId = mx::ClassId::single;
      break;
    case afft::Precision::f64:
      dstClassId = mx::ClassId::_double;
      break;
    default:
      throw mx::Exception{"afft:Plan:execute:internalError", "invalid destination precision"};
    }

    const mx::Complexity dstCmplMatlab = (dstCmpl == afft::Complexity::complex)
                                           ? mx::Complexity::complex : mx::Complexity::real;

    const auto shapeRank = desc.getShapeRank();

    auto dstDims = desc.getDstShape<std::size_t>();

    std::reverse(dstDims.data, dstDims.data + shapeRank);

    return makeDstArrayFn(mx::View<std::size_t>{dstDims.data, shapeRank}, dstClassId, dstCmplMatlab);
  };

#ifdef MATLABW_ENABLE_GPU
  if (rhs[1].isGpuArray())
  {
    mx::gpu::Array src{rhs[1]};

    checkSrcArray(src);

    mx::gpu::Array dst = makeDstArray([](auto&& dstDims, auto dstClassId, auto dstCmpl)
    {
      return mx::gpu::makeUninitNumericArray({dstDims}, dstClassId, dstCmpl);
    });

    if (plan->isDestructive())
    {
      mx::gpu::Array srcCopy{src};
      
      plan->executeUnsafe(srcCopy.getData(), dst.getData());
    }
    else
    {
      plan->executeUnsafe(src.getData(), dst.getData());
    }

    lhs[0] = dst.release();
  }
  else
#endif
  {
    mx::ArrayCref src{rhs[1]};

    checkSrcArray(src);

    mx::Array dst = makeDstArray([](auto&& dstDims, auto dstClassId, auto dstCmpl)
    {
      return mx::makeUninitNumericArray({dstDims}, dstClassId, dstCmpl);
    });

    if (plan->isDestructive())
    {
      mx::Array srcCopy{src};
      
      plan->executeUnsafe(srcCopy.getData(), dst.getData());
    }
    else
    {
      plan->executeUnsafe(src.getData(), dst.getData());
    }

    lhs[0] = std::move(dst);
  }
}

/**
 * @brief Get the transform parameters of a plan.
 * @param lhs Left-hand side array of size 1.
 *            * lhs[0] holds the transform parameters as a scalar StructArray.
 * @param rhs Right-hand side array of size 1.
 *            * rhs[0] holds the pointer to the plan as a scalar NumericArray<std::uint64_t>.
 */
void planGetTransformParameters(mx::Span<mx::Array> lhs, mx::View<mx::ArrayCref> rhs)
{
  
}

/**
 * @brief Get the target parameters of a plan.
 * @param lhs Left-hand side array of size 1.
 *            * lhs[0] holds the target parameters as a scalar StructArray.
 * @param rhs Right-hand side array of size 1.
 *            * rhs[0] holds the pointer to the plan as a scalar NumericArray<std::uint64_t>.
 */
void planGetTargetParameters(mx::Span<mx::Array> lhs, mx::View<mx::ArrayCref> rhs)
{
  
}
