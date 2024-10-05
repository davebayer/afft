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

#ifdef MATLABW_ENABLE_GPU
# include <cuda_runtime.h>
#endif

#include "planCache.hpp"
#include "transform.hpp"

using namespace matlabw;

/// @brief Default cpu backend order.
static constexpr std::array cpuBackendOrder{afft::Backend::mkl, afft::Backend::fftw3, afft::Backend::pocketfft};

/// @brief Default gpu backend order.
static constexpr std::array gpuBackendOrder{afft::Backend::cufft, afft::Backend::vkfft};

/**
 * @brief Get the transform precision from the input array.
 * @tparam ArrayT Array type. Must have a getClassId method.
 * @param[in] array Input array to get the precision from.
 * @return Transform precision.
 */
template<typename ArrayT, typename = std::void_t<decltype(std::declval<ArrayT>().getClassId())>>
[[nodiscard]] static inline afft::PrecisionTriad getTransformPrecision(const ArrayT& array)
{
  switch (array.getClassId())
  {
  case mx::ClassId::single:
    return afft::makePrecision<float>();
  default:
    return afft::makePrecision<double>();
  }
}

/**
 * @brief Split the right-hand side arguments into the args and the name-value pair arguments.
 * @param[in] rhs Right-hand side arguments to split.
 * @return Tuple of the args and the named arguments.
 */
[[nodiscard]] static inline std::tuple<mx::View<mx::ArrayCref>, mx::View<mx::ArrayCref>>
splitRhsArgs(const mx::View<mx::ArrayCref> rhs)
{
  for (std::size_t i{}; i < rhs.size(); ++i)
  {
    if (rhs[i].isChar())
    {
      return {rhs.subspan(0, i), rhs.subspan(i)};
    }
  }

  return {rhs, {}};
}

/// @brief Common named arguments parser.
struct CommonNamedArgsParser : private NormalizationParser,
                               private CpuThreadLimitParser,
                               private BackendMaskParser,
                               private SelectStrategyParser
{
  public:
    /// @brief Result of the parser.
    struct Result
    {
      afft::Normalization  normalization{afft::Normalization::none};
      std::uint32_t        cpuThreadLimit{};
      afft::BackendMask    backendMask{afft::BackendMask::all};
      afft::SelectStrategy selectStrategy{afft::SelectStrategy::first};
    };

    /**
     * @brief Parse the common named arguments.
     * @param[in] namedArgs Named arguments to parse.
     */
    Result operator()(mx::View<mx::ArrayCref> namedArgs)
    {
      Result result{};

      for (std::size_t i{}; i < namedArgs.size(); ++i)
      {
        if (const mx::ArrayCref namedArg = namedArgs[i]; namedArg.isChar())
        {
          const std::u16string_view strView{mx::CharArrayCref{namedArg}};

          if (i + 1 >= namedArgs.size())
          {
            throw mx::Exception{"afft:planCreate:invalidArgument", "missing value for named argument"};
          }

          if (strView == u"normalization" || strView == u"norm")
          {
            result.normalization = NormalizationParser::operator()(namedArgs[++i]);
          }
          else if (strView == u"threadLimit")
          {
            result.cpuThreadLimit = CpuThreadLimitParser::operator()(namedArgs[++i]);
          }
          else if (strView == u"backend")
          {
            result.backendMask = BackendMaskParser::operator()(namedArgs[++i]);
          }
          else if (strView == u"selectStrategy" || strView == u"strategy")
          {
            result.selectStrategy = SelectStrategyParser::operator()(namedArgs[++i]);
          }
        }
      }

      return result;
    }
};

/**
 * @brief Perform a 1D forward Fourier transform.
 * @param lhs Left-hand side array of size 1.
 *            * lhs[0] holds the output array.
 * @param rhs Right-hand side array of size 1-3.
 *            * rhs[0] holds the input array. Must be a real or complex floating-point array.
 *            - rhs[1] holds the optional resize parameter as a scalar numeric array.
 *            - rhs[2] holds the optional axis parameter as a scalar numeric array.
 */
void fft(mx::Span<mx::Array> lhs, mx::View<mx::ArrayCref> rhs)
{
  auto [args, namedArgs] = splitRhsArgs(rhs);

  if (args.size() < 1 || args.size() > 3)
  {
    throw mx::Exception{"afft:fft:invalidInputCount", "invalid input argument count"};
  }

  if (lhs.size() > 1)
  {
    throw mx::Exception{"afft:fft:invalidOutputCount", "invalid output argument count"};
  }

  // To be removed when resize parameter is implemented.
  if (args.size() > 1 && !args[1].isEmpty())
  {
    throw mx::Exception{"afft:fft:unimplemented", "resize parameter not yet implemented, therefor must be empty"};
  }

  CommonNamedArgsParser commonNamedArgsParser{};

  const auto commonArgs = commonNamedArgsParser(namedArgs);

  auto checkSrcArray = [&](auto&& srcArray)
  {
    if (!srcArray.isSingle() && !srcArray.isDouble())
    {
      throw mx::Exception{"afft:fft:invalidInputClass", "input array must be floating-point"};
    }

    // Should be removed when real-to-complex transforms are implemented.
    if (!srcArray.isComplex())
    {
      throw mx::Exception{"afft:fft:invalidInputComplexity", "input array must be complex"};
    }

    if (srcArray.getRank() > afft::maxDimCount)
    {
      throw mx::Exception{"afft:fft:invalidInputRank", "input array rank exceeds maximum dimension count"};
    }
  };

  auto makePlan = [&, shapeParser = ShapeParser{}](auto&& srcArray, const auto& targetParams, const auto& backendParams) mutable
  {
    using TargetParamsT = std::decay_t<decltype(targetParams)>;

    static constexpr afft::Target target{TargetParamsT::target};

    const afft::Axis axes[1]{static_cast<afft::Axis>(srcArray.getRank() - 1)};

    afft::dft::Parameters dftParams{};
    dftParams.direction     = afft::Direction::forward;
    dftParams.precision     = getTransformPrecision(srcArray);
    dftParams.shape         = shapeParser(srcArray.getDims());
    dftParams.axes          = axes;
    dftParams.normalization = commonArgs.normalization;
    dftParams.placement     = afft::Placement::outOfPlace;
    dftParams.type          = afft::dft::Type::complexToComplex;

    const afft::Description desc{dftParams, targetParams};

    switch (commonArgs.selectStrategy)
    {
    case afft::SelectStrategy::first:
    {
      afft::FirstSelectParameters selectParams{};
      selectParams.mask  = commonArgs.backendMask;
      selectParams.order = (target == afft::Target::cpu)
        ? afft::View<afft::Backend>{cpuBackendOrder} : afft::View<afft::Backend>{gpuBackendOrder};

      return afft::makePlan(desc, backendParams, selectParams);
    }
    case afft::SelectStrategy::best:
    {
      afft::BestSelectParameters selectParams{};
      selectParams.mask = commonArgs.backendMask;

      return afft::makePlan(desc, backendParams, selectParams);
    }
    default:
      throw mx::Exception{"afft:fft:invalidSelectStrategy", "invalid select strategy"};
    }
  };

#ifdef MATLABW_ENABLE_GPU
  if (args[0].isGpuArray())
  {
    mx::gpu::init();

    mx::gpu::Array src{args[0]};

    if (src.getSize() == 0)
    {
      lhs[0] = mx::gpu::makeNumericArray(0, 0, src.getClassId(), mx::Complexity::complex).release();
      return;
    }

    checkSrcArray(src);
    
    afft::cuda::BackendParameters backendParams{};
    backendParams.allowDestructive = true;

    auto plan = makePlan(src, afft::cuda::Parameters{}, backendParams);

    auto dst = mx::gpu::makeUninitNumericArray(src.getDims(), src.getClassId(), mx::Complexity::complex);

    if (plan->isDestructive())
    {
      mx::gpu::Array tmp{src};

      plan->executeUnsafe(tmp.getData(), dst.getData());
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
    mx::ArrayCref src{args[0]};

    if (src.getSize() == 0)
    {
      lhs[0] = mx::makeNumericArray(0, 0, src.getClassId(), mx::Complexity::complex);
      return;
    }

    checkSrcArray(src);

    afft::cpu::BackendParameters backendParams{};
    backendParams.allowDestructive  = true;
    backendParams.threadLimit       = commonArgs.cpuThreadLimit;
    backendParams.fftw3.plannerFlag = afft::fftw3::PlannerFlag::estimate;

    auto plan = makePlan(src, afft::cpu::Parameters{}, backendParams);

    auto dst = mx::makeUninitNumericArray(src.getDims(), src.getClassId(), mx::Complexity::complex);

    if (plan->isDestructive())
    {
      mx::Array tmp{src};

      plan->executeUnsafe(tmp.getData(), dst.getData());
    }
    else
    {
      plan->executeUnsafe(src.getData(), dst.getData());
    }

    lhs[0] = std::move(dst);
  }
}

/**
 * @brief Perform a 2D forward Fourier transform.
 * @param lhs Left-hand side array of size 1.
 *            * lhs[0] holds the output array.
 * @param rhs Right-hand side array of size 1-3.
 *            * rhs[0] holds the input array. Must be a real or complex floating-point array.
 *            - rhs[1] holds the optional M resize parameter as a scalar numeric array.
 *            - rhs[2] holds the optional N resize parameter as a scalar numeric array.
 */
void fft2(mx::Span<mx::Array> lhs, mx::View<mx::ArrayCref> rhs)
{
  auto [args, namedArgs] = splitRhsArgs(rhs);

  if (args.size() != 1 && args.size() != 3)
  {
    throw mx::Exception{"afft:fft2:invalidInputCount", "invalid input argument count"};
  }

  if (lhs.size() > 1)
  {
    throw mx::Exception{"afft:fft2:invalidOutputCount", "invalid output argument count"};
  }

  // To be removed when resize parameter is implemented.
  if (args.size() != 1)
  {
    throw mx::Exception{"afft:fft2:unimplemented", "resize parameter not yet implemented"};
  }

  CommonNamedArgsParser commonNamedArgsParser{};

  const auto commonArgs = commonNamedArgsParser(namedArgs);

  auto checkSrcArray = [&](auto&& srcArray)
  {
    if (!srcArray.isSingle() && !srcArray.isDouble())
    {
      throw mx::Exception{"afft:fft2:invalidInputClass", "input array must be floating-point"};
    }

    // Should be removed when real-to-complex transforms are implemented.
    if (!srcArray.isComplex())
    {
      throw mx::Exception{"afft:fft2:invalidInputComplexity", "input array must be complex"};
    }

    if (srcArray.getRank() > afft::maxDimCount)
    {
      throw mx::Exception{"afft:fft2:invalidInputRank", "input array rank exceeds maximum dimension count"};
    }
  };

  auto makePlan = [&, shapeParser = ShapeParser{}](auto&& srcArray, const auto& targetParams, const auto& backendParams) mutable
  {
    using TargetParamsT = std::decay_t<decltype(targetParams)>;

    static constexpr afft::Target target{TargetParamsT::target};

    auto shape = shapeParser(srcArray.getDims());

    std::size_t transformRank{};
    afft::Axis axes[2]{};

    if (shape.size() == 1)
    {
      transformRank = 1;
      axes[0] = 0;
    }
    else if (shape.size() > 2)
    {
      transformRank = 2;
      axes[0] = static_cast<afft::Axis>(shape.size() - 2);
      axes[1] = static_cast<afft::Axis>(shape.size() - 1);
    }

    afft::dft::Parameters dftParams{};
    dftParams.direction     = afft::Direction::forward;
    dftParams.precision     = getTransformPrecision(srcArray);
    dftParams.shape         = shape;
    dftParams.axes          = afft::View<afft::Axis>{axes, transformRank};
    dftParams.normalization = commonArgs.normalization;
    dftParams.placement     = afft::Placement::outOfPlace;
    dftParams.type          = afft::dft::Type::complexToComplex;

    const afft::Description desc{dftParams, targetParams};

    switch (commonArgs.selectStrategy)
    {
    case afft::SelectStrategy::first:
    {
      afft::FirstSelectParameters selectParams{};
      selectParams.mask  = commonArgs.backendMask;
      selectParams.order = (target == afft::Target::cpu)
        ? afft::View<afft::Backend>{cpuBackendOrder} : afft::View<afft::Backend>{gpuBackendOrder};

      return afft::makePlan(desc, backendParams, selectParams);
    }
    case afft::SelectStrategy::best:
    {
      afft::BestSelectParameters selectParams{};
      selectParams.mask = commonArgs.backendMask;

      return afft::makePlan(desc, backendParams, selectParams);
    }
    default:
      throw mx::Exception{"afft:fft2:invalidSelectStrategy", "invalid select strategy"};
    }
  };

  #ifdef MATLABW_ENABLE_GPU
  if (args[0].isGpuArray())
  {
    mx::gpu::init();

    mx::gpu::Array src{args[0]};

    if (src.getSize() == 0)
    {
      lhs[0] = mx::gpu::makeNumericArray(0, 0, src.getClassId(), mx::Complexity::complex).release();
      return;
    }

    checkSrcArray(src);
    
    afft::cuda::BackendParameters backendParams{};
    backendParams.allowDestructive = true;

    auto plan = makePlan(src, afft::cuda::Parameters{}, backendParams);

    auto dst = mx::gpu::makeUninitNumericArray(src.getDims(), src.getClassId(), mx::Complexity::complex);

    if (plan->isDestructive())
    {
      mx::gpu::Array tmp{src};

      plan->executeUnsafe(tmp.getData(), dst.getData());
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
    mx::ArrayCref src{args[0]};

    if (src.getSize() == 0)
    {
      lhs[0] = mx::makeNumericArray(0, 0, src.getClassId(), mx::Complexity::complex);
      return;
    }

    checkSrcArray(src);

    afft::cpu::BackendParameters backendParams{};
    backendParams.allowDestructive  = true;
    backendParams.threadLimit       = commonArgs.cpuThreadLimit;
    backendParams.fftw3.plannerFlag = afft::fftw3::PlannerFlag::estimate;

    auto plan = makePlan(src, afft::cpu::Parameters{}, backendParams);

    auto dst = mx::makeUninitNumericArray(src.getDims(), src.getClassId(), mx::Complexity::complex);

    if (plan->isDestructive())
    {
      mx::Array tmp{src};

      plan->executeUnsafe(tmp.getData(), dst.getData());
    }
    else
    {
      plan->executeUnsafe(src.getData(), dst.getData());
    }

    lhs[0] = std::move(dst);
  }
}

/**
 * @brief Perform an N-dimensional forward Fourier transform.
 * @param lhs Left-hand side array of size 1.
 *            * lhs[0] holds the output array.
 * @param rhs Right-hand side array of size 1-3.
 *            * rhs[0] holds the input array. Must be a real or complex floating-point array.
 *            - rhs[1] holds the optional resize parameter as a numeric array of size equal to input rank.
 */
void fftn(mx::Span<mx::Array> lhs, mx::View<mx::ArrayCref> rhs)
{
  auto [args, namedArgs] = splitRhsArgs(rhs);

  if (args.size() < 1 || args.size() > 2)
  {
    throw mx::Exception{"afft:fftn:invalidInputCount", "invalid input argument count"};
  }

  if (lhs.size() > 1)
  {
    throw mx::Exception{"afft:fftn:invalidOutputCount", "invalid output argument count"};
  }

  // To be removed when resize parameter is implemented.
  if (args.size() > 1 && !args[1].isEmpty())
  {
    throw mx::Exception{"afft:fftn:unimplemented", "resize parameter not yet implemented, therefor must be empty"};
  }

  CommonNamedArgsParser commonNamedArgsParser{};

  const auto commonArgs = commonNamedArgsParser(namedArgs);

  auto checkSrcArray = [&](auto&& srcArray)
  {
    if (!srcArray.isSingle() && !srcArray.isDouble())
    {
      throw mx::Exception{"afft:fftn:invalidInputClass", "input array must be floating-point"};
    }

    // Should be removed when real-to-complex transforms are implemented.
    if (!srcArray.isComplex())
    {
      throw mx::Exception{"afft:fftn:invalidInputComplexity", "input array must be complex"};
    }

    if (srcArray.getRank() > afft::maxDimCount)
    {
      throw mx::Exception{"afft:fftn:invalidInputRank", "input array rank exceeds maximum dimension count"};
    }
  };

  auto makePlan = [&, shapeParser = ShapeParser{}](auto&& srcArray, const auto& targetParams, const auto& backendParams) mutable
  {
    using TargetParamsT = std::decay_t<decltype(targetParams)>;

    static constexpr afft::Target target{TargetParamsT::target};

    afft::dft::Parameters dftParams{};
    dftParams.direction     = afft::Direction::forward;
    dftParams.precision     = getTransformPrecision(srcArray);
    dftParams.shape         = shapeParser(srcArray.getDims());
    dftParams.axes          = afft::allAxes;
    dftParams.normalization = commonArgs.normalization;
    dftParams.placement     = afft::Placement::outOfPlace;
    dftParams.type          = afft::dft::Type::complexToComplex;

    const afft::Description desc{dftParams, targetParams};

    switch (commonArgs.selectStrategy)
    {
    case afft::SelectStrategy::first:
    {
      afft::FirstSelectParameters selectParams{};
      selectParams.mask  = commonArgs.backendMask;
      selectParams.order = (target == afft::Target::cpu)
        ? afft::View<afft::Backend>{cpuBackendOrder} : afft::View<afft::Backend>{gpuBackendOrder};

      return afft::makePlan(desc, backendParams, selectParams);
    }
    case afft::SelectStrategy::best:
    {
      afft::BestSelectParameters selectParams{};
      selectParams.mask = commonArgs.backendMask;

      return afft::makePlan(desc, backendParams, selectParams);
    }
    default:
      throw mx::Exception{"afft:fftn:invalidSelectStrategy", "invalid select strategy"};
    }
  };

#ifdef MATLABW_ENABLE_GPU
  if (args[0].isGpuArray())
  {
    mx::gpu::init();

    mx::gpu::Array src{args[0]};

    if (src.getSize() == 0)
    {
      lhs[0] = mx::gpu::makeNumericArray(0, 0, src.getClassId(), mx::Complexity::complex).release();
      return;
    }

    checkSrcArray(src);
    
    afft::cuda::BackendParameters backendParams{};
    backendParams.allowDestructive = true;

    auto plan = makePlan(src, afft::cuda::Parameters{}, backendParams);

    auto dst = mx::gpu::makeUninitNumericArray(src.getDims(), src.getClassId(), mx::Complexity::complex);

    if (plan->isDestructive())
    {
      mx::gpu::Array tmp{src};

      plan->executeUnsafe(tmp.getData(), dst.getData());
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
    mx::ArrayCref src{args[0]};

    if (src.getSize() == 0)
    {
      lhs[0] = mx::makeNumericArray(0, 0, src.getClassId(), mx::Complexity::complex);
      return;
    }

    checkSrcArray(src);

    afft::cpu::BackendParameters backendParams{};
    backendParams.allowDestructive  = true;
    backendParams.threadLimit       = commonArgs.cpuThreadLimit;
    backendParams.fftw3.plannerFlag = afft::fftw3::PlannerFlag::estimate;

    auto plan = makePlan(src, afft::cpu::Parameters{}, backendParams);

    auto dst = mx::makeUninitNumericArray(src.getDims(), src.getClassId(), mx::Complexity::complex);

    if (plan->isDestructive())
    {
      mx::Array tmp{src};

      plan->executeUnsafe(tmp.getData(), dst.getData());
    }
    else
    {
      plan->executeUnsafe(src.getData(), dst.getData());
    }

    lhs[0] = std::move(dst);
  }
}

/**
 * @brief Perform a 1D inverse Fourier transform.
 * @param lhs Left-hand side array of size 1.
 *            * lhs[0] holds the output array.
 * @param rhs Right-hand side array of size 1-3.
 *            * rhs[0] holds the input array. Must be a real or complex floating-point array.
 *            - rhs[1] holds the optional resize parameter as a scalar numeric array.
 *            - rhs[2] holds the optional axis parameter as a scalar numeric array.
 *            _ 'symmetric' or 'nonsymmetric' flag to specify if C2R or C2C should be used.
 */
void ifft(mx::Span<mx::Array> lhs, mx::View<mx::ArrayCref> rhs)
{
  auto [args, namedArgs] = splitRhsArgs(rhs);

  if (args.size() < 1 || args.size() > 3)
  {
    throw mx::Exception{"afft:ifft:invalidInputCount", "invalid input argument count"};
  }

  if (lhs.size() > 1)
  {
    throw mx::Exception{"afft:ifft:invalidOutputCount", "invalid output argument count"};
  }

  // To be removed when resize parameter is implemented.
  if (args.size() > 1 && !args[1].isEmpty())
  {
    throw mx::Exception{"afft:ifft:unimplemented", "resize parameter not yet implemented, therefor must be empty"};
  }

  CommonNamedArgsParser commonNamedArgsParser{};

  const auto commonArgs = commonNamedArgsParser(namedArgs);

  auto checkSrcArray = [&](auto&& srcArray)
  {
    if (!srcArray.isSingle() && !srcArray.isDouble())
    {
      throw mx::Exception{"afft:ifft:invalidInputClass", "input array must be floating-point"};
    }

    // Should be removed when real-to-complex transforms are implemented.
    if (!srcArray.isComplex())
    {
      throw mx::Exception{"afft:ifft:invalidInputComplexity", "input array must be complex"};
    }

    if (srcArray.getRank() > afft::maxDimCount)
    {
      throw mx::Exception{"afft:ifft:invalidInputRank", "input array rank exceeds maximum dimension count"};
    }
  };

  auto makePlan = [&, shapeParser = ShapeParser{}](auto&& srcArray, const auto& targetParams, const auto& backendParams) mutable
  {
    using TargetParamsT = std::decay_t<decltype(targetParams)>;

    static constexpr afft::Target target{TargetParamsT::target};

    const afft::Axis axes[1]{static_cast<afft::Axis>(srcArray.getRank() - 1)};

    afft::dft::Parameters dftParams{};
    dftParams.direction     = afft::Direction::inverse;
    dftParams.precision     = getTransformPrecision(srcArray);
    dftParams.shape         = shapeParser(srcArray.getDims());
    dftParams.axes          = axes;
    dftParams.normalization = commonArgs.normalization;
    dftParams.placement     = afft::Placement::outOfPlace;
    dftParams.type          = afft::dft::Type::complexToComplex;

    const afft::Description desc{dftParams, targetParams};

    switch (commonArgs.selectStrategy)
    {
    case afft::SelectStrategy::first:
    {
      afft::FirstSelectParameters selectParams{};
      selectParams.mask  = commonArgs.backendMask;
      selectParams.order = (target == afft::Target::cpu)
        ? afft::View<afft::Backend>{cpuBackendOrder} : afft::View<afft::Backend>{gpuBackendOrder};

      return afft::makePlan(desc, backendParams, selectParams);
    }
    case afft::SelectStrategy::best:
    {
      afft::BestSelectParameters selectParams{};
      selectParams.mask = commonArgs.backendMask;

      return afft::makePlan(desc, backendParams, selectParams);
    }
    default:
      throw mx::Exception{"afft:ifft:invalidSelectStrategy", "invalid select strategy"};
    }
  };

#ifdef MATLABW_ENABLE_GPU
  if (args[0].isGpuArray())
  {
    mx::gpu::init();

    mx::gpu::Array src{args[0]};

    if (src.getSize() == 0)
    {
      lhs[0] = mx::gpu::makeNumericArray(0, 0, src.getClassId(), mx::Complexity::complex).release();
      return;
    }

    checkSrcArray(src);
    
    afft::cuda::BackendParameters backendParams{};
    backendParams.allowDestructive = true;

    auto plan = makePlan(src, afft::cuda::Parameters{}, backendParams);

    auto dst = mx::gpu::makeUninitNumericArray(src.getDims(), src.getClassId(), mx::Complexity::complex);

    if (plan->isDestructive())
    {
      mx::gpu::Array tmp{src};

      plan->executeUnsafe(tmp.getData(), dst.getData());
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
    mx::ArrayCref src{args[0]};

    if (src.getSize() == 0)
    {
      lhs[0] = mx::makeNumericArray(0, 0, src.getClassId(), mx::Complexity::complex);
      return;
    }

    checkSrcArray(src);

    afft::cpu::BackendParameters backendParams{};
    backendParams.allowDestructive  = true;
    backendParams.threadLimit       = commonArgs.cpuThreadLimit;
    backendParams.fftw3.plannerFlag = afft::fftw3::PlannerFlag::estimate;

    auto plan = makePlan(src, afft::cpu::Parameters{}, backendParams);

    auto dst = mx::makeUninitNumericArray(src.getDims(), src.getClassId(), mx::Complexity::complex);

    if (plan->isDestructive())
    {
      mx::Array tmp{src};

      plan->executeUnsafe(tmp.getData(), dst.getData());
    }
    else
    {
      plan->executeUnsafe(src.getData(), dst.getData());
    }

    lhs[0] = std::move(dst);
  }
}

/**
 * @brief Perform a 2D inverse Fourier transform.
 * @param lhs Left-hand side array of size 1.
 *            * lhs[0] holds the output array.
 * @param rhs Right-hand side array of size 1-3.
 *            * rhs[0] holds the input array. Must be a real or complex floating-point array.
 *            - rhs[1] holds the optional M resize parameter as a scalar numeric array.
 *            - rhs[2] holds the optional N resize parameter as a scalar numeric array.
 *            _ 'symmetric' or 'nonsymmetric' flag to specify if C2R or C2C should be used.
 */
void ifft2(mx::Span<mx::Array> lhs, mx::View<mx::ArrayCref> rhs)
{
  auto [args, namedArgs] = splitRhsArgs(rhs);

  if (args.size() != 1 && args.size() != 3)
  {
    throw mx::Exception{"afft:ifft2:invalidInputCount", "invalid input argument count"};
  }

  if (lhs.size() > 1)
  {
    throw mx::Exception{"afft:ifft2:invalidOutputCount", "invalid output argument count"};
  }

  // To be removed when resize parameter is implemented.
  if (args.size() != 1)
  {
    throw mx::Exception{"afft:ifft2:unimplemented", "resize parameter not yet implemented"};
  }

  CommonNamedArgsParser commonNamedArgsParser{};

  const auto commonArgs = commonNamedArgsParser(namedArgs);

  auto checkSrcArray = [&](auto&& srcArray)
  {
    if (!srcArray.isSingle() && !srcArray.isDouble())
    {
      throw mx::Exception{"afft:ifft2:invalidInputClass", "input array must be floating-point"};
    }

    // Should be removed when real-to-complex transforms are implemented.
    if (!srcArray.isComplex())
    {
      throw mx::Exception{"afft:ifft2:invalidInputComplexity", "input array must be complex"};
    }

    if (srcArray.getRank() > afft::maxDimCount)
    {
      throw mx::Exception{"afft:ifft2:invalidInputRank", "input array rank exceeds maximum dimension count"};
    }
  };

  auto makePlan = [&, shapeParser = ShapeParser{}](auto&& srcArray, const auto& targetParams, const auto& backendParams) mutable
  {
    using TargetParamsT = std::decay_t<decltype(targetParams)>;

    static constexpr afft::Target target{TargetParamsT::target};

    auto shape = shapeParser(srcArray.getDims());

    std::size_t transformRank{};
    afft::Axis axes[2]{};

    if (shape.size() == 1)
    {
      transformRank = 1;
      axes[0] = 0;
    }
    else if (shape.size() > 2)
    {
      transformRank = 2;
      axes[0] = static_cast<afft::Axis>(shape.size() - 2);
      axes[1] = static_cast<afft::Axis>(shape.size() - 1);
    }

    afft::dft::Parameters dftParams{};
    dftParams.direction     = afft::Direction::inverse;
    dftParams.precision     = getTransformPrecision(srcArray);
    dftParams.shape         = shape;
    dftParams.axes          = afft::View<afft::Axis>{axes, transformRank};
    dftParams.normalization = commonArgs.normalization;
    dftParams.placement     = afft::Placement::outOfPlace;
    dftParams.type          = afft::dft::Type::complexToComplex;

    const afft::Description desc{dftParams, targetParams};

    switch (commonArgs.selectStrategy)
    {
    case afft::SelectStrategy::first:
    {
      afft::FirstSelectParameters selectParams{};
      selectParams.mask  = commonArgs.backendMask;
      selectParams.order = (target == afft::Target::cpu)
        ? afft::View<afft::Backend>{cpuBackendOrder} : afft::View<afft::Backend>{gpuBackendOrder};

      return afft::makePlan(desc, backendParams, selectParams);
    }
    case afft::SelectStrategy::best:
    {
      afft::BestSelectParameters selectParams{};
      selectParams.mask = commonArgs.backendMask;

      return afft::makePlan(desc, backendParams, selectParams);
    }
    default:
      throw mx::Exception{"afft:ifft2:invalidSelectStrategy", "invalid select strategy"};
    }
  };

  #ifdef MATLABW_ENABLE_GPU
  if (args[0].isGpuArray())
  {
    mx::gpu::init();

    mx::gpu::Array src{args[0]};

    if (src.getSize() == 0)
    {
      lhs[0] = mx::gpu::makeNumericArray(0, 0, src.getClassId(), mx::Complexity::complex).release();
      return;
    }

    checkSrcArray(src);
    
    afft::cuda::BackendParameters backendParams{};
    backendParams.allowDestructive = true;

    auto plan = makePlan(src, afft::cuda::Parameters{}, backendParams);

    auto dst = mx::gpu::makeUninitNumericArray(src.getDims(), src.getClassId(), mx::Complexity::complex);

    if (plan->isDestructive())
    {
      mx::gpu::Array tmp{src};

      plan->executeUnsafe(tmp.getData(), dst.getData());
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
    mx::ArrayCref src{args[0]};

    if (src.getSize() == 0)
    {
      lhs[0] = mx::makeNumericArray(0, 0, src.getClassId(), mx::Complexity::complex);
      return;
    }

    checkSrcArray(src);

    afft::cpu::BackendParameters backendParams{};
    backendParams.allowDestructive  = true;
    backendParams.threadLimit       = commonArgs.cpuThreadLimit;
    backendParams.fftw3.plannerFlag = afft::fftw3::PlannerFlag::estimate;

    auto plan = makePlan(src, afft::cpu::Parameters{}, backendParams);

    auto dst = mx::makeUninitNumericArray(src.getDims(), src.getClassId(), mx::Complexity::complex);

    if (plan->isDestructive())
    {
      mx::Array tmp{src};

      plan->executeUnsafe(tmp.getData(), dst.getData());
    }
    else
    {
      plan->executeUnsafe(src.getData(), dst.getData());
    }

    lhs[0] = std::move(dst);
  }
}

/**
 * @brief Perform an N-dimensional inverse Fourier transform.
 * @param lhs Left-hand side array of size 1.
 *            * lhs[0] holds the output array.
 * @param rhs Right-hand side array of size 1-3.
 *            * rhs[0] holds the input array. Must be a real or complex floating-point array.
 *            - rhs[1] holds the optional resize parameter as a numeric array of size equal to input rank.
 *            _ 'symmetric' or 'nonsymmetric' flag to specify if C2R or C2C should be used.
 */
void ifftn(mx::Span<mx::Array> lhs, mx::View<mx::ArrayCref> rhs)
{
  auto [args, namedArgs] = splitRhsArgs(rhs);

  if (args.size() < 1 || args.size() > 2)
  {
    throw mx::Exception{"afft:ifftn:invalidInputCount", "invalid input argument count"};
  }

  if (lhs.size() > 1)
  {
    throw mx::Exception{"afft:ifftn:invalidOutputCount", "invalid output argument count"};
  }

  // To be removed when resize parameter is implemented.
  if (args.size() > 1 && !args[1].isEmpty())
  {
    throw mx::Exception{"afft:ifftn:unimplemented", "resize parameter not yet implemented, therefor must be empty"};
  }

  CommonNamedArgsParser commonNamedArgsParser{};

  const auto commonArgs = commonNamedArgsParser(namedArgs);

  auto checkSrcArray = [&](auto&& srcArray)
  {
    if (!srcArray.isSingle() && !srcArray.isDouble())
    {
      throw mx::Exception{"afft:ifftn:invalidInputClass", "input array must be floating-point"};
    }

    if (!srcArray.isComplex())
    {
      throw mx::Exception{"afft:ifftn:invalidInputComplexity", "input array must be complex"};
    }

    if (srcArray.getRank() > afft::maxDimCount)
    {
      throw mx::Exception{"afft:ifftn:invalidInputRank", "input array rank exceeds maximum dimension count"};
    }
  };

  auto makePlan = [&, shapeParser = ShapeParser{}](auto&& srcArray, const auto& targetParams, const auto& backendParams) mutable
  {
    using TargetParamsT = std::decay_t<decltype(targetParams)>;

    static constexpr afft::Target target{TargetParamsT::target};

    afft::dft::Parameters dftParams{};
    dftParams.direction     = afft::Direction::inverse;
    dftParams.precision     = getTransformPrecision(srcArray);
    dftParams.shape         = shapeParser(srcArray.getDims());
    dftParams.axes          = afft::allAxes;
    dftParams.normalization = commonArgs.normalization;
    dftParams.placement     = afft::Placement::outOfPlace;
    dftParams.type          = afft::dft::Type::complexToComplex;

    const afft::Description desc{dftParams, targetParams};

    switch (commonArgs.selectStrategy)
    {
    case afft::SelectStrategy::first:
    {
      afft::FirstSelectParameters selectParams{};
      selectParams.mask  = commonArgs.backendMask;
      selectParams.order = (target == afft::Target::cpu)
        ? afft::View<afft::Backend>{cpuBackendOrder} : afft::View<afft::Backend>{gpuBackendOrder};

      return afft::makePlan(desc, backendParams, selectParams);
    }
    case afft::SelectStrategy::best:
    {
      afft::BestSelectParameters selectParams{};
      selectParams.mask = commonArgs.backendMask;

      return afft::makePlan(desc, backendParams, selectParams);
    }
    default:
      throw mx::Exception{"afft:ifftn:invalidSelectStrategy", "invalid select strategy"};
    }
  };

  #ifdef MATLABW_ENABLE_GPU
  if (args[0].isGpuArray())
  {
    mx::gpu::init();

    mx::gpu::Array src{args[0]};

    if (src.getSize() == 0)
    {
      lhs[0] = mx::gpu::makeNumericArray(0, 0, src.getClassId(), mx::Complexity::complex).release();
      return;
    }

    checkSrcArray(src);

    afft::cuda::BackendParameters backendParams{};
    backendParams.allowDestructive = true;

    auto plan = makePlan(src, afft::cuda::Parameters{}, backendParams);

    auto dst = mx::gpu::makeUninitNumericArray(src.getDims(), src.getClassId(), mx::Complexity::complex);

    if (plan->isDestructive())
    {
      mx::gpu::Array tmp{src};

      plan->executeUnsafe(tmp.getData(), dst.getData());
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
    mx::ArrayCref src{args[0]};

    if (src.getSize() == 0)
    {
      lhs[0] = mx::makeNumericArray(0, 0, src.getClassId(), mx::Complexity::complex);
      return;
    }

    checkSrcArray(src);

    afft::cpu::BackendParameters backendParams{};
    backendParams.allowDestructive  = true;
    backendParams.threadLimit       = commonArgs.cpuThreadLimit;
    backendParams.fftw3.plannerFlag = afft::fftw3::PlannerFlag::estimate;

    auto plan = makePlan(src, afft::cpu::Parameters{}, backendParams);

    auto dst = mx::makeUninitNumericArray(src.getDims(), src.getClassId(), mx::Complexity::complex);

    if (plan->isDestructive())
    {
      mx::Array tmp{src};

      plan->executeUnsafe(tmp.getData(), dst.getData());
    }
    else
    {
      plan->executeUnsafe(src.getData(), dst.getData());
    }

    lhs[0] = std::move(dst);
  }
}

/**
 * @brief Perform a 1D forward Fourier transform on real input.
 * @param lhs Left-hand side array of size 1.
 *            * lhs[0] holds the output array.
 * @param rhs Right-hand side array of size 1-3.
 *            * rhs[0] holds the input array. Must be a real or complex floating-point array.
 *            - rhs[1] holds the optional resize parameter as a scalar numeric array.
 *            - rhs[2] holds the optional axis parameter as a scalar numeric array.
 */
void rfft(mx::Span<mx::Array> lhs, mx::View<mx::ArrayCref> rhs)
{
  throw mx::Exception{"afft:rfft:unimplemented", "not yet implemented"};
}

/**
 * @brief Perform a 2D forward Fourier transform on real input.
 * @param lhs Left-hand side array of size 1.
 *            * lhs[0] holds the output array.
 * @param rhs Right-hand side array of size 1-3.
 *            * rhs[0] holds the input array. Must be a real or complex floating-point array.
 *            - rhs[1] holds the optional M resize parameter as a scalar numeric array.
 *            - rhs[2] holds the optional N resize parameter as a scalar numeric array.
 */
void rfft2(mx::Span<mx::Array> lhs, mx::View<mx::ArrayCref> rhs)
{
  throw mx::Exception{"afft:rfft2:unimplemented", "not yet implemented"};
}

/**
 * @brief Perform an N-dimensional forward Fourier transform on real input.
 * @param lhs Left-hand side array of size 1.
 *            * lhs[0] holds the output array.
 * @param rhs Right-hand side array of size 1-3.
 *            * rhs[0] holds the input array. Must be a real or complex floating-point array.
 *            - rhs[1] holds the optional resize parameter as a numeric array of size equal to input rank.
 */
void rfftn(mx::Span<mx::Array> lhs, mx::View<mx::ArrayCref> rhs)
{
  auto [args, namedArgs] = splitRhsArgs(rhs);

  if (args.size() < 1 || args.size() > 2)
  {
    throw mx::Exception{"afft:fftn:invalidInputCount", "invalid input argument count"};
  }

  if (lhs.size() > 1)
  {
    throw mx::Exception{"afft:rfftn:invalidOutputCount", "invalid output argument count"};
  }

  // To be removed when resize parameter is implemented.
  if (args.size() > 1 && !args[1].isEmpty())
  {
    throw mx::Exception{"afft:rfftn:unimplemented", "resize parameter not yet implemented, therefor must be empty"};
  }

  CommonNamedArgsParser commonNamedArgsParser{};

  const auto commonArgs = commonNamedArgsParser(namedArgs);

  auto checkSrcArray = [&](auto&& srcArray)
  {
    if (!srcArray.isSingle() && !srcArray.isDouble())
    {
      throw mx::Exception{"afft:rfftn:invalidInputClass", "input array must be floating-point"};
    }

    if (srcArray.isComplex())
    {
      throw mx::Exception{"afft:rfftn:invalidInputComplexity", "input array must be real"};
    }

    if (srcArray.getRank() > afft::maxDimCount)
    {
      throw mx::Exception{"afft:rfftn:invalidInputRank", "input array rank exceeds maximum dimension count"};
    }
  };

  auto makePlan = [&, shapeParser = ShapeParser{}](auto&& srcArray, const auto& targetParams, const auto& backendParams) mutable
  {
    using TargetParamsT = std::decay_t<decltype(targetParams)>;

    static constexpr afft::Target target{TargetParamsT::target};

    afft::dft::Parameters dftParams{};
    dftParams.direction     = afft::Direction::forward;
    dftParams.precision     = getTransformPrecision(srcArray);
    dftParams.shape         = shapeParser(srcArray.getDims());
    dftParams.axes          = afft::allAxes;
    dftParams.normalization = commonArgs.normalization;
    dftParams.placement     = afft::Placement::outOfPlace;
    dftParams.type          = afft::dft::Type::realToComplex;

    const afft::Description desc{dftParams, targetParams};

    switch (commonArgs.selectStrategy)
    {
    case afft::SelectStrategy::first:
    {
      afft::FirstSelectParameters selectParams{};
      selectParams.mask  = commonArgs.backendMask;
      selectParams.order = (target == afft::Target::cpu)
        ? afft::View<afft::Backend>{cpuBackendOrder} : afft::View<afft::Backend>{gpuBackendOrder};

      return afft::makePlan(desc, backendParams, selectParams);
    }
    case afft::SelectStrategy::best:
    {
      afft::BestSelectParameters selectParams{};
      selectParams.mask = commonArgs.backendMask;

      return afft::makePlan(desc, backendParams, selectParams);
    }
    default:
      throw mx::Exception{"afft:rfftn:invalidSelectStrategy", "invalid select strategy"};
    }
  };

  auto makeDstArray = [&](auto&& srcArray, auto&& makeDstArrayFn)
  {
    static_assert(std::is_invocable_v<decltype(makeDstArrayFn), mx::View<std::size_t>>);

    afft::detail::MaxDimBuffer<std::size_t> dstDims{};

    std::copy(srcArray.getDims().begin(), srcArray.getDims().end(), dstDims.data);

    dstDims[0] = dstDims[0] / 2 + 1;

    return makeDstArrayFn(mx::View<std::size_t>{dstDims.data, srcArray.getRank()});
  };

#ifdef MATLABW_ENABLE_GPU
  if (args[0].isGpuArray())
  {
    mx::gpu::init();

    mx::gpu::Array src{args[0]};

    if (src.getSize() == 0)
    {
      lhs[0] = mx::gpu::makeNumericArray(0, 0, src.getClassId(), mx::Complexity::complex).release();
      return;
    }

    checkSrcArray(src);

    afft::cuda::BackendParameters backendParams{};
    backendParams.allowDestructive = true;

    auto plan = makePlan(src, afft::cuda::Parameters{}, backendParams);

    auto dst = makeDstArray(src, [&](auto&& dstDims)
    {
      return mx::gpu::makeUninitNumericArray(dstDims, src.getClassId(), mx::Complexity::complex);
    });

    if (plan->isDestructive())
    {
      mx::gpu::Array tmp{src};

      plan->executeUnsafe(tmp.getData(), dst.getData());
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
    mx::ArrayCref src{args[0]};

    if (src.getSize() == 0)
    {
      lhs[0] = mx::makeNumericArray(0, 0, src.getClassId(), mx::Complexity::complex);
      return;
    }

    checkSrcArray(src);

    afft::cpu::BackendParameters backendParams{};
    backendParams.allowDestructive  = true;
    backendParams.threadLimit       = commonArgs.cpuThreadLimit;
    backendParams.fftw3.plannerFlag = afft::fftw3::PlannerFlag::estimate;

    auto plan = makePlan(src, afft::cpu::Parameters{}, backendParams);

    auto dst = makeDstArray(src, [&](auto&& dstDims)
    {
      return mx::makeUninitNumericArray(dstDims, src.getClassId(), mx::Complexity::complex);
    });

    if (plan->isDestructive())
    {
      mx::Array tmp{src};

      plan->executeUnsafe(tmp.getData(), dst.getData());
    }
    else
    {
      plan->executeUnsafe(src.getData(), dst.getData());
    }

    lhs[0] = std::move(dst);
  }
}

/**
 * @brief Perform a 1D inverse Fourier transform on real input.
 * @param lhs Left-hand side array of size 1.
 *            * lhs[0] holds the output array.
 * @param rhs Right-hand side array of size 1-3.
 *            * rhs[0] holds the input array. Must be a real or complex floating-point array.
 *            - rhs[1] holds the output size in transformed axis as a scalar numeric array.
 *            - rhs[2] holds the optional axis parameter as a scalar numeric array.
 */
void irfft(mx::Span<mx::Array> lhs, mx::View<mx::ArrayCref> rhs)
{
  throw mx::Exception{"afft:irfft:unimplemented", "not yet implemented"};
}

/**
 * @brief Perform a 2D inverse Fourier transform on real input.
 * @param lhs Left-hand side array of size 1.
 *            * lhs[0] holds the output array.
 * @param rhs Right-hand side array of size 1-3.
 *            * rhs[0] holds the input array. Must be a real or complex floating-point array.
 *            - rhs[1] holds the output n size in transformed as a scalar numeric array.
 *            - rhs[2] holds the output m size in transformed as a scalar numeric array.
 */
void irfft2(mx::Span<mx::Array> lhs, mx::View<mx::ArrayCref> rhs)
{
  throw mx::Exception{"afft:irfft2:unimplemented", "not yet implemented"};
}

/**
 * @brief Perform an N-dimensional inverse Fourier transform on real input.
 * @param lhs Left-hand side array of size 1.
 *            * lhs[0] holds the output array.
 * @param rhs Right-hand side array of size 1-3.
 *            * rhs[0] holds the input array. Must be a real or complex floating-point array.
 *            * rhs[1] holds the output size in transformed axis as a numeric array of size equal to input rank.
 */
void irfftn(mx::Span<mx::Array> lhs, mx::View<mx::ArrayCref> rhs)
{
  auto [args, namedArgs] = splitRhsArgs(rhs);

  if (args.size() != 2)
  {
    throw mx::Exception{"afft:irfftn:invalidInputCount", "invalid input argument count"};
  }

  if (lhs.size() > 1)
  {
    throw mx::Exception{"afft:irfftn:invalidOutputCount", "invalid output argument count"};
  }

  ShapeParser shapeParser{};

  std::size_t dstRank{};
  std::size_t dstDims[afft::maxDimCount]{};

  const auto shape = shapeParser(rhs[1]);

  dstRank = shape.size();
  std::transform(shape.rbegin(), shape.rend(), dstDims, [](auto&& dim)
  {
    return static_cast<std::size_t>(dim);
  });

  CommonNamedArgsParser commonNamedArgsParser{};

  const auto commonArgs = commonNamedArgsParser(namedArgs);

  auto checkSrcArray = [&](auto&& srcArray)
  {
    if (!srcArray.isSingle() && !srcArray.isDouble())
    {
      throw mx::Exception{"afft:irfftn:invalidInputClass", "input array must be floating-point"};
    }

    if (!srcArray.isComplex())
    {
      throw mx::Exception{"afft:irfftn:invalidInputComplexity", "input array must be complex"};
    }

    if (srcArray.getRank() > afft::maxDimCount)
    {
      throw mx::Exception{"afft:irfftn:invalidInputRank", "input array rank exceeds maximum dimension count"};
    }
  };

  auto makePlan = [&](auto&& srcArray, const auto& targetParams, const auto& backendParams) mutable
  {
    using TargetParamsT = std::decay_t<decltype(targetParams)>;

    static constexpr afft::Target target{TargetParamsT::target};

    afft::dft::Parameters dftParams{};
    dftParams.direction     = afft::Direction::inverse;
    dftParams.precision     = getTransformPrecision(srcArray);
    dftParams.shape         = shape;
    dftParams.axes          = afft::allAxes;
    dftParams.normalization = commonArgs.normalization;
    dftParams.placement     = afft::Placement::outOfPlace;
    dftParams.type          = afft::dft::Type::complexToReal;

    const afft::Description desc{dftParams, targetParams};

    switch (commonArgs.selectStrategy)
    {
    case afft::SelectStrategy::first:
    {
      afft::FirstSelectParameters selectParams{};
      selectParams.mask  = commonArgs.backendMask;
      selectParams.order = (target == afft::Target::cpu)
        ? afft::View<afft::Backend>{cpuBackendOrder} : afft::View<afft::Backend>{gpuBackendOrder};

      return afft::makePlan(desc, backendParams, selectParams);
    }
    case afft::SelectStrategy::best:
    {
      afft::BestSelectParameters selectParams{};
      selectParams.mask = commonArgs.backendMask;

      return afft::makePlan(desc, backendParams, selectParams);
    }
    default:
      throw mx::Exception{"afft:irfftn:invalidSelectStrategy", "invalid select strategy"};
    }
  };

  #ifdef MATLABW_ENABLE_GPU
  if (args[0].isGpuArray())
  {
    mx::gpu::init();

    mx::gpu::Array src{args[0]};

    if (src.getSize() == 0)
    {
      lhs[0] = mx::gpu::makeNumericArray(0, 0, src.getClassId()).release();
      return;
    }

    checkSrcArray(src);

    afft::cuda::BackendParameters backendParams{};
    backendParams.allowDestructive = true;

    auto plan = makePlan(src, afft::cuda::Parameters{}, backendParams);

    auto dst = mx::gpu::makeUninitNumericArray({dstDims, dstRank}, src.getClassId());

    if (plan->isDestructive())
    {
      mx::gpu::Array tmp{src};

      plan->executeUnsafe(tmp.getData(), dst.getData());
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
    mx::ArrayCref src{args[0]};

    if (src.getSize() == 0)
    {
      lhs[0] = mx::makeNumericArray(0, 0, src.getClassId());
      return;
    }

    checkSrcArray(src);

    afft::cpu::BackendParameters backendParams{};
    backendParams.allowDestructive  = true;
    backendParams.threadLimit       = commonArgs.cpuThreadLimit;
    backendParams.fftw3.plannerFlag = afft::fftw3::PlannerFlag::estimate;

    auto plan = makePlan(src, afft::cpu::Parameters{}, backendParams);

    auto dst = mx::makeUninitNumericArray({dstDims, dstRank}, src.getClassId());

    if (plan->isDestructive())
    {
      mx::Array tmp{src};

      plan->executeUnsafe(tmp.getData(), dst.getData());
    }
    else
    {
      plan->executeUnsafe(src.getData(), dst.getData());
    }

    lhs[0] = std::move(dst);
  }
}

/**
 * @brief Perform a 1D discrete cosine transform.
 * @param lhs Left-hand side array of size 1.
 *            * lhs[0] holds the output array.
 * @param rhs Right-hand side array of size 1.
 *            * rhs[0] holds the input array. Must be a real floating-point array.
 *            - rhs[1] holds the optional resize parameter as a scalar numeric array.
 *            - rhs[2] holds the optional axis parameter as a scalar numeric array.
 *            _ 'type' + value specifies the dct type.
 */
void dct(matlabw::mx::Span<matlabw::mx::Array> lhs, matlabw::mx::View<matlabw::mx::ArrayCref> rhs)
{
  throw mx::Exception{"afft:dct:unimplemented", "not yet implemented"};
}

/**
 * @brief Perform a 2D discrete cosine transform.
 * @param lhs Left-hand side array of size 1.
 *            * lhs[0] holds the output array.
 * @param rhs Right-hand side array of size 1.
 *            * rhs[0] holds the input array. Must be a real floating-point array.
 *            - rhs[1] holds the optional M resize parameter as a scalar numeric array.
 *            - rhs[2] holds the optional N resize parameter as a scalar numeric array.
 *            _ 'type' + value specifies the dct type.
 */
void dct2(matlabw::mx::Span<matlabw::mx::Array> lhs, matlabw::mx::View<matlabw::mx::ArrayCref> rhs)
{
  throw mx::Exception{"afft:dct2:unimplemented", "not yet implemented"};
}

class DctTypeConverter
{
  public:
    /**
     * @brief Convert the dct type from the named arguments.
     * @param[in] namedArgs Named arguments to convert.
     * @param[in] errorId Error identifier to throw.
     * @return Dtt type.
     */
    [[nodiscard]] afft::View<afft::dtt::Type> operator()(const mx::View<mx::ArrayCref> namedArgs, std::size_t transformRank, const char* errorId)
    {
      for (std::size_t i{}; i < namedArgs.size(); ++i)
      {
        if (namedArgs[i].isChar())
        {
          std::u16string_view argName{mx::CharArrayCref{namedArgs[i]}};

          if (argName == u"type" || argName == u"Type")
          {
            if (i + 1 < namedArgs.size())
            {
              return convertType(namedArgs[i + 1], transformRank);
            }
            else
            {
              throw mx::Exception{errorId, "missing value for 'type' named argument"};
            }
          }
        }
      }

      return {mType, 1};
    }
  private:
    [[nodiscard]] afft::View<afft::dtt::Type> convertType(mx::ArrayCref typeArray, std::size_t transformRank)
    {
      auto cvt = [](std::uint64_t value)
      {
        switch (value)
        {
        case 1:
          return afft::dtt::Type::dct1;
        case 2:
          return afft::dtt::Type::dct2;
        case 3:
          return afft::dtt::Type::dct3;
        case 4:
          return afft::dtt::Type::dct4;
        default:
          throw mx::Exception{"afft:dctn:invalidDctType", "invalid dct type"};
        }
      };

      std::size_t typeCount = typeArray.getSize();

      if (typeCount != 1 && typeCount != transformRank)
      {
        throw mx::Exception{"afft:dctn:invalidDctTypeCount", "invalid dct type count"};
      }

      mx::visit(typeArray, [&](auto&& typedTypeArray) -> void
      {
        using TypedArrayT = std::decay_t<decltype(typedTypeArray)>;

        if constexpr (mx::isRealNumeric<typename TypedArrayT::value_type>)
        {
          for (std::size_t i{}; i < typeCount; ++i)
          {
            if (typedTypeArray[i] < 1 || typedTypeArray[i] > 4)
            {
              throw mx::Exception{"afft:dctn:invalidDctType", "invalid dct type"};
            }

            mType[i] = cvt(static_cast<std::uint64_t>(typedTypeArray[i]));
          }
        }
        else
        {
          throw mx::Exception{"afft:dctn:invalidDctType", "invalid dct type"};
        }
      });

      return {mType, typeCount};
    }

    afft::dtt::Type mType[afft::maxDimCount]{afft::dtt::Type::dct}; ///< Dct type.
};

/**
 * @brief Perform an N-dimensional discrete cosine transform.
 * @param lhs Left-hand side array of size 1.
 *            * lhs[0] holds the output array.
 * @param rhs Right-hand side array of size 1.
 *            * rhs[0] holds the input array. Must be a real floating-point array.
 *            - rhs[1] holds the optional resize parameter as a numeric array of size equal to input rank.
 *            _ 'type' + value specifies the dct type.
 */
void dctn(matlabw::mx::Span<matlabw::mx::Array> lhs, matlabw::mx::View<matlabw::mx::ArrayCref> rhs)
{
  auto [args, namedArgs] = splitRhsArgs(rhs);

  if (args.size() < 1 || args.size() > 2)
  {
    throw mx::Exception{"afft:dctn:invalidInputCount", "invalid input argument count"};
  }

  if (lhs.size() > 1)
  {
    throw mx::Exception{"afft:dctn:invalidOutputCount", "invalid output argument count"};
  }

  // To be removed when resize parameter is implemented.
  if (args.size() != 1)
  {
    throw mx::Exception{"afft:dctn:unimplemented", "resize parameter not yet implemented"};
  }

  auto checkSrcArray = [&](auto&& srcArray)
  {
    if (!srcArray.isSingle() && !srcArray.isDouble())
    {
      throw mx::Exception{"afft:dctn:invalidInputClass", "input array must be floating-point"};
    }

    if (srcArray.isComplex())
    {
      throw mx::Exception{"afft:dctn:invalidInputComplexity", "input array must be real"};
    }

    if (srcArray.getRank() > afft::maxDimCount)
    {
      throw mx::Exception{"afft:dctn:invalidInputRank", "input array rank exceeds maximum dimension count"};
    }
  };

  auto makeDttParams = [&,
                        shapeParser      = ShapeParser{},
                        dctTypeConverter = DctTypeConverter{}](auto&& srcArray) mutable
  {
    afft::dtt::Parameters dftParams{};
    dftParams.direction     = afft::Direction::forward;
    dftParams.precision     = getTransformPrecision(srcArray);
    dftParams.shape         = shapeParser(srcArray.getDims());
    dftParams.axes          = afft::allAxes;
    dftParams.normalization = afft::Normalization::none;
    dftParams.placement     = afft::Placement::outOfPlace;
    dftParams.types         = dctTypeConverter(namedArgs, dftParams.shape.size(), "afft:dctn:invalidDctType");

    return dftParams;
  };

#ifdef MATLABW_ENABLE_GPU
  if (rhs[0].isGpuArray())
  {
    mx::gpu::init();

    mx::gpu::Array src{rhs[0]};

    if (src.getSize() == 0)
    {
      lhs[0] = mx::gpu::makeNumericArray(0, 0, src.getClassId()).release();
      return;
    }

    checkSrcArray(src);

    const afft::Description desc{makeDttParams(src), afft::cuda::Parameters{}};

    afft::cuda::BackendParameters backendParams{};
    backendParams.allowDestructive = true;

    afft::FirstSelectParameters selectParams{};
    selectParams.order = gpuBackendOrder;

    auto plan = afft::makePlan(desc, backendParams, selectParams);

    auto dst = mx::gpu::makeUninitNumericArray(src.getDims(), src.getClassId());

    if (plan->isDestructive())
    {
      mx::gpu::Array tmp{src};

      plan->executeUnsafe(tmp.getData(), dst.getData());
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
    mx::ArrayCref src{rhs[0]};

    if (src.getSize() == 0)
    {
      lhs[0] = mx::makeNumericArray(0, 0, src.getClassId());
      return;
    }

    checkSrcArray(src);

    const afft::Description desc{makeDttParams(src), afft::cpu::Parameters{}};

    afft::cpu::BackendParameters backendParams{};
    backendParams.allowDestructive  = true;
    backendParams.threadLimit       = 4;
    backendParams.fftw3.plannerFlag = afft::fftw3::PlannerFlag::estimate;

    afft::FirstSelectParameters selectParams{};
    selectParams.order = cpuBackendOrder;

    auto plan = afft::makePlan(desc, backendParams, selectParams);

    auto dst = mx::makeUninitNumericArray(src.getDims(), src.getClassId());

    if (plan->isDestructive())
    {
      mx::Array tmp{src};

      plan->executeUnsafe(tmp.getData(), dst.getData());
    }
    else
    {
      plan->executeUnsafe(src.getData(), dst.getData());
    }

    lhs[0] = std::move(dst);
  }
}

/**
 * @brief Perform a 1D inverse discrete cosine transform.
 * @param lhs Left-hand side array of size 1.
 *            * lhs[0] holds the output array.
 * @param rhs Right-hand side array of size 1.
 *            * rhs[0] holds the input array. Must be a real floating-point array.
 *            - rhs[1] holds the output size in transformed axis as a scalar numeric array.
 *            - rhs[2] holds the optional axis parameter as a scalar numeric array.
 *            _ 'type' + value specifies the dct type.
 */
void idct(matlabw::mx::Span<matlabw::mx::Array> lhs, matlabw::mx::View<matlabw::mx::ArrayCref> rhs)
{
  throw mx::Exception{"afft:idct:unimplemented", "not yet implemented"};
}

/**
 * @brief Perform a 2D inverse discrete cosine transform.
 * @param lhs Left-hand side array of size 1.
 *            * lhs[0] holds the output array.
 * @param rhs Right-hand side array of size 1.
 *            * rhs[0] holds the input array. Must be a real floating-point array.
 *            - rhs[1] holds the output n size in transformed as a scalar numeric array.
 *            - rhs[2] holds the output m size in transformed as a scalar numeric array.
 *            _ 'type' + value specifies the dct type.
 */
void idct2(matlabw::mx::Span<matlabw::mx::Array> lhs, matlabw::mx::View<matlabw::mx::ArrayCref> rhs)
{
  throw mx::Exception{"afft:idct2:unimplemented", "not yet implemented"};
}

/**
 * @brief Perform an N-dimensional inverse discrete cosine transform.
 * @param lhs Left-hand side array of size 1.
 *            * lhs[0] holds the output array.
 * @param rhs Right-hand side array of size 1.
 *            * rhs[0] holds the input array. Must be a real floating-point array.
 *            - rhs[1] holds the output size in transformed axis as a numeric array of size equal to input rank.
 *            _ 'type' + value specifies the dct type.
 */
void idctn(matlabw::mx::Span<matlabw::mx::Array> lhs, matlabw::mx::View<matlabw::mx::ArrayCref> rhs)
{
  throw mx::Exception{"afft:idctn:unimplemented", "not yet implemented"};
}

/**
 * @brief Perform a 1D discrete sine transform.
 * @param lhs Left-hand side array of size 1.
 *            * lhs[0] holds the output array.
 * @param rhs Right-hand side array of size 1.
 *            * rhs[0] holds the input array. Must be a real floating-point array.
 *            - rhs[1] holds the output size in transformed axis as a scalar numeric array.
 *            - rhs[2] holds the optional axis parameter as a scalar numeric array.
 *            _ 'type' + value specifies the dst type.
 */
void dst(matlabw::mx::Span<matlabw::mx::Array> lhs, matlabw::mx::View<matlabw::mx::ArrayCref> rhs)
{
  throw mx::Exception{"afft:dst:unimplemented", "not yet implemented"};
}

/**
 * @brief Perform a 2D discrete sine transform.
 * @param lhs Left-hand side array of size 1.
 *            * lhs[0] holds the output array.
 * @param rhs Right-hand side array of size 1.
 *            * rhs[0] holds the input array. Must be a real floating-point array.
 *            - rhs[1] holds the output n size in transformed as a scalar numeric array.
 *            - rhs[2] holds the output m size in transformed as a scalar numeric array.
 *            _ 'type' + value specifies the dst type.
 */
void dst2(matlabw::mx::Span<matlabw::mx::Array> lhs, matlabw::mx::View<matlabw::mx::ArrayCref> rhs)
{
  throw mx::Exception{"afft:dst2:unimplemented", "not yet implemented"};
}

class DstTypeConverter
{
  public:
    /**
     * @brief Convert the dst type from the named arguments.
     * @param[in] namedArgs Named arguments to convert.
     * @param[in] errorId Error identifier to throw.
     * @return Dtt type.
     */
    [[nodiscard]] afft::View<afft::dtt::Type> operator()(const mx::View<mx::ArrayCref> namedArgs, std::size_t transformRank, const char* errorId)
    {
      for (std::size_t i{}; i < namedArgs.size(); ++i)
      {
        if (namedArgs[i].isChar())
        {
          std::u16string_view argName{mx::CharArrayCref{namedArgs[i]}};

          if (argName == u"type" || argName == u"Type")
          {
            if (i + 1 < namedArgs.size())
            {
              return convertType(namedArgs[i + 1], transformRank);
            }
            else
            {
              throw mx::Exception{errorId, "missing value for 'type' named argument"};
            }
          }
        }
      }

      return {mType, 1};
    }
  private:
    [[nodiscard]] afft::View<afft::dtt::Type> convertType(mx::ArrayCref typeArray, std::size_t transformRank)
    {
      auto cvt = [](std::uint64_t value)
      {
        switch (value)
        {
        case 1:
          return afft::dtt::Type::dst1;
        case 2:
          return afft::dtt::Type::dst2;
        case 3:
          return afft::dtt::Type::dst3;
        case 4:
          return afft::dtt::Type::dst4;
        default:
          throw mx::Exception{"afft:dstn:invalidDstType", "invalid dst type"};
        }
      };

      std::size_t typeCount = typeArray.getSize();

      if (typeCount != 1 && typeCount != transformRank)
      {
        throw mx::Exception{"afft:dstn:invalidDstTypeCount", "invalid dst type count"};
      }

      mx::visit(typeArray, [&](auto&& typedTypeArray) -> void
      {
        using TypedArrayT = std::decay_t<decltype(typedTypeArray)>;

        if constexpr (mx::isRealNumeric<typename TypedArrayT::value_type>)
        {
          for (std::size_t i{}; i < typeCount; ++i)
          {
            if (typedTypeArray[i] < 1 || typedTypeArray[i] > 4)
            {
              throw mx::Exception{"afft:dstn:invalidDstType", "invalid dst type"};
            }

            mType[i] = cvt(static_cast<std::uint64_t>(typedTypeArray[i]));
          }
        }
        else
        {
          throw mx::Exception{"afft:dstn:invalidDstType", "invalid dst type"};
        }
      });

      return {mType, typeCount};
    }

    afft::dtt::Type mType[afft::maxDimCount]{afft::dtt::Type::dst}; ///< Dst type.
};

/**
 * @brief Perform an N-dimensional discrete sine transform.
 * @param lhs Left-hand side array of size 1.
 *            * lhs[0] holds the output array.
 * @param rhs Right-hand side array of size 1.
 *            * rhs[0] holds the input array. Must be a real floating-point array.
 *            - rhs[1] holds the optional resize parameter as a numeric array of size equal to input rank.
 *            _ 'type' + value specifies the dst type.
 */
void dstn(matlabw::mx::Span<matlabw::mx::Array> lhs, matlabw::mx::View<matlabw::mx::ArrayCref> rhs)
{
  auto [args, namedArgs] = splitRhsArgs(rhs);

  if (args.size() < 1 || args.size() > 2)
  {
    throw mx::Exception{"afft:dstn:invalidInputCount", "invalid input argument count"};
  }

  if (lhs.size() > 1)
  {
    throw mx::Exception{"afft:dstn:invalidOutputCount", "invalid output argument count"};
  }

  // To be removed when resize parameter is implemented.
  if (args.size() != 1)
  {
    throw mx::Exception{"afft:dstn:unimplemented", "resize parameter not yet implemented"};
  }

  auto checkSrcArray = [&](auto&& srcArray)
  {
    if (!srcArray.isSingle() && !srcArray.isDouble())
    {
      throw mx::Exception{"afft:dstn:invalidInputClass", "input array must be floating-point"};
    }

    if (srcArray.isComplex())
    {
      throw mx::Exception{"afft:dstn:invalidInputComplexity", "input array must be real"};
    }

    if (srcArray.getRank() > afft::maxDimCount)
    {
      throw mx::Exception{"afft:dstn:invalidInputRank", "input array rank exceeds maximum dimension count"};
    }
  };

  auto makeDttParams = [&,
                        shapeParser      = ShapeParser{},
                        dstTypeConverter = DstTypeConverter{}](auto&& srcArray) mutable
  {
    afft::dtt::Parameters dftParams{};
    dftParams.direction     = afft::Direction::forward;
    dftParams.precision     = getTransformPrecision(srcArray);
    dftParams.shape         = shapeParser(srcArray.getDims());
    dftParams.axes          = afft::allAxes;
    dftParams.normalization = afft::Normalization::none;
    dftParams.placement     = afft::Placement::outOfPlace;
    dftParams.types         = dstTypeConverter(namedArgs, dftParams.shape.size(), "afft:dstn:invalidDstType");

    return dftParams;
  };

#ifdef MATLABW_ENABLE_GPU
  if (rhs[0].isGpuArray())
  {
    mx::gpu::init();

    mx::gpu::Array src{rhs[0]};

    if (src.getSize() == 0)
    {
      lhs[0] = mx::gpu::makeNumericArray(0, 0, src.getClassId()).release();
      return;
    }

    checkSrcArray(src);

    const afft::Description desc{makeDttParams(src), afft::cuda::Parameters{}};

    afft::cuda::BackendParameters backendParams{};
    backendParams.allowDestructive = true;

    afft::FirstSelectParameters selectParams{};
    selectParams.order = gpuBackendOrder;

    auto plan = afft::makePlan(desc, backendParams, selectParams);

    auto dst = mx::gpu::makeUninitNumericArray(src.getDims(), src.getClassId());

    if (plan->isDestructive())
    {
      mx::gpu::Array tmp{src};

      plan->executeUnsafe(tmp.getData(), dst.getData());
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
    mx::ArrayCref src{rhs[0]};

    if (src.getSize() == 0)
    {
      lhs[0] = mx::makeNumericArray(0, 0, src.getClassId());
      return;
    }

    checkSrcArray(src);

    const afft::Description desc{makeDttParams(src), afft::cpu::Parameters{}};

    afft::cpu::BackendParameters backendParams{};
    backendParams.allowDestructive  = true;
    backendParams.threadLimit       = 4;
    backendParams.fftw3.plannerFlag = afft::fftw3::PlannerFlag::estimate;

    afft::FirstSelectParameters selectParams{};
    selectParams.order = cpuBackendOrder;

    auto plan = afft::makePlan(desc, backendParams, selectParams);

    auto dst = mx::makeUninitNumericArray(src.getDims(), src.getClassId());

    if (plan->isDestructive())
    {
      mx::Array tmp{src};

      plan->executeUnsafe(tmp.getData(), dst.getData());
    }
    else
    {
      plan->executeUnsafe(src.getData(), dst.getData());
    }

    lhs[0] = std::move(dst);
  }
}
