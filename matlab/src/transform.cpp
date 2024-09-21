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

/// @brief Shape converter.
class ShapeConverter
{
  public:
    /**
     * @brief Convert the shape to the afft shape.
     * @param[in] dims Dimensions to convert.
     * @param[in] errorId Error identifier to throw.
     * @return Afft shape.
     */
    [[nodiscard]] afft::View<afft::Size> operator()(const mx::View<std::size_t> dims, const char* errorId)
    {
      if (dims.size() > afft::maxDimCount)
      {
        throw mx::Exception{errorId, "input array rank exceeds maximum dimension count"};
      }

      std::transform(dims.rbegin(),
                     dims.rend(),
                     mShape,
                     [](const auto dim) { return static_cast<afft::Size>(dim); });

      return {mShape, dims.size()};
    }
  private:
    afft::Size mShape[afft::maxDimCount]{}; ///< Storage for converted shape.
};

/**
 * @brief Check if the shape rank is within the maximum dimension count.
 * @param[in] shapeRank Shape rank to check.
 * @param[in] errorId Error identifier to throw.
 */
static constexpr void checkShapeRank(const std::size_t shapeRank, const char* errorId)
{
  if (shapeRank >= afft::maxDimCount)
  {
    throw mx::Exception{errorId, "input array rank exceeds maximum dimension count"};
  }
}

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
  throw mx::Exception("afft:fft:unimplemented", "not yet implemented");
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
  throw mx::Exception("afft:fft2:unimplemented", "not yet implemented");
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
  if (rhs.size() < 1 || rhs.size() > 2)
  {
    throw mx::Exception{"afft:fftn:invalidInputCount", "invalid input argument count"};
  }

  if (lhs.size() > 1)
  {
    throw mx::Exception{"afft:fftn:invalidOutputCount", "invalid output argument count"};
  }

  // To be removed when resize parameter is implemented.
  if (rhs.size() != 1)
  {
    throw mx::Exception{"afft:fftn:unimplemented", "resize parameter not yet implemented"};
  }

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

  auto makeDftParams = [&, shapeConverter = ShapeConverter{}](auto&& srcArray) mutable
  {
    afft::dft::Parameters dftParams{};
    dftParams.direction     = afft::Direction::forward;
    dftParams.precision     = getTransformPrecision(srcArray);
    dftParams.shape         = shapeConverter(srcArray.getDims(), "afft:fftn:invalidInputDims");
    dftParams.axes          = afft::allAxes;
    dftParams.normalization = afft::Normalization::none;
    dftParams.placement     = afft::Placement::outOfPlace;
    dftParams.type          = afft::dft::Type::complexToComplex;

    return dftParams;
  };

#ifdef MATLABW_ENABLE_GPU
  if (rhs[0].isGpuArray())
  {
    mx::gpu::init();

    mx::gpu::Array src{rhs[0]};

    if (src.getSize() == 0)
    {
      lhs[0] = mx::gpu::makeNumericArray(0, 0, src.getClassId(), mx::Complexity::complex).release();
      return;
    }

    checkSrcArray(src);

    const afft::Description desc{makeDftParams(src), afft::cuda::Parameters{}};

    afft::cuda::BackendParameters backendParams{};
    backendParams.allowDestructive = true;

    afft::FirstSelectParameters selectParams{};
    selectParams.order = gpuBackendOrder;

    auto plan = afft::makePlan(desc, backendParams, selectParams);

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
    mx::ArrayCref src{rhs[0]};

    if (src.getSize() == 0)
    {
      lhs[0] = mx::makeNumericArray(0, 0, src.getClassId(), mx::Complexity::complex);
      return;
    }

    checkSrcArray(src);

    const afft::Description desc{makeDftParams(src), afft::cpu::Parameters{}};

    afft::cpu::BackendParameters backendParams{};
    backendParams.allowDestructive  = true;
    backendParams.threadLimit       = 4;
    backendParams.fftw3.plannerFlag = afft::fftw3::PlannerFlag::estimate;

    afft::FirstSelectParameters selectParams{};
    selectParams.order = cpuBackendOrder;

    auto plan = afft::makePlan(desc, backendParams, selectParams);

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
  throw mx::Exception("afft:ifft:unimplemented", "not yet implemented");
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
  throw mx::Exception("afft:ifft2:unimplemented", "not yet implemented");
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
  if (rhs.size() < 1 || rhs.size() > 2)
  {
    throw mx::Exception{"afft:ifftn:invalidInputCount", "invalid input argument count"};
  }

  if (lhs.size() > 1)
  {
    throw mx::Exception{"afft:ifftn:invalidOutputCount", "invalid output argument count"};
  }

  // To be removed when resize parameter is implemented.
  if (rhs.size() != 1)
  {
    throw mx::Exception{"afft:ifftn:unimplemented", "resize parameter not yet implemented"};
  }

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

  auto makeDftParams = [&, shapeConverter = ShapeConverter{}](auto&& srcArray) mutable
  {
    afft::dft::Parameters dftParams{};
    dftParams.direction     = afft::Direction::inverse;
    dftParams.precision     = getTransformPrecision(srcArray);
    dftParams.shape         = shapeConverter(srcArray.getDims(), "afft:ifftn:invalidInputDims");
    dftParams.axes          = afft::allAxes;
    dftParams.normalization = afft::Normalization::none;
    dftParams.placement     = afft::Placement::outOfPlace;
    dftParams.type          = afft::dft::Type::complexToComplex;

    return dftParams;
  };

  #ifdef MATLABW_ENABLE_GPU
  if (rhs[0].isGpuArray())
  {
    mx::gpu::init();

    mx::gpu::Array src{rhs[0]};

    if (src.getSize() == 0)
    {
      lhs[0] = mx::gpu::makeNumericArray(0, 0, src.getClassId(), mx::Complexity::complex).release();
      return;
    }

    checkSrcArray(src);

    const afft::Description desc{makeDftParams(src), afft::cuda::Parameters{}};

    afft::cuda::BackendParameters backendParams{};
    backendParams.allowDestructive = true;

    afft::FirstSelectParameters selectParams{};
    selectParams.order = gpuBackendOrder;

    auto plan = afft::makePlan(desc, backendParams, selectParams);

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
    mx::ArrayCref src{rhs[0]};

    if (src.getSize() == 0)
    {
      lhs[0] = mx::makeNumericArray(0, 0, src.getClassId(), mx::Complexity::complex);
      return;
    }

    checkSrcArray(src);

    const afft::Description desc{makeDftParams(src), afft::cpu::Parameters{}};

    afft::cpu::BackendParameters backendParams{};
    backendParams.allowDestructive  = true;
    backendParams.threadLimit       = 4;
    backendParams.fftw3.plannerFlag = afft::fftw3::PlannerFlag::estimate;

    afft::FirstSelectParameters selectParams{};
    selectParams.order = cpuBackendOrder;

    auto plan = afft::makePlan(desc, backendParams, selectParams);

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
  if (rhs.size() < 1 || rhs.size() > 2)
  {
    throw mx::Exception{"afft:fftn:invalidInputCount", "invalid input argument count"};
  }

  if (lhs.size() > 1)
  {
    throw mx::Exception{"afft:rfftn:invalidOutputCount", "invalid output argument count"};
  }

  // To be removed when resize parameter is implemented.
  if (rhs.size() != 1)
  {
    throw mx::Exception{"afft:rfftn:unimplemented", "resize parameter not yet implemented"};
  }

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

  auto makeDftParams = [&, shapeConverter = ShapeConverter{}](auto&& srcArray) mutable
  {
    afft::dft::Parameters dftParams{};
    dftParams.direction     = afft::Direction::forward;
    dftParams.precision     = getTransformPrecision(srcArray);
    dftParams.shape         = shapeConverter(srcArray.getDims(), "afft:rfftn:invalidInputDims");
    dftParams.axes          = afft::allAxes;
    dftParams.normalization = afft::Normalization::none;
    dftParams.placement     = afft::Placement::outOfPlace;
    dftParams.type          = afft::dft::Type::realToComplex;

    return dftParams;
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
  if (rhs[0].isGpuArray())
  {
    mx::gpu::init();

    mx::gpu::Array src{rhs[0]};

    if (src.getSize() == 0)
    {
      lhs[0] = mx::gpu::makeNumericArray(0, 0, src.getClassId(), mx::Complexity::complex).release();
      return;
    }

    checkSrcArray(src);

    const afft::Description desc{makeDftParams(src), afft::cuda::Parameters{}};

    afft::cuda::BackendParameters backendParams{};
    backendParams.allowDestructive = true;

    afft::FirstSelectParameters selectParams{};
    selectParams.order = gpuBackendOrder;

    auto plan = afft::makePlan(desc, backendParams, selectParams);

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
    mx::ArrayCref src{rhs[0]};

    if (src.getSize() == 0)
    {
      lhs[0] = mx::makeNumericArray(0, 0, src.getClassId(), mx::Complexity::complex);
      return;
    }

    checkSrcArray(src);

    const afft::Description desc{makeDftParams(src), afft::cpu::Parameters{}};

    afft::cpu::BackendParameters backendParams{};
    backendParams.allowDestructive  = true;
    backendParams.threadLimit       = 4;
    backendParams.fftw3.plannerFlag = afft::fftw3::PlannerFlag::estimate;

    afft::FirstSelectParameters selectParams{};
    selectParams.order = cpuBackendOrder;

    auto plan = afft::makePlan(desc, backendParams, selectParams);

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
 *            - rhs[1] holds the output size in transformed axis as a numeric array of size equal to input rank.
 */
void irfftn(mx::Span<mx::Array> lhs, mx::View<mx::ArrayCref> rhs)
{
  throw mx::Exception{"afft:irfftn:unimplemented", "not yet implemented"};
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
                        shapeConverter   = ShapeConverter{},
                        dctTypeConverter = DctTypeConverter{}](auto&& srcArray) mutable
  {
    afft::dtt::Parameters dftParams{};
    dftParams.direction     = afft::Direction::forward;
    dftParams.precision     = getTransformPrecision(srcArray);
    dftParams.shape         = shapeConverter(srcArray.getDims(), "afft:dctn:invalidInputDims");
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
                        shapeConverter   = ShapeConverter{},
                        dstTypeConverter = DstTypeConverter{}](auto&& srcArray) mutable
  {
    afft::dtt::Parameters dftParams{};
    dftParams.direction     = afft::Direction::forward;
    dftParams.precision     = getTransformPrecision(srcArray);
    dftParams.shape         = shapeConverter(srcArray.getDims(), "afft:dstn:invalidInputDims");
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
