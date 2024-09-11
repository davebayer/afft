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

#ifdef MATLABW_ENABLE_GPU
/**
 * @brief Get the current GPU device.
 * @param[in] errorId Error identifier to throw.
 * @return Current GPU device.
 */
[[nodiscard]] static inline int getCurrentGpuDevice(const char* errorId)
{
  mx::gpu::init();

  int device{};

  if (cudaGetDevice(&device) != cudaSuccess)
  {
    throw mx::Exception{errorId, "failed to get current CUDA device"};
  }

  return device;
}
#endif

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
 * @param[in] array Input array to get the precision from.
 * @return Transform precision.
 */
[[nodiscard]] static inline afft::PrecisionTriad getTransformPrecision(const mx::ArrayCref array)
{
  switch (array.getClassId())
  {
  case mx::ClassId::single:
    return afft::makePrecision<float>();
  default:
    return afft::makePrecision<double>();
  }
}

#ifdef MATLABW_ENABLE_GPU
/**
 * @brief Get the transform precision from the input array.
 * @param[in] array Input array to get the precision from.
 * @return Transform precision.
 */
[[nodiscard]] static inline afft::PrecisionTriad getTransformPrecision(const mx::gpu::ArrayCref array)
{
  switch (array.getClassId())
  {
  case mx::ClassId::single:
    return afft::makePrecision<float>();
  default:
    return afft::makePrecision<double>();
  }
}
#endif

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

  ShapeConverter shapeConverter{};

  afft::dft::Parameters dftParams{};
  dftParams.direction     = afft::Direction::forward;
  dftParams.axes          = afft::allAxes;
  dftParams.normalization = afft::Normalization::none;
  dftParams.placement     = afft::Placement::outOfPlace;
  dftParams.destructive   = false;
  dftParams.type          = afft::dft::Type::complexToComplex;


#ifdef MATLABW_ENABLE_GPU
  if (rhs[0].isGpuArray())
  {
    mx::gpu::init();

    const auto cudaDevice = getCurrentGpuDevice("afft:fftn:failedToGetGpuDevice");

    mx::gpu::Array src{rhs[0]};

    if (src.getSize() == 0)
    {
      lhs[0] = mx::gpu::makeUninitNumericArray<std::complex<double>>(0, 0).release();
      return;
    }

    if (!src.isSingle() && !src.isDouble())
    {
      throw mx::Exception{"afft:fftn:invalidInputClass", "input array must be floating-point"};
    }

    if (!src.isComplex())
    {
      throw mx::Exception{"afft:fftn:invalidInputComplexity", "input array must be complex"};
    }
    
    dftParams.precision = getTransformPrecision(src);
    dftParams.shape     = shapeConverter(src.getDims(), "afft:fftn:invalidInputDims");

    afft::cuda::Parameters cudaParams{};
    cudaParams.devices = afft::makeScalarView(cudaDevice);

    const afft::Description desc{dftParams, cudaParams};

    auto it = planCache.find(desc);
    if (it == planCache.end())
    {
      it = planCache.insert(afft::makePlan(desc));
    }

    auto dst = mx::gpu::makeUninitNumericArray(src.getDims(), src.getClassId(), mx::Complexity::complex);

    (*it)->executeUnsafe(src.getData(), dst.getData());

    mex::printf("Used backend: %s", afft::getBackendName((*it)->getBackend()).data());

    lhs[0] = dst.release();
  }
  else
#endif
  {
    mx::ArrayCref src{rhs[0]};

    if (src.getSize() == 0)
    {
      lhs[0] = src;
      return;
    }

    if (!src.isSingle() && !src.isDouble())
    {
      throw mx::Exception{"afft:fftn:invalidInputClass", "input array must be floating-point"};
    }

    if (!src.isComplex())
    {
      throw mx::Exception{"afft:fftn:invalidInputComplexity", "input array must be complex"};
    }

    dftParams.precision = getTransformPrecision(src);
    dftParams.shape     = shapeConverter(src.getDims(), "afft:fftn:invalidInputDims");

    const afft::Description desc{dftParams, afft::cpu::Parameters{}};

    auto it = planCache.find(desc);
    if (it == planCache.end())
    {
      afft::cpu::BackendParameters backendParams{};
      backendParams.threadLimit       = 4;
      backendParams.fftw3.plannerFlag = afft::fftw3::PlannerFlag::estimate;

      it = planCache.insert(afft::makePlan(desc, backendParams));
    }

    auto dst = mx::makeUninitNumericArray(src.getDims(), src.getClassId(), mx::Complexity::complex);

    (*it)->executeUnsafe(src.getData(), dst.getData());

    mex::printf("Used backend: %s", afft::getBackendName((*it)->getBackend()).data());

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

  ShapeConverter shapeConverter{};

  afft::dft::Parameters dftParams{};
  dftParams.direction     = afft::Direction::inverse;
  dftParams.axes          = afft::allAxes;
  dftParams.normalization = afft::Normalization::unitary;
  dftParams.placement     = afft::Placement::outOfPlace;
  dftParams.destructive   = false;
  dftParams.type          = afft::dft::Type::complexToComplex;

#ifdef MATLABW_ENABLE_GPU
  if (rhs[0].isGpuArray())
  {
    mx::gpu::init();

    const auto cudaDevice = getCurrentGpuDevice("afft:ifftn:failedToGetGpuDevice");

    mx::gpu::Array src{rhs[0]};

    if (src.getSize() == 0)
    {
      lhs[0] = mx::gpu::makeUninitNumericArray<std::complex<double>>(0, 0).release();
      return;
    }

    if (!src.isSingle() && !src.isDouble())
    {
      throw mx::Exception{"afft:ifftn:invalidInputClass", "input array must be floating-point"};
    }

    if (!src.isComplex())
    {
      throw mx::Exception{"afft:ifftn:invalidInputComplexity", "input array must be complex"};
    }
    
    dftParams.precision = getTransformPrecision(src);
    dftParams.shape     = shapeConverter(src.getDims(), "afft:ifftn:invalidInputDims");

    afft::cuda::Parameters cudaParams{};
    cudaParams.devices = afft::makeScalarView(cudaDevice);

    const afft::Description desc{dftParams, cudaParams};

    auto it = planCache.find(desc);
    if (it == planCache.end())
    {
      it = planCache.insert(afft::makePlan(desc));
    }

    auto dst = mx::gpu::makeUninitNumericArray(src.getDims(), src.getClassId(), mx::Complexity::complex);

    (*it)->executeUnsafe(src.getData(), dst.getData());

    mex::printf("Used backend: %s", afft::getBackendName((*it)->getBackend()).data());

    lhs[0] = dst.release();
  }
  else
#endif
  {
    mx::ArrayCref src{rhs[0]};

    if (src.getSize() == 0)
    {
      lhs[0] = src;
      return;
    }

    if (!src.isSingle() && !src.isDouble())
    {
      throw mx::Exception{"afft:ifftn:invalidInputClass", "input array must be floating-point"};
    }

    if (!src.isComplex())
    {
      throw mx::Exception{"afft:ifftn:invalidInputComplexity", "input array must be complex"};
    }

    dftParams.precision = getTransformPrecision(src);
    dftParams.shape     = shapeConverter(src.getDims(), "afft:ifftn:invalidInputDims");

    const afft::Description desc{dftParams, afft::cpu::Parameters{}};

    auto it = planCache.find(desc);
    if (it == planCache.end())
    {
      afft::cpu::BackendParameters backendParams{};
      backendParams.threadLimit       = 4;
      backendParams.fftw3.plannerFlag = afft::fftw3::PlannerFlag::estimate;

      it = planCache.insert(afft::makePlan(desc, backendParams));
    }

    auto dst = mx::makeUninitNumericArray(src.getDims(), src.getClassId(), mx::Complexity::complex);

    (*it)->executeUnsafe(src.getData(), dst.getData());

    mex::printf("Used backend: %s", afft::getBackendName((*it)->getBackend()).data());

    lhs[0] = std::move(dst);
  }
}
