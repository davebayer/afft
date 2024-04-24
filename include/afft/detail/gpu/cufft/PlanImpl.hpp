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

#ifndef AFFT_DETAIL_GPU_CUFFT_PLAN_IMPL_HPP
#define AFFT_DETAIL_GPU_CUFFT_PLAN_IMPL_HPP

#include <array>
#include <cstddef>
#include <string_view>

#include <cufftXt.h>

#include "error.hpp"
#include "Handle.hpp"
#include "../../PlanImpl.hpp"
#include "../../../gpu.hpp"

namespace afft::detail::gpu::cufft
{
  using namespace afft::gpu;

  /// @brief cuFFT size type
  using SizeT = long long;

  /// @brief cuFFT callback source code
  constexpr std::string_view callbackSrcCode
  {R"(
#define PRECISION_F32      3 // must match afft::Precision::f32
#define PRECISION_F64      4 // must match afft::Precision::f64

#define COMPLEXITY_REAL    1 // must match afft::Complexity::real
#define COMPLEXITY_COMPLEX 2 // must match afft::Complexity::complex

#ifndef PRECISION
# error "PRECISION must be defined"
#elif !(PRECISION == PRECISION_F32 || PRECISION == PRECISION_F64)
# error "PRECISION must be either PRECISION_F32 or PRECISION_F64"
#endif

#ifndef COMPLEXITY
# error "COMPLEXITY must be defined"
#elif !(COMPLEXITY == COMPLEXITY_REAL || COMPLEXITY == COMPLEXITY_COMPLEX)
# error "COMPLEXITY must be either COMPLEXITY_REAL or COMPLEXITY_COMPLEX" 
#endif

#ifndef SCALE
# error "SCALE must be defined"
#endif

#include <cufftXt.h>

#if PRECISION == PRECISION_F32
constexpr cufftReal scale{SCALE};
#else
constexpr cufftDoubleReal scale{SCALE};
#endif

extern "C" __device__
#if PRECISION == PRECISION_F32
# if COMPLEXITY == COMPLEXITY_REAL
    void cufftJITCallbackStoreReal(void* dataOut, size_t offset, cufftReal elem, void*, void*)
# else
    void cufftJITCallbackStoreComplex(void* dataOut, size_t offset, cufftComplex elem, void*, void*)
# endif
#elif PRECISION == PRECISION_F64
# if COMPLEXITY == COMPLEXITY_REAL
    void cufftJITCallbackStoreDoubleReal(void* dataOut, size_t offset, cufftDoubleReal elem, void*, void*)
# else
    void cufftJITCallbackStoreDoubleComplex(void* dataOut, size_t offset, cufftDoubleComplex elem, void*, void*)
# endif
#endif
{
#if COMPLEXITY == COMPLEXITY_REAL
  elem *= scale;
#else
  elem.x *= scale;
  elem.y *= scale;
#endif

  reinterpret_cast<decltype(elem)*>(dataOut)[offset] = elem;
}

// Define the callback function pointer if not using JIT callbacks
extern "C" __device__ __constant__
#ifndef USE_JIT_CALLBACKS
# if PRECISION == PRECISION_F32
#   if COMPLEXITY == COMPLEXITY_REAL
      cufftCallbackStoreR cufftCallbackStoreFnPtr = cufftJITCallbackStoreReal;
#   else
      cufftCallbackStoreC cufftCallbackStoreFnPtr = cufftJITCallbackStoreComplex;
#   endif
# elif PRECISION == PRECISION_F64
#   if COMPLEXITY == COMPLEXITY_REAL
      cufftCallbackStoreD cufftCallbackStoreFnPtr = cufftJITCallbackStoreDoubleReal;
#   else
      cufftCallbackStoreZ cufftCallbackStoreFnPtr = cufftJITCallbackStoreDoubleComplex;
#   endif
# endif
#endif /* USE_JIT_CALLBACKS */
  )"};
  
  /// @brief cuFFT callback function pointer name
  constexpr cuda::rtc::CSymbolName storeCallbackPtrName{"cufftCallbackStoreFnPtr"};

  /**
   * @brief Get the cuFFT data type.
   * @param prec The precision of the data type.
   * @param comp The complexity of the data type.
   * @return The cuFFT data type.
   */
  [[nodiscard]] inline constexpr cudaDataType makeCudaDatatype(const Precision prec, const Complexity comp)
  {
    auto pickType = [comp](cudaDataType realType, cudaDataType complexType) -> cudaDataType
    {
      switch (comp)
      {
      case Complexity::real:    return realType;
      case Complexity::complex: return complexType;
      default:
        throw std::runtime_error("cuFFT does not recognize given complexity");
      }
    };

    switch (prec)
    {
    case Precision::bf16: return pickType(CUDA_R_16BF, CUDA_C_16BF);
    case Precision::f16:  return pickType(CUDA_R_16F,  CUDA_C_16F);
    case Precision::f32:  return pickType(CUDA_R_32F,  CUDA_C_32F);
    case Precision::f64:  return pickType(CUDA_R_64F,  CUDA_C_64F);
    default:
      throw std::runtime_error("cuFFT does not support given precision");
    }
  }

  /**
   * @brief Get the cuFFT workspace policy.
   * @param policy The workspace policy.
   * @return The cuFFT workspace policy.
   */
  [[nodiscard]] constexpr cufftXtWorkAreaPolicy getWorkspacePolicy(WorkspacePolicy policy)
  {
    switch (policy)
    {
      case WorkspacePolicy::minimal:     return CUFFT_WORKAREA_MINIMAL;
      case WorkspacePolicy::performance: return CUFFT_WORKAREA_PERFORMANCE;
      default:
        throw std::runtime_error("cuFFT does not support given workspace policy");
    }
  }

  /**
   * @brief Get the cuFFT callback type.
   * @param prec The precision of the data type.
   * @param comp The complexity of the data type.
   * @return The cuFFT callback type.
   */
  [[nodiscard]] inline constexpr cufftXtCallbackType makeStoreCallbackType(const Precision prec, const Complexity comp)
  {
    switch (prec)
    {
    case Precision::f32: return (comp == Complexity::real) ? CUFFT_CB_ST_REAL : CUFFT_CB_ST_COMPLEX;
    case Precision::f64: return (comp == Complexity::real) ? CUFFT_CB_ST_REAL_DOUBLE : CUFFT_CB_ST_COMPLEX_DOUBLE;
    default:
      throw std::runtime_error("cuFFT does not support given precision");
    }
  }

  /**
   * @brief cuFFT plan implementation.
   */
  class PlanImpl final : public afft::detail::PlanImpl
  {
    private:
      /// @brief Alias for the parent class.
      using Parent = afft::detail::PlanImpl;

    public:
      /// @brief Inherit constructors from the parent class.
      using Parent::Parent;

      /**
       * @brief Construct a cuFFT plan implementation.
       * @param config The configuration of the plan.
       */
      PlanImpl(const Config& config)
      : Parent(config)
      {
        cuda::ScopedDevice device{getConfig().getTargetConfig<Target::gpu>().device};

        const auto& precision         = getConfig().getTransformPrecision();
        const auto& dftParams         = getConfig().getTransformConfig<Transform::dft>();

        const auto rank               = static_cast<int>(getConfig().getTransformRank());
        const auto n                  = getConfig().getTransformDims<SizeT>();
        const auto [inembed, istride] = getConfig().getTransformSrcNembedAndStride<SizeT>();
        const auto idist              = (config.getTransformHowManyRank() == 1)
                                          ? getConfig().getTransformHowManySrcStrides<SizeT>().front()
                                          : SizeT{1};
        const auto inputType          = makeCudaDatatype(precision.execution,
                                                        (dftParams.type == dft::Type::realToComplex)
                                                          ? Complexity::real : Complexity::complex);
        const auto [onembed, ostride] = getConfig().getTransformDstNembedAndStride<SizeT>();
        const auto odist              = (config.getTransformHowManyRank() == 1)
                                          ? getConfig().getTransformHowManyDstStrides<SizeT>().front()
                                          : SizeT{1};
        const auto outputType         = makeCudaDatatype(precision.execution,
                                                        (dftParams.type == dft::Type::complexToReal)
                                                          ? Complexity::real : Complexity::complex);
        const auto batch              = (config.getTransformHowManyRank() == 1)
                                          ? getConfig().getTransformHowManyDims<SizeT>().front() : SizeT{1};
        const auto executionType      = makeCudaDatatype(precision.execution, Complexity::complex);

        std::size_t workSize{};

        Error::check(cufftSetAutoAllocation(mPlan, !getConfig().getTargetConfig<Target::gpu>().externalWorkspace));
        Error::check(cufftXtMakePlanMany(mPlan,
                                         rank,
                                         const_cast<SizeT*>(n.data()),
                                         const_cast<SizeT*>(inembed.data()),
                                         istride,
                                         idist,
                                         inputType,
                                         const_cast<SizeT*>(onembed.data()),
                                         ostride,
                                         odist,
                                         outputType,
                                         batch,
                                         &workSize,
                                         executionType));
        
        // FIXME: This is not working
        // Error::check(cufftXtSetWorkAreaPolicy(mPlan, getWorkspacePolicy(commonParams.workspacePolicy), &workSize));

        if (const auto normalize = getConfig().getCommonParameters().normalize;
            normalize != Normalize::none)
        {
          cuda::rtc::Program program{callbackSrcCode, "cufftCallbackFn.cu"};

          std::array options
          {
            cuda::rtc::makeDefinitionOption("PRECISION", to_underlying(precision.execution)),
            cuda::rtc::makeDefinitionOption("COMPLEXITY", dftParams.type == dft::Type::complexToReal
                                                              ? to_underlying(Complexity::real)
                                                              : to_underlying(Complexity::complex)),
            cuda::rtc::makeDefinitionOption("SCALE", getConfig().getTransformNormFactor<Precision::f64>()),
            cuda::rtc::makeArchOption(device),
            cuda::rtc::makeIncludePathOption(cuda::getIncludePath()),
          };

          std::array optionPtrs = {options[0].c_str(), options[1].c_str(), options[2].c_str(), options[3].c_str(), "-dc"};

          if (!program.compile(optionPtrs))
          {
            throw makeException<std::runtime_error>("Failed to compile callback function");
          }

          mModule.load(program.getCode(cuda::rtc::CodeType::CUBIN));

          auto [dStoreCallbackPtr, storeCallbackPtrSize] = mModule.getGlobal(storeCallbackPtrName);

          void** hStoreCallbackPtr{};
          Error::check(cuMemcpyDtoH(&hStoreCallbackPtr, dStoreCallbackPtr, storeCallbackPtrSize));

          Error::check(cufftXtSetCallback(mPlan,
                                          hStoreCallbackPtr,
                                          makeStoreCallbackType(precision.execution,
                                                                (dftParams.type == dft::Type::complexToReal)
                                                                  ? Complexity::real : Complexity::complex),
                                          nullptr));
        }
      }

      /// @brief Default destructor.
      ~PlanImpl() override = default;

      /// @brief Inherit assignment operator from the parent class.
      using Parent::operator=;

      /**
       * @brief Execute the plan.
       * @param src The source data.
       * @param dst The destination data.
       * @param execParams The execution parameters.
       */
      void executeImpl(ExecParam src, ExecParam dst, const afft::gpu::ExecutionParameters& execParams) override
      {
        if (src.isSplit() || dst.isSplit())
        {
          throw makeException<std::runtime_error>("vkfft does not support planar complex format");
        }

        const int direction = (getConfig().getTransformDirection() == Direction::forward)
                                ? CUFFT_FORWARD : CUFFT_INVERSE;

        cuda::ScopedDevice device{getConfig().getTargetConfig<Target::gpu>().device};

        // Set stream
        Error::check(cufftSetStream(mPlan, execParams.stream));

        // Set workspace area if defined
        if (getConfig().getTargetConfig<Target::gpu>().externalWorkspace)
        {
          Error::check(cufftSetWorkArea(mPlan, execParams.workspace));
        }

        Error::check(cufftXtExec(mPlan, src.getRealImag(), dst.getRealImag(), direction));
      }

      /**
       * @brief Get the size of the workspace required for the plan.
       * @return The size of the workspace required for the plan.
       */
      [[nodiscard]] std::size_t getWorkspaceSize() const override
      {
        std::size_t size{};

        Error::check(cufftGetSize(mPlan, &size));

        return size;
      }
    protected:
    private:
      cuda::Module mModule{}; ///< The module containing the callback function.
      Handle       mPlan{};   ///< The cuFFT plan.
  };

  /**
   * @brief Create a cuFFT plan implementation.
   * @param config The configuration of the plan.
   * @return The cuFFT plan implementation.
   */
  [[nodiscard]] std::unique_ptr<PlanImpl> makePlanImpl(const Config& config)
  {
    if (config.getTransformRank() == 0 || config.getTransformRank() > 3)
    {
      throw makeException<std::runtime_error>("cuFFT only supports 1D, 2D, and 3D transforms");
    }

    if (config.getTransformHowManyRank() > 1)
    {
      throw makeException<std::runtime_error>("cuFFT does not support multi-dimensional batched transforms");
    }

    const auto& commonParams = config.getCommonParameters();

    switch (commonParams.complexFormat)
    {
    case ComplexFormat::interleaved:
      break;
    default:
      throw makeException<std::runtime_error>("cuFFT only supports interleaved complex format");
    }

    switch (config.getTransform())
    {
    case Transform::dft:
    {
      const auto& dftConfig = config.getTransformConfig<Transform::dft>();

      if (dftConfig.type == dft::Type::complexToReal && !commonParams.destroySource)
      {
        throw makeException<std::runtime_error>("cuFFT requires source data to be destroyed when transforming c2r");
      }
      break;
    }
    default:
      throw std::runtime_error("cuFFT only supports DFT plans");
    }

    const auto& precision = config.getTransformPrecision();

    if (precision.execution != precision.source || precision.execution != precision.destination)
    {
      throw makeException<std::runtime_error>("cuFFT does not support type conversions");
    }

    switch (precision.execution)
    {
    case Precision::bf16:
    case Precision::f16:
    case Precision::f32:
    case Precision::f64:
      break;
    default:
      throw makeException<std::runtime_error>("cuFFT does not support given precision");
    }

    switch (commonParams.normalize)
    {
    case Normalize::none:
      break;
    case Normalize::unitary:
    case Normalize::orthogonal:
      switch (precision.execution)
      {
      case Precision::f32:
      case Precision::f64:
        break;
      default:
        throw makeException<std::runtime_error>("cuFFT does not support given precision for normalization");
      }
    default:
      throw makeException<std::runtime_error>("cuFFT does not support given normalization");
    }

    return std::make_unique<PlanImpl>(config);
  }
} // namespace afft::detail::gpu::cufft

#endif /* AFFT_DETAIL_GPU_CUFFT_PLAN_IMPL_HPP */
