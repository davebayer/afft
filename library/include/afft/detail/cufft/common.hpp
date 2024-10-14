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

#ifndef AFFT_DETAIL_CUFFT_COMMON_HPP
#define AFFT_DETAIL_CUFFT_COMMON_HPP

#ifndef AFFT_TOP_LEVEL_INCLUDE
# include "../include.hpp"
#endif

namespace afft::detail::cufft
{
  /// @brief cuFFT size type
  using SizeT = long long;

  /**
   * @class Handle
   * @brief RAII wrapper for cuFFT handle.
   */
  class Handle
  {
    public:
      /// @brief Default constructor.
      Handle()
      {
        checkError(cufftCreate(&mHandle));
      }

      /// @brief Deleted copy constructor.
      Handle(const Handle&) = delete;

      /// @brief Default move constructor.
      Handle(Handle&&) = default;

      /// @brief Deleted copy assignment operator.
      Handle& operator=(const Handle&) = delete;

      /// @brief Default move assignment operator.
      Handle& operator=(Handle&&) = default;

      /// @brief Destructor.
      ~Handle()
      {
        if (mHandle)
        {
          cufftDestroy(mHandle);
        }
      }

      /// @brief Get cuFFT handle.
      [[nodiscard]] operator cufftHandle() const
      {
        return mHandle;
      }
    private:
      cufftHandle mHandle{}; ///< cuFFT handle.
  };

  [[nodiscard]] inline cuda::rtc::Code makeNormalizationStoreCallbackCode(const int        device,
                                                                          const Precision  prec,
                                                                          const Complexity cmpl,
                                                                          const double     normFactor)
  {
    static constexpr std::string_view srcCode{R"(
#define PREC_F32 (0) // single precision
#define PREC_F64 (1) // double precision

#define CMPL_R   (0) // real complexity
#define CMPL_C   (1) // complex complexity

#ifdef PREC
# if PREC != PREC_F32 && PREC != PREC_F64
#   error "unsupported precision"
# endif
#else
# error "PREC must be defined"
#endif

#ifdef CMPL
# if CMPL != CMPL_R && CMPL != CMPL_C
#   error "unsupported complexity"
# endif
#else
# error "CMPL must be defined"
#endif

#ifndef NORM_FACT
# error "NORM_FACT must be defined"
#endif

// cuFFT data types
using cufftReal          = float;
using cufftComplex       = float2;
using cufftDoubleReal    = double;
using cufftDoubleComplex = double2;

#if PREC == PREC_F32
using Real    = cufftReal;
using Complex = cufftComplex;
#elif PREC == PREC_F64
using Real    = cufftDoubleReal;
using Complex = cufftDoubleComplex;
#endif

// Normalization factor
inline constexpr Real normFactor = static_cast<Real>(NORM_FACT);

#if CMPL == CMPL_R
// cuFFT callback function to store real normalized data
__device__ void normStoreCallback(void* dataOut, unsigned long long offset, Real element, void*, void*)
{
  element *= normFactor;

  reinterpret_cast<Real*>(dataOut)[offset] = element;
}
#else
// cuFFT callback function to store complex normalized data
__device__ void normStoreCallback(void* dataOut, unsigned long long offset, Complex element, void*, void*)
{
  element.x *= normFactor;
  element.y *= normFactor;

  reinterpret_cast<Complex*>(dataOut)[offset] = element;
}
#endif
    )"};

    if (prec != Precision::f32 && prec != Precision::f64)
    {
      throw Exception{Error::cufft, "unsupported precision for normalization callback"};
    }

    cuda::rtc::Program program{srcCode, "cufftNormCallback.cu"};

    const auto precisionDef  = cuda::rtc::makeDefinitionOption("PREC", prec == Precision::f32 ? "PREC_F32" : "PREC_F64");
    const auto complexityDef = cuda::rtc::makeDefinitionOption("CMPL", cmpl == Complexity::real ? "CMPL_R" : "CMPL_C");

    std::array<char, 32> normFactorStr{};
    const auto [normFactorStrEnd, ec] = std::to_chars(normFactorStr.data(),
                                                      normFactorStr.data() + normFactorStr.size(),
                                                      normFactor,
                                                      std::chars_format::general, 
                                                      16);

    if (ec != std::errc{})
    {
      throw Exception{Error::cufft, "failed to convert normalization factor to string"};
    }
    
    const auto normFactorDef  = cuda::rtc::makeDefinitionOption("NORM_FACT", {normFactorStr.data(), normFactorStrEnd});
    const auto archOption     = cuda::rtc::makeRealArchOption(device);
    const auto rdcOption      = cuda::rtc::makeRelocatableDeviceCodeOption(true);
    const auto dltoOption     = cuda::rtc::makeLinkTimeOptimizationOption();
    const auto cpp17Option    = cuda::rtc::makeCppStandardOption(17);
    const auto fastMathOption = cuda::rtc::makeFastMathOption();

    const char* options[]{precisionDef.c_str(),
                          complexityDef.c_str(),
                          normFactorDef.c_str(),
                          archOption.c_str(),
                          rdcOption.c_str(),
                          dltoOption.c_str(),
                          cpp17Option.c_str(),
                          fastMathOption.c_str()};

    if (!program.compile(options))
    {
      throw Exception{Error::cufft, "failed to compile the normalization callback"};
    }

    return program.getCode(cuda::rtc::CodeType::LTOIR);
  }

  /**
   * @brief Make the cuFFT direction.
   * @param dir The direction.
   * @return The cuFFT direction.
   */
  [[nodiscard]] inline constexpr int makeDirection(const Direction dir)
  {
    switch (dir)
    {
    case Direction::forward:
      return CUFFT_FORWARD;
    case Direction::inverse:
      return CUFFT_INVERSE;
    default:
      cxx::unreachable();
    }
  }

  /**
   * @brief Get the cuFFT data type.
   * @param prec The precision of the data type.
   * @param comp The complexity of the data type.
   * @return The cuFFT data type.
   */
  [[nodiscard]] inline constexpr cudaDataType makeCudaDataType(const Precision prec, const Complexity comp)
  {
    validate(comp);

    switch (prec)
    {
    case Precision::bf16:
      return (comp == Complexity::real) ? CUDA_R_16BF : CUDA_C_16BF;
    case Precision::f16:
      return (comp == Complexity::real) ? CUDA_R_16F : CUDA_C_16F;
    case Precision::f32:
      return (comp == Complexity::real) ? CUDA_R_32F : CUDA_C_32F;
    case Precision::f64:
      return (comp == Complexity::real) ? CUDA_R_64F : CUDA_C_64F;
    default:
      throw Exception{Error::cufft, "unsupported precision"};
    }
  }

  /**
   * @brief Make the cuFFT type.
   * @param prec The precision of the data type.
   * @param dftType The DFT type.
   * @return The cuFFT type.
   */
  [[nodiscard]] inline constexpr cufftType makeCufftType(const Precision prec, dft::Type dftType)
  {
    if (prec != Precision::f32 && prec != Precision::f64)
    {
      throw Exception{Error::cufft, "unsupported precision for cuFFT type"};
    }

    switch (dftType)
    {
    case dft::Type::complexToComplex:
      return (prec == Precision::f32) ? CUFFT_C2C : CUFFT_Z2Z;
    case dft::Type::realToComplex:
      return (prec == Precision::f32) ? CUFFT_R2C : CUFFT_D2Z;
    case dft::Type::complexToReal:
      return (prec == Precision::f32) ? CUFFT_C2R : CUFFT_Z2D;
    default:
      throw Exception{Error::cufft, "unsupported DFT type"};
    }
  }

  /**
   * @brief Make the cuFFT workspace policy.
   * @param policy The workspace policy.
   * @return The cuFFT workspace policy.
   */
  [[nodiscard]] constexpr cufftXtWorkAreaPolicy makeWorkAreaPolicy(afft::cufft::WorkspacePolicy policy)
  {
    switch (policy)
    {
#   if CUFFT_VERSION >= 9200
      case afft::cufft::WorkspacePolicy::minimal:
        return CUFFT_WORKAREA_MINIMAL;
      case afft::cufft::WorkspacePolicy::user:
        return CUFFT_WORKAREA_USER;
#   endif
      case afft::cufft::WorkspacePolicy::performance:
      default:
        return CUFFT_WORKAREA_PERFORMANCE;
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
    case Precision::f32:
      return (comp == Complexity::real) ? CUFFT_CB_ST_REAL : CUFFT_CB_ST_COMPLEX;
    case Precision::f64:
      return (comp == Complexity::real) ? CUFFT_CB_ST_REAL_DOUBLE : CUFFT_CB_ST_COMPLEX_DOUBLE;
    default:
      throw Exception{Error::cufft, "unsupported precision for callback"};
    }
  }
} // namespace afft::detail::cufft

#endif /* AFFT_DETAIL_CUFFT_COMMON_HPP */
