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

#include "../../gpu.hpp"

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
        Error::check(cufftCreate(&mHandle));
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

  /// @brief cuFFT callback source code
  inline constexpr std::string_view callbackSrcCode
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

/**********************************************************************************************************************/
/* Copied from cufft.h and cufftXt.h to prevent include                                                               */
/**********************************************************************************************************************/
// cufftReal is a single-precision, floating-point real data type.
// cufftDoubleReal is a double-precision, real data type.
typedef float cufftReal;
typedef double cufftDoubleReal;

// cufftComplex is a single-precision, floating-point complex data type that
// consists of interleaved real and imaginary components.
// cufftDoubleComplex is the double-precision equivalent.
typedef cuComplex cufftComplex;
typedef cuDoubleComplex cufftDoubleComplex;

typedef cufftComplex (*cufftCallbackLoadC)(void *dataIn, size_t offset, void *callerInfo, void *sharedPointer);
typedef cufftDoubleComplex (*cufftCallbackLoadZ)(void *dataIn, size_t offset, void *callerInfo, void *sharedPointer);
typedef cufftReal (*cufftCallbackLoadR)(void *dataIn, size_t offset, void *callerInfo, void *sharedPointer);
typedef cufftDoubleReal(*cufftCallbackLoadD)(void *dataIn, size_t offset, void *callerInfo, void *sharedPointer);

typedef void (*cufftCallbackStoreC)(void *dataOut, size_t offset, cufftComplex element, void *callerInfo, void *sharedPointer);
typedef void (*cufftCallbackStoreZ)(void *dataOut, size_t offset, cufftDoubleComplex element, void *callerInfo, void *sharedPointer);
typedef void (*cufftCallbackStoreR)(void *dataOut, size_t offset, cufftReal element, void *callerInfo, void *sharedPointer);
typedef void (*cufftCallbackStoreD)(void *dataOut, size_t offset, cufftDoubleReal element, void *callerInfo, void *sharedPointer);

/**********************************************************************************************************************/

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
  inline constexpr cuda::rtc::CSymbolName storeCallbackPtrName{"cufftCallbackStoreFnPtr"};

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
      case Complexity::real:
        return realType;
      case Complexity::complex:
        return complexType;
      default:
        cxx::unreachable();
      }
    };

    switch (prec)
    {
    case Precision::bf16:
      return pickType(CUDA_R_16BF, CUDA_C_16BF);
    case Precision::f16:
      return pickType(CUDA_R_16F,  CUDA_C_16F);
    case Precision::f32:
      return pickType(CUDA_R_32F,  CUDA_C_32F);
    case Precision::f64:
      return pickType(CUDA_R_64F,  CUDA_C_64F);
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
      case WorkspacePolicy::minimal:
        return CUFFT_WORKAREA_MINIMAL;
      case WorkspacePolicy::performance:
        return CUFFT_WORKAREA_PERFORMANCE;
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
    case Precision::f32:
      return (comp == Complexity::real) ? CUFFT_CB_ST_REAL : CUFFT_CB_ST_COMPLEX;
    case Precision::f64:
      return (comp == Complexity::real) ? CUFFT_CB_ST_REAL_DOUBLE : CUFFT_CB_ST_COMPLEX_DOUBLE;
    default:
      throw std::runtime_error("cuFFT does not support given precision");
    }
  }
} // namespace afft::detail::cufft

#endif /* AFFT_DETAIL_CUFFT_COMMON_HPP */
