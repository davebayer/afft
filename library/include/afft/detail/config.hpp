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

#ifndef AFFT_DETAIL_CONFIG_HPP
#define AFFT_DETAIL_CONFIG_HPP

#include "../config.h"

// Define the AFFT_CXX_VERSION version macro
#ifdef _MSVC_LANG
#  define AFFT_CXX_VERSION         _MSVC_LANG
#else
#  define AFFT_CXX_VERSION         __cplusplus
#endif

// Define AFFT_EXPORT macro to expand to nothing if not defined
#ifndef AFFT_EXPORT
# define AFFT_EXPORT
#endif

// Define AFFT_HEADER_ONLY_INLINE macro to expand to inline if not defined
#ifndef AFFT_HEADER_ONLY_INLINE
# ifdef AFFT_HEADER_ONLY
#   define AFFT_HEADER_ONLY_INLINE inline
# else
#   define AFFT_HEADER_ONLY_INLINE
# endif
#endif

// If C++ version is 20, try to include <version> header
#if (AFFT_CXX_VERSION >= 202002L) && __has_include(<version>)
# define AFFT_CXX_HAS_VERSION
# include <version>
#endif

// implementation of C++20 requires clause for older C++ versions, should be used as:
// AFFT_TEMPL_REQUIRES(typename T, std::is_integral_v<T>)
// auto func(...) { ...}
#if defined(__cpp_concepts) && (__cpp_concepts >= 201907L)
  /// @brief Macro for requires clause
# define AFFT_TEMPL_REQUIRES(templParam, requiredExpr) \
    template<templParam> requires(requiredExpr)
#else
  /// @brief Macro for requires clause, using std::enable_if_t for older C++ versions
# define AFFT_TEMPL_REQUIRES(templParam, requiredExpr) \
    template<templParam, std::enable_if_t<requiredExpr, int> = 0>
#endif

// implementation of C++20 requires clause for older C++ versions, should be used as:
// template<typename T>
// auto func() -> AFFT_RET_REQUIRES(returnType, std::is_integral_v<T>) { ... }
#if defined(__cpp_concepts) && (__cpp_concepts >= 201907L)
  /// @brief Macro for requires clause
# define AFFT_RET_REQUIRES(retType, requiredExpr) \
    retType requires(requiredExpr)
#else
  /// @brief Macro for requires clause, using std::enable_if_t for older C++ versions
# define AFFT_RET_REQUIRES(retType, requiredExpr) \
    std::enable_if_t<requiredExpr, retType>
#endif

// Check if C++20 <span> is supported
#if defined(AFFT_CXX_HAS_VERSION) && defined(__cpp_lib_span) && (__cpp_lib_span >= 202002L)
# define AFFT_CXX_HAS_SPAN
#endif

// Check if C++23 `import std` is supported
#if defined(AFFT_CXX_HAS_VERSION) && defined(__cpp_lib_modules) && (__cpp_lib_modules >= 202207L)
# define AFFT_CXX_HAS_IMPORT_STD
#endif

// Check if C++23 <stdfloat> is implemented
#if (AFFT_CXX_VERSION >= 202302L) && __has_include(<stdfloat>)
# define AFFT_CXX_HAS_STD_FLOAT
#endif

#endif /* AFFT_DETAIL_CONFIG_HPP */
