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

#ifndef AFFT_FORMATTERS_HPP
#define AFFT_FORMATTERS_HPP

#ifndef AFFT_TOP_LEVEL_INCLUDE
# include <afft/detail/include.hpp>
#endif

#include <afft/backend.hpp>
#include <afft/error.hpp>
#include <afft/memory.hpp>
#include <afft/mp.hpp>
#include <afft/target.hpp>
#include <afft/transform.hpp>
#include <afft/type.hpp>
#include <afft/version.hpp>
#include <afft/detail/cxx.hpp>
#include <afft/detail/formatters.hpp>
#include <afft/detail/validate.hpp>

/**********************************************************************************************************************/
// Backend
/**********************************************************************************************************************/
/// @brief Formatter for afft::Backend
AFFT_EXPORT template<>
struct std::formatter<afft::Backend> : afft::detail::ParseFormatterSkipper<>
{
  /**
   * @brief Format a backend.
   * @param[in] backend Backend
   * @param[in] ctx     Format context
   * @return decltype(ctx.out())
   */
  template<typename FormatContext>
  [[nodiscard]] auto format(const afft::Backend& backend, FormatContext& ctx) const
    -> decltype(ctx.out())
  {
    afft::detail::validate(backend);
    return std::format_to(ctx.out(), "{}", afft::getBackendName(backend));
  }
};

/// @brief Formatter for afft::BackendMask
AFFT_EXPORT template<>
struct std::formatter<afft::BackendMask> : afft::detail::ParseFormatterSkipper<>
{
  /**
   * @brief Format a backend mask.
   * @param[in] mask Backend mask
   * @param[in] ctx  Format context
   * @return decltype(ctx.out())
   */
  template<typename FormatContext>
  [[nodiscard]] auto format(const afft::BackendMask& backendMask, FormatContext& ctx) const
    -> decltype(ctx.out())
  {    
    bool first = true;

    for (std::size_t i{}; i < afft::backendCount; ++i)
    {
      if ((backendMask & afft::makeBackendMask(static_cast<afft::Backend>(i))) != afft::BackendMask::empty)
      {
        if (!first)
        {
          std::format_to(ctx.out(), " | ");
          first = false;
        }
        
        std::format_to(ctx.out(), "{}", afft::getBackendName(static_cast<afft::Backend>(i)));
      }
    }

    return ctx.out();
  }
};

/// @brief Formatter for afft::SelectStrategy
AFFT_EXPORT template<>
struct std::formatter<afft::SelectStrategy> : afft::detail::ParseFormatterSkipper<>
{
  /**
   * @brief Format a select strategy.
   * @param[in] strategy Select strategy
   * @param[in] ctx      Format context
   * @return decltype(ctx.out())
   */
  template<typename FormatContext>
  [[nodiscard]] auto format(const afft::SelectStrategy& strategy, FormatContext& ctx) const
    -> decltype(ctx.out())
  {
    afft::detail::validate(strategy);

    auto getName = [](afft::SelectStrategy strategy) -> const char*
    {
      switch (strategy)
      {
      case afft::SelectStrategy::first:
        return "first";
      case afft::SelectStrategy::best:
        return "best";
      default:
        afft::detail::cxx::unreachable();
      }
    };

    return std::format_to(ctx.out(), "{}", getName(strategy));
  }
};

/**********************************************************************************************************************/
// Error
/**********************************************************************************************************************/
/// @brief Formatter for afft::Error
AFFT_EXPORT template<>
struct std::formatter<afft::Error> : afft::detail::ParseFormatterSkipper<>
{
  /**
   * @brief Format an error.
   * @param[in] error Error
   * @param[in] ctx   Format context
   * @return decltype(ctx.out())
   */
  template<typename FormatContext>
  [[nodiscard]] auto format(const afft::Error& error, FormatContext& ctx) const
    -> decltype(ctx.out())
  {
    auto getName = [](afft::Error error) -> const char*
    {
      switch (error)
      {
      case afft::Error::internal:
        return "internal";
      case afft::Error::invalidArgument:
        return "invalidArgument";
      case afft::Error::mpi:
        return "mpi";
      case afft::Error::cudaDriver:
        return "cudaDriver";
      case afft::Error::cudaRuntime:
        return "cudaRuntime";
      case afft::Error::cudaRtc:
        return "cudaRtc";
      case afft::Error::hip:
        return "hip";
      case afft::Error::opencl:
        return "opencl";
      case afft::Error::clfft:
        return "clfft";
      case afft::Error::cufft:
        return "cufft";
      case afft::Error::fftw3:
        return "fftw3";
      case afft::Error::heffte:
        return "heffte";
      case afft::Error::hipfft:
        return "hipfft";
      case afft::Error::mkl:
        return "mkl";
      case afft::Error::pocketfft:
        return "pocketfft";
      case afft::Error::rocfft:
        return "rocfft";
      case afft::Error::vkfft:
        return "vkfft";
      default:
        return "unknown";
      }
    };

    return std::format_to(ctx.out(), "{}", getName(error));
  }
};

/**********************************************************************************************************************/
// Memory
/**********************************************************************************************************************/
/// @brief Formatter for afft::MemoryLayout
AFFT_EXPORT template<>
struct std::formatter<afft::MemoryLayout> : afft::detail::ParseFormatterSkipper<>
{
  /**
   * @brief Format a memory layout.
   * @param[in] layout Memory layout
   * @param[in] ctx    Format context
   * @return decltype(ctx.out())
   */
  template<typename FormatContext>
  [[nodiscard]] auto format(const afft::MemoryLayout& layout, FormatContext& ctx) const
    -> decltype(ctx.out())
  {
    afft::detail::validate(layout);

    auto getName = [](afft::MemoryLayout layout) -> const char*
    {
      switch (layout)
      {
      case afft::MemoryLayout::centralized:
        return "centralized";
      case afft::MemoryLayout::distributed:
        return "distributed";
      default:
        afft::detail::cxx::unreachable();
      }
    };

    return std::format_to(ctx.out(), "{}", getName(layout));
  }
};

/// @brief Formatter for afft::Alignment
AFFT_EXPORT template<>
struct std::formatter<afft::Alignment> : afft::detail::ParseFormatterSkipper<>
{
  /**
   * @brief Format an alignment.
   * @param[in] alignment Alignment
   * @param[in] ctx       Format context
   * @return decltype(ctx.out())
   */
  template<typename FormatContext>
  [[nodiscard]] auto format(const afft::Alignment& alignment, FormatContext& ctx) const
    -> decltype(ctx.out())
  {
    return std::format_to(ctx.out(), "{}", afft::detail::cxx::to_underlying(alignment));
  }
};

/// @brief Formatter for afft::ComplexFormat
AFFT_EXPORT template<>
struct std::formatter<afft::ComplexFormat> : afft::detail::ParseFormatterSkipper<>
{
  /**
   * @brief Format a complex format.
   * @param[in] format Complex format
   * @param[in] ctx    Format context
   * @return decltype(ctx.out())
   */
  template<typename FormatContext>
  [[nodiscard]] auto format(const afft::ComplexFormat& format, FormatContext& ctx) const
    -> decltype(ctx.out())
  {
    afft::detail::validate(format);

    auto getName = [](afft::ComplexFormat format) -> const char*
    {
      switch (format)
      {
      case afft::ComplexFormat::interleaved:
        return "interleaved";
      case afft::ComplexFormat::planar:
        return "planar";
      default:
        afft::detail::cxx::unreachable();
      }
    };

    return std::format_to(ctx.out(), "{}", getName(format));
  }
};

/**********************************************************************************************************************/
// Multi-process
/**********************************************************************************************************************/
/// @brief Formatter for afft::MpBackend
AFFT_EXPORT template<>
struct std::formatter<afft::MpBackend> : afft::detail::ParseFormatterSkipper<>
{
  /**
   * @brief Format a multi-process backend.
   * @param[in] backend Multi-process backend
   * @param[in] ctx     Format context
   * @return decltype(ctx.out())
   */
  template<typename FormatContext>
  [[nodiscard]] auto format(const afft::MpBackend& backend, FormatContext& ctx) const
    -> decltype(ctx.out())
  {
    afft::detail::validate(backend);

    auto getName = [](afft::MpBackend backend) -> const char*
    {
      switch (backend)
      {
      case afft::MpBackend::none:
        return "none";
      case afft::MpBackend::mpi:
        return "mpi";
      default:
        afft::detail::cxx::unreachable();
      }
    };

    return std::format_to(ctx.out(), "{}", getName(backend));
  }
};

/**********************************************************************************************************************/
// Target
/**********************************************************************************************************************/
/// @brief Formatter for afft::Target
AFFT_EXPORT template<>
struct std::formatter<afft::Target> : afft::detail::ParseFormatterSkipper<>
{
  /**
   * @brief Format a target.
   * @param[in] target Target
   * @param[in] ctx    Format context
   * @return decltype(ctx.out())
   */
  template<typename FormatContext>
  [[nodiscard]] auto format(const afft::Target& target, FormatContext& ctx) const
    -> decltype(ctx.out())
  {
    afft::detail::validate(target);

    auto getName = [](afft::Target target) -> const char*
    {
      switch (target)
      {
      case afft::Target::cpu:
        return "cpu";
      case afft::Target::cuda:
        return "cuda";
      case afft::Target::hip:
        return "hip";
      case afft::Target::opencl:
        return "opencl";
      default:
        afft::detail::cxx::unreachable();
      }
    };

    return std::format_to(ctx.out(), "{}", getName(target));
  }
};

/**********************************************************************************************************************/
// Transform
/**********************************************************************************************************************/
/// @brief Formatter for afft::Transform
AFFT_EXPORT template<>
struct std::formatter<afft::Transform> : afft::detail::ParseFormatterSkipper<>
{
  /**
   * @brief Format a transform.
   * @param[in] transform Transform
   * @param[in] ctx       Format context
   * @return decltype(ctx.out())
   */
  template<typename FormatContext>
  [[nodiscard]] auto format(const afft::Transform& transform, FormatContext& ctx) const
    -> decltype(ctx.out())
  {
    afft::detail::validate(transform);

    auto getName = [](afft::Transform transform) -> const char*
    {
      switch (transform)
      {
      case afft::Transform::dft:
        return "dft";
      case afft::Transform::dht:
        return "dht";
      case afft::Transform::dtt:
        return "dtt";
      default:
        afft::detail::cxx::unreachable();
      }
    };

    return std::format_to(ctx.out(), "{}", getName(transform));
  }
};

/// @brief Formatter for afft::Direction
AFFT_EXPORT template<>
struct std::formatter<afft::Direction> : afft::detail::ParseFormatterSkipper<>
{
  /**
   * @brief Format a direction.
   * @param[in] direction Direction
   * @param[in] ctx       Format context
   * @return decltype(ctx.out())
   */
  template<typename FormatContext>
  [[nodiscard]] auto format(const afft::Direction& direction, FormatContext& ctx) const
    -> decltype(ctx.out())
  {
    afft::detail::validate(direction);

    auto getName = [](afft::Direction direction) -> const char*
    {
      switch (direction)
      {
      case afft::Direction::forward:
        return "forward";
      case afft::Direction::inverse:
        return "inverse";
      default:
        afft::detail::cxx::unreachable();
      }
    };

    return std::format_to(ctx.out(), "{}", getName(direction));
  }
};

/// @brief Formatter for afft::Normalization
AFFT_EXPORT template<>
struct std::formatter<afft::Normalization> : afft::detail::ParseFormatterSkipper<>
{
  /**
   * @brief Format a normalization.
   * @param[in] normalization Normalization
   * @param[in] ctx           Format context
   * @return decltype(ctx.out())
   */
  template<typename FormatContext>
  [[nodiscard]] auto format(const afft::Normalization& normalization, FormatContext& ctx) const
    -> decltype(ctx.out())
  {
    afft::detail::validate(normalization);

    auto getName = [](afft::Normalization normalization) -> const char*
    {
      switch (normalization)
      {
      case afft::Normalization::none:
        return "none";
      case afft::Normalization::orthogonal:
        return "orthogonal";
      case afft::Normalization::unitary:
        return "unitary";
      default:
        afft::detail::cxx::unreachable();
      }
    };

    return std::format_to(ctx.out(), "{}", getName(normalization));
  }
};

/// @brief Formatter for afft::Placement
AFFT_EXPORT template<>
struct std::formatter<afft::Placement> : afft::detail::ParseFormatterSkipper<>
{
  /**
   * @brief Format a placement.
   * @param[in] placement Placement
   * @param[in] ctx       Format context
   * @return decltype(ctx.out())
   */
  template<typename FormatContext>
  [[nodiscard]] auto format(const afft::Placement& placement, FormatContext& ctx) const
    -> decltype(ctx.out())
  {
    afft::detail::validate(placement);

    auto getName = [](afft::Placement placement) -> const char*
    {
      switch (placement)
      {
      case afft::Placement::outOfPlace:
        return "outOfPlace";
      case afft::Placement::inPlace:
        return "inPlace";
      default:
        afft::detail::cxx::unreachable();
      }
    };

    return std::format_to(ctx.out(), "{}", getName(placement));
  }
};

/// @brief Formatter for afft::PrecisionTriad
AFFT_EXPORT template<>
struct std::formatter<afft::PrecisionTriad> : afft::detail::ParseFormatterSkipper<>
{
  /**
   * @brief Format a precision triad.
   * @param[in] triad Precision triad
   * @param[in] ctx   Format context
   * @return decltype(ctx.out())
   */
  template<typename FormatContext>
  [[nodiscard]] auto format(const afft::PrecisionTriad& triad, FormatContext& ctx) const
    -> decltype(ctx.out())
  {
    return std::format_to(ctx.out(), "{{{}, {}, {}}}", triad.execution, triad.source, triad.destination);
  }
};

/// @brief Formatter for afft::dft::Type
AFFT_EXPORT template<>
struct std::formatter<afft::dft::Type> : afft::detail::ParseFormatterSkipper<>
{
  /**
   * @brief Format a DFT type.
   * @param[in] type DFT type
   * @param[in] ctx  Format context
   * @return decltype(ctx.out())
   */
  template<typename FormatContext>
  [[nodiscard]] auto format(const afft::dft::Type& type, FormatContext& ctx) const
    -> decltype(ctx.out())
  {
    afft::detail::validate(type);

    auto getName = [](afft::dft::Type type) -> const char*
    {
      switch (type)
      {
      case afft::dft::Type::complexToComplex:
        return "c2c";
      case afft::dft::Type::realToComplex:
        return "r2c";
      case afft::dft::Type::complexToReal:
        return "c2r";
      default:
        afft::detail::cxx::unreachable();
      }
    };

    return std::format_to(ctx.out(), "{}", getName(type));
  }
};

/// @brief Formatter for afft::dht::Type
AFFT_EXPORT template<>
struct std::formatter<afft::dht::Type> : afft::detail::ParseFormatterSkipper<>
{
  /**
   * @brief Format a DHT type.
   * @param[in] type DHT type
   * @param[in] ctx  Format context
   * @return decltype(ctx.out())
   */
  template<typename FormatContext>
  [[nodiscard]] auto format(const afft::dht::Type& type, FormatContext& ctx) const
    -> decltype(ctx.out())
  {
    afft::detail::validate(type);

    auto getName = [](afft::dht::Type type) -> const char*
    {
      switch (type)
      {
      case afft::dht::Type::separable:
        return "separable";
      default:
        afft::detail::cxx::unreachable();
      }
    };

    return std::format_to(ctx.out(), "{}", getName(type));
  }
};

/// @brief Formatter for afft::dtt::Type
AFFT_EXPORT template<>
struct std::formatter<afft::dtt::Type> : afft::detail::ParseFormatterSkipper<>
{
  /**
   * @brief Format a DTT type.
   * @param[in] type DTT type
   * @param[in] ctx  Format context
   * @return decltype(ctx.out())
   */
  template<typename FormatContext>
  [[nodiscard]] auto format(const afft::dtt::Type& type, FormatContext& ctx) const
    -> decltype(ctx.out())
  {
    afft::detail::validate(type);

    auto getName = [](afft::dtt::Type type) -> const char*
    {
      switch (type)
      {
      case afft::dtt::Type::dct1:
        return "dct1";
      case afft::dtt::Type::dct2:
        return "dct2";
      case afft::dtt::Type::dct3:
        return "dct3";
      case afft::dtt::Type::dct4:
        return "dct4";
      case afft::dtt::Type::dct5:
        return "dct5";
      case afft::dtt::Type::dct6:
        return "dct6";
      case afft::dtt::Type::dct7:
        return "dct7";
      case afft::dtt::Type::dct8:
        return "dct8";
      case afft::dtt::Type::dst1:
        return "dst1";
      case afft::dtt::Type::dst2:
        return "dst2";
      case afft::dtt::Type::dst3:
        return "dst3";
      case afft::dtt::Type::dst4:
        return "dst4";
      case afft::dtt::Type::dst5:
        return "dst5";
      case afft::dtt::Type::dst6:
        return "dst6";
      case afft::dtt::Type::dst7:
        return "dst7";
      case afft::dtt::Type::dst8:
        return "dst8";
      default:
        afft::detail::cxx::unreachable();
      }
    };

    return std::format_to(ctx.out(), "{}", getName(type));
  }
};

/**********************************************************************************************************************/
// Type
/**********************************************************************************************************************/
/// @brief Formatter for afft::Precision
AFFT_EXPORT template<>
struct std::formatter<afft::Precision> : afft::detail::ParseFormatterSkipper<>
{
  /**
   * @brief Format a precision.
   * @param[in] precision Precision
   * @param[in] ctx       Format context
   * @return decltype(ctx.out())
   */
  template<typename FormatContext>
  [[nodiscard]] auto format(const afft::Precision& precision, FormatContext& ctx) const
    -> decltype(ctx.out())
  {
    afft::detail::validate(precision);

    auto getName = [](afft::Precision precision) -> const char*
    {
      switch (precision)
      {
      case afft::Precision::bf16:
        return "bf16";
      case afft::Precision::f16:
        return "f16";
      case afft::Precision::f32:
        return "f32";
      case afft::Precision::f64:
        return "f64";
      case afft::Precision::f80:
        return "f80";
      case afft::Precision::f64f64:
        return "f64f64";
      case afft::Precision::f128:
        return "f128";
      default:
        afft::detail::cxx::unreachable();
      }
    };

    return std::format_to(ctx.out(), "{}", getName(precision));
  }
};

/// @brief Formatter for afft::Complexity
AFFT_EXPORT template<>
struct std::formatter<afft::Complexity> : afft::detail::ParseFormatterSkipper<>
{
  /**
   * @brief Format a complexity.
   * @param[in] complexity Complexity
   * @param[in] ctx        Format context
   * @return decltype(ctx.out())
   */
  template<typename FormatContext>
  [[nodiscard]] auto format(const afft::Complexity& complexity, FormatContext& ctx) const
    -> decltype(ctx.out())
  {
    afft::detail::validate(complexity);

    auto getName = [](afft::Complexity complexity) -> const char*
    {
      switch (complexity)
      {
      case afft::Complexity::real:
        return "real";
      case afft::Complexity::complex:
        return "complex";
      default:
        afft::detail::cxx::unreachable();
      }
    };

    return std::format_to(ctx.out(), "{}", getName(complexity));
  }
};

/// @brief Formatter for afft::TypeProperties
AFFT_EXPORT template<typename T>
struct std::formatter<afft::TypeProperties<T>> : afft::detail::ParseFormatterSkipper<>
{
  /**
   * @brief Format a type properties.
   * @param[in] properties Type properties
   * @param[in] ctx        Format context
   * @return decltype(ctx.out())
   */
  template<typename FormatContext>
  [[nodiscard]] auto format(const afft::TypeProperties<T>& properties, FormatContext& ctx) const
    -> decltype(ctx.out())
  {
    return std::format_to(ctx.out(), "{{{}, {}}}", properties.precision, properties.complexity);
  }
};

/**********************************************************************************************************************/
// Version
/**********************************************************************************************************************/
/// @brief Formatter for afft::Version
AFFT_EXPORT template<>
struct std::formatter<afft::Version> : afft::detail::ParseFormatterSkipper<>
{
  /**
   * @brief Format a version.
   * @param[in] version Version
   * @param[in] ctx     Format context
   * @return decltype(ctx.out())
   */
  template<typename FormatContext>
  [[nodiscard]] auto format(const afft::Version& version, FormatContext& ctx) const
    -> decltype(ctx.out())
  {
    return std::format_to(ctx.out(), "{}.{}.{}", version.major, version.minor, version.patch);
  }
};

#endif /* AFFT_FORMATTERS_HPP */
