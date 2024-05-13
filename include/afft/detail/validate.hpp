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

#ifndef AFFT_DETAIL_VALIDATE_HPP
#define AFFT_DETAIL_VALIDATE_HPP

#include <stdexcept>
#include <typeinfo>

#include "cxx.hpp"
#include "../backend.hpp"
#include "../common.hpp"
#include "../distrib.hpp"
#include "../Span.hpp"

namespace afft::detail
{
  /**
   * @brief Validator class template declaration. Specializations must provide a call operator that takes a value of type T and returns true if the value is valid, false otherwise.
   * @tparam T Type of the value to validate.
   */
  template<typename T>
  struct Validator;

  /// @brief Validator for the Backend enum class.
  template<>
  struct Validator<Backend>
  {
    constexpr bool operator()(Backend backend) const noexcept
    {
      switch (backend)
      {
      case Backend::clfft:
      case Backend::cufft:
      case Backend::fftw3:
      case Backend::hipfft:
      case Backend::mkl:
      case Backend::pocketfft:
      case Backend::rocfft:
      case Backend::vkfft:
        return true;
      default:
        return false;
      }
    }
  };

  /// @brief Validator for the BackendSelectStrategy enum class.
  template<>
  struct Validator<BackendSelectStrategy>
  {
    constexpr bool operator()(BackendSelectStrategy strategy) const noexcept
    {
      switch (strategy)
      {
      case BackendSelectStrategy::first:
      case BackendSelectStrategy::best:
        return true;
      default:
        return false;
      }
    }
  };

  /// @brief Validator for the Precision enum class.
  template<>
  struct Validator<Precision>
  {
    constexpr bool operator()(Precision prec) const noexcept
    {
      switch (prec)
      {
      case Precision::bf16:
      case Precision::f16:
      case Precision::f32:
      case Precision::f64:
      case Precision::f64f64:
      case Precision::f80:
      case Precision::f128:
        return true;
      default:
        return false;
      }
    }
  };

  /// @brief Validator for the Alignment enum class.
  template<>
  struct Validator<Alignment>
  {
    constexpr bool operator()(Alignment alignment) const noexcept
    {
      return cxx::has_single_bit(cxx::to_underlying(alignment));
    }
  };

  /// @brief Validator for the Complexity enum class.
  template<>
  struct Validator<Complexity>
  {
    constexpr bool operator()(Complexity complexity) const noexcept
    {
      switch (complexity)
      {
      case Complexity::real:
      case Complexity::complex:
        return true;
      default:
        return false;
      }
    }
  };

  /// @brief Validator for the ComplexFormat enum class.
  template<>
  struct Validator<ComplexFormat>
  {
    constexpr bool operator()(ComplexFormat format) const noexcept
    {
      switch (format)
      {
      case ComplexFormat::interleaved:
      case ComplexFormat::planar:
        return true;
      default:
        return false;
      }
    }
  };

  /// @brief Validator for the Direction enum class.
  template<>
  struct Validator<Direction>
  {
    constexpr bool operator()(Direction direction) const noexcept
    {
      switch (direction)
      {
      case Direction::forward:
      case Direction::inverse:
        return true;
      default:
        return false;
      }
    }
  };

  /// @brief Validator for the Placement enum class.
  template<>
  struct Validator<Placement>
  {
    constexpr bool operator()(Placement placement) const noexcept
    {
      switch (placement)
      {
      case Placement::inPlace:
      case Placement::outOfPlace:
        return true;
      default:
        return false;
      }
    }
  };

  /// @brief Validator for the Transform enum class.
  template<>
  struct Validator<Transform>
  {
    constexpr bool operator()(Transform transform) const noexcept
    {
      switch (transform)
      {
      case Transform::dft:
      case Transform::dht:
      case Transform::dtt:
        return true;
      default:
        return false;
      }
    }
  };

  /// @brief Validator for the Target enum class.
  template<>
  struct Validator<Target>
  {
    constexpr bool operator()(Target target) const noexcept
    {
      switch (target)
      {
      case Target::cpu:
      case Target::gpu:
        return true;
      default:
        return false;
      }
    }
  };

  /// @brief Validator for the InitEffort enum class.
  template<>
  struct Validator<InitEffort>
  {
    constexpr bool operator()(InitEffort effort) const noexcept
    {
      switch (effort)
      {
      case InitEffort::low:
      case InitEffort::med:
      case InitEffort::high:
      case InitEffort::max:
        return true;
      default:
        return false;
      }
    }
  };

  /// @brief Validator for the Normalization enum class.
  template<>
  struct Validator<Normalization>
  {
    constexpr bool operator()(Normalization normalization) const noexcept
    {
      switch (normalization)
      {
      case Normalization::none:
      case Normalization::orthogonal:
      case Normalization::unitary:
        return true;
      default:
        return false;
      }
    }
  };

  /// @brief Validator for the WorkspacePolicy enum class.
  template<>
  struct Validator<WorkspacePolicy>
  {
    constexpr bool operator()(WorkspacePolicy policy) const noexcept
    {
      switch (policy)
      {
      case WorkspacePolicy::minimal:
      case WorkspacePolicy::performance:
        return true;
      default:
        return false;
      }
    }
  };

  /// @brief Validator for the PrecisionTriad struct.
  template<>
  struct Validator<PrecisionTriad>
  {
    constexpr bool operator()(const PrecisionTriad& triad) const noexcept
    {
      return Validator<Precision>{}(triad.execution) &&
             Validator<Precision>{}(triad.source) &&
             Validator<Precision>{}(triad.destination);
    }
  };

  /// @brief Validator for the dft::Type enum class.
  template<>
  struct Validator<dft::Type>
  {
    constexpr bool operator()(dft::Type dftType) const noexcept
    {
      switch (dftType)
      {
      case dft::Type::complexToComplex:
      case dft::Type::realToComplex:
      case dft::Type::complexToReal:
        return true;
      default:
        return false;
      }
    }
  };

  /// @brief Validator for the dtt::Type enum class.
  template<>
  struct Validator<dtt::Type>
  {
    constexpr bool operator()(dtt::Type dttType) const noexcept
    {
      switch (dttType)
      {
      case dtt::Type::dct1:
      case dtt::Type::dct2:
      case dtt::Type::dct3:
      case dtt::Type::dct4:
      case dtt::Type::dst1:
      case dtt::Type::dst2:
      case dtt::Type::dst3:
      case dtt::Type::dst4:
        return true;
      default:
        return false;
      }
    }
  };

  /// @brief Validator for the distrib::Type enum class.
  template<>
  struct Validator<distrib::Type>
  {
    constexpr bool operator()(distrib::Type distribType) const noexcept
    {
      switch (distribType)
      {
      case distrib::Type::spst:
      case distrib::Type::spmt:
      case distrib::Type::mpst:
        return true;
      default:
        return false;
      }
    }
  };

  /**
   * @brief Validates a value.
   * @tparam T Type of the value.
   * @param value Value to validate.
   */
  template<typename T>
  constexpr void validate(const T& value)
  {
    if (!Validator<T>{}(value))
    {
      throw std::invalid_argument("Invalid value of type " + typeid(T).name());
    }
  }

  /**
   * @brief Validates a span of values.
   * @tparam T Type of the values.
   * @param span Span of values to validate.
   */
  template<typename T>
  constexpr void validate(const Span<T>& span)
  {
    for (const auto& value : span)
    {
      validate(value);
    }
  }
} // namespace afft::detail

#endif /* AFFT_DETAIL_VALIDATE_HPP */
