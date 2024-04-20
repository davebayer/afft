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

#ifndef AFFT_DETAIL_COMMON_HPP
#define AFFT_DETAIL_COMMON_HPP

#include <array>

#include "../common.hpp"
#include "../cpu.hpp"
#include "../gpu.hpp"

namespace afft::detail
{
  /**
   * @brief std::array with size equal to the maximum number of dimensions.
   * @tparam T Type of the elements.
   */
  template<typename T>
  using MaxDimArray = std::array<T, maxDimCount>;

  /**
   * @brief Returns true if the given precision is valid.
   * @param prec Precision to check.
   * @return True if the precision is valid, false otherwise.
   */
  [[nodiscard]] constexpr bool isValidPrecision(Precision prec) noexcept
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
  
  /**
   * @brief Returns true if the given complexity is valid.
   * @param cmpl Complexity to check.
   * @return True if the complexity is valid, false otherwise.
   */
  [[nodiscard]] constexpr bool isValidComplexity(Complexity cmpl) noexcept
  {
    switch (cmpl)
    {
    case Complexity::real:
    case Complexity::complex:
      return true;
    default:
      return false;
    }
  }

  /**
   * @brief Returns true if the given complex format is valid.
   * @param format Complex format to check.
   * @return True if the complex format is valid, false otherwise.
   */
  [[nodiscard]] constexpr bool isValidComplexFormat(ComplexFormat format) noexcept
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

  /**
   * @brief Returns true if the given direction is valid.
   * @param dir Direction to check.
   * @return True if the direction is valid, false otherwise.
   */
  [[nodiscard]] constexpr bool isValidDirection(Direction dir) noexcept
  {
    switch (dir)
    {
    case Direction::forward:
    case Direction::inverse:
      return true;
    default:
      return false;
    }
  }

  /**
   * @brief Returns true if the given target is valid.
   * @param target Target to check.
   * @return True if the target is valid, false otherwise.
   */
  [[nodiscard]] constexpr bool isValidTarget(Target target) noexcept
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

  /**
   * @brief Returns true if the given backend select strategy is valid.
   * @param strategy Backend select strategy to check.
   * @return True if the backend select strategy is valid, false otherwise.
   */
  [[nodiscard]] constexpr bool isValidBackendSelectStrategy(BackendSelectStrategy strategy) noexcept
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

  /**
   * @brief Returns true if the given placement is valid.
   * @param place Placement to check.
   * @return True if the placement is valid, false otherwise.
   */
  [[nodiscard]] constexpr bool isValidPlacement(Placement place) noexcept
  {
    switch (place)
    {
    case Placement::inPlace:
    case Placement::outOfPlace:
      return true;
    default:
      return false;
    }
  }

  /**
   * @brief Returns true if the given transform is valid.
   * @param transform Transform to check.
   * @return True if the transform is valid, false otherwise.
   */
  [[nodiscard]] constexpr bool isValidTransform(Transform transform) noexcept
  {
    switch (transform)
    {
    case Transform::dft:
    case Transform::dtt:
      return true;
    default:
      return false;
    }
  }

  /**
   * @brief Returns true if the given init effort is valid.
   * @param effort Init effort to check.
   * @return True if the init effort is valid, false otherwise.
   */
  [[nodiscard]] constexpr bool isValidInitEffort(InitEffort effort) noexcept
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

  /**
   * @brief Returns true if the given normalize is valid.
   * @param normalize Normalize to check.
   * @return True if the normalize is valid, false otherwise.
   */
  [[nodiscard]] constexpr bool isValidNormalize(Normalize normalize) noexcept
  {
    switch (normalize)
    {
    case Normalize::none:
    case Normalize::orthogonal:
    case Normalize::unitary:
      return true;
    default:
      return false;
    }
  }

  /**
   * @brief Returns true if the given workspace policy is valid.
   * @param policy Workspace policy to check.
   * @return True if the workspace policy is valid, false otherwise.
   */
  [[nodiscard]] constexpr bool isValidWorkspacePolicy(WorkspacePolicy policy) noexcept
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

  /**
   * @brief Returns true if the given DFT type is valid.
   * @param dftType DFT type to check.
   * @return True if the DFT type is valid, false otherwise.
   */
  [[nodiscard]] constexpr bool isValidDftType(dft::Type dftType) noexcept
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

  /**
   * @brief Returns true if the given DTT type is valid.
   * @param dttType DTT type to check.
   * @return True if the DTT type is valid, false otherwise.
   */
  [[nodiscard]] constexpr bool isValidDttType(dtt::Type dttType) noexcept
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
} // namespace afft::detail

#endif /* AFFT_DETAIL_COMMON_HPP */
