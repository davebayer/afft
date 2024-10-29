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

#ifndef AFFT_SELECT_HPP
#define AFFT_SELECT_HPP

#ifndef AFFT_TOP_LEVEL_INCLUDE
# include <afft/detail/include.hpp>
#endif

#include <afft/backend.hpp>

namespace afft
{
  /// @brief Backend select strategy
  enum class SelectStrategy : ::afft_SelectStrategy
  {
    first    = afft_SelectStrategy_first,   ///< Select the first available backend
    best     = afft_SelectStrategy_best,    ///< Select the best available backend
    _default = afft_SelectStrategy_default, ///< Default select strategy
  };

  /**
   * @brief Backend select strategy constant.
   * @tparam _selectStrategy Select strategy.
   */
  template<SelectStrategy _selectStrategy>
  struct SelectStrategyConstant
  {
    static constexpr SelectStrategy selectStratgy = _selectStrategy; ///< Select strategy
  };

  /// @brief Select parameters for selecting first backend supporting the transform
  struct FirstSelectParameters : SelectStrategyConstant<SelectStrategy::first>
  {
    BackendMask    mask{BackendMask::all}; ///< Backend mask
    const Backend* order{};                ///< Backend initialization order
    std::size_t    orderSize{};            ///< Backend initialization order size
  };

  /// @brief Select parameters for selecting best of all the backends supporting the transform
  struct BestSelectParameters : SelectStrategyConstant<SelectStrategy::best>
  {
    BackendMask                   mask{BackendMask::all};   ///< Backend mask
    std::chrono::duration<double> destructiveTimePenalty{}; ///< Time penalty for destructive backends
  };
  
  /// @brief Default select parameters
  using DefaultSelectParameters = FirstSelectParameters;

  /// @brief Select parameters variant
  using SelectParametersVariant = std::variant<std::monostate, FirstSelectParameters, BestSelectParameters>;
} // namespace afft

#endif /* AFFT_SELECT_HPP */
