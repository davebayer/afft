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

#ifndef AFFT_DETAIL_POCKETFFT_PLAN_HPP
#define AFFT_DETAIL_POCKETFFT_PLAN_HPP

#ifndef AFFT_TOP_LEVEL_INCLUDE
# include "../include.hpp"
#endif

#include "common.hpp"
#include "../../Plan.hpp"

namespace afft::detail::pocketfft
{
  /// @brief The pocketfft plan implementation base class.
  class Plan : public afft::Plan
  {
    private:
      /// @brief Alias for the parent class.
      using Parent = afft::Plan;

    public:
      /// @brief Inherit constructor.
      using Parent::Parent;

      /// @brief Inherit assignment operator.
      using Parent::operator=;

      /// @brief Default destructor.
      virtual ~Plan() = default;

      /**
       * @brief Get the backend.
       * @return The backend.
       */
      [[nodiscard]] Backend getBackend() const noexcept override
      {
        return Backend::pocketfft;
      }
    protected:
      /**
       * @brief Get the pocketfft direction from the plan description.
       * @return The pocketfft direction.
       */
      [[nodiscard]] auto getDirection() const noexcept
      {
        switch (mDesc.getDirection())
        {
        case Direction::forward:
          return ::pocketfft::FORWARD;
        case Direction::backward:
          return ::pocketfft::BACKWARD;
        default:
          cxx::unreachable();
        }
      }

      /**
       * @brief Get the thread count from the plan description.
       * @return The thread count.
       */
      [[nodiscard]] constexpr std::size_t getThreadCount() const
      {
        return static_cast<std::size_t>(mDesc.template getArchDesc<Target::cpu, Distribution::spst>().threadLimit);
      }
  };
} // namespace afft::detail::pocketfft

#endif /* AFFT_DETAIL_POCKETFFT_PLAN_HPP */
