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

#ifndef AFFT_DETAIL_PLAN_IMPL_HPP
#define AFFT_DETAIL_PLAN_IMPL_HPP

#include <cstddef>
#include <variant>

#include "Config.hpp"
#include "../gpu.hpp"

namespace afft::detail
{
  /// @brief A variant type for the source and destination buffers
  using ExecParam = std::variant<void*, PlannarComplex<void>>;

  /**
   * @class PlanImpl
   * @brief The abstract base class for all plan implementations
   */
  class PlanImpl
  {
    public:
      /// @brief The default constructor is deleted
      PlanImpl() = delete;

      /// @brief The copy constructor is deleted
      PlanImpl(const PlanImpl&) = delete;

      /// @brief The move constructor is defaulted
      PlanImpl(PlanImpl&&) = default;

      /// @brief The destructor is defaulted
      virtual ~PlanImpl() = default;

      /// @brief The copy assignment operator is deleted
      PlanImpl& operator=(const PlanImpl&) = delete;

      /// @brief The move assignment operator is defaulted
      PlanImpl& operator=(PlanImpl&&) = default;

      /**
       * @brief Get the configuration of the plan
       * @return const reference to the configuration of the plan
       */
      const Config& getConfig() const noexcept
      {
        return mConfig;
      }
      
      /**
       * @brief Execute the plan
       * @param src the source buffer
       * @param dst the destination buffer
       */
      virtual void execute(ExecParam src, ExecParam dst) = 0;

      /**
       * @brief Execute the plan
       * @param src the source buffer
       * @param dst the destination buffer
       * @param execParams the GPU execution parameters
       */
      virtual void execute(ExecParam src, ExecParam dst, const afft::gpu::ExecutionParameters& execParams) = 0;

      /**
       * @brief Get the size of the workspace required for the plan
       * @return std::size_t the size of the workspace in bytes
       */
      virtual std::size_t getWorkspaceSize() const { return {}; }
    protected:
      /**
       * @brief Construct a new PlanImpl object
       * @param config the configuration of the plan
       */
      PlanImpl(const Config& config) noexcept
      : mConfig(config)
      {}

    private:
      Config mConfig; ///< The configuration of the plan
  };
};

#endif /* AFFT_DETAIL_PLAN_IMPL_HPP */
