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

#ifndef AFFT_DETAIL_PLAN_HPP
#define AFFT_DETAIL_PLAN_HPP

#ifndef AFFT_TOP_LEVEL_INCLUDE
# include "../include.hpp"
#endif

#include "../Plan.hpp"

namespace afft::detail
{
  /**
   * @brief Backend data implementation.
   * @tparam mpBackend Multi-process backend.
   * @tparam target Target.
   */
  template<MpBackend mpBackend, Target target>
  class BackendData
  {
    private:
      /// @brief Alias for the backend parameters
      using BackendParameters = afft::BackendParameters<mpBackend, target>;

    public:
      /// @brief Default constructor
      BackendData() = default;

      /**
       * @brief Constructor
       * @param backendParams Backend parameters.
       */
      BackendData(const BackendParameters& backendParams)
      : mBackendParams{backendParams}
      {}

      /// @brief Copy constructor
      BackendData(const BackendData&) = default;

      /// @brief Move constructor
      BackendData(BackendData&&) = default;

      /// @brief Destructor
      virtual ~BackendData() = default;

      /// @brief Copy assignment operator
      BackendData& operator=(const BackendData&) = default;

      /// @brief Move assignment operator
      BackendData& operator=(BackendData&&) = default;

    protected:
      BackendParameters mBackendParams{}; ///< The backend parameters
  };

  /**
   * @brief The plan implementation base class.
   * @tparam mpBackend Multi-process backend.
   * @tparam target Target.
   */
  template<MpBackend mpBackend, Target target>
  class Plan : public afft::Plan, protected BackendData<mpBackend, target>
  {
    private:
      /// @brief Alias for the backend data implementation
      using BackendData = detail::BackendData<mpBackend, target>;

    public:
      /// @brief Default constructor
      Plan() = delete;

      /**
       * @brief Constructor
       * @param desc Plan description.
       * @param backendParams Backend parameters.
       */
      Plan(const Description& desc, const BackendParameters<mpBackend, target>& backendParams)
      : afft::Plan{desc},
        BackendData{backendParams}
      {} 

      /**
       * @brief Copy constructor
       * @param other Other plan.
       */
      Plan(const Plan&) = delete;

      /**
       * @brief Move constructor
       * @param other Other plan.
       */
      Plan(Plan&&) = default;

      /// @brief Destructor
      virtual ~Plan() = default;

      /**
       * @brief Copy assignment operator
       * @param other Other plan.
       * @return Reference to this plan.
       */
      Plan& operator=(const Plan&) = delete;

      /**
       * @brief Move assignment operator
       * @param other Other plan.
       * @return Reference to this plan.
       */
      Plan& operator=(Plan&&) = default;

      /**
       * @brief Get the backend parameters implementation.
       * @return Backend parameters implementation.
       */
      [[nodiscard]] const void* getBackendParametersImpl() const noexcept override
      {
        return std::addressof(BackendData::mBackendParams);
      }

      /**
       * @brief Get the backend parameters variant.
       * @return Backend parameters variant.
       */
      [[nodiscard]] BackendParametersVariant getBackendParametersVariant() const noexcept override
      {
        return BackendData::mBackendParams;
      }
  };
} // namespace afft::detail

#endif /* AFFT_DETAIL_PLAN_HPP */
