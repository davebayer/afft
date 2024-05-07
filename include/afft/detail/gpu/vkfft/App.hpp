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

#ifndef AFFT_DETAIL_GPU_VKFFT_APP_HPP
#define AFFT_DETAIL_GPU_VKFFT_APP_HPP

#include <utility>

#include "include.hpp"
#include "error.hpp"

namespace afft::detail::gpu::vkfft
{
  /**
   * @class App
   * @brief Wrapper for VkFFTApplication
   */
  class App
  {
    public:
      /// @brief Default constructor
      constexpr App() noexcept = default;

      /**
       * @brief Constructor
       * @param config VkFFTConfiguration
       */
      App(VkFFTConfiguration&& config)
      {
        init(std::move(config));
      }

      /// @brief Copy constructor is not allowed
      constexpr App(const App&) noexcept = delete;

      /// @brief Move constructor
      constexpr App(App&&) noexcept = default;

      /// @brief Destructor
      ~App() noexcept
      {
        if (isValid())
        {
          deleteVkFFT(&mApp);
        }
      }

      /// @brief Copy assignment operator is not allowed
      App& operator=(const App&) = delete;

      /// @brief Move assignment operator
      App& operator=(App&&) = default;

      /**
       * @brief Initialize VkFFTApplication
       * @param config VkFFTConfiguration
       */
      void init(VkFFTConfiguration&& config)
      {
        if (isValid())
        {
          deleteVkFFT(&mApp);
        }

        Error::check(initializeVkFFT(&mApp, std::move(config)));
        mValid = true;
      }

      /**
       * @brief Check if VkFFTApplication is valid
       * @return bool
       */
      constexpr bool isValid() const noexcept
      {
        return mValid;
      }

      /**
       * @brief Get VkFFTApplication
       * @return VkFFTApplication&
       */
      constexpr VkFFTApplication& get() noexcept
      {
        return mApp;
      }

      /**
       * @brief Get VkFFTApplication
       * @return const VkFFTApplication&
       */
      constexpr const VkFFTApplication& get() const noexcept
      {
        return mApp;
      }
    protected:
    private:
      bool             mValid{}; ///< Flag to check if VkFFTApplication is valid
      VkFFTApplication mApp{};   ///< VkFFTApplication
  };
} // namespace afft::detail::gpu::vkfft

#endif /* AFFT_DETAIL_GPU_VKFFT_APP_HPP */
