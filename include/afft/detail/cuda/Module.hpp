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

#ifndef AFFT_DETAIL_CUDA_MODULE_HPP
#define AFFT_DETAIL_CUDA_MODULE_HPP

#ifndef AFFT_TOP_LEVEL_INCLUDE
# include "../include.hpp"
#endif

#include "error.hpp"
#include "rtc/rtc.hpp"

namespace afft::detail::cuda
{
  /**
   * @class Module
   * @brief A wrapper around a CUDA module.
   */
  class Module
  {
    public:
      /// @brief Default constructor.
      Module() noexcept = default;

      /**
       * @brief Constructor.
       * @param code The CUDA code to load.
       */
      explicit Module(const rtc::Code& code)
      {
        load(code);
      }

      /// @brief Copy constructor.
      Module(const Module&) noexcept = default;

      /// @brief Move constructor.
      Module(Module&&) noexcept = default;

      /// @brief Destructor.
      ~Module() = default;

      /// @brief Copy assignment operator.
      Module& operator=(const Module&) noexcept = default;

      /// @brief Move assignment operator.
      Module& operator=(Module&& other) noexcept = default;

      /**
       * @brief Checks if the module is valid.
       * @return True if the module is valid, false otherwise.
       */
      [[nodiscard]] bool valid() const noexcept
      {
        return (mModule != nullptr);
      }

      /**
       * @brief Loads the CUDA code.
       * @param code The CUDA code to load.
       */
      void load(const rtc::Code& code)
      {
        CUmodule mod{};

        checkError(cuModuleLoadData(&mod, code.data()));

        mModule.reset(mod, Deleter{});
      }

      /**
       * @brief Gets a global variable from the module.
       * @param name The name of the global variable.
       * @return A tuple containing the device pointer and the size of the global variable.
       */
      [[nodiscard]] std::tuple<CUdeviceptr, std::size_t> getGlobal(const rtc::CSymbolName& name) const
      {
        requireValid();

        CUdeviceptr ptr{};
        std::size_t size{};

        checkError(cuModuleGetGlobal(&ptr, &size, mModule.get(), name.data()));

        return std::make_tuple(ptr, size);
      }

      /**
       * @brief Gets a global variable from the module.
       * @param name The name of the global variable.
       * @return A tuple containing the device pointer and the size of the global variable.
       */
      [[nodiscard]] std::tuple<CUdeviceptr, std::size_t> getGlobal(const rtc::CppLoweredSymbolName& name) const
      {
        requireValid();

        CUdeviceptr ptr{};
        std::size_t size{};

        checkError(cuModuleGetGlobal(&ptr, &size, mModule.get(), name.data()));

        return std::make_tuple(ptr, size);
      }

      /**
       * @brief Gets a function from the module.
       * @param name The name of the function.
       * @return The function.
       */
      [[nodiscard]] CUfunction getFunction(const rtc::CSymbolName& name) const
      {
        requireValid();

        CUfunction function{};

        checkError(cuModuleGetFunction(&function, mModule.get(), name.data()));

        return function;
      }

      /**
       * @brief Gets a function from the module.
       * @param name The name of the function.
       * @return The function.
       */
      [[nodiscard]] CUfunction getFunction(const rtc::CppLoweredSymbolName& name) const
      {
        requireValid();

        CUfunction function{};

        checkError(cuModuleGetFunction(&function, mModule.get(), name.data()));

        return function;
      }

      /**
       * @brief Gets the CUDA module.
       * @return The CUDA module.
       */
      [[nodiscard]] operator CUmodule() const
      {
        return mModule.get();
      }
    private:
      /**
       * @struct Deleter
       * @brief A custom deleter for a CUDA module.
       */
      struct Deleter
      {
        /**
         * @brief Deletes the CUDA module.
         * @param mod The CUDA module to delete.
         */
        void operator()(CUmodule mod) const
        {
          if (mod != nullptr)
          {
            cuModuleUnload(mod);
          }
        }
      };

      /**
       * @brief Checks if the module is valid and throws an exception if it is not.
       */
      void requireValid() const
      {
        if (!valid())
        {
          throw makeException<std::runtime_error>("Invalid CUDA module.");
        }
      }

      std::shared_ptr<std::remove_pointer_t<CUmodule>> mModule{}; ///< The CUDA module.
  };
} // afft::detail::cuda

#endif /* AFFT_DETAIL_CUDA_MODULE_HPP */
