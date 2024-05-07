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

#ifndef AFFT_DETAIL_TARGET_CONFIG_HPP
#define AFFT_DETAIL_TARGET_CONFIG_HPP

#include <thread>
#include <variant>

#include "common.hpp"
#include "cxx.hpp"
#include "error.hpp"
#include "../cpu.hpp"
#include "../gpu.hpp"

namespace afft::detail
{
  /**
   * @struct CpuConfig
   * @brief CPU configuration.
   */
  struct CpuConfig
  {
    Alignment alignment{};   ///< Memory alignment.
    unsigned  threadLimit{}; ///< Number of threads.

    /// @brief Equality operator.
    friend bool operator==(const CpuConfig&, const CpuConfig&) noexcept = default;
  };

  /**
   * @struct GpuConfig
   * @brief GPU configuration.
   */
  struct GpuConfig
  {
# if AFFT_GPU_FRAMEWORK_IS_CUDA
    int          device{};    ///< CUDA device.
# elif AFFT_GPU_FRAMEWORK_IS_HIP
    int          device{};    ///< HIP device.
# elif AFFT_GPU_FRAMEWORK_IS_OPENCL
    cl_context   context{};   ///< OpenCL context.
    cl_device_id device{};    ///< OpenCL device.
# endif

    bool externalWorkspace{}; ///< Use external workspace.

    /// @brief Equality operator.
    friend bool operator==(const GpuConfig&, const GpuConfig&) noexcept = default;
  };

  /**
   * @struct TargetConfig
   * @brief Target configuration.
   */
  class TargetConfig
  {
    public:
      /// @brief Default constructor not allowed.
      TargetConfig() = delete;

      /**
       * @brief Constructor.
       * @param cpuParams CPU parameters.
       */
      TargetConfig(const TargetParametersType auto& targetParams)
      : mVariant{makeTargetVariant(targetParams)}
      {}

      /// @brief Copy constructor.
      TargetConfig(const TargetConfig&) = default;

      /// @brief Move constructor.
      TargetConfig(TargetConfig&&) = default;

      /// @brief Destructor.
      ~TargetConfig() = default;

      /// @brief Copy assignment operator.
      TargetConfig& operator=(const TargetConfig&) = default;

      /// @brief Move assignment operator.
      TargetConfig& operator=(TargetConfig&&) = default;

      /**
       * @brief Get the target.
       * @return Target.
       */
      [[nodiscard]] constexpr Target getTarget() const noexcept
      {
        return static_cast<Target>(mVariant.index());
      }

      /**
       * @brief Get target configuration.
       * @tparam target Target.
       * @return Target configuration.
       */
      template<Target target>
      [[nodiscard]] constexpr const auto& getConfig() const
      {
        static_assert(isValidTarget(target), "Invalid target.");

        if constexpr (target == Target::cpu)
        {
          return std::get<CpuConfig>(mVariant);
        }
        else if constexpr (target == Target::gpu)
        {
          return std::get<GpuConfig>(mVariant);
        }
      }

      /**
       * @brief Get target parameters.
       * @tparam target Target.
       * @return Target parameters.
       */
      template<Target target>
      [[nodiscard]] constexpr TargetParameters<target> getParameters() const
      {
        static_assert(isValidTarget(target), "Invalid target.");

        if constexpr (target == Target::cpu)
        {
          const auto& cpuConfig = getConfig<Target::cpu>();

          return afft::cpu::Parameters{/* .alignment   = */ cpuConfig.alignment,
                                       /* .threadLimit = */ cpuConfig.threadLimit};
        }
        else if constexpr (target == Target::gpu)
        {
          const auto& gpuConfig = getConfig<Target::gpu>();

          return afft::gpu::Parameters
          {
#         if AFFT_GPU_BACKEND_IS_CUDA
            .device            = gpuConfig.device,
#         elif AFFT_GPU_BACKEND_IS_HIP
            .device            = gpuConfig.device,
#         endif
            .externalWorkspace = gpuConfig.externalWorkspace,
          };
        }
        else
        {
          cxx::unreachable();
        }
      }

      /// @brief Equality operator.
      friend bool operator==(const TargetConfig&, const TargetConfig&) noexcept = default;

      /// @brief Inequality operator.
      friend bool operator!=(const TargetConfig&, const TargetConfig&) noexcept = default;
    protected:
    private:
      using ConfigVariant = std::variant<CpuConfig, GpuConfig>;

      /**
       * @brief Make target configuration.
       * @param cpuParams CPU parameters.
       * @return Target configuration.
       */
      [[nodiscard]] static CpuConfig makeTargetVariant(const afft::cpu::Parameters& cpuParams)
      {
        return CpuConfig{/* .alignment   = */ cpuParams.alignment,
                         /* .threadLimit = */ std::min(cpuParams.threadLimit, std::thread::hardware_concurrency())};
      }

      /**
       * @brief Make target configuration.
       * @param gpuParams GPU parameters.
       * @return Target configuration.
       */
      [[nodiscard]] static GpuConfig makeTargetVariant(const afft::gpu::Parameters& gpuParams)
      {
        GpuConfig config{};
        config.externalWorkspace = gpuParams.externalWorkspace;

        {
#       if AFFT_GPU_FRAMEWORK_IS_CUDA
          if (!gpu::cuda::isValidDevice(gpuParams.device))
          {
            throw makeException<std::runtime_error>("Invalid CUDA device");
          }

          config.device = gpuParams.device;
#       elif AFFT_GPU_FRAMEWORK_IS_HIP
          if (!gpu::hip::isValidDevice(gpuParams.device))
          {
            throw makeException<std::runtime_error>("Invalid HIP device");
          }

          config.device = gpuParams.device;
#       elif AFFT_GPU_FRAMEWORK_IS_OPENCL
          if (config.context == nullptr)
          {
            throw makeException<std::runtime_error>("Invalid OpenCL context");
          }

          if (config.device == nullptr)
          {
            throw makeException<std::runtime_error>("Invalid OpenCL device");
          }

          config.context = gpuParams.context;
          config.device  = gpuParams.device;
#       else
          throw makeException<std::runtime_error>("Invalid GPU backend");
#       endif

          return config;
        }
      }

      ConfigVariant mVariant; ///< Target configuration.
  };
} // namespace afft::detail

#endif /* AFFT_DETAIL_TARGET_CONFIG_HPP */
