#ifndef AFFT_DETAIL_TARGET_CONFIG_HPP
#define AFFT_DETAIL_TARGET_CONFIG_HPP

#include <thread>
#include <variant>

#include "common.hpp"
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
    std::size_t alignment{};   ///< Memory alignment.
    unsigned    threadLimit{}; ///< Number of threads.
  };

  /**
   * @struct GpuConfig
   * @brief GPU configuration.
   */
  struct GpuConfig
  {
# if AFFT_GPU_BACKEND_IS_CUDA
    int device{};             ///< CUDA device.
# elif AFFT_GPU_BACKEND_IS_HIP
    int device{};             ///< HIP device.
# endif

    bool externalWorkspace{}; ///< Use external workspace.
  };

  /**
   * @struct TargetConfig
   * @brief Target configuration.
   */
  class TargetConfig
  {
    public:
      /**
       * @brief Make a CPU target configuration.
       * @param cpuParams CPU parameters.
       * @return CPU target configuration.
       */
      static TargetConfig make(const afft::cpu::Parameters& cpuParams)
      {
        CpuConfig config{.alignment   = cpuParams.alignment,
                         .threadLimit = std::min(cpuParams.threadLimit, std::thread::hardware_concurrency())};

        return TargetConfig{std::move(config)};
      }

      /**
       * @brief Make a GPU target configuration.
       * @param gpuParams GPU parameters.
       * @return GPU target configuration.
       */
      TargetConfig make(const afft::gpu::Parameters& gpuParams)
      {
        GpuConfig config{.externalWorkspace = gpuParams.externalWorkspace};

        {
#       if AFFT_GPU_BACKEND_IS_CUDA
          if (!gpu::cuda::isValidDevice(gpuParams.device))
          {
            throw makeException<std::runtime_error>("Invalid CUDA device");
          }

          config.device = gpuParams.device;
#       elif AFFT_GPU_BACKEND_IS_HIP
          if (!gpu::hip::isValidDevice(gpuParams.device))
          {
            throw makeException<std::runtime_error>("Invalid HIP device");
          }

          config.device = gpuParams.device;
#       else
          throw makeException<std::runtime_error>("Invalid GPU backend");
#       endif
        }

        return TargetConfig{std::move(config)};
      }

      /// @brief Default constructor not allowed.
      TargetConfig() = delete;

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
      [[nodiscard]] constexpr const auto& getConfig() noexcept
      {
        if constexpr (target == Target::cpu)
        {
          return std::get<CpuConfig>(mVariant);
        }
        else if constexpr (target == Target::gpu)
        {
          return std::get<GpuConfig>(mVariant);
        }
      }

      /// @brief Equality operator.
      friend bool operator==(const TargetConfig&, const TargetConfig&) noexcept = default;

      /// @brief Inequality operator.
      friend bool operator!=(const TargetConfig&, const TargetConfig&) noexcept = default;
    protected:
    private:
      using ConfigVariant = std::variant<CpuConfig, GpuConfig>;

      static_assert(variant_alternative_index<ConfigVariant, CpuConfig>() == to_underlying(Target::cpu));
      static_assert(variant_alternative_index<ConfigVariant, GpuConfig>() == to_underlying(Target::gpu));

      /**
       * @brief Constructor.
       * @param config Target configuration.
       */
      TargetConfig(auto&& config)
      : mVariant{std::forward<decltype(config)>(config)}
      {}

      ConfigVariant mVariant; ///< Target configuration.
  };
} // namespace afft::detail

#endif /* AFFT_DETAIL_TARGET_CONFIG_HPP */
