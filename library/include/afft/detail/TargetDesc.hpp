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

#ifndef AFFT_DETAIL_TARGET_DESC_HPP
#define AFFT_DETAIL_TARGET_DESC_HPP

#ifndef AFFT_TOP_LEVEL_INCLUDE
# include "include.hpp"
#endif

#include "common.hpp"
#include "validate.hpp"
#include "typeTraits.hpp"
#include "../target.hpp"

namespace afft::detail
{
  /// @brief Dummy target description
  struct DummyTargetDesc
  {
    /**
       * @brief Get the target.
       * @return Target.
       */
      [[nodiscard]] Target getTarget() const
      {
        throw Exception{Error::internal, "Dummy target descriptor does not have a target"};
      }

    /**
     * @brief Get the number of targets.
     * @return Number of targets.
     */
    [[nodiscard]] std::size_t getTargetCount() const
    {
      throw Exception{Error::internal, "Dummy target descriptor does not have a target count"};
    }
    
    /**
     * @brief Equality operator.
     * @param[in] lhs Left-hand side.
     * @param[in] rhs Right-hand side.
     * @return True if equal, false otherwise.
     */
    [[nodiscard]] constexpr friend bool operator==(const DummyTargetDesc&, const DummyTargetDesc&) noexcept
    {
      return true;
    }

    /**
     * @brief Inequality operator.
     * @param[in] lhs Left-hand side.
     * @param[in] rhs Right-hand side.
     * @return True if not equal, false otherwise.
     */
    [[nodiscard]] constexpr friend bool operator!=(const DummyTargetDesc&, const DummyTargetDesc&) noexcept
    {
      return false;
    }
  };

#ifdef AFFT_ENABLE_CPU
  /// @brief CPU description
  class CpuDesc
  {
    public:
      /**
       * @brief Get the target.
       * @return Target.
       */
      [[nodiscard]] constexpr Target getTarget() const noexcept
      {
        return Target::cpu;
      }

      /**
       * @brief Get the number of targets.
       * @return Number of targets.
       */
      [[nodiscard]] constexpr std::size_t getTargetCount() const noexcept
      {
        return 1;
      }

      /**
       * @brief Equality operator.
       * @param[in] lhs Left-hand side.
       * @param[in] rhs Right-hand side.
       * @return True if equal, false otherwise.
       */
      [[nodiscard]] constexpr friend bool operator==(const CpuDesc&, const CpuDesc&) noexcept
      {
        return true;
      }

      /**
       * @brief Inequality operator.
       * @param[in] lhs Left-hand side.
       * @param[in] rhs Right-hand side.
       * @return True if not equal, false otherwise.
       */
      [[nodiscard]] constexpr friend bool operator!=(const CpuDesc& lhs, const CpuDesc& rhs) noexcept
      {
        return !(lhs == rhs);
      }
  };
#endif /* AFFT_ENABLE_CPU */

#ifdef AFFT_ENABLE_CUDA
  /// @brief CUDA descriptor
  class CudaDesc
  {
    public:
      /// @brief Default constructor.
      CudaDesc() = default;

      /**
       * @brief Constructor from CUDA devices.
       * @param[in] devices CUDA devices.
       */
      CudaDesc(const View<int> devices)
      : mTargetCount{devices.size()}
      {
        if (mTargetCount < Devices::maxLocDevices)
        {
          std::copy(devices.begin(), devices.end(), mDevices.loc);
        }
        else
        {
          mDevices.ext = new int[mTargetCount];
          std::copy(devices.begin(), devices.end(), mDevices.ext);
        }
      }

      /**
       * @brief Copy constructor.
       * @param[in] other Other.
       */
      CudaDesc(const CudaDesc& other)
      : CudaDesc{other.getDevices()}
      {}

      /**
       * @brief Move constructor.
       * @param[in] other Other.
       */
      CudaDesc(CudaDesc&& other)
      : mTargetCount{std::exchange(other.mTargetCount, 0)},
        mDevices{std::exchange(other.mDevices, {})}
      {}

      /// @brief Destructor.
      ~CudaDesc()
      {
        destroy();
      }

      /**
       * @brief Copy assignment operator.
       * @param[in] other Other.
       * @return Reference to this.
       */
      CudaDesc& operator=(const CudaDesc& other)
      {
        if (this != std::addressof(other))
        {
          destroy();

          const auto otherDevices = other.getDevices();

          mTargetCount = otherDevices.size();

          if (mTargetCount < Devices::maxLocDevices)
          {
            std::copy(otherDevices.begin(), otherDevices.end(), mDevices.loc);
          }
          else
          {
            mDevices.ext = new int[mTargetCount];
            std::copy(otherDevices.begin(), otherDevices.end(), mDevices.ext);
          }
        }

        return *this;
      }

      /**
       * @brief Move assignment operator.
       * @param[in] other Other.
       * @return Reference to this.
       */
      CudaDesc& operator=(CudaDesc&& other)
      {
        if (this != std::addressof(other))
        {
          destroy();

          mTargetCount = std::exchange(other.mTargetCount, 0);
          mDevices     = std::exchange(other.mDevices, {});
        }

        return *this;
      }

      /**
       * @brief Get the target.
       * @return Target.
       */
      [[nodiscard]] constexpr Target getTarget() const noexcept
      {
        return Target::cuda;
      }

      /**
       * @brief Get the number of targets.
       * @return Number of targets.
       */
      [[nodiscard]] constexpr std::size_t getTargetCount() const noexcept
      {
        return mTargetCount;
      }

      /**
       * @brief Get the CUDA devices.
       * @return CUDA devices.
       */
      [[nodiscard]] constexpr View<int> getDevices() const noexcept
      {
        return View<int>{(mTargetCount < Devices::maxLocDevices) ? mDevices.loc : mDevices.ext, mTargetCount};
      }

      /**
       * @brief Equality operator.
       * @param[in] lhs Left-hand side.
       * @param[in] rhs Right-hand side.
       * @return True if equal, false otherwise.
       */
      [[nodiscard]] friend bool operator==(const CudaDesc& lhs, const CudaDesc& rhs) noexcept
      {
        const auto lhsDevices = lhs.getDevices();
        const auto rhsDevices = rhs.getDevices();

        return std::equal(lhsDevices.begin(), lhsDevices.end(), rhsDevices.begin(), rhsDevices.end());
      }

      /**
       * @brief Inequality operator.
       * @param[in] lhs Left-hand side.
       * @param[in] rhs Right-hand side.
       * @return True if not equal, false otherwise.
       */
      [[nodiscard]] friend bool operator!=(const CudaDesc& lhs, const CudaDesc& rhs) noexcept
      {
        return !(lhs == rhs);
      }

    private:
      /// @brief Devices union. Enables small buffer optimization.
      union Devices
      {
        /// @brief Maximum number of local devices
        static constexpr std::size_t maxLocDevices{sizeof(int*) / sizeof(int)};

        int  loc[maxLocDevices]; ///< Local devices
        int* ext;                ///< External devices
      };

      /// @brief Destroy the object.
      void destroy()
      {
        if (mTargetCount >= Devices::maxLocDevices)
        {
          delete[] mDevices.ext;
        }
      }

      std::size_t mTargetCount{}; ///< Number of targets
      Devices     mDevices{};     ///< CUDA devices
  };
#endif /* AFFT_ENABLE_CUDA */

#ifdef AFFT_ENABLE_HIP
  // TODO: first test cuda, then copy and adapt for hip
#endif /* AFFT_ENABLE_HIP */
  
#ifdef AFFT_ENABLE_OPENCL
  // TODO: rework

  /// @brief OpenCL description
  struct OpenclDesc
  {
    std::size_t                                                                targetCount{}; ///< Number of targets
# ifdef AFFT_ENABLE_OPENCL
    std::unique_ptr<std::remove_pointer_t<cl_context>, opencl::ContextDeleter> context{}; ///< OpenCL context
    std::unique_ptr<cl_device_id[]>                                            devices{}; ///< OpenCL devices
# endif

    OpenclDesc(const OpenclDesc& other)
    : targetCount{other.targetCount}
    {
#   ifdef AFFT_ENABLE_OPENCL
      opencl::checkError(clRetainContext(other.context.get()));
      context.reset(other.context.get());
      devices = std::make_unique<int[]>(targetCount);
      std::copy(other.devices.get(), other.devices.get() + targetCount, devices.get());
#   endif
    }

    [[nodiscard]] constexpr friend bool operator==([[maybe_unused]] const OpenclDesc& lhs,
                                                   [[maybe_unused]] const OpenclDesc& rhs)
    {
#   ifdef AFFT_ENABLE_OPENCL
      return lhs.targetCount == rhs.targetCount &&
             lhs.context == rhs.context &&
             std::equal(lhs.devices.get(), lhs.devices.get() + lhs.targetCount, rhs.devices.get());
#   else
      return true;
#   endif
    }
  };
#endif /* AFFT_ENABLE_OPENCL */

#ifdef AFFT_ENABLE_OPENMP
  // TODO: rework

  /// @brief OpenMP description
  struct OpenmpDesc
  {
    static constexpr std::size_t targetCount{1}; ///< Number of targets
# ifdef AFFT_ENABLE_OPENMP
    int device{};                                ///< OpenMP device
# endif

    [[nodiscard]] constexpr friend bool operator==([[maybe_unused]] const OpenmpDesc& lhs,
                                                   [[maybe_unused]] const OpenmpDesc& rhs)
    {
#   ifdef AFFT_ENABLE_OPENMP
      return lhs.device == rhs.device;
#   else
      return true;
#   endif
    }
  };
#endif /* AFFT_ENABLE_OPENMP */

  /// @brief Target description
  class TargetDesc
  {
    public:
      /// @brief Default constructor is deleted
      TargetDesc() = delete;

      /**
       * @brief Constructor from target parameters
       * @tparam TargetParamsT Target parameters type
       * @param targetParams Target parameters
       */
      AFFT_TEMPL_REQUIRES(typename TargetParamsT,
                          isCxxTargetParameters<TargetParamsT> || isCTargetParameters<TargetParamsT>)
      TargetDesc(const TargetParamsT& targetParams)
      : mTargetVariant{makeTargetVariant(targetParams)}
      {}

      /**
       * @brief Constructor from target parameters variant
       * @param targetParamsVariant Target parameters variant
       */
      TargetDesc(const TargetParametersVariant& targetParamsVariant)
      : TargetDesc{[&]() -> TargetVariant
          {
#         ifdef AFFT_ENABLE_CPU
            if (std::holds_alternative<afft::cpu::Parameters>(targetParamsVariant))
            {
              return TargetDesc{std::get<afft::cpu::Parameters>(targetParamsVariant)};
            }
#         endif
#         ifdef AFFT_ENABLE_CUDA
            if (std::holds_alternative<afft::cuda::Parameters>(targetParamsVariant))
            {
              return TargetDesc{std::get<afft::cuda::Parameters>(targetParamsVariant)};
            }
#         endif
#         ifdef AFFT_ENABLE_HIP
            if (std::holds_alternative<afft::hip::Parameters>(targetParamsVariant))
            {
              return TargetDesc{std::get<afft::hip::Parameters>(targetParamsVariant)};
            }
#         endif
#         ifdef AFFT_ENABLE_OPENCL
            if (std::holds_alternative<afft::opencl::Parameters>(targetParamsVariant))
            {
              return TargetDesc{std::get<afft::opencl::Parameters>(targetParamsVariant)};
            }
#         endif
#         ifdef AFFT_ENABLE_OPENMP
            if (std::holds_alternative<afft::openmp::Parameters>(targetParamsVariant))
            {
              return TargetDesc{std::get<afft::openmp::Parameters>(targetParamsVariant)};
            }
#         endif
            throw Exception{Error::invalidArgument, "invalid target parameters variant"};
          }()}
      {}

      /// @brief Copy constructor is default
      TargetDesc(const TargetDesc&) = default;

      /// @brief Move constructor is default
      TargetDesc(TargetDesc&&) = default;

      /// @brief Destructor is default
      ~TargetDesc() = default;

      /// @brief Copy assignment operator is default
      TargetDesc& operator=(const TargetDesc&) = default;

      /// @brief Move assignment operator is default
      TargetDesc& operator=(TargetDesc&&) = default;

      /**
       * @brief Get the target
       * @return Target
       */
      [[nodiscard]] constexpr Target getTarget() const
      {
        return std::visit([](const auto& desc) { return desc.getTarget(); }, mTargetVariant);
      }

      /**
       * @brief Get the number of targets
       * @return Number of targets
       */
      [[nodiscard]] constexpr std::size_t getTargetCount() const
      {
        return std::visit([](const auto& desc) { return desc.getTargetCount(); }, mTargetVariant);
      }

      /**
       * @brief Get the target description for the given target
       * @tparam target Target
       * @return Target description
       */
      template<Target target>
      [[nodiscard]] constexpr const auto& getTargetDesc() const
      {
        static_assert(isValid(target), "invalid target");

#     ifdef AFFT_ENABLE_CPU
        if constexpr (target == Target::cpu)
        {
          return std::get<CpuDesc>(mTargetVariant);
        }
#     endif
#     ifdef AFFT_ENABLE_CUDA
        if constexpr (target == Target::cuda)
        {
          return std::get<CudaDesc>(mTargetVariant);
        }
#     endif
#     ifdef AFFT_ENABLE_HIP
        if constexpr (target == Target::hip)
        {
          return std::get<HipDesc>(mTargetVariant);
        }
#     endif
#     ifdef AFFT_ENABLE_OPENCL
        if constexpr (target == Target::opencl)
        {
          return std::get<OpenclDesc>(mTargetVariant);
        }
#     endif
#     ifdef AFFT_ENABLE_OPENMP
        if constexpr (target == Target::openmp)
        {
          return std::get<OpenmpDesc>(mTargetVariant);
        }
#     endif

        throw Exception{Error::internal, "calling getTargetDesc() on disabled target"};
      }

      /**
       * @brief Get the C++ target parameters for the given target
       * @tparam target Target
       * @return Target parameters
       */
      template<Target target>
      [[nodiscard]] constexpr TargetParameters<target> getCxxTargetParameters() const
      {
        static_assert(isValid(target), "invalid target");

        TargetParameters<target> targetParams{};

        if constexpr (target == Target::cpu)
        {
#       ifdef AFFT_ENABLE_CPU
          targetParams.threadLimit = getTargetDesc<Target::cpu>().getThreadLimit();
#       endif
        }
        else if constexpr (target == Target::cuda)
        {
#       ifdef AFFT_ENABLE_CUDA
          targetParams.devices = getTargetDesc<Target::cuda>().getDevices();
#       endif
        }
        else if constexpr (target == Target::hip)
        {
#       ifdef AFFT_ENABLE_HIP
          targetParams.devices = getTargetDesc<Target::hip>().getDevices();
#       endif
        }
        else if constexpr (target == Target::opencl)
        {
#       ifdef AFFT_ENABLE_OPENCL
          const auto& openclDesc = getTargetDesc<Target::opencl>();
          targetParams.context = openclDesc.getContext();
          targetParams.devices = openclDesc.getDevices();
#       endif
        }
        else if constexpr (target == Target::openmp)
        {
#       ifdef AFFT_ENABLE_OPENMP
          targetParams.devices = getTargetDesc<Target::opencl>().getDevice();
#       endif
        }
        else
        {
          cxx::unreachable();
        }

        return targetParams;
      }

      /**
       * @brief Get the C target parameters for the given target
       * @tparam target Target
       * @return Target parameters
       */
      template<Target target>
      [[nodiscard]] constexpr typename TargetParametersSelect<target>::CType
      getCTargetParameters() const
      {
        static_assert(isValid(target), "invalid target");

        typename TargetParametersSelect<target>::CType targetParams{};

        if constexpr (target == Target::cpu)
        {
#       ifdef AFFT_ENABLE_CPU
          const auto& cpuDesc = std::get<CpuDesc>(mTargetVariant);
          targetParams.threadLimit = cpuDesc.getThreadLimit();
#       endif
        }
        else if constexpr (target == Target::cuda)
        {
#       ifdef AFFT_ENABLE_CUDA
          const auto cudaDevices = std::get<CudaDesc>(mTargetVariant).getDevices();
          targetParams.deviceCount = cudaDevices.size();
          targetParams.devices     = cudaDevices.data();
#       endif
        }
        else if constexpr (target == Target::hip)
        {
#       ifdef AFFT_ENABLE_HIP
          const auto& hipDesc = std::get<HipDesc>(mTargetVariant);
          targetParams.deviceCount = hipDesc.targetCount;
          targetParams.devices     = hipDesc.devices.get();
#       endif
        }
        else if constexpr (target == Target::opencl)
        {
#       ifdef AFFT_ENABLE_OPENCL
          const auto& openclDesc = std::get<OpenclDesc>(mTargetVariant);
          targetParams.context     = openclDesc.context.get();
          targetParams.deviceCount = openclDesc.targetCount;
          targetParams.devices     = openclDesc.devices.get();
#       endif
        }
        else if constexpr (target == Target::openmp)
        {
#       ifdef AFFT_ENABLE_OPENMP
          const auto& openmpDesc = std::get<OpenmpDesc>(mTargetVariant);
          targetParams.device = openmpDesc.device;
#       endif
        }
        else
        {
          cxx::unreachable();
        }

        return targetParams;
      }

      /**
       * @brief Equality operator
       * @param lhs Left-hand side
       * @param rhs Right-hand side
       * @return True if equal, false otherwise
       */
      [[nodiscard]] constexpr friend bool operator==(const TargetDesc& lhs, const TargetDesc& rhs)
      {
        return lhs.mTargetVariant == rhs.mTargetVariant;
      }

    private:
      /// @brief The variant type that holds the target description.
      using TargetVariant = std::variant<
        DummyTargetDesc
#     ifdef AFFT_ENABLE_CPU
      , CpuDesc
#     endif
#     ifdef AFFT_ENABLE_CUDA
      , CudaDesc
#     endif
#     ifdef AFFT_ENABLE_HIP
      , HipDesc
#     endif
#     ifdef AFFT_ENABLE_OPENCL
      , OpenclDesc
#     endif
#     ifdef AFFT_ENABLE_OPENMP
      , OpenmpDesc
#     endif
      >;

#   ifdef AFFT_ENABLE_CPU
      /// @brief Make a target variant from the given target parameters.
      [[nodiscard]] static TargetVariant makeTargetVariant(const afft::cpu::Parameters& cpuParams)
      {
        return CpuDesc{};
      }
#   endif

#   ifdef AFFT_ENABLE_CUDA
      [[nodiscard]] static TargetVariant makeTargetVariant(const afft::cuda::Parameters& cudaParams)
      {
        return CudaDesc{cudaParams.devices};
      }
#   endif

#   ifdef AFFT_ENABLE_HIP
      [[nodiscard]] static TargetVariant makeTargetVariant(const afft::hip::Parameters& hipParams)
      {
        HipDesc hipDesc{};
        hipDesc.targetCount = hipParams.devices.size();
        hipDesc.devices     = std::make_unique<int[]>(hipDesc.targetCount);

        std::copy(hipParams.devices.begin(), hipParams.devices.end(), hipDesc.devices.get());

        return hipDesc;
      }
#   endif

#   ifdef AFFT_ENABLE_OPENCL
      [[nodiscard]] static TargetVariant makeTargetVariant(const afft::opencl::Parameters& openclParams)
      {
        OpenclDesc openclDesc{};
        openclDesc.targetCount = openclParams.devices.size();
        opencl::checkError(clRetainContext(openclParams.context));
        openclDesc.context.reset(openclParams.context);
        openclDesc.devices = std::make_unique<cl_device_id[]>(openclDesc.targetCount);

        std::copy(openclParams.devices.begin(), openclParams.devices.end(), openclDesc.devices.get());

        return openclDesc;
      }
#   endif

#   ifdef AFFT_ENABLE_OPENMP
      [[nodiscard]] static TargetVariant makeTargetVariant(const afft::openmp::Parameters& openmpParams)
      {
        OpenmpDesc openmpDesc{};
        openmpDesc.device = openmpParams.device;

        return openmpDesc;
      }
#   endif

#   ifdef AFFT_ENABLE_CPU
      /// @brief Make a target variant from the given target parameters.
      [[nodiscard]] static TargetVariant makeTargetVariant(const afft_cpu_Parameters& cpuParams)
      {
        return CpuDesc{};
      }
#   endif

#   ifdef AFFT_ENABLE_CUDA
      [[nodiscard]] static TargetVariant makeTargetVariant(const afft_cuda_Parameters& cudaParams)
      {
        return CudaDesc{View<int>{cudaParams.devices, cudaParams.deviceCount}};
      }
#   endif

#   ifdef AFFT_ENABLE_HIP
      [[nodiscard]] static TargetVariant makeTargetVariant(const afft_hip_Parameters& hipParams)
      {
        HipDesc hipDesc{};
        hipDesc.targetCount = hipParams.deviceCount;
        hipDesc.devices     = std::make_unique<int[]>(hipDesc.targetCount);

        std::copy_n(hipParams.devices, hipDesc.targetCount, hipDesc.devices.get());

        return hipDesc;
      }
#   endif

#   ifdef AFFT_ENABLE_OPENCL
      [[nodiscard]] static TargetVariant makeTargetVariant(const afft_opencl_Parameters& openclParams)
      {
        OpenclDesc openclDesc{};
        openclDesc.targetCount = openclParams.deviceCount;
        opencl::checkError(clRetainContext(openclParams.context));
        openclDesc.context.reset(openclParams.context);
        openclDesc.devices = std::make_unique<cl_device_id[]>(openclDesc.targetCount);

        std::copy_n(openclParams.devices, openclDesc.targetCount, openclDesc.devices.get());

        return openclDesc;
      }
#   endif

#   ifdef AFFT_ENABLE_OPENMP
      [[nodiscard]] static TargetVariant makeTargetVariant(const afft_openmp_Parameters& openmpParams)
      {
        OpenmpDesc openmpDesc{};
        openmpDesc.device = openmpParams.device;

        return openmpDesc;
      }
#   endif

      TargetVariant mTargetVariant; ///< Target variant
  };
} // namespace afft::detail

#endif /* AFFT_DETAIL_TARGET_DESC_HPP */
