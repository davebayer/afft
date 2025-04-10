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
      CudaDesc(const int* devices, const std::size_t targetCount)
      {
        construct(devices, targetCount);
      }

      /**
       * @brief Copy constructor.
       * @param[in] other Other.
       */
      CudaDesc(const CudaDesc& other)
      : CudaDesc{other.getDevices(), other.getTargetCount()}
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
          construct(other.getDevices(), other.getTargetCount());
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
      [[nodiscard]] constexpr const int* getDevices() const noexcept
      {
        return (mTargetCount <= Devices::maxLocDevices) ? mDevices.loc : mDevices.ext;
      }

      /**
       * @brief Equality operator.
       * @param[in] lhs Left-hand side.
       * @param[in] rhs Right-hand side.
       * @return True if equal, false otherwise.
       */
      [[nodiscard]] friend bool operator==(const CudaDesc& lhs, const CudaDesc& rhs) noexcept
      {
        return std::equal(lhs.getDevices(),
                          lhs.getDevices() + lhs.getTargetCount(),
                          rhs.getDevices(),
                          rhs.getDevices() + rhs.getTargetCount());
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
        static constexpr std::size_t maxLocDevices{sizeof(void*) / sizeof(int)};

        int  loc[maxLocDevices]; ///< Local devices
        int* ext;                ///< External devices
      };

      /**
       * @brief Construct the object.
       * @param[in] devices CUDA devices.
       * @param[in] targetCount Number of targets.
       */
      void construct(const int* devices, const std::size_t targetCount)
      {
        mTargetCount = targetCount;

        if (mTargetCount <= Devices::maxLocDevices)
        {
          std::copy_n(devices, mTargetCount, mDevices.loc);
        }
        else
        {
          mDevices.ext = new int[mTargetCount];
          std::copy_n(devices, mTargetCount, mDevices.ext);
        }
      }

      /// @brief Destroy the object.
      void destroy()
      {
        if (mTargetCount > Devices::maxLocDevices)
        {
          delete[] mDevices.ext;
        }
      }

      std::size_t mTargetCount{}; ///< Number of targets
      Devices     mDevices{};     ///< CUDA devices
  };
#endif /* AFFT_ENABLE_CUDA */

#ifdef AFFT_ENABLE_HIP
  /// @brief HIP descriptor
  class HipDesc
  {
    public:
      /// @brief Default constructor.
      HipDesc() = default;

      /**
       * @brief Constructor from HIP devices.
       * @param[in] devices HIP devices.
       */
      HipDesc(const int* devices, const std::size_t targetCount)
      {
        construct(devices, targetCount);
      }

      /**
       * @brief Copy constructor.
       * @param[in] other Other.
       */
      HipDesc(const HipDesc& other)
      : HipDesc{other.getDevices(), other.getTargetCount()}
      {}

      /**
       * @brief Move constructor.
       * @param[in] other Other.
       */
      HipDesc(HipDesc&& other)
      : mTargetCount{std::exchange(other.mTargetCount, 0)},
        mDevices{std::exchange(other.mDevices, {})}
      {}

      /// @brief Destructor.
      ~HipDesc()
      {
        destroy();
      }

      /**
       * @brief Copy assignment operator.
       * @param[in] other Other.
       * @return Reference to this.
       */
      HipDesc& operator=(const HipDesc& other)
      {
        if (this != std::addressof(other))
        {
          destroy();
          construct(other.getDevices(), other.getTargetCount());
        }

        return *this;
      }

      /**
       * @brief Move assignment operator.
       * @param[in] other Other.
       * @return Reference to this.
       */
      HipDesc& operator=(HipDesc&& other)
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
        return Target::hip;
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
       * @brief Get the HIP devices.
       * @return HIP devices.
       */
      [[nodiscard]] constexpr const int* getDevices() const noexcept
      {
        return (mTargetCount <= Devices::maxLocDevices) ? mDevices.loc : mDevices.ext;
      }

      /**
       * @brief Equality operator.
       * @param[in] lhs Left-hand side.
       * @param[in] rhs Right-hand side.
       * @return True if equal, false otherwise.
       */
      [[nodiscard]] friend bool operator==(const HipDesc& lhs, const HipDesc& rhs) noexcept
      {
        return std::equal(lhs.getDevices(),
                          lhs.getDevices() + lhs.getTargetCount(),
                          rhs.getDevices(),
                          rhs.getDevices() + rhs.getTargetCount());
      }

      /**
       * @brief Inequality operator.
       * @param[in] lhs Left-hand side.
       * @param[in] rhs Right-hand side.
       * @return True if not equal, false otherwise.
       */
      [[nodiscard]] friend bool operator!=(const HipDesc& lhs, const HipDesc& rhs) noexcept
      {
        return !(lhs == rhs);
      }

    private:
      /// @brief Devices union. Enables small buffer optimization.
      union Devices
      {
        /// @brief Maximum number of local devices
        static constexpr std::size_t maxLocDevices{sizeof(void*) / sizeof(int)};

        int  loc[maxLocDevices]; ///< Local devices
        int* ext;                ///< External devices
      };

      /**
       * @brief Construct the object.
       * @param[in] devices HIP devices.
       * @param[in] targetCount Number of targets.
       */
      void construct(const int* devices, const std::size_t targetCount)
      {
        mTargetCount = targetCount;

        if (mTargetCount <= Devices::maxLocDevices)
        {
          std::copy_n(devices, mTargetCount, mDevices.loc);
        }
        else
        {
          mDevices.ext = new int[mTargetCount];
          std::copy_n(devices, mTargetCount, mDevices.ext);
        }
      }

      /// @brief Destroy the object.
      void destroy()
      {
        if (mTargetCount > Devices::maxLocDevices)
        {
          delete[] mDevices.ext;
        }
      }

      std::size_t mTargetCount{}; ///< Number of targets
      Devices     mDevices{};     ///< HIP devices
  };
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
#       endif
        }
        else if constexpr (target == Target::cuda)
        {
#       ifdef AFFT_ENABLE_CUDA
          const auto& cudaDesc = getTargetDesc<Target::cuda>();

          targetParams.devices     = cudaDesc.getDevices();
          targetParams.targetCount = cudaDesc.getTargetCount();
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
#       endif
        }
        else if constexpr (target == Target::cuda)
        {
#       ifdef AFFT_ENABLE_CUDA
          const auto& cudaDesc = std::get<CudaDesc>(mTargetVariant);

          targetParams.targetCount = cudaDesc.getTargetCount();
          targetParams.devices     = cudaDesc.getDevices();
#       endif
        }
        else if constexpr (target == Target::hip)
        {
#       ifdef AFFT_ENABLE_HIP
          const auto& hipDesc = std::get<HipDesc>(mTargetVariant);

          targetParams.targetCount = hipDesc.getTargetCount();
          targetParams.devices     = hipDesc.devices.get();
#       endif
        }
        else if constexpr (target == Target::opencl)
        {
#       ifdef AFFT_ENABLE_OPENCL
          const auto& openclDesc = std::get<OpenclDesc>(mTargetVariant);

          targetParams.targetCount = openclDesc.getTargetCount();
          targetParams.context     = openclDesc.context.get();
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
      /**
       * @brief Make a target variant from the cpu parameters.
       * @param cpuParams CPU parameters
       * @return Target variant
       */
      [[nodiscard]] static TargetVariant makeTargetVariant(const afft::cpu::Parameters&)
      {
        return CpuDesc{};
      }

      /**
       * @brief Make a target variant from the CPU parameters.
       * @param cpuParams CPU parameters
       * @return Target variant
       */
      [[nodiscard]] static TargetVariant makeTargetVariant(const afft_cpu_Parameters&)
      {
        return CpuDesc{};
      }
#   endif

#   ifdef AFFT_ENABLE_CUDA
      /**
       * @brief Make a target variant from the CUDA parameters.
       * @param cudaParams CUDA parameters
       * @return Target variant
       */
      [[nodiscard]] static TargetVariant makeTargetVariant(const afft::cuda::Parameters& cudaParams)
      {
        if (cudaParams.targetCount == 0)
        {
          throw Exception{Error::invalidArgument, "no targets specified"};
        }

        if (cudaParams.devices == nullptr)
        {
          if (cudaParams.targetCount != 1)
          {
            throw Exception{Error::invalidArgument, "invalid CUDA devices"};
          }

          const int device = cuda::getCurrentDevice();

          return CudaDesc{std::addressof(device), 1};
        }
        else
        {
          if (!std::all_of(cudaParams.devices, cudaParams.devices + cudaParams.targetCount, cuda::isValidDevice))
          {
            throw Exception{Error::invalidArgument, "invalid CUDA device"};
          }

          return CudaDesc{cudaParams.devices, cudaParams.targetCount};
        }
      }

      /**
       * @brief Make a target variant from the CUDA parameters.
       * @param cudaParams CUDA parameters
       * @return Target variant
       */
      [[nodiscard]] static TargetVariant makeTargetVariant(const afft_cuda_Parameters& cudaParams)
      {
        afft::cuda::Parameters cxxCudaParams{};
        cxxCudaParams.targetCount = cudaParams.targetCount;
        cxxCudaParams.devices     = cudaParams.devices;

        return makeTargetVariant(cxxCudaParams);
      }
#   endif

#   ifdef AFFT_ENABLE_HIP
      /**
       * @brief Make a target variant from the HIP parameters.
       * @param hipParams HIP parameters
       * @return Target variant
       */
      [[nodiscard]] static TargetVariant makeTargetVariant(const afft::hip::Parameters& hipParams)
      {
        if (hipParams.targetCount == 0)
        {
          throw Exception{Error::invalidArgument, "no targets specified"};
        }

        if (hipParams.devices == nullptr)
        {
          if (hipParams.targetCount != 1)
          {
            throw Exception{Error::invalidArgument, "invalid HIP devices"};
          }

          const int device = hip::getCurrentDevice();

          return HipDesc{std::addressof(device), 1};
        }
        else
        {
          if (!std::all_of(hipParams.devices, hipParams.devices + hipParams.targetCount, hip::isValidDevice))
          {
            throw Exception{Error::invalidArgument, "invalid HIP device"};
          }

          return HipDesc{hipParams.devices, hipParams.targetCount};
        }
      }

      /**
       * @brief Make a target variant from the HIP parameters.
       * @param hipParams HIP parameters
       * @return Target variant
       */
      [[nodiscard]] static TargetVariant makeTargetVariant(const afft_hip_Parameters& hipParams)
      {
        afft::hip::Parameters cxxHipParams{};
        cxxHipParams.targetCount = hipParams.targetCount;
        cxxHipParams.devices     = hipParams.devices;

        return makeTargetVariant(cxxHipParams);
      }
#   endif

#   ifdef AFFT_ENABLE_OPENCL
      /**
       * @brief Make a target variant from the OpenCL parameters.
       * @param openclParams OpenCL parameters
       * @return Target variant
       */
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

      /**
       * @brief Make a target variant from the OpenCL parameters.
       * @param openclParams OpenCL parameters
       * @return Target variant
       */
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
      /**
       * @brief Make a target variant from the OpenMP parameters.
       * @param openmpParams OpenMP parameters
       * @return Target variant
       */
      [[nodiscard]] static TargetVariant makeTargetVariant(const afft::openmp::Parameters& openmpParams)
      {
        OpenmpDesc openmpDesc{};
        openmpDesc.device = openmpParams.device;

        return openmpDesc;
      }

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
