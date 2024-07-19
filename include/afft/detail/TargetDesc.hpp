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
#include "../target.hpp"

namespace afft::detail
{
  /// @brief CPU description
  struct CpuDesc
  {
    static constexpr std::size_t targetCount{1}; ///< Number of targets
    unsigned threadLimit{};                      ///< Thread limit

    [[nodiscard]] constexpr friend bool operator==(const CpuDesc& lhs, const CpuDesc& rhs)
    {
      return lhs.threadLimit == rhs.threadLimit;
    }
  };

  /// @brief CUDA description
  struct CudaDesc
  {
    std::size_t            targetCount{}; ///< Number of targets
# ifdef AFFT_ENABLE_CUDA
    std::unique_ptr<int[]> devices{};     ///< CUDA devices
# endif

    [[nodiscard]] constexpr friend bool operator==([[maybe_unused]] const CudaDesc& lhs,
                                                   [[maybe_unused]] const CudaDesc& rhs)
    {
#   ifdef AFFT_ENABLE_CUDA
      return lhs.targetCount == rhs.targetCount &&
             std::equal(lhs.devices.get(), lhs.devices.get() + lhs.targetCount, rhs.devices.get());
#   else
      return true;
#   endif
    }
  };

  /// @brief HIP description
  struct HipDesc
  {
    std::size_t            targetCount{}; ///< Number of targets
# ifdef AFFT_ENABLE_HIP
    std::unique_ptr<int[]> devices{}; ///< HIP devices
# endif

    [[nodiscard]] constexpr friend bool operator==([[maybe_unused]] const HipDesc& lhs,
                                                   [[maybe_unused]] const HipDesc& rhs)
    {
#   ifdef AFFT_ENABLE_HIP
      return lhs.targetCount == rhs.targetCount &&
             std::equal(lhs.devices.get(), lhs.devices.get() + lhs.targetCount, rhs.devices.get());
#   else
      return true;
#   endif
    }
  };
  
  /// @brief OpenCL description
  struct OpenclDesc
  {
    std::size_t                                                                targetCount{}; ///< Number of targets
# ifdef AFFT_ENABLE_OPENCL
    std::unique_ptr<std::remove_pointer_t<cl_context>, opencl::ContextDeleter> context{}; ///< OpenCL context
    std::unique_ptr<cl_device_id[]>                                            devices{}; ///< OpenCL devices
# endif

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
      template<typename TargetParamsT>
      TargetDesc(const TargetParamsT& targetParams)
      : mTargetVariant{makeTargetVariant(targetParams)}
      {
        static_assert(isTargetParameters<TargetParamsT>, "invalid target parameters type");
      }

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
        switch (mTargetVariant.index())
        {
          case 0:
            return Target::cpu;
          case 1:
            return Target::cuda;
          case 2:
            return Target::hip;
          case 3:
            return Target::opencl;
          default:
            throw std::runtime_error("invalid target variant index");
        }
      }

      /**
       * @brief Get the number of targets
       * @return Number of targets
       */
      [[nodiscard]] constexpr std::size_t getTargetCount() const
      {
        return std::visit([](const auto& desc) { return desc.targetCount; }, mTargetVariant);
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

        if constexpr (target == Target::cpu)
        {
          return std::get<CpuDesc>(mTargetVariant);
        }
        else if constexpr (target == Target::cuda)
        {
          return std::get<CudaDesc>(mTargetVariant);
        }
        else if constexpr (target == Target::hip)
        {
          return std::get<HipDesc>(mTargetVariant);
        }
        else if constexpr (target == Target::opencl)
        {
          return std::get<OpenclDesc>(mTargetVariant);
        }
      }

      /**
       * @brief Get the target parameters for the given target
       * @tparam target Target
       * @return Target parameters
       */
      template<Target target>
      [[nodiscard]] constexpr const TargetParameters<target> getTargetParameters() const
      {
        static_assert(isValid(target), "invalid target");

        TargetParameters<target> targetParams{};

        if constexpr (target == Target::cpu)
        {
          const auto& cpuDesc = std::get<CpuDesc>(mTargetVariant);
          targetParams.threadLimit = cpuDesc.threadLimit;
        }
        else if constexpr (target == Target::cuda)
        {
#       ifdef AFFT_ENABLE_CUDA
          const auto& cudaDesc = std::get<CudaDesc>(mTargetVariant);
          targetParams.devices = View{cudaDesc.devices.get(), cudaDesc.targetCount};
#       endif
        }
        else if constexpr (target == Target::hip)
        {
#       ifdef AFFT_ENABLE_HIP
          const auto& hipDesc = std::get<HipDesc>(mTargetVariant);
          targetParams.devices = View{hipDesc.devices.get(), hipDesc.targetCount};
#       endif
        }
        else if constexpr (target == Target::opencl)
        {
#       ifdef AFFT_ENABLE_OPENCL
          const auto& openclDesc = std::get<OpenclDesc>(mTargetVariant);
          targetParams.context = openclDesc.context.get();
          targetParams.devices = View{openclDesc.devices.get(), openclDesc.targetCount};
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
      using TargetVariant = std::variant<CpuDesc, CudaDesc, HipDesc, OpenclDesc>;

      /// @brief Make a target variant from the given target parameters.
      [[nodiscard]] constexpr static TargetVariant makeTargetVariant(const afft::cpu::Parameters& cpuParams)
      {
        CpuDesc cpuDesc{};
        cpuDesc.threadLimit = cpuParams.threadLimit;

        return cpuDesc;
      }

#     ifdef AFFT_ENABLE_CUDA
      [[nodiscard]] static TargetVariant makeTargetVariant(const afft::cuda::Parameters& cudaParams)
      {
        CudaDesc cudaDesc{};
        cudaDesc.targetCount = cudaParams.devices.size();
        cudaDesc.devices     = std::make_unique<int[]>(targetCount);

        std::copy(cudaParams.devices.begin(), cudaParams.devices.end(), cudaDesc.devices.get());

        return cudaDesc;
      }
#     endif

#     ifdef AFFT_ENABLE_HIP
      [[nodiscard]] static TargetVariant makeTargetVariant(const afft::hip::Parameters& hipParams)
      {
        HipDesc hipDesc{};
        hipDesc.targetCount = hipParams.devices.size();
        hipDesc.devices     = std::make_unique<int[]>(targetCount);

        std::copy(hipParams.devices.begin(), hipParams.devices.end(), hipDesc.devices.get());

        return hipDesc;
      }
#     endif

#     ifdef AFFT_ENABLE_OPENCL
      [[nodiscard]] static TargetVariant makeTargetVariant(const afft::opencl::Parameters& openclParams)
      {
        OpenclDesc openclDesc{};
        openclDesc.targetCount = openclParams.devices.size();
        opencl::checkError(clRetainContext(openclParams.context));
        openclDesc.context.reset(openclParams.context);
        openclDesc.devices = std::make_unique<cl_device_id[]>(targetCount);

        std::copy(openclParams.devices.begin(), openclParams.devices.end(), openclDesc.devices.get());

        return openclDesc;
      }
#     endif

      TargetVariant mTargetVariant; ///< Target variant
  };
} // namespace afft::detail

#endif /* AFFT_DETAIL_TARGET_DESC_HPP */
