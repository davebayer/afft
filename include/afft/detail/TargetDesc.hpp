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
#include "../target.hpp"

namespace afft::detail
{
  /// @brief CPU description
  struct CpuDesc
  {
    unsigned threadLimit{}; ///< Thread limit

    /**
     * @brief Equality operator
     * @param lhs Left-hand side
     * @param rhs Right-hand side
     * @return True if equal, false otherwise
     */
    [[nodiscard]] constexpr friend bool operator==(const CpuDesc& lhs, const CpuDesc& rhs)
    {
      return lhs.threadLimit == rhs.threadLimit;
    }
  };

  /// @brief CUDA description
  struct CudaDesc
  {
# ifdef AFFT_ENABLE_CUDA
    std::vector<int> devices{}; ///< CUDA devices
# endif

    /**
     * @brief Equality operator
     * @param lhs Left-hand side
     * @param rhs Right-hand side
     * @return True if equal, false otherwise
     */
    [[nodiscard]] constexpr friend bool operator==([[maybe_unused]] const CudaDesc& lhs,
                                                   [[maybe_unused]] const CudaDesc& rhs)
    {
# ifdef AFFT_ENABLE_CUDA
      return lhs.devices == rhs.devices;
# else
      return true;
# endif
    }
  };

  /// @brief HIP description
  struct HipDesc
  {
# ifdef AFFT_ENABLE_HIP
    std::vector<int> devices{}; ///< HIP devices
# endif

    /**
     * @brief Equality operator
     * @param lhs Left-hand side
     * @param rhs Right-hand side
     * @return True if equal, false otherwise
     */
    [[nodiscard]] constexpr friend bool operator==([[maybe_unused]] const HipDesc& lhs,
                                                   [[maybe_unused]] const HipDesc& rhs)
    {
# ifdef AFFT_ENABLE_HIP
      return lhs.devices == rhs.devices;
# else
      return true;
# endif
    }
  };
  
  /// @brief OpenCL description
  struct OpenclDesc
  {
# ifdef AFFT_ENABLE_OPENCL
    std::unique_ptr<std::remove_pointer_t<cl_context>, opencl::ContextDeleter> context{}; ///< OpenCL context
    std::vector<cl_device_id>                                                  devices{}; ///< OpenCL devices
# endif

    /**
     * @brief Equality operator
     * @param lhs Left-hand side
     * @param rhs Right-hand side
     * @return True if equal, false otherwise
     */
    [[nodiscard]] constexpr friend bool operator==([[maybe_unused]] const OpenclDesc& lhs, [[maybe_unused]] const OpenclDesc& rhs)
    {
# ifdef AFFT_ENABLE_OPENCL
      return lhs.context == rhs.context && lhs.devices == rhs.devices;
# else
      return true;
# endif
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
      : mTargetVariant(makeTargetVariant(targetParams))
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
        switch (mTargetVariant.index())
        {
        case 0:
          return 1;
#       ifdef AFFT_ENABLE_CUDA
        case 1:
          return std::get<CudaDesc>(mTargetVariant).devices.size();
#       endif
#       ifdef AFFT_ENABLE_HIP
        case 2:
          return std::get<HipDesc>(mTargetVariant).devices.size();
#       endif
#       ifdef AFFT_ENABLE_OPENCL
        case 3:
          return std::get<OpenclDesc>(mTargetVariant).devices.size();
#       endif        
        default:
          return 0;
        }
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
        else
        {
          throw std::runtime_error("invalid target");
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
        
        switch (target)
        {
          case Target::cpu:
          {
            const auto& cpuDesc = std::get<CpuDesc>(mTargetVariant);
            targetParams.threadLimit = cpuDesc.threadLimit;
            break;
          }
          case Target::cuda:
          {
#         ifdef AFFT_ENABLE_CUDA
            const auto& cudaDesc = std::get<CudaDesc>(mTargetVariant);
            targetParams.devices = cudaDesc.devices;
#         endif
            break;
          }
          case Target::hip:
          {
#         ifdef AFFT_ENABLE_HIP
            const auto& hipDesc = std::get<HipDesc>(mTargetVariant);
            targetParams.devices = hipDesc.devices;
#         endif
            break;
          }
          case Target::opencl:
          {
#         ifdef AFFT_ENABLE_OPENCL
            const auto& openclDesc = std::get<OpenclDesc>(mTargetVariant);
            targetParams.context = openclDesc.context.get();
            targetParams.devices = openclDesc.devices;
#         endif
            break;
          }
          default:
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
      [[nodiscard]] static TargetVariant makeTargetVariant([[maybe_unused]] const afft::cuda::Parameters& cudaParams)
      {
        CudaDesc cudaDesc{};
        cudaDesc.devices.resize(cudaParams.devices.size());
        std::copy(cudaParams.devices.begin(), cudaParams.devices.end(), cudaDesc.devices.begin());

        return cudaDesc;
      }
#     endif

#     ifdef AFFT_ENABLE_HIP
      [[nodiscard]] static TargetVariant makeTargetVariant([[maybe_unused]] const afft::hip::Parameters& hipParams)
      {
        HipDesc hipDesc{};
        hipDesc.devices.resize(hipParams.devices.size());
        std::copy(hipParams.devices.begin(), hipParams.devices.end(), hipDesc.devices.begin());

        return hipDesc;
      }
#     endif

#     ifdef AFFT_ENABLE_OPENCL
      [[nodiscard]] static TargetVariant makeTargetVariant([[maybe_unused]] const afft::opencl::Parameters& openclParams)
      {
        OpenclDesc openclDesc{};
        opencl::checkError(clRetainContext(openclParams.context));
        openclDesc.context.reset(openclParams.context);
        openclDesc.devices.resize(openclParams.devices.size());
        std::copy(openclParams.devices.begin(), openclParams.devices.end(), openclDesc.devices.begin());

        return openclDesc;
      }
#     endif

      TargetVariant mTargetVariant;
  };
} // namespace afft::detail

#endif /* AFFT_DETAIL_TARGET_DESC_HPP */
