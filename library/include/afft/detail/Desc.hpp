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

#ifndef AFFT_DETAIL_DESC_HPP
#define AFFT_DETAIL_DESC_HPP

#ifndef AFFT_TOP_LEVEL_INCLUDE
# include "include.hpp"
#endif

#include "MemDesc.hpp"
#include "MpDesc.hpp"
#include "TargetDesc.hpp"
#include "TransformDesc.hpp"
#include "../utils.hpp"

namespace afft::detail
{
  class Desc : public TransformDesc, public MpDesc, public TargetDesc, public MemDesc
  {
    public:
      /// @brief Default constructor is deleted.
      Desc() = delete;

      /// @brief Constructor.
      template<typename TransformParamsT,
               typename MpBackendParamsT,
               typename TargetParamsT,
               typename MemoryLayoutT>
      Desc(const TransformParamsT& transformParams,
           const MpBackendParamsT& mpBackendParams,
           const TargetParamsT&    targetParams,
           const MemoryLayoutT&    memoryLayout)
      : TransformDesc{transformParams},
        MpDesc{mpBackendParams},
        TargetDesc{targetParams},
        MemDesc{memoryLayout,
                static_cast<TransformDesc&>(*this),
                static_cast<MpDesc&>(*this),
                static_cast<TargetDesc&>(*this)}
      {
        // static_assert(isTransformParameters<TransformParamsT>, "TransformParamsT must be a transform parameters type");
        // static_assert(isMpBackendParameters<MpBackendParamsT>, "MpBackendParamsT must be an MPI backend parameters type");
        // static_assert(isTargetParameters<TargetParamsT>, "TargetParamsT must be a target parameters type");
        // static_assert(isMemoryLayout<MemoryLayoutT>, "MemoryLayoutT must be a memory layout type");
      }

      // /**
      //  * @brief Constructor.
      //  * @param planParams The plan parameters.
      //  */
      // Desc(const ::afft_PlanParameters& planParams)
      // : TransformDesc{makeTransformDesc(static_cast<Transform>(planParams.transform), planParams.transformParams)},
      //   MpDesc{makeMpDesc(static_cast<MpBackend>(planParams.mpBackend), planParams.mpBackendParams)},
      //   TargetDesc{makeTargetDesc(static_cast<Target>(planParams.target), planParams.targetParams)},
      //   MemDesc{makeMemDesc(planParams.memoryLayout,
      //                       static_cast<TransformDesc&>(*this),
      //                       static_cast<MpDesc&>(*this),
      //                       static_cast<TargetDesc&>(*this))}
      // {}

      /// @brief Copy constructor.
      Desc(const Desc&) = default;

      /// @brief Move constructor.
      Desc(Desc&&) = default;

      /// @brief Destructor.
      ~Desc() = default;

      /// @brief Copy assignment operator.
      Desc& operator=(const Desc&) = default;

      /// @brief Move assignment operator.
      Desc& operator=(Desc&&) = default;

      /**
       * @brief Get the number of buffers required for the source and destination.
       * @return A pair of the number of source and destination buffers.
       */
      [[nodiscard]] constexpr std::pair<std::size_t, std::size_t> getSrcDstBufferCount() const
      {
        const auto complexFormat      = getComplexFormat();
        const auto [srcCmpl, dstCmpl] = getSrcDstComplexity();
        const auto targetCount        = getTargetCount();

        auto bufferCounts = std::make_pair(targetCount, targetCount);

        switch (complexFormat)
        {
        case ComplexFormat::planar:
          if (srcCmpl == Complexity::complex)
          {
            bufferCounts.first *= 2;
          }
          if (dstCmpl == Complexity::complex)
          {
            bufferCounts.second *= 2;
          }
          break;
        default:
          break;
        }

        return bufferCounts;
      }

      /**
       * @brief Get the number of elements required for the source and destination.
       * @param srcElemCounts The source element counts.
       * @param dstElemCounts The destination element counts.
       */
      void getRefElemCounts(Span<std::size_t> srcElemCounts, Span<std::size_t> dstElemCounts) const
      {
        const auto [srcBufferCount, dstBufferCount] = getSrcDstBufferCount();

        if (srcElemCounts.size() < srcBufferCount)
        {
          throw Exception{Error::internal, "srcElemCounts is too small"};
        }

        if (dstElemCounts.size() < dstBufferCount)
        {
          throw Exception{Error::internal, "dstElemCounts is too small"};
        }

        std::fill(srcElemCounts.begin(), srcElemCounts.end(), 0);
        std::fill(dstElemCounts.begin(), dstElemCounts.end(), 0);

        const auto [srcComplexity, dstComplexity] = getSrcDstComplexity();
        const auto srcShape                       = getSrcShape();
        const auto dstShape                       = getDstShape();

        switch (getMemoryLayout())
        {
        case MemoryLayout::centralized:
        {
          const auto& memDesc = getMemDesc<MemoryLayout::centralized>();

          std::size_t currentSrcElemCount{1};
          std::size_t currentDstElemCount{1};

          for (std::size_t i{}; i < getShapeRank(); ++i)
          {
            currentSrcElemCount *= safeIntCast<std::size_t>(srcShape[i]);
            currentDstElemCount *= safeIntCast<std::size_t>(dstShape[i]);

            srcElemCounts[0] = std::max(srcElemCounts[0], currentSrcElemCount * static_cast<std::size_t>(memDesc.getSrcStrides()[i]));
            dstElemCounts[0] = std::max(dstElemCounts[0], currentDstElemCount * static_cast<std::size_t>(memDesc.getDstStrides()[i]));
          }

          if (srcComplexity == Complexity::complex)
          {
            srcElemCounts[1] = srcElemCounts[0];
          }

          if (dstComplexity == Complexity::complex)
          {
            dstElemCounts[1] = dstElemCounts[0];
          }

          break;
        }
        case MemoryLayout::distributed:
          throw Exception{Error::internal, "unimplemented getRefElemCounts for distributed memory layout"};
        default:
          cxx::unreachable();
        }
      }

      // [[nodiscard]] friend bool operator==(const Desc& lhs, const Desc& rhs) noexcept
      // {
      //   return static_cast<const TransformDesc&>(lhs) == static_cast<const TransformDesc&>(rhs) &&
      //          static_cast<const ArchDesc&>(lhs) == static_cast<const ArchDesc&>(rhs);
      // }

      // [[nodiscard]] friend bool operator!=(const Desc& lhs, const Desc& rhs) noexcept
      // {
      //   return !(lhs == rhs);
      // }

      /**
       * @brief Equality operator.
       * @param lhs Left-hand side.
       * @param rhs Right-hand side.
       * @return True if equal, false otherwise.
       */
      [[nodiscard]] friend bool operator==(const Desc& lhs, const Desc& rhs)
      {
        return static_cast<const TransformDesc&>(lhs) == static_cast<const TransformDesc&>(rhs) &&
               static_cast<const MpDesc&>(lhs) == static_cast<const MpDesc&>(rhs) &&
               static_cast<const TargetDesc&>(lhs) == static_cast<const TargetDesc&>(rhs) &&
               static_cast<const MemDesc&>(lhs) == static_cast<const MemDesc&>(rhs);
      }

      /**
       * @brief Inequality operator.
       * @param lhs Left-hand side.
       * @param rhs Right-hand side.
       * @return True if not equal, false otherwise.
       */
      [[nodiscard]] friend bool operator!=(const Desc& lhs, const Desc& rhs) noexcept
      {
        return !(lhs == rhs);
      }

    private:
      [[nodiscard]] static TransformDesc makeTransformDesc(const Transform transform, const void* cTransformParams)
      {
        if (cTransformParams == nullptr)
        {
          throw Exception{Error::invalidArgument, "invalid transform parameters"};
        }

        switch (transform)
        {
        case Transform::dft:
          return TransformDesc{*static_cast<const afft_dft_Parameters*>(cTransformParams)};
        case Transform::dht:
          return TransformDesc{*static_cast<const afft_dht_Parameters*>(cTransformParams)};
        case Transform::dtt:
          return TransformDesc{*static_cast<const afft_dtt_Parameters*>(cTransformParams)};
        default:
          throw Exception{Error::invalidArgument, "invalid transform type"};
        }
      }

      [[nodiscard]] static MpDesc makeMpDesc(const MpBackend mpBackend, [[maybe_unused]] const void* cMpBackendParams)
      {
        switch (mpBackend)
        {
        case MpBackend::none:
          return MpDesc{SingleProcessParameters{}};
        case MpBackend::mpi:
#       ifdef AFFT_ENABLE_MPI
          if (cMpBackendParams == nullptr)
          {
            return MpDesc{afft::mpi::Parameters{}};
          }
          else
          {
            return MpDesc{*static_cast<const afft_mpi_Parameters*>(cMpBackendParams)};
          }
#       else
          throw Exception{Error::invalidArgument, "MPI backend is not enabled"};
#       endif
        default:
          throw Exception{Error::invalidArgument, "invalid MPI backend type"};
        }
      }

      [[nodiscard]] static TargetDesc makeTargetDesc(const Target target, const void* cTargetParams)
      {
        if (cTargetParams == nullptr)
        {
          throw Exception{Error::invalidArgument, "invalid target parameters"};
        }

        switch (target)
        {
        case Target::cpu:
#       ifdef AFFT_ENABLE_CPU
          if (cTargetParams == nullptr)
          {
            return TargetDesc{cpu::Parameters{}};
          }
          else
          {
            return TargetDesc{*static_cast<const afft_cpu_Parameters*>(cTargetParams)};
          }
#       else
          throw Exception{Error::invalidArgument, "CPU target is not enabled"};
#       endif
        case Target::cuda:
#       ifdef AFFT_ENABLE_CUDA
          if (cTargetParams == nullptr)
          {
            throw Exception{Error::invalidArgument, "invalid CUDA target parameters"};
          }
          return TargetDesc{*static_cast<const afft_cuda_Parameters*>(cTargetParams)};
#       else
          throw Exception{Error::invalidArgument, "CUDA target is not enabled"};
#       endif
        case Target::hip:
#       ifdef AFFT_ENABLE_HIP
          if (cTargetParams == nullptr)
          {
            throw Exception{Error::invalidArgument, "invalid HIP target parameters"};
          }
          return TargetDesc{*static_cast<const afft_hip_Parameters*>(cTargetParams)};
#       else
          throw Exception{Error::invalidArgument, "HIP target is not enabled"};
#       endif
        case Target::opencl:
#       ifdef AFFT_ENABLE_OPENCL
          if (cTargetParams == nullptr)
          {
            throw Exception{Error::invalidArgument, "invalid OpenCL target parameters"};
          }
          return TargetDesc{*static_cast<const afft_opencl_Parameters*>(cTargetParams)};
#       else
          throw Exception{Error::invalidArgument, "OpenCL target is not enabled"};
#       endif
        default:
          throw Exception{Error::invalidArgument, "invalid target type"};
        }
      }

      [[nodiscard]] static MemDesc makeMemDesc(const void*          cMemLayout,
                                               const TransformDesc& transformDesc,
                                               const MpDesc&        mpDesc,
                                               const TargetDesc&    targetDesc)
      {
        if (mpDesc.getMpBackend() != MpBackend::none || targetDesc.getTargetCount() > 1)
        {
          if (cMemLayout == nullptr)
          {
            return MemDesc{CentralizedMemoryLayout{}, transformDesc, mpDesc, targetDesc};
          }
          else
          {
            return MemDesc{*static_cast<const afft_CentralizedMemoryLayout*>(cMemLayout),
                           transformDesc,
                           mpDesc,
                           targetDesc};
          }
        }
        else
        {
          if (cMemLayout == nullptr)
          {
            return MemDesc{DistributedMemoryLayout{}, transformDesc, mpDesc, targetDesc};
          }
          else
          {
            return MemDesc{*static_cast<const afft_DistributedMemoryLayout*>(cMemLayout),
                           transformDesc,
                           mpDesc,
                           targetDesc};
          }
        }
      }
  };

  /// @brief Helper struct to limit the access to the Desc object.
  class DescToken
  {
    public:
      /// @brief Make a Desc object.
      [[nodiscard]] static constexpr DescToken make() noexcept
      {
        return DescToken{};
      }

    private:
      /// @brief Default constructor.
      DescToken() = default;
  };
} // namespace afft::detail

/// @brief Specialization of std::hash for afft::detail::Desc.
template<>
struct std::hash<afft::detail::Desc>
{
  /**
   * @brief Hash function for afft::detail::Desc.
   * @param desc The afft::detail::Desc object.
   * @return The hash value.
   */
  [[nodiscard]] constexpr std::size_t operator()(const afft::detail::Desc&) const noexcept
  {
    return 0;
    // return std::hash<TransformDesc>{}(desc) ^ std::hash<ArchDesc>{}(desc);
  }
};

#endif /* AFFT_DETAIL_DESC_HPP */
