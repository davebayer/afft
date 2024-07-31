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
        static_assert(isTransformParameters<TransformParamsT>, "TransformParamsT must be a transform parameters type");
        static_assert(isMpBackendParameters<MpBackendParamsT>, "MpBackendParamsT must be an MPI backend parameters type");
        static_assert(isTargetParameters<TargetParamsT>, "TargetParamsT must be a target parameters type");
        static_assert(isMemoryLayout<MemoryLayoutT>, "MemoryLayoutT must be a memory layout type");
      }

      /**
       * @brief Constructor.
       * @param planParams The plan parameters.
       */
      Desc(const ::afft_PlanParameters& planParams)
      : TransformDesc{makeTransformDesc(static_cast<Transform>(planParams.transform), planParams.transformParams)},
        MpDesc{makeMpDesc(static_cast<MpBackend>(planParams.mpBackend), planParams.mpBackendParams)},
        TargetDesc{makeTargetDesc(static_cast<Target>(planParams.target), planParams.targetParams)},
        MemDesc{makeMemDesc(planParams.memoryLayout,
                            static_cast<TransformDesc&>(*this),
                            static_cast<MpDesc&>(*this),
                            static_cast<TargetDesc&>(*this))}
      {}

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

      // [[nodiscard]] friend bool operator==(const Desc& lhs, const Desc& rhs) noexcept
      // {
      //   return static_cast<const TransformDesc&>(lhs) == static_cast<const TransformDesc&>(rhs) &&
      //          static_cast<const ArchDesc&>(lhs) == static_cast<const ArchDesc&>(rhs);
      // }

      // [[nodiscard]] friend bool operator!=(const Desc& lhs, const Desc& rhs) noexcept
      // {
      //   return !(lhs == rhs);
      // }
  };

  /// @brief Helper struct to get the Desc object from an object.
  struct DescGetter
  {
    /**
     * @brief Get the Desc object from an object.
     * @tparam T The type of the object.
     * @param obj The object.
     * @return The Desc object.
     */
    template<typename T>
    [[nodiscard]] static const Desc& get(T& obj)
    {
      return obj.getDesc();
    }
  };
} // namespace afft::detail

/// @brief Specialization of std::hash for afft::detail::Desc.
template<>
struct std::hash<afft::detail::Desc>
{
  [[nodiscard]] constexpr std::size_t operator()(const afft::detail::Desc&) const noexcept
  {
    return 0;
    // return std::hash<TransformDesc>{}(desc) ^ std::hash<ArchDesc>{}(desc);
  }
};

#endif /* AFFT_DETAIL_DESC_HPP */
