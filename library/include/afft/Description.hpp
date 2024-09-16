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

#ifndef AFFT_DESCRIPTION_HPP
#define AFFT_DESCRIPTION_HPP

#ifndef AFFT_TOP_LEVEL_INCLUDE
# include "detail/include.hpp"
#endif

#include "backend.hpp"
#include "common.hpp"
#include "memory.hpp"
#include "target.hpp"
#include "transform.hpp"
#include "typeTraits.hpp"
#include "detail/Desc.hpp"

AFFT_EXPORT namespace afft
{
  /// @brief Plan description.
  class Description
  {
    friend struct DescGetter;
    friend std::hash<afft::Description>;
    
    public:
      /// @brief Default constructor is deleted.
      Description() = delete;

      /**
       * @brief Constructor.
       * @tparam TransformParamsT Transform parameters type.
       * @tparam TargetParamsT Target parameters type.
       * @param[in] transformParams Transform parameters.
       * @param[in] targetParams    Target parameters.
       */
      template<typename TransformParamsT, typename TargetParamsT>
      Description(const TransformParamsT& transformParams,
                  const TargetParamsT&    targetParams)
      : mDesc{transformParams, targetParams}
      {
        static_assert(isTransformParameters<TransformParamsT>, "invalid transform parameters type");
        static_assert(isTargetParameters<TargetParamsT>, "invalid target parameters type");
      }

      /**
       * @brief Constructor.
       * @tparam TransformParamsT Transform parameters type.
       * @tparam TargetParamsT Target parameters type.
       * @tparam MemoryLayoutParamsT Memory layout parameters type.
       * @param[in] transformParams    Transform parameters.
       * @param[in] targetParams       Target parameters.
       * @param[in] memoryLayoutParams Memory layout parameters.
       */
      template<typename TransformParamsT,
               typename TargetParamsT,
               typename MemoryLayoutParamsT>
      Description(const TransformParamsT&    transformParams,
                  const TargetParamsT&       targetParams,
                  const MemoryLayoutParamsT& memoryLayoutParams)
      : mDesc{transformParams, targetParams, memoryLayoutParams}
      {
        static_assert(isTransformParameters<TransformParamsT>, "invalid transform parameters type");
        static_assert(isTargetParameters<TargetParamsT>, "invalid target parameters type");
        static_assert(isMemoryLayout<MemoryLayoutParamsT>, "invalid memory layout parameters type");
      }

      /**
       * @brief Constructor.
       * @tparam TransformParamsT Transform parameters type.
       * @tparam MpBackendParamsT Multi process backend parameters type.
       * @tparam TargetParamsT Target parameters type.
       * @tparam MemoryLayoutParamsT Memory layout parameters type.
       * @param[in] transformParams    Transform parameters.
       * @param[in] mpBackendParams    Multi-process backend parameters.
       * @param[in] targetParams       Target parameters.
       * @param[in] memoryLayoutParams Memory layout parameters.
       */
      template<typename TransformParamsT,
               typename MpBackendParamsT,
               typename TargetParamsT,
               typename MemoryLayoutT>
      Description(const TransformParamsT& transformParams,
                  const MpBackendParamsT& mpBackendParams,
                  const TargetParamsT&    targetParams,
                  const MemoryLayoutT&    memoryLayoutParams)
      : mDesc{transformParams, mpBackendParams, targetParams, memoryLayoutParams}
      {
        static_assert(isTransformParameters<TransformParamsT>, "invalid transform parameters type");
        static_assert(isMpBackendParameters<MpBackendParamsT>, "invalid multi-process backend parameters type");
        static_assert(isTargetParameters<TargetParamsT>, "invalid target parameters type");
        static_assert(isMemoryLayout<MemoryLayoutT>, "invalid memory layout type");
      }

      /**
       * @brief Constructor. Internal use only.
       * @param[in] desc  Plan description.
       * @param[in] token Description token.
       */
      Description(const detail::Desc& desc, detail::DescToken)
      : mDesc{desc}
      {}

      /// @brief Copy constructor.
      Description(const Description&) = default;

      /// @brief Move constructor.
      Description(Description&&) = default;

      /**
       * @brief Destructor.
       */
      ~Description() = default;

      /// @brief Copy assignment operator.
      Description& operator=(const Description&) = default;

      /// @brief Move assignment operator.
      Description& operator=(Description&&) = default;

      /**
       * @brief Get the detailed plan description. Only for internal use.
       * @return Plan description.
       */
      [[nodiscard]] constexpr const detail::Desc& get(detail::DescToken) const noexcept
      {
        return mDesc;
      }

      /**
       * @brief Get the detailed plan description. Only for internal use.
       * @return Plan description.
       */
      [[nodiscard]] constexpr detail::Desc& get(detail::DescToken) noexcept
      {
        return mDesc;
      }

      /**
       * @brief Get the transform.
       * @return Transform.
       */
      [[nodiscard]] constexpr Transform getTransform() const noexcept
      {
        return mDesc.getTransform();
      }

      /**
       * @brief Get transform parameters.
       * @tparam transform Transform type
       * @return Transform parameters
       */
      template<Transform transform>
      [[nodiscard]] constexpr TransformParameters<transform> getTransformParameters() const
      {
        static_assert(detail::isValid(transform), "invalid transform type");

        if (transform != getTransform())
        {
          throw Exception{Error::invalidArgument, "plan transform does not match requested transform"};
        }

        return mDesc.getCxxTransformParameters<transform>();
      }

      /**
       * @brief Get transform parameters variant.
       * @return Transform parameters variant.
       */
      [[nodiscard]] constexpr TransformParametersVariant getTransformParametersVariant() const
      {
        switch (getTransform())
        {
        case Transform::dft:
          return getTransformParameters<Transform::dft>();
        case Transform::dht:
          return getTransformParameters<Transform::dht>();
        case Transform::dtt:
          return getTransformParameters<Transform::dtt>();
        default:
          detail::cxx::unreachable();
        }
      }

      /**
       * @brief Get target.
       * @return Target
       */
      [[nodiscard]] constexpr Target getTarget() const noexcept
      {
        return mDesc.getTarget();
      }

      /**
       * @brief Get target count.
       * @return Target count.
       */
      [[nodiscard]] constexpr std::size_t getTargetCount() const noexcept
      {
        return mDesc.getTargetCount();
      }

      /**
       * @brief Get target parameters.
       * @tparam target Target type
       * @return Target parameters
       */
      template<Target target>
      [[nodiscard]] constexpr TargetParameters<target> getTargetParameters() const
      {
        static_assert(detail::isValid(target), "invalid target type");

        if (target != getTarget())
        {
          throw Exception{Error::invalidArgument, "plan target does not match requested target"};
        }

        return mDesc.getCxxTargetParameters<target>();
      }

      /**
       * @brief Get target parameters variant.
       * @return Target parameters variant.
       */
      [[nodiscard]] constexpr TargetParametersVariant getTargetParametersVariant() const
      {
        switch (getTarget())
        {
#       ifdef AFFT_ENABLE_CPU
        case Target::cpu:
          return getTargetParameters<Target::cpu>();
#       endif
#       ifdef AFFT_ENABLE_CUDA
        case Target::cuda:
          return getTargetParameters<Target::cuda>();
#       endif
#       ifdef AFFT_ENABLE_HIP
        case Target::hip:
          return getTargetParameters<Target::hip>();
#       endif
#       ifdef AFFT_ENABLE_OPENCL
        case Target::opencl:
          return getTargetParameters<Target::opencl>();
#       endif
#       ifdef AFFT_ENABLE_OPENMP
        case Target::openmp:
          return getTargetParameters<Target::openmp>();
#       endif
        default:
          detail::cxx::unreachable();
        }
      }

      /**
       * @brief Get the multi-process backend.
       * @return Multi-process backend.
       */
      [[nodiscard]] constexpr MpBackend getMpBackend() const noexcept
      {
        return mDesc.getMpBackend();
      }

      /**
       * @brief Get multi-process backend parameters.
       * @tparam mpBackend Multi-process backend type
       * @return Multi-process backend parameters
       */
      template<MpBackend mpBackend>
      [[nodiscard]] constexpr MpBackendParameters<mpBackend> getMpBackendParameters() const
      {
        static_assert(detail::isValid(mpBackend), "invalid multi-process backend type");

        if (mpBackend != getMpBackend())
        {
          throw Exception{Error::invalidArgument, "plan multi-process backend does not match requested multi-process backend"};
        }

        return mDesc.getCxxMpParameters<mpBackend>();
      }

      /**
       * @brief Get multi-process backend parameters variant.
       * @return Multi-process backend parameters variant.
       */
      [[nodiscard]] constexpr MpBackendParametersVariant getMpBackendParametersVariant() const
      {
        switch (getMpBackend())
        {
        case MpBackend::none:
          return getMpBackendParameters<MpBackend::none>();
#       ifdef AFFT_ENABLE_MPI
        case MpBackend::mpi:
          return getMpBackendParameters<MpBackend::mpi>();
#       endif
        default:
          detail::cxx::unreachable();
        }
      }

      /**
       * @brief Get the memory layout.
       * @return Memory layout.
       */
      [[nodiscard]] constexpr MemoryLayout getMemoryLayout() const noexcept
      {
        return mDesc.getMemoryLayout();
      }

      /**
       * @brief Equality operator.
       * @param[in] lhs Left-hand side.
       * @param[in] rhs Right-hand side.
       * @return True if equal, false otherwise.
       */
      [[nodiscard]] friend bool operator==(const Description& lhs, const Description& rhs) noexcept
      {
        return lhs.mDesc == rhs.mDesc;
      }

      /**
       * @brief Inequality operator.
       * @param[in] lhs Left-hand side.
       * @param[in] rhs Right-hand side.
       * @return True if not equal, false otherwise.
       */
      [[nodiscard]] friend bool operator!=(const Description& lhs, const Description& rhs) noexcept
      {
        return !(lhs == rhs);
      }

    protected:
      detail::Desc mDesc; ///< Plan description.
  };
} // namespace afft

/// @brief Hash function for afft::Description.
AFFT_EXPORT template<>
struct std::hash<afft::Description>
{
  /**
   * @brief Hash function.
   * @param[in] desc Description.
   * @return Hash value.
   */
  [[nodiscard]] std::size_t operator()(const afft::Description& desc) const noexcept
  {
    return std::hash<afft::detail::Desc>{}(desc.mDesc);
  }
};

#endif /* AFFT_DESCRIPTION_HPP */
