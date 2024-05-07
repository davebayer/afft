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

#ifndef AFFT_PLAN_HPP
#define AFFT_PLAN_HPP

#include <memory>

#include "common.hpp"
#include "concepts.hpp"
#include "detail/makePlanImpl.hpp"
#include "detail/Config.hpp"
#include "detail/PlanImpl.hpp"

namespace afft
{
  /**
   * @class Plan
   * @brief Plan class
   */
  class Plan
  {
    public:
      /// Default constructor is not allowed
      Plan() = delete;

      /// Copy constructor
      Plan(const Plan&) = default;

      /// Move constructor
      Plan(Plan&&) = default;

      /// Destructor
      virtual ~Plan() = default;

      /// Copy assignment operator
      Plan& operator=(const Plan&) = default;

      /// Move assignment operator
      Plan& operator=(Plan&&) = default;

      /**
       * @brief Get the transform type
       * @return Transform type
       */
      [[nodiscard]] Transform getTransform() const
      {
        return mImpl->getConfig().getTransform();
      }

      /**
       * @brief Get the transform parameters
       * @tparam transform Transform type
       * @return Transform parameters
       */
      template<Transform transform>
      [[nodiscard]] TransformParameters<transform> getTransformParameters() const
      {
        static_assert(detail::isValidTransform(transform), "Invalid transform type");

        return mImpl->getConfig().getTransformParameters<transform>();
      }

      /**
       * @brief Get target
       * @return Target
       */
      [[nodiscard]] Target getTarget() const
      {
        return mImpl->getConfig().getTarget();
      }

      /**
       * @brief Get target parameters
       * @tparam target Target type
       * @return Target parameters
       */
      template<Target target>
      [[nodiscard]] TargetParameters<target> getTargetParameters() const
      {
        static_assert(detail::isValidTarget(target), "Invalid target type");

        return mImpl->getConfig().getTargetParameters<target>();
      }

      /**
       * @brief Execute in-place plan with default execution parameters
       * @param srcDst Source and destination
       */
      template<typename SrcDstT>
      void execute(SrcDstT* srcDst)
      {
        static_assert(isKnownType<SrcDstT>, "A known type is required");

        mImpl->executeWithDefaultTargetParameters(srcDst);
      }

      /**
       * @brief Execute in-place plan with default execution parameters
       * @param srcDst Source and destination
       */
      template<typename SrcDstT>
      void execute(PlanarComplex<SrcDstT*> srcDst)
      {
        static_assert(isRealType<SrcDstT>, "A real type is required");

        mImpl->executeWithDefaultTargetParameters(srcDst);
      }

      /**
       * @brief Execute in-place plan
       * @tparam SrcDstT Source or destination buffer type
       * @tparam ExecParamsT Execution parameters type
       * @param srcDst Source and destination buffer
       * @param execParams Execution parameters
       */
      template<typename SrcDstT, typename ExecParamsT>
      void execute(SrcDstT* srcDst, const ExecParamsT& execParams)
      {
        static_assert(isKnownType<SrcDstT>, "A known type is required");
        static_assert(isExecutionParameters<ExecParamsT>, "Unknown execution parameters type");

        mImpl->execute(srcDst, execParams);
      }

      /**
       * @brief Execute in-place plan
       * @tparam SrcDstT Source or destination buffer type
       * @tparam ExecParamsT Execution parameters type
       * @param srcDst Source and destination buffer
       * @param execParams Execution parameters
       */
      template<typename SrcDstT, typename ExecParamsT>
      void execute(PlanarComplex<SrcDstT*> srcDst, const ExecParamsT& execParams)
      {
        static_assert(isRealType<SrcDstT>, "A real type is required");
        static_assert(isExecutionParameters<ExecParamsT>, "Unknown execution parameters type");

        mImpl->execute(srcDst, execParams);
      }

      /**
       * @brief Execute out-of-place plan
       * @tparam SrcT Source buffer type
       * @tparam DstT Destination buffer type
       * @param src Source buffer
       * @param dst Destination buffer
       */
      template<typename SrcT, typename DstT>
      void execute(SrcT* src, DstT* dst)
      {
        static_assert(isKnownType<SrcT>, "A known type is required");
        static_assert(isKnownType<DstT>, "A known type is required");

        mImpl->executeWithDefaultTargetParameters(src, dst);
      }

      /**
       * @brief Execute out-of-place plan with source buffer as PlanarComplex
       * @tparam SrcT Source buffer type
       * @tparam DstT Destination buffer type
       * @param src Source buffer in PlanarComplex format
       * @param dst Destination buffer
       */
      template<typename SrcT, typename DstT>
      void execute(PlanarComplex<SrcT*> src, DstT* dst)
      {
        static_assert(isRealType<SrcT>, "A real type is required");
        static_assert(isKnownType<DstT>, "A known type is required");

        mImpl->executeWithDefaultTargetParameters(src, dst);
      }

      /**
       * @brief Execute out-of-place plan with destination buffer as PlanarComplex
       * @tparam SrcT Source buffer type
       * @tparam DstT Destination buffer type
       * @param src Source buffer
       * @param dst Destination buffer in PlanarComplex format
       */
      template<typename SrcT, typename DstT>
      void execute(SrcT* src, PlanarComplex<DstT*> dst)
      {
        static_assert(isKnownType<SrcT>, "A known type is required");
        static_assert(isRealType<DstT>, "A real type is required");

        mImpl->executeWithDefaultTargetParameters(src, dst);
      }

      /**
       * @brief Execute out-of-place plan with source and destination buffers as PlanarComplex
       * @tparam SrcT Source buffer type
       * @tparam DstT Destination buffer type
       * @param src Source buffer in PlanarComplex format
       * @param dst Destination buffer in PlanarComplex format
       */
      template<typename SrcT, typename DstT>
      void execute(PlanarComplex<SrcT*> src, PlanarComplex<DstT*> dst)
      {
        static_assert(isRealType<SrcT>, "A real type is required");
        static_assert(isRealType<DstT>, "A real type is required");

        mImpl->executeWithDefaultTargetParameters(src, dst);
      }

      /**
       * @brief Execute out-of-place plan
       * @tparam SrcT Source buffer type
       * @tparam DstT Destination buffer type
       * @tparam ExecParamsT Execution parameters type
       * @param src Source buffer
       * @param dst Destination buffer
       * @param execParams Execution parameters
       */
      template<typename SrcT, typename DstT, typename ExecParamsT>
      void execute(SrcT* src, DstT* dst, const ExecParamsT& execParams)
      {
        static_assert(isKnownType<SrcT>, "A known type is required");
        static_assert(isKnownType<DstT>, "A known type is required");
        static_assert(isExecutionParameters<ExecParamsT>, "Unknown execution parameters type");

        mImpl->execute(src, dst, execParams);
      }

      /**
       * @brief Execute out-of-place plan with source buffer as PlanarComplex
       * @tparam SrcT Source buffer type
       * @tparam DstT Destination buffer type
       * @tparam ExecParamsT Execution parameters type
       * @param src Source buffer in PlanarComplex format
       * @param dst Destination buffer
       * @param execParams Execution parameters
       */
      template<typename SrcT, typename DstT, typename ExecParamsT>
      void execute(PlanarComplex<SrcT*> src, DstT* dst, const ExecParamsT& execParams)
      {
        static_assert(isRealType<SrcT>, "A real type is required");
        static_assert(isKnownType<DstT>, "A known type is required");
        static_assert(isExecutionParameters<ExecParamsT>, "Unknown execution parameters type");

        mImpl->execute(src, dst, execParams);
      }

      /**
       * @brief Execute out-of-place plan with destination buffer as PlanarComplex
       * @tparam SrcT Source buffer type
       * @tparam DstT Destination buffer type
       * @tparam ExecParamsT Execution parameters type
       * @param src Source buffer
       * @param dst Destination buffer in PlanarComplex format
       * @param execParams Execution parameters
       */
      template<typename SrcT, typename DstT, typename ExecParamsT>
      void execute(SrcT* src, PlanarComplex<DstT*> dst, const ExecParamsT& execParams)
      {
        static_assert(isKnownType<SrcT>, "A known type is required");
        static_assert(isRealType<DstT>, "A real type is required");
        static_assert(isExecutionParameters<ExecParamsT>, "Unknown execution parameters type");

        mImpl->execute(src, dst, execParams);
      }

      /**
       * @brief Execute out-of-place plan with source and destination buffers as PlanarComplex
       * @tparam SrcT Source buffer type
       * @tparam DstT Destination buffer type
       * @tparam ExecParamsT Execution parameters type
       * @param src Source buffer in PlanarComplex format
       * @param dst Destination buffer in PlanarComplex format
       * @param execParams Execution parameters
       */
      template<typename SrcT, typename DstT, typename ExecParamsT>
      void execute(PlanarComplex<SrcT*> src, PlanarComplex<DstT*> dst, const ExecParamsT& execParams)
      {
        static_assert(isRealType<SrcT>, "A real type is required");
        static_assert(isRealType<DstT>, "A real type is required");
        static_assert(isExecutionParameters<ExecParamsT>, "Unknown execution parameters type");

        mImpl->execute(src, dst, execParams);
      }

      /**
       * @brief Execute out-of-place plan without type checking
       * @tparam SrcT Source buffer type
       * @tparam DstT Destination buffer type
       * @param src Source buffer
       * @param dst Destination buffer
       */
      template<typename SrcT, typename DstT>
      void executeUnsafe(SrcT* src, DstT* dst)
      {
        mImpl->executeUnsafeWithDefaultTargetParameters(src, dst);
      }

      /**
       * @brief Execute out-of-place plan without type checking and with source buffer as PlanarComplex
       * @tparam SrcT Source buffer type in PlanarComplex format
       * @tparam DstT Destination buffer type
       * @param src Source buffer
       * @param dst Destination buffer
       */
      template<typename SrcT, typename DstT>
      void executeUnsafe(PlanarComplex<SrcT*> src, DstT* dst)
      {
        mImpl->executeUnsafeWithDefaultTargetParameters(src, dst);
      }

      /**
       * @brief Execute out-of-place plan without type checking and with source and destination buffers as PlanarComplex
       * @tparam SrcT Source buffer type in PlanarComplex format
       * @tparam DstT Destination buffer type in PlanarComplex format
       * @param src Source buffer
       * @param dst Destination buffer
       */
      template<typename SrcT, typename DstT>
      void executeUnsafe(SrcT* src, PlanarComplex<DstT*> dst)
      {
        mImpl->executeUnsafeWithDefaultTargetParameters(src, dst);
      }

      /**
       * @brief Execute out-of-place plan without type checking and with source and destination buffers as PlanarComplex
       * @tparam SrcT Source buffer type in PlanarComplex format
       * @tparam DstT Destination buffer type in PlanarComplex format
       * @param src Source buffer
       * @param dst Destination buffer
       */
      template<typename SrcT, typename DstT>
      void executeUnsafe(PlanarComplex<SrcT*> src, PlanarComplex<DstT*> dst)
      {
        mImpl->executeUnsafeWithDefaultTargetParameters(src, dst);
      }

      /**
       * @brief Execute out-of-place plan without type checking
       * @tparam SrcT Source buffer type
       * @tparam DstT Destination buffer type
       * @tparam ExecParamsT Execution parameters type
       * @param src Source buffer
       * @param dst Destination buffer
       * @param execParams Execution parameters
       */
      template<typename SrcT, typename DstT, typename ExecParamsT>
      void executeUnsafe(SrcT* src, DstT* dst, const ExecParamsT& execParams)
      {
        static_assert(isExecutionParameters<ExecParamsT>, "Unknown execution parameters type");

        mImpl->executeUnsafe(src, dst, execParams);
      }

      /**
       * @brief Execute out-of-place plan without type checking and with source buffer as PlanarComplex
       * @tparam SrcT Source buffer type in PlanarComplex format
       * @tparam DstT Destination buffer type
       * @tparam ExecParamsT Execution parameters type
       * @param src Source buffer
       * @param dst Destination buffer
       * @param execParams Execution parameters
       */
      template<typename SrcT, typename DstT, typename ExecParamsT>
      void executeUnsafe(PlanarComplex<SrcT*> src, DstT* dst, const ExecParamsT& execParams)
      {
        static_assert(isExecutionParameters<ExecParamsT>, "Unknown execution parameters type");

        mImpl->executeUnsafe(src, dst, execParams);
      }

      /**
       * @brief Execute out-of-place plan without type checking and with source and destination buffers as PlanarComplex
       * @tparam SrcT Source buffer type in PlanarComplex format
       * @tparam DstT Destination buffer type in PlanarComplex format
       * @tparam ExecParamsT Execution parameters type
       * @param src Source buffer
       * @param dst Destination buffer
       * @param execParams Execution parameters
       */
      template<typename SrcT, typename DstT, typename ExecParamsT>
      void executeUnsafe(SrcT* src, PlanarComplex<DstT*> dst, const ExecParamsT& execParams)
      {
        static_assert(isExecutionParameters<ExecParamsT>, "Unknown execution parameters type");

        mImpl->executeUnsafe(src, dst, execParams);
      }

      /**
       * @brief Execute out-of-place plan without type checking and with source and destination buffers as PlanarComplex
       * @tparam SrcT Source buffer type in PlanarComplex format
       * @tparam DstT Destination buffer type in PlanarComplex format
       * @tparam ExecParamsT Execution parameters type
       * @param src Source buffer
       * @param dst Destination buffer
       * @param execParams Execution parameters
       */
      template<typename SrcT, typename DstT, typename ExecParamsT>
      void executeUnsafe(PlanarComplex<SrcT*> src, PlanarComplex<DstT*> dst, const ExecParamsT& execParams)
      {
        static_assert(isExecutionParameters<ExecParamsT>, "Unknown execution parameters type");

        mImpl->executeUnsafe(src, dst, execParams);
      }
    protected:
    private:
      // Allow makePlan to create Plan objects
      template<typename TransformParametersT>
      friend Plan makePlan(const TransformParametersT&         transformParams,
                           const cpu::Parameters&              cpuParams,
                           const cpu::BackendSelectParameters& cpuBackendSelectParams);
    
      template<typename TransformParametersT>
      friend Plan makePlan(const TransformParametersT&         transformParams,
                           const gpu::Parameters&              gpuParams,
                           const gpu::BackendSelectParameters& gpuBackendSelectParams);

      // Allow PlanCache to create Plan objects
      friend class PlanCache;

      /**
       * @brief Construct a new Plan object with the given implementation
       * @param impl Plan implementation
       */
      explicit Plan(std::shared_ptr<detail::PlanImpl> impl)
      : mImpl{std::move(impl)}
      {}

      std::shared_ptr<detail::PlanImpl> mImpl; ///< Plan implementation
  };

  /**
   * @brief Create a plan for the given transform parameters and CPU backend parameters
   * @param transformParams Transform parameters
   * @param cpuParams CPU backend parameters
   * @param cpuBackendSelectParams CPU backend selection parameters
   * @return Plan
   */
  template<typename TransformParametersT>
  Plan makePlan(const TransformParametersT&         transformParams,
                const cpu::Parameters&              cpuParams,
                const cpu::BackendSelectParameters& cpuBackendSelectParams)
  {
    static_assert(isTransformParameters<TransformParametersT>, "Invalid transform parameters type");

    return Plan{detail::makePlanImpl(detail::Config(transformParams, cpuParams), cpuBackendSelectParams)};
  }

  /**
   * @brief Create a plan for the given transform parameters and GPU backend parameters
   * @param transformParams Transform parameters
   * @param gpuParams GPU backend parameters
   * @param gpuBackendSelectParams GPU backend selection parameters
   * @return Plan
   */
  template<typename TransformParametersT>
  Plan makePlan(const TransformParametersT&         transformParams,
                const gpu::Parameters&              gpuParams,
                const gpu::BackendSelectParameters& gpuBackendSelectParams)
  {
    static_assert(isTransformParameters<TransformParametersT>, "Invalid transform parameters type");

    return Plan{detail::makePlanImpl(detail::Config(transformParams, gpuParams), gpuBackendSelectParams)};
  }

  /**
   * @brief Create a plan for the given transform parameters
   * @param transformParams Transform parameters
   * @param targetParams Target parameters
   * @return Plan
   */
  template<TransformParametersType TransformParamsT, TargetParametersType TargetParamsT>
  Plan makePlan(const TransformParamsT& transformParams, const TargetParamsT& targetParams)
  {
    static_assert(isTransformParameters<TransformParamsT>, "Invalid transform parameters type");
    static_assert(isTargetParameters<TargetParamsT>, "Invalid target parameters type");

    if constexpr (std::same_as<TargetParamsT, cpu::Parameters>)
    {
      return makePlan(transformParams, targetParams, cpu::BackendSelectParameters{});
    }
    else if constexpr (std::same_as<TargetParamsT, gpu::Parameters>)
    {
      return makePlan(transformParams, targetParams, gpu::BackendSelectParameters{});
    }
    else
    {
      detail::unreachable();
    }
  }
} // namespace afft

#endif /* AFFT_PLAN_HPP */
