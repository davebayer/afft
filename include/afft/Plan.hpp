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
  // Forward declaration
  class Plan;

  // Forward declarations with default parameters
  Plan makePlan(const TransformParametersType auto& transformParams,
                const cpu::Parameters&              cpuParams,
                const cpu::BackendSelectParameters& cpuBackendSelectParams = {});

  Plan makePlan(const TransformParametersType auto& transformParams,
                const gpu::Parameters&              gpuParams,
                const gpu::BackendSelectParameters& gpuBackendSelectParams = {});

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
        return mImpl->getConfig().getTargetParameters<target>();
      }

      /**
       * @brief Execute the plan without specifying execution parameters
       * @param src Source
       * @param dst Destination
       */
      void execute(auto src, auto dst)
      {
        switch (getTarget())
        {
        case Target::cpu:
          execute(src, dst, cpu::ExecutionParameters{});
          break;
        case Target::gpu:
          execute(src, dst, gpu::ExecutionParameters{});
          break;
        default:
          detail::unreachable();
        }
      }

      /**
       * @brief Execute the plan
       * @tparam SrcT Source buffer type
       * @tparam DstT Destination buffer type
       * @tparam ExecParamsT Execution parameters type
       * @param src Source buffer
       * @param dst Destination buffer
       * @param execParams Execution parameters
       */
      template<KnownType SrcT, KnownType DstT, ExecutionParametersType ExecParamsT>
      void execute(SrcT* src, DstT* dst, const ExecParamsT& execParams)
      {
        mImpl->checkExecTypes(typePrecision<SrcT>, typeComplexity<SrcT>, typePrecision<DstT>, typeComplexity<DstT>);
        executeUnsafe(src, dst, execParams);
      }

      /**
       * @brief Execute the plan with source buffer as PlanarComplex
       * @tparam SrcT Source buffer type
       * @tparam DstT Destination buffer type
       * @tparam ExecParamsT Execution parameters type
       * @param src Source buffer in PlanarComplex format
       * @param dst Destination buffer
       * @param execParams Execution parameters
       */
      template<RealType SrcT, KnownType DstT, ExecutionParametersType ExecParamsT>
      void execute(PlanarComplex<SrcT*> src, DstT* dst, const ExecParamsT& execParams)
      {
        mImpl->checkExecTypes(typePrecision<SrcT>, Complexity::complex, typePrecision<DstT>, typeComplexity<DstT>);
        executeUnsafe(src, dst, execParams);
      }

      /**
       * @brief Execute the plan with destination buffer as PlanarComplex
       * @tparam SrcT Source buffer type
       * @tparam DstT Destination buffer type
       * @tparam ExecParamsT Execution parameters type
       * @param src Source buffer
       * @param dst Destination buffer in PlanarComplex format
       * @param execParams Execution parameters
       */
      template<KnownType SrcT, RealType DstT, ExecutionParametersType ExecParamsT>
      void execute(SrcT* src, PlanarComplex<DstT*> dst, const ExecParamsT& execParams)
      {
        mImpl->checkExecTypes(typePrecision<SrcT>, typeComplexity<SrcT>, typePrecision<DstT>, Complexity::complex);
        executeUnsafe(src, dst, execParams);
      }

      /**
       * @brief Execute the plan with source and destination buffers as PlanarComplex
       * @tparam SrcT Source buffer type
       * @tparam DstT Destination buffer type
       * @tparam ExecParamsT Execution parameters type
       * @param src Source buffer in PlanarComplex format
       * @param dst Destination buffer in PlanarComplex format
       * @param execParams Execution parameters
       */
      template<RealType SrcT, RealType DstT, ExecutionParametersType ExecParamsT>
      void execute(PlanarComplex<SrcT*> src, PlanarComplex<DstT*> dst, const ExecParamsT& execParams)
      {
        mImpl->checkExecTypes(typePrecision<SrcT>, Complexity::complex, typePrecision<DstT>, Complexity::complex);
        executeUnsafe(src, dst, execParams);
      }

      /**
       * @brief Execute the plan without specifying execution parameters and without type checking
       * @param src Source
       * @param dst Destination
       */
      void executeUnsafe(auto src, auto dst)
      {
        switch (getTarget())
        {
        case Target::cpu:
          executeUnsafe(src, dst, cpu::ExecutionParameters{});
          break;
        case Target::gpu:
          executeUnsafe(src, dst, gpu::ExecutionParameters{});
          break;
        default:
          detail::unreachable();
        }
      }

      /**
       * @brief Execute the plan without type checking
       * @tparam SrcT Source buffer type
       * @tparam DstT Destination buffer type
       * @tparam ExecParamsT Execution parameters type
       * @param src Source buffer
       * @param dst Destination buffer
       * @param execParams Execution parameters
       */
      template<ExecutionParametersType ExecParamsT>
      void executeUnsafe(void* src, void* dst, const ExecParamsT& execParams)
      {
        mImpl->execute(src, dst, execParams);
      }

      /**
       * @brief Execute the plan without type checking
       * @tparam SrcT Source buffer type
       * @tparam DstT Destination buffer type
       * @tparam ExecParamsT Execution parameters type
       * @param src Source buffer
       * @param dst Destination buffer
       * @param execParams Execution parameters
       */
      template<ExecutionParametersType ExecParamsT>
      void executeUnsafe(const void* src, void* dst, const ExecParamsT& execParams)
      {
        mImpl->requireNonDestructiveTransform();
        executeUnsafe(detail::removeConstFromPtr(src), dst, execParams);
      }

      /**
       * @brief Execute the plan without type checking and with source buffer as PlanarComplex
       * @tparam SrcT Source buffer type in PlanarComplex format
       * @tparam DstT Destination buffer type
       * @tparam ExecParamsT Execution parameters type
       * @param src Source buffer
       * @param dst Destination buffer
       * @param execParams Execution parameters
       */
      template<ExecutionParametersType ExecParamsT>
      void executeUnsafe(PlanarComplex<void*> src, void* dst, const ExecParamsT& execParams)
      {
        mImpl->execute(src, dst, execParams);
      }

      /**
       * @brief Execute the plan without type checking and with source buffer as PlanarComplex
       * @tparam SrcT Source buffer type in PlanarComplex format
       * @tparam DstT Destination buffer type
       * @tparam ExecParamsT Execution parameters type
       * @param src Source buffer
       * @param dst Destination buffer
       * @param execParams Execution parameters
       */
      template<ExecutionParametersType ExecParamsT>
      void executeUnsafe(PlanarComplex<const void*> src, void* dst, const ExecParamsT& execParams)
      {
        mImpl->requireNonDestructiveTransform();
        executeUnsafe(PlanarComplex{detail::removeConstFromPtr(src.real),
                                    detail::removeConstFromPtr(src.imag)}, dst, execParams);
      }

      /**
       * @brief Execute the plan without type checking and with source and destination buffers as PlanarComplex
       * @tparam SrcT Source buffer type in PlanarComplex format
       * @tparam DstT Destination buffer type in PlanarComplex format
       * @tparam ExecParamsT Execution parameters type
       * @param src Source buffer
       * @param dst Destination buffer
       * @param execParams Execution parameters
       */
      template<ExecutionParametersType ExecParamsT>
      void executeUnsafe(void* src, PlanarComplex<void*> dst, const ExecParamsT& execParams)
      {
        mImpl->execute(src, dst, execParams);
      }

      /**
       * @brief Execute the plan without type checking and with source and destination buffers as PlanarComplex
       * @tparam SrcT Source buffer type in PlanarComplex format
       * @tparam DstT Destination buffer type in PlanarComplex format
       * @tparam ExecParamsT Execution parameters type
       * @param src Source buffer
       * @param dst Destination buffer
       * @param execParams Execution parameters
       */
      template<ExecutionParametersType ExecParamsT>
      void executeUnsafe(const void* src, PlanarComplex<void*> dst, const ExecParamsT& execParams)
      {
        mImpl->requireNonDestructiveTransform();
        executeUnsafe(detail::removeConstFromPtr(src), dst, execParams);
      }

      /**
       * @brief Execute the plan without type checking and with source and destination buffers as PlanarComplex
       * @tparam SrcT Source buffer type in PlanarComplex format
       * @tparam DstT Destination buffer type in PlanarComplex format
       * @tparam ExecParamsT Execution parameters type
       * @param src Source buffer
       * @param dst Destination buffer
       * @param execParams Execution parameters
       */
      template<ExecutionParametersType ExecParamsT>
      void executeUnsafe(PlanarComplex<void*> src, PlanarComplex<void*> dst, const ExecParamsT& execParams)
      {
        mImpl->execute(src, dst, execParams);
      }

      /**
       * @brief Execute the plan without type checking and with source and destination buffers as PlanarComplex
       * @tparam SrcT Source buffer type in PlanarComplex format
       * @tparam DstT Destination buffer type in PlanarComplex format
       * @tparam ExecParamsT Execution parameters type
       * @param src Source buffer
       * @param dst Destination buffer
       * @param execParams Execution parameters
       */
      template<ExecutionParametersType ExecParamsT>
      void executeUnsafe(PlanarComplex<const void*> src, PlanarComplex<void*> dst, const ExecParamsT& execParams)
      {
        mImpl->requireNonDestructiveTransform();
        executeUnsafe(PlanarComplex{detail::removeConstFromPtr(src.real),
                                    detail::removeConstFromPtr(src.imag)}, dst, execParams);
      }

#   if AFFT_GPU_FRAMEWORK_IS_OPENCL
      // TODO: Implement GPU execution for OpenCL buffers
      void executeUnsafe(cl_mem src, cl_mem dst, const gpu::Parameters& gpuParams)
      {
        
      }
#   endif
    protected:
    private:
      // Allow makePlan to create Plan objects
      friend Plan makePlan(const TransformParametersType auto& transformParams,
                           const cpu::Parameters&              cpuParams,
                           const cpu::BackendSelectParameters& cpuBackendSelectParams);
    
      friend Plan makePlan(const TransformParametersType auto& transformParams,
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
  Plan makePlan(const TransformParametersType auto& transformParams,
                const cpu::Parameters&              cpuParams,
                const cpu::BackendSelectParameters& cpuBackendSelectParams)
  {
    return Plan{detail::makePlanImpl(detail::Config(transformParams, cpuParams), cpuBackendSelectParams)};
  }

  /**
   * @brief Create a plan for the given transform parameters and GPU backend parameters
   * @param transformParams Transform parameters
   * @param gpuParams GPU backend parameters
   * @param gpuBackendSelectParams GPU backend selection parameters
   * @return Plan
   */
  Plan makePlan(const TransformParametersType auto& transformParams,
                const gpu::Parameters&              gpuParams,
                const gpu::BackendSelectParameters& gpuBackendSelectParams)
  {
    return Plan{detail::makePlanImpl(detail::Config(transformParams, gpuParams), gpuBackendSelectParams)};
  }
} // namespace afft

#endif /* AFFT_PLAN_HPP */
