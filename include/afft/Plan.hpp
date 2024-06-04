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

#ifndef AFFT_TOP_LEVEL_INCLUDE
# include "detail/include.hpp"
#endif

#include "common.hpp"
#include "distrib.hpp"
#include "typeTraits.hpp"
#include "detail/cxx.hpp"
#include "detail/Desc.hpp"
#include "detail/makePlanImpl.hpp"
#include "detail/PlanImpl.hpp"

AFFT_EXPORT namespace afft
{
  /**
   * @class Plan
   * @brief Plan class
   */
  class Plan
  {
    public:
      /// Default constructor. Creates an uninitialized plan.
      Plan() = default;

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
       * @brief Check if the plan is initialized
       * @return True if the plan is initialized, false otherwise
       */
      [[nodiscard]] bool isInitialized() const
      {
        return static_cast<bool>(mImpl);
      }

      /**
       * @brief Get the transform type
       * @return Transform type
       */
      [[nodiscard]] Transform getTransform() const
      {
        checkInitialized();

        return mImpl->getDesc().getTransform();
      }

      /**
       * @brief Get the transform parameters
       * @tparam transform Transform type
       * @return Transform parameters
       */
      template<Transform transform>
      [[nodiscard]] TransformParameters<transform> getTransformParameters() const
      {
        static_assert(detail::isValid(transform), "Invalid transform type");

        checkInitialized();

        if (transform != getTransform())
        {
          throw std::runtime_error("Plan transform does not match requested transform");
        }

        return mImpl->getDesc().getTransformParameters<transform>();
      }

      /**
       * @brief Get target
       * @return Target
       */
      [[nodiscard]] Target getTarget() const
      {
        checkInitialized();

        return mImpl->getDesc().getTarget();
      }

      /**
       * @brief Get the distribution
       * @return Distribution
       */
      [[nodiscard]] Distribution getDistribution() const
      {
        checkInitialized();

        return mImpl->getDesc().getDistribution();
      }

      /**
       * @brief Get target parameters
       * @tparam target Target type
       * @return Target parameters
       */
      template<Target target, Distribution distrib = Distribution::spst>
      [[nodiscard]] ArchitectureParameters<target, distrib> getArchitectureParameters() const
      {
        static_assert(detail::isValid(target), "Invalid target type");
        static_assert(detail::isValid(distrib), "Invalid distribution");

        checkInitialized();

        if (target != getTarget())
        {
          throw std::runtime_error("Plan target does not match requested target");
        }

        if (distrib != getDistribution())
        {
          throw std::runtime_error("Plan distribution does not match requested distribution");
        }

        return mImpl->getDesc().getArchitectureParameters<target, distrib>();
      }

      /**
       * @brief Get plan backend.
       * @return Backend.
       */
      [[nodiscard]] Backend getBackend() const
      {
        checkInitialized();

        return mImpl->getBackend();
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
        static_assert(!std::is_const_v<SrcDstT>, "A non-const type is required for the source and destination buffer");
        static_assert(isExecutionParameters<ExecParamsT>, "Unknown execution parameters type");

        checkInitialized();

        mImpl->execute(detail::makeView(srcDst), execParams);
      }

      /**
       * @brief Execute in-place plan
       * @tparam SrcDstT Source or destination buffer type
       * @tparam ExecParamsT Execution parameters type
       * @param srcDst Source and destination buffers
       * @param execParams Execution parameters
       */
      template<typename SrcDstT, typename ExecParamsT>
      void execute(View<SrcDstT*> srcDst, const ExecParamsT& execParams)
      {
        static_assert(isKnownType<SrcDstT>, "A known type is required");
        static_assert(!std::is_const_v<SrcDstT>, "A non-const type is required for the source and destination buffer");
        static_assert(isExecutionParameters<ExecParamsT>, "Unknown execution parameters type");

        checkInitialized();

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
        static_assert(!std::is_const_v<SrcDstT>, "A non-const type is required for the source and destination buffer");
        static_assert(isExecutionParameters<ExecParamsT>, "Unknown execution parameters type");

        checkInitialized();

        mImpl->execute(detail::makeView(srcDst), execParams);
      }

      /**
       * @brief Execute in-place plan
       * @tparam SrcDstT Source or destination buffer type
       * @tparam ExecParamsT Execution parameters type
       * @param srcDst Source and destination buffers
       * @param execParams Execution parameters
       */
      template<typename SrcDstT, typename ExecParamsT>
      void execute(View<PlanarComplex<SrcDstT*>> srcDst, const ExecParamsT& execParams)
      {
        static_assert(isRealType<SrcDstT>, "A real type is required");
        static_assert(!std::is_const_v<SrcDstT>, "A non-const type is required for the source and destination buffer");
        static_assert(isExecutionParameters<ExecParamsT>, "Unknown execution parameters type");

        checkInitialized();

        mImpl->execute(srcDst, execParams);
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
        static_assert(!std::is_const_v<DstT>, "A non-const type is required for the destination buffer");
        static_assert(isExecutionParameters<ExecParamsT>, "Unknown execution parameters type");

        checkInitialized();

        mImpl->execute(detail::makeView(src), detail::makeView(dst), execParams);
      }

      /**
       * @brief Execute out-of-place plan
       * @tparam SrcT Source buffer type
       * @tparam DstT Destination buffer type
       * @tparam ExecParamsT Execution parameters type
       * @param src Source buffers
       * @param dst Destination buffers
       * @param execParams Execution parameters
       */
      template<typename SrcT, typename DstT, typename ExecParamsT>
      void execute(View<SrcT*> src, View<DstT*> dst, const ExecParamsT& execParams)
      {
        static_assert(isKnownType<SrcT>, "A known type is required");
        static_assert(isKnownType<DstT>, "A known type is required");
        static_assert(!std::is_const_v<DstT>, "A non-const type is required for the destination buffer");
        static_assert(isExecutionParameters<ExecParamsT>, "Unknown execution parameters type");

        checkInitialized();

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
        static_assert(!std::is_const_v<DstT>, "A non-const type is required for the destination buffer");
        static_assert(isExecutionParameters<ExecParamsT>, "Unknown execution parameters type");

        checkInitialized();

        mImpl->execute(detail::makeView(src), detail::makeView(dst), execParams);
      }

      /**
       * @brief Execute out-of-place plan with source buffer as PlanarComplex
       * @tparam SrcT Source buffer type
       * @tparam DstT Destination buffer type
       * @tparam ExecParamsT Execution parameters type
       * @param src Source buffers in PlanarComplex format
       * @param dst Destination buffers
       * @param execParams Execution parameters
       */
      template<typename SrcT, typename DstT, typename ExecParamsT>
      void execute(View<PlanarComplex<SrcT*>> src, View<DstT*> dst, const ExecParamsT& execParams)
      {
        static_assert(isRealType<SrcT>, "A real type is required");
        static_assert(isKnownType<DstT>, "A known type is required");
        static_assert(!std::is_const_v<DstT>, "A non-const type is required for the destination buffer");
        static_assert(isExecutionParameters<ExecParamsT>, "Unknown execution parameters type");

        checkInitialized();

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
        static_assert(!std::is_const_v<DstT>, "A non-const type is required for the destination buffer");
        static_assert(isExecutionParameters<ExecParamsT>, "Unknown execution parameters type");

        checkInitialized();

        mImpl->execute(detail::makeView(src), detail::makeView(dst), execParams);
      }

      /**
       * @brief Execute out-of-place plan with destination buffer as PlanarComplex
       * @tparam SrcT Source buffer type
       * @tparam DstT Destination buffer type
       * @tparam ExecParamsT Execution parameters type
       * @param src Source buffers
       * @param dst Destination buffers in PlanarComplex format
       * @param execParams Execution parameters
       */
      template<typename SrcT, typename DstT, typename ExecParamsT>
      void execute(View<SrcT*> src, View<PlanarComplex<DstT*>> dst, const ExecParamsT& execParams)
      {
        static_assert(isKnownType<SrcT>, "A known type is required");
        static_assert(isRealType<DstT>, "A real type is required");
        static_assert(!std::is_const_v<DstT>, "A non-const type is required for the destination buffer");
        static_assert(isExecutionParameters<ExecParamsT>, "Unknown execution parameters type");

        checkInitialized();

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
        static_assert(!std::is_const_v<DstT>, "A non-const type is required for the destination buffer");
        static_assert(isExecutionParameters<ExecParamsT>, "Unknown execution parameters type");

        checkInitialized();

        mImpl->execute(detail::makeView(src), detail::makeView(dst), execParams);
      }

      /**
       * @brief Execute out-of-place plan with source and destination buffers as PlanarComplex
       * @tparam SrcT Source buffer type
       * @tparam DstT Destination buffer type
       * @tparam ExecParamsT Execution parameters type
       * @param src Source buffers in PlanarComplex format
       * @param dst Destination buffers in PlanarComplex format
       * @param execParams Execution parameters
       */
      template<typename SrcT, typename DstT, typename ExecParamsT>
      void execute(View<PlanarComplex<SrcT*>> src, View<PlanarComplex<DstT*>> dst, const ExecParamsT& execParams)
      {
        static_assert(isRealType<SrcT>, "A real type is required");
        static_assert(isRealType<DstT>, "A real type is required");
        static_assert(!std::is_const_v<DstT>, "A non-const type is required for the destination buffer");
        static_assert(isExecutionParameters<ExecParamsT>, "Unknown execution parameters type");

        checkInitialized();

        mImpl->execute(src, dst, execParams);
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
        static_assert(!std::is_const_v<DstT>, "A non-const type is required for the destination buffer");
        static_assert(isExecutionParameters<ExecParamsT>, "Unknown execution parameters type");

        checkInitialized();

        mImpl->executeUnsafe(detail::makeView(src), detail::makeView(dst), execParams);
      }

      /**
       * @brief Execute out-of-place plan without type checking
       * @tparam SrcT Source buffer type
       * @tparam DstT Destination buffer type
       * @tparam ExecParamsT Execution parameters type
       * @param src Source buffers
       * @param dst Destination buffers
       * @param execParams Execution parameters
       */
      template<typename SrcT, typename DstT, typename ExecParamsT>
      void executeUnsafe(View<SrcT*> src, View<DstT*> dst, const ExecParamsT& execParams)
      {
        static_assert(!std::is_const_v<DstT>, "A non-const type is required for the destination buffer");
        static_assert(isExecutionParameters<ExecParamsT>, "Unknown execution parameters type");

        checkInitialized();

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
        static_assert(!std::is_const_v<DstT>, "A non-const type is required for the destination buffer");
        static_assert(isExecutionParameters<ExecParamsT>, "Unknown execution parameters type");

        checkInitialized();

        mImpl->executeUnsafe(detail::makeView(src), detail::makeView(dst), execParams);
      }

      /**
       * @brief Execute out-of-place plan without type checking and with source buffer as PlanarComplex
       * @tparam SrcT Source buffer type in PlanarComplex format
       * @tparam DstT Destination buffer type
       * @tparam ExecParamsT Execution parameters type
       * @param src Source buffers
       * @param dst Destination buffers
       * @param execParams Execution parameters
       */
      template<typename SrcT, typename DstT, typename ExecParamsT>
      void executeUnsafe(View<PlanarComplex<SrcT*>> src, View<DstT*> dst, const ExecParamsT& execParams)
      {
        static_assert(!std::is_const_v<DstT>, "A non-const type is required for the destination buffer");
        static_assert(isExecutionParameters<ExecParamsT>, "Unknown execution parameters type");

        checkInitialized();

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
        static_assert(!std::is_const_v<DstT>, "A non-const type is required for the destination buffer");
        static_assert(isExecutionParameters<ExecParamsT>, "Unknown execution parameters type");

        checkInitialized();

        mImpl->executeUnsafe(detail::makeView(src), detail::makeView(dst), execParams);
      }

      /**
       * @brief Execute out-of-place plan without type checking and with source and destination buffers as PlanarComplex
       * @tparam SrcT Source buffer type in PlanarComplex format
       * @tparam DstT Destination buffer type in PlanarComplex format
       * @tparam ExecParamsT Execution parameters type
       * @param src Source buffers
       * @param dst Destination buffers
       * @param execParams Execution parameters
       */
      template<typename SrcT, typename DstT, typename ExecParamsT>
      void executeUnsafe(View<SrcT*> src, View<PlanarComplex<DstT*>> dst, const ExecParamsT& execParams)
      {
        static_assert(!std::is_const_v<DstT>, "A non-const type is required for the destination buffer");
        static_assert(isExecutionParameters<ExecParamsT>, "Unknown execution parameters type");

        checkInitialized();

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
        static_assert(!std::is_const_v<DstT>, "A non-const type is required for the destination buffer");
        static_assert(isExecutionParameters<ExecParamsT>, "Unknown execution parameters type");

        checkInitialized();

        mImpl->executeUnsafe(detail::makeView(src), detail::makeView(dst), execParams);
      }

      /**
       * @brief Execute out-of-place plan without type checking and with source and destination buffers as PlanarComplex
       * @tparam SrcT Source buffer type in PlanarComplex format
       * @tparam DstT Destination buffer type in PlanarComplex format
       * @tparam ExecParamsT Execution parameters type
       * @param src Source buffers
       * @param dst Destination buffers
       * @param execParams Execution parameters
       */
      template<typename SrcT, typename DstT, typename ExecParamsT>
      void executeUnsafe(View<PlanarComplex<SrcT*>> src, View<PlanarComplex<DstT*>> dst, const ExecParamsT& execParams)
      {
        static_assert(!std::is_const_v<DstT>, "A non-const type is required for the destination buffer");
        static_assert(isExecutionParameters<ExecParamsT>, "Unknown execution parameters type");

        checkInitialized();

        mImpl->executeUnsafe(src, dst, execParams);
      }
    protected:
    private:
      // Allow makePlan to create Plan objects
      template<typename TransformParametersT, typename ArchParametersT, typename BackendParametersT>
      friend Plan makePlan(const TransformParametersT& transformParams,
                           ArchParametersT&            archParams,
                           const BackendParametersT&   backendParams);

      // Allow makePlanWithFeedback to create Plan objects
      template<typename TransformParametersT, typename ArchParametersT, typename BackendParametersT>
      friend std::pair<Plan, std::vector<Feedback>> makePlanWithFeedback(const TransformParametersT& transformParams,
                                                                         ArchParametersT&            archParams,
                                                                         const BackendParametersT&   backendParams);

      // Allow PlanCache to create Plan objects
      friend class PlanCache;

      /**
       * @brief Construct a new Plan object with the given implementation
       * @param impl Plan implementation
       */
      explicit Plan(std::shared_ptr<detail::PlanImpl> impl)
      : mImpl{std::move(impl)}
      {}

      /**
       * @brief Check if the plan is initialized
       */
      void checkInitialized() const
      {
        if (!mImpl)
        {
          throw std::runtime_error("Plan is not initialized");
        }
      }

      std::shared_ptr<detail::PlanImpl> mImpl; ///< Plan implementation
  };

  /**
   * @brief Create a plan for the given transform, architecture and backend parameters
   * @tparam TransformParametersT Transform parameters type
   * @tparam ArchParametersT Architecture parameters type
   * @tparam BackendParametersT Backend parameters type
   * @param transformParams Transform parameters
   * @param archParams Architecutre parameters
   * @param backendParams Backend parameters
   * @return Plan
   */
  template<typename TransformParametersT, typename ArchParametersT, typename BackendParametersT>
  Plan makePlan(const TransformParametersT& transformParams,
                ArchParametersT&            archParams,
                const BackendParametersT&   backendParams)
  {
    static_assert(isTransformParameters<TransformParametersT>, "Invalid transform parameters type");
    static_assert(isArchitectureParameters<ArchParametersT>, "Invalid architecture parameters type");
    static_assert(isBackendParameters<BackendParametersT>, "Invalid backend parameters type");

    // static_assert(detail::isCompatible<ArchParametersT, BackendParametersT>,
    //              "Architecture and backend parameters must share the same target and distribution");

    static constexpr auto transformParamsShapeRank = detail::TransformParametersTemplateRanks<TransformParametersT>::shape;
    static constexpr auto archParamsShapeRank      = detail::ArchParametersTemplateRanks<ArchParametersT>::shape;

    static_assert((transformParamsShapeRank == dynamicRank) ||
                  (archParamsShapeRank == dynamicRank) ||
                  (transformParamsShapeRank == archParamsShapeRank),
                  "Transform and target parameters must have the same shape rank");

    return Plan{detail::makePlanImpl(detail::Desc{transformParams, archParams}, backendParams)};
  }

  /**
   * @brief Create a plan for the given transform and architecture parameters with default backend parameters
   * @tparam TransformParametersT Transform parameters type
   * @tparam ArchParametersT Architecture parameters type
   * @param transformParams Transform parameters
   * @param archParams Architecutre parameters
   * @return Plan
   */
  template<typename TransformParametersT, typename ArchParametersT>
  Plan makePlan(const TransformParametersT& transformParams,
                ArchParametersT&            archParams)
  {
    static_assert(isTransformParameters<TransformParametersT>, "Invalid transform parameters type");
    static_assert(isArchitectureParameters<ArchParametersT>, "Invalid architecture parameters type");

    return makePlan(transformParams, archParams, BackendParameters<ArchParametersT::target, ArchParametersT::distribution>{});
  }

  /**
   * @brief Create a plan with feedback for the given transform, architecture and backend parameters
   * @tparam TransformParametersT Transform parameters type
   * @tparam ArchParametersT Architecture parameters type
   * @tparam BackendParametersT Backend parameters type
   * @param transformParams Transform parameters
   * @param archParams Architecutre parameters
   * @param backendParams Backend parameters
   * @return Plan and feedback
   */
  template<typename TransformParametersT, typename ArchParametersT, typename BackendParametersT>
  std::pair<Plan, std::vector<Feedback>> makePlanWithFeedback(const TransformParametersT& transformParams,
                                                              ArchParametersT&            archParams,
                                                              const BackendParametersT&   backendParams)
  {
    static_assert(isTransformParameters<TransformParametersT>, "Invalid transform parameters type");
    static_assert(isArchitectureParameters<ArchParametersT>, "Invalid architecture parameters type");
    static_assert(isBackendParameters<BackendParametersT>, "Invalid backend parameters type");

    // static_assert(detail::isCompatible<ArchParametersT, BackendParametersT>,
    //              "Architecture and backend parameters must share the same target and distribution");

    static constexpr auto transformParamsShapeRank = detail::TransformParametersTemplateRanks<TransformParametersT>::shape;
    static constexpr auto archParamsShapeRank      = detail::ArchParametersTemplateRanks<ArchParametersT>::shape;

    static_assert((transformParamsShapeRank == dynamicRank) ||
                  (archParamsShapeRank == dynamicRank) ||
                  (transformParamsShapeRank == archParamsShapeRank),
                  "Transform and target parameters must have the same shape rank");

    std::pair<Plan, std::vector<Feedback>> result{};

    result.first = Plan{detail::makePlanImpl(detail::Desc{transformParams, archParams},
                                             backendParams,
                                             &result.second)};

    return result;
  }

  /**
   * @brief Create a plan with feedback for the given transform and architecture parameters with default backend parameters
   * @tparam TransformParametersT Transform parameters type
   * @tparam ArchParametersT Architecture parameters type
   * @param transformParams Transform parameters
   * @param archParams Architecutre parameters
   * @return Plan and feedback
   */
  template<typename TransformParametersT, typename ArchParametersT>
  std::pair<Plan, std::vector<Feedback>> makePlanWithFeedback(const TransformParametersT& transformParams,
                                                              ArchParametersT&          archParams)
  {
    static_assert(isTransformParameters<TransformParametersT>, "Invalid transform parameters type");
    static_assert(isArchitectureParameters<ArchParametersT>, "Invalid architecture parameters type");

    return makePlanWithFeedback(transformParams,
                                archParams,
                                BackendParameters<ArchParametersT::target, ArchParametersT::distribution>{});
  }
} // namespace afft

#endif /* AFFT_PLAN_HPP */
