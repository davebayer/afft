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

#ifndef AFFT_DETAIL_MKL_SP_HPP
#define AFFT_DETAIL_MKL_SP_HPP

#ifndef AFFT_TOP_LEVEL_INCLUDE
# include "../include.hpp"
#endif

#include "../../Plan.hpp"

namespace afft::detail::mkl::sp
{
  namespace cpu
  {
    /**
     * @brief Create a mkl single process cpu plan implementation.
     * @param desc Plan description.
     * @return Plan implementation.
     */
    [[nodiscard]] std::unique_ptr<afft::Plan> makePlan(const Desc& desc, Workspace workspace);
  } // namespace cpu

  namespace openmp
  {
    /**
     * @brief Create a mkl single process openmp plan implementation.
     * @param desc Plan description.
     * @return Plan implementation.
     */
    [[nodiscard]] std::unique_ptr<afft::Plan> makePlan(const Desc& desc, Workspace workspace);
  } // namespace openmp
} // namespace afft::detail::mkl::sp

#ifdef AFFT_HEADER_ONLY

#include "Plan.hpp"

namespace afft::detail::mkl::sp
{
  static_assert(std::is_pointer_v<DFTI_DESCRIPTOR_HANDLE>, "Implementation relies on DFTI_DESCRIPTOR_HANDLE being a pointer");

  namespace cpu
  {
    /**
     * @class Plan
     * @brief The mkl single process cpu plan implementation.
     */
    class Plan;
  } // namespace cpu

  namespace openmp
  {
    /**
     * @class Plan
     * @brief The mkl single process openmp plan implementation.
     */
    class Plan;
  } // namespace openmp
  
  /**
   * @class Plan
   * @brief The mkl single process cpu plan implementation.
   */
  class cpu::Plan final : public mkl::Plan
  {
    private:
      /// @brief Alias for the parent class
      using Parent = mkl::Plan;

    public:
      /// @brief Inherit constructor.
      using Parent::Parent;

      /**
       * @brief Constructor
       * @param Desc The plan description
       */
      Plan(const Desc& desc, Workspace workspace)
      : Parent{desc, workspace}
      {
        const auto& memDesc = mDesc.getMemDesc<MemoryLayout::centralized>();

        {
          DFTI_DESCRIPTOR_HANDLE dftiHandle{};

          const auto transformDims = mDesc.getTransformDimsAs<MKL_LONG>();

          if (const std::size_t transformRank = mDesc.getTransformRank(); transformRank == 1)
          {
            checkError(DftiCreateDescriptor(&dftiHandle,
                                            getPrecision(),
                                            getForwardDomain(),
                                            1,
                                            transformDims[0]));
          }
          else
          {
            checkError(DftiCreateDescriptor(&dftiHandle,
                                            getPrecision(),
                                            getForwardDomain(),
                                            static_cast<MKL_LONG>(transformRank),
                                            transformDims.data()));
          }

          mDftiHandle.reset(dftiHandle);
        }

        checkError(DftiSetValue(mDftiHandle.get(),
                                DFTI_PLACEMENT,
                                getPlacement()));

        const auto scaleConfigParam = (mDesc.getDirection() == Direction::forward) ? DFTI_FORWARD_SCALE : DFTI_BACKWARD_SCALE;
        if (getPrecision() == DFTI_DOUBLE)
        {
          checkError(DftiSetValue(mDftiHandle.get(),
                                  scaleConfigParam,
                                  mDesc.getNormalizationFactor<double>()));
        }
        else
        {
          checkError(DftiSetValue(mDftiHandle.get(),
                                  scaleConfigParam,
                                  mDesc.getNormalizationFactor<float>()));
        }

        checkError(DftiSetValue(mDftiHandle.get(),
                                DFTI_THREAD_LIMIT,
                                getThreadLimit()));

        if (!memDesc.hasDefaultSrcStrides())
        {
          throw Exception{Error::mkl, "custom src strides are not supported yet"};
        }

        if (!memDesc.hasDefaultDstStrides())
        {
          throw Exception{Error::mkl, "custom dst strides are not supported yet"};
        }

        if (const std::size_t howManyRank = mDesc.getTransformHowManyRank(); howManyRank > 0)
        {
          throw Exception{Error::mkl, "howManyRank is not supported yet"};
        }

        switch (mDesc.getTransformDesc<Transform::dft>().type)
        {
        case dft::Type::complexToComplex:
          if (mDesc.getComplexFormat() == ComplexFormat::interleaved)
          {
            checkError(DftiSetValue(mDftiHandle.get(), DFTI_COMPLEX_STORAGE, DFTI_COMPLEX_COMPLEX));
          }
          else
          {
            checkError(DftiSetValue(mDftiHandle.get(), DFTI_COMPLEX_STORAGE, DFTI_REAL_REAL));
          }
          break;
        case dft::Type::realToComplex:
        case dft::Type::complexToReal:
          if (mDesc.getComplexFormat() == ComplexFormat::planar)
          {
            throw Exception{Error::mkl, "Planar format is not supported for real-to-complex or complex-to-real transforms"};
          }

          checkError(DftiSetValue(mDftiHandle.get(), DFTI_CONJUGATE_EVEN_STORAGE, DFTI_COMPLEX_COMPLEX));
          break;
        default:
          cxx::unreachable();
        }

        if (getWorkspace() == Workspace::none)
        {
          checkError(DftiSetValue(mDftiHandle.get(),
                                  DFTI_WORKSPACE,
                                  DFTI_AVOID));
        }

        if (mDesc.isDestructive())
        {
          checkError(DftiSetValue(mDftiHandle.get(),
                                  DFTI_DESTROY_INPUT,
                                  DFTI_ALLOW));
        }

        checkError(DftiCommitDescriptor(mDftiHandle.get()));

        mDesc.getRefElemCounts(mSrcElemCount, mDstElemCount);
      }

      /// @brief Default destructor.
      ~Plan() = default;

      /// @brief Inherit assignment operator.
      using Parent::operator=;

      /**
       * @brief Get element count of the source buffers.
       * @return Element count of the source buffers.
       */
      [[nodiscard]] View<std::size_t> getSrcElemCounts() const noexcept override
      {
        return makeScalarView(mSrcElemCount);
      }

      /**
       * @brief Get element count of the destination buffers.
       * @return Element count of the destination buffers.
       */
      [[nodiscard]] View<std::size_t> getDstElemCounts() const noexcept override
      {
        return makeScalarView(mDstElemCount);
      }

      /**
       * @brief Execute the plan
       * @param src The source buffer
       * @param dst The destination buffer
       */
      void executeBackendImpl(View<void*> src, View<void*> dst, const afft::cpu::ExecutionParameters&) override
      {
        const auto computeFn = (mDesc.getDirection() == Direction::forward)
                                 ? DftiComputeForward : DftiComputeBackward;

        if (mDesc.getComplexFormat() == ComplexFormat::interleaved)
        {
          if (mDesc.getPlacement() == Placement::inPlace)
          {
            checkError(computeFn(mDftiHandle.get(), src[0]));
          }
          else
          {
            checkError(computeFn(mDftiHandle.get(), src[0], dst[0]));
          }
        }
        else
        {
          if (mDesc.getPlacement() == Placement::inPlace)
          {
            checkError(computeFn(mDftiHandle.get(), src[0], src[1]));
          }
          else
          {
            checkError(computeFn(mDftiHandle.get(), src[0], src[1], dst[0], dst[1]));
          }
        }
      }
    
    private:
      /// @brief Alias for the dfti descriptor.
      using DftiDesc = std::remove_pointer_t<DFTI_DESCRIPTOR_HANDLE>;

      /// @brief Delete the dfi descriptor.
      struct DftiDescDeleter
      {
        /**
         * @brief Delete the descriptor.
         * @param desc The descriptor.
         */
        void operator()(DftiDesc* desc) const
        {
          DftiFreeDescriptor(&desc);
        }
      };

      std::unique_ptr<DftiDesc, DftiDescDeleter> mDftiHandle{};   ///< MKL DFTI descriptor handle
      std::size_t                                mSrcElemCount{}; ///< The number of elements in the source buffer
      std::size_t                                mDstElemCount{}; ///< The number of elements in the destination buffer
  };

  /**
   * @brief Create a mkl single process cpu plan implementation.
   * @param desc Plan description.
   * @return Plan implementation.
   */
  [[nodiscard]] AFFT_HEADER_ONLY_INLINE std::unique_ptr<afft::Plan> cpu::makePlan(const Desc& desc, Workspace workspace)
  {
    // MKL DFTI for cpu supports up to 7 dimensions
    static constexpr std::size_t dftiMaxDimCount{7};

    if (desc.getTransformRank() > dftiMaxDimCount)
    {
      throw Exception{Error::mkl, "only up to 7 transformed dimensions are supported"};
    }

    switch (workspace)
    {
    case Workspace::any:
      workspace = Workspace::internal;
      break;
    case Workspace::internal:
    case Workspace::none:
      break;
    default:
      throw Exception{Error::mkl, "only internal, none or any workspace is supported"};
    }

    if (!desc.hasUniformPrecision())
    {
      throw Exception{Error::mkl, "only same precision for execution, source and destination is supported"};
    }

    return std::make_unique<Plan>(desc, workspace);
  }
} // namespace afft::detail::mkl::sp

#endif /* AFFT_HEADER_ONLY */

#endif /* AFFT_DETAIL_MKL_SP_HPP */