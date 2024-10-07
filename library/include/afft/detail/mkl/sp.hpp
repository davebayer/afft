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
  /**
   * @brief Create a mkl single process plan implementation.
   * @tparam BackendParamsT Backend parameters type.
   * @param desc Plan description.
   * @param backendParams Backend parameters.
   * @return Plan implementation.
   */
  template<typename BackendParamsT>
  [[nodiscard]] std::unique_ptr<afft::Plan>
  makePlan(const Description& desc, const BackendParamsT& backendParams);
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

    /**
     * @brief Create a mkl single process cpu plan implementation.
     * @param desc Plan description.
     * @param backendParams Backend parameters.
     * @return Plan implementation.
     */
    [[nodiscard]] std::unique_ptr<afft::Plan>
    makePlan(const Description&                  desc,
             const afft::cpu::BackendParameters& backendParams);
  } // namespace cpu

  namespace openmp
  {
    /**
     * @class Plan
     * @brief The mkl single process openmp plan implementation.
     */
    class Plan;

    /**
     * @brief Create a mkl single process openmp plan implementation.
     * @param desc Plan description.
     * @param backendParams Backend parameters.
     * @return Plan implementation.
     */
    [[nodiscard]] std::unique_ptr<afft::Plan>
    makePlan(const Description&                     desc,
             const afft::openmp::BackendParameters& backendParams);
  } // namespace openmp

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

#ifdef AFFT_ENABLE_CPU  
  /**
   * @class Plan
   * @brief The mkl single process cpu plan implementation.
   */
  class cpu::Plan final : public mkl::Plan<MpBackend::none, Target::cpu>
  {
    private:
      /// @brief Alias for the parent class
      using Parent = mkl::Plan<MpBackend::none, Target::cpu>;

    public:
      /// @brief Inherit constructor.
      using Parent::Parent;

      /**
       * @brief Constructor
       * @param desc The plan description
       * @param backendParams Backend parameters
       */
      Plan(const Description& desc, const afft::cpu::BackendParameters& backendParams)
      : Parent{desc, backendParams}
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
                                            transformDims.data));
          }

          mDftiHandle.reset(dftiHandle);
        }

        checkError(DftiSetValue(mDftiHandle.get(), DFTI_PLACEMENT, getPlacement()));

        if (getPrecision() == DFTI_DOUBLE)
        {
          checkError(DftiSetValue(mDftiHandle.get(), getScaleConfigParam(), mDesc.getNormalizationFactor<double>()));
        }
        else
        {
          checkError(DftiSetValue(mDftiHandle.get(), getScaleConfigParam(), mDesc.getNormalizationFactor<float>()));
        }

        checkError(DftiSetValue(mDftiHandle.get(), DFTI_THREAD_LIMIT, static_cast<MKL_LONG>(Parent::mBackendParams.threadLimit)));

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

        {
          MKL_LONG srcStrides[maxDimCount + 1]{};
          MKL_LONG dstStrides[maxDimCount + 1]{};

          std::size_t i{1};

          for (auto axis : mDesc.getTransformAxes())
          {
            srcStrides[i] = safeIntCast<MKL_LONG>(memDesc.getSrcStrides()[axis]);
            dstStrides[i] = safeIntCast<MKL_LONG>(memDesc.getDstStrides()[axis]);
            ++i;
          }
          
          checkError(DftiSetValue(mDftiHandle.get(), DFTI_INPUT_STRIDES, srcStrides));
          checkError(DftiSetValue(mDftiHandle.get(), DFTI_OUTPUT_STRIDES, dstStrides));
        }
        
        {
          const auto srcStrides = memDesc.getSrcStrides();
          const auto dstStrides = memDesc.getDstStrides();

          MKL_LONG numberOfTransforms{1};
          MKL_LONG srcDist{1};
          MKL_LONG dstDist{1};

          if (const auto howManyRank = mDesc.getTransformHowManyRank(); howManyRank == 1)
          {
            const auto howManyAxis = mDesc.getTransformHowManyAxes().front();

            numberOfTransforms = safeIntCast<MKL_LONG>(Parent::mDesc.getShape()[howManyAxis]);
            srcDist            = safeIntCast<MKL_LONG>(srcStrides[howManyAxis]);
            dstDist            = safeIntCast<MKL_LONG>(dstStrides[howManyAxis]);
          }
          else if (howManyRank > 1)
          {
            const auto shape       = Parent::mDesc.getShape();
            const auto srcShape    = Parent::mDesc.getSrcShape();
            const auto dstShape    = Parent::mDesc.getDstShape();
            const auto howManyAxes = mDesc.getTransformHowManyAxes();

            numberOfTransforms = shape[howManyAxes.front()];
            srcDist            = safeIntCast<MKL_LONG>(srcStrides[howManyAxes.back()]);
            dstDist            = safeIntCast<MKL_LONG>(dstStrides[howManyAxes.back()]);

            for (std::size_t i = howManyAxes.size() - 1; i > 0; --i)
            {
              if (howManyAxes[i] != howManyAxes[i - 1] + 1)
              {
                throw Exception{Error::cufft, "unsupported how many axes"};
              }

              if (srcStrides[howManyAxes[i]] * srcShape[howManyAxes[i]] != srcStrides[howManyAxes[i - 1]])
              {
                throw Exception{Error::cufft, "unsupported how many strides"};
              }

              if (dstStrides[howManyAxes[i]] * dstShape[howManyAxes[i]] != dstStrides[howManyAxes[i - 1]])
              {
                throw Exception{Error::cufft, "unsupported how many strides"};
              }

              numberOfTransforms *= safeIntCast<MKL_LONG>(shape[howManyAxes[i]]);
            }
          }

          checkError(DftiSetValue(mDftiHandle.get(), DFTI_NUMBER_OF_TRANSFORMS, numberOfTransforms));
          checkError(DftiSetValue(mDftiHandle.get(), DFTI_INPUT_DISTANCE, srcDist));
          checkError(DftiSetValue(mDftiHandle.get(), DFTI_OUTPUT_DISTANCE, dstDist));
        }

        if (Parent::mBackendParams.mkl.avoidWorkspace)
        {
          checkError(DftiSetValue(mDftiHandle.get(), DFTI_WORKSPACE, DFTI_AVOID));
        }

        if (Parent::mBackendParams.allowDestructive)
        {
          checkError(DftiSetValue(mDftiHandle.get(), DFTI_DESTROY_INPUT, DFTI_ALLOW));
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
        return {mSrcElemCount.data(), mDesc.getSrcDstBufferCount().first};
      }

      /**
       * @brief Get element count of the destination buffers.
       * @return Element count of the destination buffers.
       */
      [[nodiscard]] View<std::size_t> getDstElemCounts() const noexcept override
      {
        return {mDstElemCount.data(), mDesc.getSrcDstBufferCount().second};
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
      std::unique_ptr<DftiDesc, DftiDescDeleter> mDftiHandle{};   ///< MKL DFTI descriptor handle
      std::array<std::size_t, 2>                 mSrcElemCount{}; ///< The number of elements in the source buffer
      std::array<std::size_t, 2>                 mDstElemCount{}; ///< The number of elements in the destination buffer
  };
#endif /* AFFT_ENABLE_CPU */

#if defined(AFFT_ENABLE_OPENMP) && defined(AFFT_MKL_HAS_OMP_OFFLOAD)
  /**
   * @class Plan
   * @brief The mkl single process openmp plan implementation.
   */
  class openmp::Plan final : public mkl::Plan<MpBackend::none, Target::openmp>
  {
    private:
      /// @brief Alias for the parent class
      using Parent = mkl::Plan<MpBackend::none, Target::openmp>;

    public:
      /// @brief Inherit constructor.
      using Parent::Parent;

      /**
       * @brief Constructor
       * @param desc The plan description
       * @param backendParams Backend parameters
       */
      Plan(const Description& desc, afft::openmp::BackendParameters backendParams)
      : Parent{desc, backendParams}
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

        if (getPrecision() == DFTI_DOUBLE)
        {
          checkError(DftiSetValue(mDftiHandle.get(), getScaleConfigParam(), mDesc.getNormalizationFactor<double>()));
        }
        else
        {
          checkError(DftiSetValue(mDftiHandle.get(), getScaleConfigParam(), mDesc.getNormalizationFactor<float>()));
        }

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
          checkError(DftiSetValue(mDftiHandle.get(), DFTI_COMPLEX_STORAGE, DFTI_COMPLEX_COMPLEX));
          break;
        case dft::Type::realToComplex:
        case dft::Type::complexToReal:
          checkError(DftiSetValue(mDftiHandle.get(), DFTI_CONJUGATE_EVEN_STORAGE, DFTI_COMPLEX_COMPLEX));
          break;
        default:
          cxx::unreachable();
        }

        if (getWorkspace() == Workspace::none)
        {
          checkError(DftiSetValue(mDftiHandle.get(), DFTI_WORKSPACE, DFTI_AVOID));
        }

        if (mDesc.isDestructive())
        {
          checkError(DftiSetValue(mDftiHandle.get(), DFTI_DESTROY_INPUT, DFTI_ALLOW));
        }

        {
          const int dev = mDesc.getTargetDesc<Target::openmp>().device;
          MKL_LONG retval{};

#         pragma omp dispatch device(dev)
          retval = DftiCommitDescriptor(mDftiHandle.get());

          checkError(retval);
        }

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
        return {mSrcElemCount.data(), mDesc.getSrcDstBufferCount().first};
      }

      /**
       * @brief Get element count of the destination buffers.
       * @return Element count of the destination buffers.
       */
      [[nodiscard]] View<std::size_t> getDstElemCounts() const noexcept override
      {
        return {mDstElemCount.data(), mDesc.getSrcDstBufferCount().second};
      }

      /**
       * @brief Execute the plan
       * @param src The source buffer
       * @param dst The destination buffer
       */
      void executeBackendImpl(View<void*> src, View<void*> dst, const afft::openmp::ExecutionParameters& execParams) override
      {
        const int dev = mDesc.getTargetDesc<Target::openmp>().device;

        MKL_LONG retval{};

        if (execParams.nowait)
        {
          if (mDesc.getPlacement() == Placement::inPlace)
          {
            if (mDesc.getDirection() == Direction::forward)
            {
#             pragma omp dispatch device(dev) need_device_ptr(2) nowait
              retval = DftiComputeForward(mDftiHandle.get(), src[0]);
            }
            else
            {
#             pragma omp dispatch device(dev) need_device_ptr(2) nowait
              retval = DftiComputeBackward(mDftiHandle.get(), src[0]);
            }
          }
          else
          {
            if (mDesc.getDirection() == Direction::forward)
            {
#             pragma omp dispatch device(dev) need_device_ptr(2, 3) nowait
              retval = DftiComputeForward(mDftiHandle.get(), src[0], dst[0]);
            }
            else
            {
#             pragma omp dispatch device(dev) need_device_ptr(2, 3) nowait
              retval = DftiComputeBackward(mDftiHandle.get(), src[0], dst[0]);
            }
          }
        }
        else
        {
          if (mDesc.getPlacement() == Placement::inPlace)
          {
            if (mDesc.getDirection() == Direction::forward)
            {
#             pragma omp dispatch device(dev) need_device_ptr(2)
              retval = DftiComputeForward(mDftiHandle.get(), src[0]);
            }
            else
            {
#             pragma omp dispatch device(dev) need_device_ptr(2)
              retval = DftiComputeBackward(mDftiHandle.get(), src[0]);
            }
          }
          else
          {
            if (mDesc.getDirection() == Direction::forward)
            {
#             pragma omp dispatch device(dev) need_device_ptr(2, 3)
              retval = DftiComputeForward(mDftiHandle.get(), src[0], dst[0]);
            }
            else
            {
#             pragma omp dispatch device(dev) need_device_ptr(2, 3)
              retval = DftiComputeBackward(mDftiHandle.get(), src[0], dst[0]);
            }
          }
        }

        checkError(retval);
      }
    
    private:
      std::unique_ptr<DftiDesc, DftiDescDeleter> mDftiHandle{};   ///< MKL DFTI descriptor handle
      std::array<std::size_t, 2>                 mSrcElemCount{}; ///< The number of elements in the source buffer
      std::array<std::size_t, 2>                 mDstElemCount{}; ///< The number of elements in the destination buffer
  };
#endif /* AFFT_ENABLE_OPENMP */

  /**
   * @brief Create a mkl single process cpu plan implementation.
   * @param desc Plan description.
   * @param backendParams Backend parameters.
   * @return Plan implementation.
   */
  [[nodiscard]] AFFT_HEADER_ONLY_INLINE std::unique_ptr<afft::Plan>
  cpu::makePlan(const Description&                  desc,
                const afft::cpu::BackendParameters& backendParams)
  {
# ifdef AFFT_ENABLE_CPU
    const auto& descImpl = desc.get(DescToken::make());

    if (descImpl.getTransformRank() > 7)
    {
      throw Exception{Error::mkl, "only up to 7 transformed dimensions are supported"};
    }

    return std::make_unique<Plan>(desc, backendParams);
# else
    throw Exception{Error::mkl, "cpu backend is not enabled"};
# endif
  }

  /**
   * @brief Create a mkl single process openmp plan implementation.
   * @param desc Plan description.
   * @param backendParams Backend parameters.
   * @return Plan implementation.
   */
  [[nodiscard]] AFFT_HEADER_ONLY_INLINE std::unique_ptr<afft::Plan>
  openmp::makePlan([[maybe_unused]] const Description&                     desc,
                   [[maybe_unused]] const afft::openmp::BackendParameters& backendParams)
  {
# ifdef AFFT_ENABLE_OPENMP
#   ifdef AFFT_MKL_HAS_OMP_OFFLOAD
    const auto& descImpl = desc.get(DescToken::make());

    if (descImpl.getTransformRank() > 3)
    {
      throw Exception{Error::mkl, "only up to 3 transformed dimensions are supported"};
    }

    if (descImpl.getTransform() != Transform::dft)
    {
      throw Exception{Error::mkl, "only DFT transform is supported"};
    }    

    if (descImpl.getComplexFormat() != ComplexFormat::interleaved)
    {
      throw Exception{Error::mkl, "only interleaved complex format is supported"};
    }

    return std::make_unique<Plan>(desc, backendParams);
#   else
    throw Exception{Error::mkl, "mkl omp offload is disabled"};
#   endif /* AFFT_MKL_HAS_OMP_OFFLOAD */
# else
    throw Exception{Error::mkl, "openmp backend is not enabled"};
# endif /* AFFT_ENABLE_OPENMP */
  }

  /**
   * @brief Create a mkl single process plan implementation.
   * @tparam BackendParamsT Backend parameters type.
   * @param desc Plan description.
   * @param backendParams Backend parameters.
   * @return Plan implementation.
   */
  template<typename BackendParamsT>
  [[nodiscard]] std::unique_ptr<afft::Plan>
  makePlan(const Description& desc, const BackendParamsT& backendParams)
  {
    const auto& descImpl = desc.get(DescToken::make());

    if (!descImpl.hasUniformPrecision())
    {
      throw Exception{Error::mkl, "only same precision for execution, source and destination is supported"};
    }

    if constexpr (BackendParamsT::target == Target::cpu)
    {
      return cpu::makePlan(desc, backendParams);
    }
    else if constexpr (BackendParamsT::target == Target::openmp)
    {
      if (descImpl.getTransformHowManyRank() > 1)
      {
        throw Exception{Error::mkl, "no more than one dimension can be omitted"};
      }

      return openmp::makePlan(desc, backendParams);
    }
    else
    {
      throw Exception{Error::mkl, "unsupported target"};
    }
  }
} // namespace afft::detail::mkl::sp

#endif /* AFFT_HEADER_ONLY */

#endif /* AFFT_DETAIL_MKL_SP_HPP */