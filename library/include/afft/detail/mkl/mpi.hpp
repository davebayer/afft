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

#ifndef AFFT_DETAIL_MKL_MPI_HPP
#define AFFT_DETAIL_MKL_MPI_HPP

#ifndef AFFT_TOP_LEVEL_INCLUDE
# include "../include.hpp"
#endif

#include "../../Plan.hpp"

namespace afft::detail::mkl::mpi
{
  /**
   * @brief Create a mkl mpi plan implementation.
   * @tparam BackendParamsT Backend parameters type.
   * @param desc Plan description.
   * @param backendParams Backend parameters.
   * @return Plan implementation.
   */
  template<typename BackendParamsT>
  [[nodiscard]] std::unique_ptr<afft::Plan>
  makePlan(const Description& desc, const BackendParamsT& backendParams);
} // namespace afft::detail::mkl::mpi

#ifdef AFFT_HEADER_ONLY

#include "Plan.hpp"

namespace afft::detail::mkl::mpi
{
  static_assert(std::is_pointer_v<DFTI_DESCRIPTOR_DM_HANDLE>, "Implementation relies on DFTI_DESCRIPTOR_DM_HANDLE being a pointer");

  namespace cpu
  {
    /**
     * @class Plan
     * @brief The mkl mpi cpu plan implementation.
     */
    class Plan;

    /**
     * @brief Create a mkl mpi cpu plan implementation.
     * @param desc Plan description.
     * @param backendParams Backend parameters.
     * @return Plan implementation.
     */
    [[nodiscard]] std::unique_ptr<afft::Plan>
    makePlan(const Description&                       desc,
             const afft::mpi::cpu::BackendParameters& backendParams);
  } // namespace cpu

#ifdef AFFT_ENABLE_CPU
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
        const auto& memDesc = mDesc.getMemDesc<MemoryLayout::distributed>();

        {
          DFTI_DESCRIPTOR_DM_HANDLE cdftHandle{};

          const auto mpiComm       = mDesc.getMpDesc<MpBackend::mpi>().comm;
          const auto transformDims = mDesc.getTransformDimsAs<MKL_LONG>();

          if (const std::size_t transformRank = mDesc.getTransformRank(); transformRank == 1)
          {
            checkError(DftiCreateDescriptorDM(mpiComm,
                                              &cdftHandle,
                                              getPrecision(),
                                              getForwardDomain(),
                                              1,
                                              transformDims[0]));
          }
          else
          {
            checkError(DftiCreateDescriptorDM(mpiComm,
                                              &cdftHandle,
                                              getPrecision(),
                                              getForwardDomain(),
                                              static_cast<MKL_LONG>(transformRank),
                                              transformDims.data()));
          }

          mCdftHandle.reset(cdftHandle);
        }

        checkError(DftiSetValueDM(mCdftHandle.get(), DFTI_PLACEMENT, getPlacement()));

        if (getPrecision() == DFTI_DOUBLE)
        {
          checkError(DftiSetValueDM(mCdftHandle.get(), getScaleConfigParam(), mDesc.getNormalizationFactor<double>()));
        }
        else
        {
          checkError(DftiSetValueDM(mCdftHandle.get(), getScaleConfigParam(), mDesc.getNormalizationFactor<float>()));
        }

        if (const std::size_t howManyRank = mDesc.getTransformHowManyRank(); howManyRank > 0)
        {
          throw Exception{Error::mkl, "howManyRank is not supported yet"};
        }

        checkError(DftiCommitDescriptorDM(mCdftHandle.get()));

        MKL_LONG n0Start{};
        checkError(DftiGetValueDM(mCdftHandle.get(), CDFT_LOCAL_X_START, &n0Start));

        MKL_LONG n0Size{};
        checkError(DftiGetValueDM(mCdftHandle.get(), CDFT_LOCAL_NX, &n0Size));

        MKL_LONG bufferSize{};
        checkError(DftiGetValueDM(mCdftHandle.get(), CDFT_LOCAL_SIZE, &bufferSize));

        if (mDesc.getTransformRank() == 1)
        {
          MKL_LONG n0FreqStart{};
          checkError(DftiGetValueDM(mCdftHandle.get(), CDFT_LOCAL_OUT_X_START, &n0FreqStart));

          MKL_LONG n0FreqSize{};
          checkError(DftiGetValueDM(mCdftHandle.get(), CDFT_LOCAL_OUT_NX, &n0FreqSize));

          // TODO: Compare values to memDesc, or just fill memDesc with them
        }
        else
        {
          // TODO: Compare values to memDesc, or just fill memDesc with them

          // TODO: set transpositions if passed
          // checkError(DftiSetValueDM(mCdftHandle.get(), DFTI_TRANSPOSE, DFTI_ALLOW));
        }        

        mSrcElemCount = static_cast<std::size_t>(bufferSize);
        mDstElemCount = static_cast<std::size_t>(bufferSize);

        if (mDesc.getPlacement() == Placement::inPlace)
        {
          if (mDesc.getPrecision().execution == Precision::f32)
          {
            mWorkspaceSize = static_cast<std::size_t>(bufferSize) * sizeof(std::complex<float>);
          }
          else
          {
            mWorkspaceSize = static_cast<std::size_t>(bufferSize) * sizeof(std::complex<double>);
          }
        }
      }

      /// @brief Default destructor.
      ~Plan() = default;

      /// @brief Inherit assignment operator.
      using Parent::operator=;

      /**
       * @brief Get element count of the source buffers.
       * @return Element count of the source buffers.
       */
      [[nodiscard]] const std::size_t* getSrcElemCounts() const noexcept override
      {
        return std::addressof(mSrcElemCount);
      }

      /**
       * @brief Get element count of the destination buffers.
       * @return Element count of the destination buffers.
       */
      [[nodiscard]] const std::size_t* getDstElemCounts() const noexcept override
      {
        return std::addressof(mDstElemCount);
      }

      /**
       * @brief Get external workspace sizes. Only valid if the workspace is external.
       * @return External workspace sizes.
       */
      [[nodiscard]] virtual const std::size_t* getExternalWorkspaceSizes() const noexcept override
      {
        return (getWorkspace() == Workspace::external) ? std::addressof(mWorkspaceSize) : nullptr;
      }

      /**
       * @brief Execute the plan
       * @param src The source buffer
       * @param dst The destination buffer
       */
      void executeBackendImpl(void* const* src, void* const* dst, const afft::cpu::ExecutionParameters& execParams) override
      {
        if (getWorkspace() == Workspace::external)
        {
          checkError(DftiSetValueDM(mCdftHandle.get(), CDFT_WORKSPACE, execParams.externalWorkspace));
          checkError(DftiCommitDescriptorDM(mCdftHandle.get()));
        }

        const auto computeFn = (mDesc.getDirection() == Direction::forward)
                                 ? DftiComputeForwardDM : DftiComputeBackwardDM;

        checkError(computeFn(mCdftHandle.get(), src[0], dst[0]));
      }
    
    private:
      /// @brief Alias for the cdft descriptor.
      using CdftDesc = std::remove_pointer_t<DFTI_DESCRIPTOR_DM_HANDLE>;

      /// @brief Delete the cdft descriptor.
      struct CdftDescDeleter
      {
        /**
         * @brief Delete the descriptor.
         * @param desc The descriptor.
         */
        void operator()(CdftDesc* desc) const
        {
          DftiFreeDescriptorDM(&desc);
        }
      };

      std::unique_ptr<CdftDesc, CdftDescDeleter> mCdftHandle{};    ///< MKL CDFT descriptor handle
      std::size_t                                mSrcElemCount{};  ///< The number of elements in the source buffer
      std::size_t                                mDstElemCount{};  ///< The number of elements in the destination buffer
      std::size_t                                mWorkspaceSize{}; ///< The workspace size
  };
#endif /* AFFT_ENABLE_CPU */

  /**
   * @brief Create a mkl mpi cpu plan implementation.
   * @param desc Plan description.
   * @param backendParams Backend parameters.
   * @return Plan implementation.
   */
  [[nodiscard]] std::unique_ptr<afft::Plan>
  cpu::makePlan(const Description&                       desc,
                const afft::mpi::cpu::BackendParameters& backendParams)
  {
# ifdef AFFT_ENABLE_CPU
    // MKL DFTI for cpu supports up to 7 dimensions
    static constexpr std::size_t dftiMaxDimCount{7};

    if (desc.getTransformRank() > dftiMaxDimCount)
    {
      throw Exception{Error::mkl, "only up to 7 transformed dimensions are supported"};
    }

    // if (desc.getPlacement() == Placement::outOfPlace)
    // {
    //   workspace = Workspace::none;
    // }
    // else
    // {
    //   switch (workspace)
    //   {
    //   case Workspace::any:
    //   case Workspace::internal:
    //     workspace = Workspace::internal;
    //     break;
    //   case Workspace::none:
    //     throw Exception{Error::mkl, "in-place transforms requires workspace"};
    //   case Workspace::external:
    //     if (desc.getPlacement() != Placement::inPlace)
    //     {
    //       workspace = 
    //     }
    //     break;
    //   default:
    //     throw Exception{Error::mkl, "only internal, none or any workspace is supported"};
    //   }
    // }

    if (!desc.hasUniformPrecision())
    {
      throw Exception{Error::mkl, "only same precision for execution, source and destination is supported"};
    }

    if (desc.getComplexFormat() != ComplexFormat::interleaved)
    {
      throw Exception{Error::mkl, "only interleaved complex format is supported"};
    }

    return std::make_unique<Plan>(desc, workspace);
# else
    throw Exception{Error::mkl, "cpu backend is not enabled"};
# endif
  }

  /**
   * @brief Create a mkl mpi plan implementation.
   * @tparam BackendParamsT Backend parameters type.
   * @param desc Plan description.
   * @param backendParams Backend parameters.
   * @return Plan implementation.
   */
  template<typename BackendParamsT>
  [[nodiscard]] AFFT_HEADER_ONLY_INLINE std::unique_ptr<afft::Plan>
  makePlan(const Description& desc, const BackendParamsT& backendParams)
  {
    if constexpr (BackendParamsT::target == Target::cpu)
    {
      return cpu::makePlan(desc, backendParams);
    }
    else
    {
      throw Exception{Error::mkl, "unsupported mpi target"};
    }
  }
} // namespace afft::detail::mkl::mpi

#endif /* AFFT_HEADER_ONLY */

#endif /* AFFT_DETAIL_MKL_MPI_HPP */
