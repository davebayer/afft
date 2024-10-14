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

#ifndef AFFT_DETAIL_CUFFT_MPI_HPP
#define AFFT_DETAIL_CUFFT_MPI_HPP

#ifndef AFFT_TOP_LEVEL_INCLUDE
# include "../include.hpp"
#endif

#include "../../Plan.hpp"

namespace afft::detail::cufft::sp
{
  /**
   * @brief Create a cufft sp plan.
   * @param desc The plan descriptor.
   * @param backendParams The backend parameters.
   * @return The plan.
   */
  [[nodiscard]] std::unique_ptr<afft::Plan>
  makePlan(const Description& desc, const afft::cuda::BackendParameters& backendParams);
} // namespace afft::detail::cufft::sp

#ifdef AFFT_HEADER_ONLY

#include "common.hpp"
#include "error.hpp"
#include "Plan.hpp"

namespace afft::detail::cufft::mpi
{
  /**
   * @class PlanImpl
   * @brief Implementation of the mpi plan interface for cuFFT.
   */
  class Plan final : public cufft::Plan<MpBackend::mpi>
  {
    private:
      /// @brief Alias for the parent class.
      using Parent = cufft::Plan<MpBackend::mpi>;

    public:
      /// @brief inherit constructors
      using Parent::Parent;

      /**
       * @brief Constructor.
       * @param desc The plan descriptor.
       * @param cufftParams The cuFFT parameters.
       */
      Plan(const Description& desc, const afft::mpi::cuda::BackendParameters& backendParams)
      : Parent{desc, backendParams},
        mMpiComm{Parent::mDesc.template getMpDesc<MpBackend::mpi>().comm}
      {
        const int device = Parent::mDesc.template getTargetDesc<Target::cuda>().getDevices()[0];

        cuda::ScopedDevice scopedDevice{device};

        checkError(cufftMpAttachComm(mHandle, CUFFT_COMM_MPI, &mMpiComm));

        // if (desc.useExternalWorkspace())
        // {
        //   checkError(cufftSetAutoAllocation(mHandle, 0));
        // }

//         if (dftDesc.type == dft::Type::complexToComplex && std::all_of(n.begin(), n.end(), [](auto size){ return size <= 4096}))
//         {
// #       if CUFFT_VERSION >= 9200
//           checkError(cufftXtSetWorkAreaPolicy(planImpl->mHandle,
//                                               makeWorkAreaPolicy(cufftParams.workspacePolicy)),
//                                               &cufftParams.userWorkspaceSize);
// #       endif
//        }

        const auto precision          = Parent::mDesc.getPrecision().execution;
        const auto dftType            = Parent::mDesc.getTransformDesc<Transform::dft>().type;
        const auto [srcCmpl, dstCmpl] = Parent::mDesc.getSrcDstComplexity();

        const auto& memDesc = Parent::mDesc.template getMemDesc<MemoryLayout::centralized>();

        const auto shapeRank     = Parent::mDesc.getShapeRank();
        const auto transformRank = Parent::mDesc.getTransformRank();
        const auto transformAxes = Parent::mDesc.getTransformAxes();
        const auto srcShape      = Parent::mDesc.getSrcShape();
        const auto dstShape      = Parent::mDesc.getDstShape();
        const auto srcStrides    = memDesc.getSrcStrides();
        const auto dstStrides    = memDesc.getDstStrides();

        auto n                  = Parent::mDesc.template getTransformDimsAs<SizeT>();
        auto srcNEmbedAndStride = makeNEmbedAndStride<SizeT>({srcShape.data, shapeRank},
                                                             transformAxes,
                                                             srcStrides);
        auto dstNEmbedAndStride = makeNEmbedAndStride<SizeT>({dstShape.data, shapeRank},
                                                             transformAxes,
                                                             dstStrides);

        SizeT batch{1};
        SizeT srcDist{1};
        SizeT dstDist{1};

        if (const auto howManyRank = Parent::mDesc.getTransformHowManyRank(); howManyRank == 1)
        {
          const auto howManyAxis = mDesc.getTransformHowManyAxes().front();

          batch   = safeIntCast<SizeT>(Parent::mDesc.getShape()[howManyAxis]);
          srcDist = safeIntCast<SizeT>(srcStrides[howManyAxis]);
          dstDist = safeIntCast<SizeT>(dstStrides[howManyAxis]);
        }
        else if (howManyRank > 1)
        {
          const auto shape       = Parent::mDesc.getShape();
          const auto howManyAxes = mDesc.getTransformHowManyAxes();

          batch   = shape[howManyAxes.front()];
          srcDist = safeIntCast<SizeT>(srcStrides[howManyAxes.back()]);
          dstDist = safeIntCast<SizeT>(dstStrides[howManyAxes.back()]);

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

            batch *= safeIntCast<SizeT>(shape[howManyAxes[i]]);
          }
        }

        if (transformRank == 2)
        {
          checkError(cufftMakePlan2d(mHandle,
                                     n[0],
                                     n[1],
                                     makeCufftType(precision, dftType),
                                     &mWorkspaceSize));
        }
        else
        {
          checkError(cufftMakePlan3d(mHandle,
                                     n[0],
                                     n[1],
                                     n[2],
                                     makeCufftType(precision, dftType),
                                     &mWorkspaceSize));
        }

        checkError(cufftXtSetSubformatDefault(mHandle,
                                              CUFFT_XT_FORMAT_DISTRIBUTED_INPUT,
                                              CUFFT_XT_FORMAT_DISTRIBUTED_OUTPUT));


      }

      /// @brief Destructor.
      ~Plan() override = default;

      /// @brief Inherit assignment operator
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
       * @brief Get the external workspace sizes
       * @return The workspace sizes
       */
      [[nodiscard]] const std::size_t* getExternalWorkspaceSizes() const noexcept override
      {
        return std::addressof(mWorkspaceSize);
      }

    private:
      /**
       * @brief Implementation of the executeImpl method.
       * @param src The source data pointers.
       * @param dst The destination data pointers.
       * @param execParams The execution parameters.
       */
      void executeBackendImpl(void* const* src, void* const* dst, const afft::cuda::ExecutionParameters& execParams) override
      {
        checkError(cufftSetStream(mHandle, execParams.stream));

        // if (getDesc().useExternalWorkspace())
        // {
        //   checkError(cufftSetWorkArea(mHandle, execParams.externalWorkspaces.front())); 
        // }

        checkError(cufftXtExec(mHandle, src.front(), dst.front(), getDirection()));
      }

      MPI_Comm    mMpiComm{};       ///< The MPI communicator.
      Handle      mHandle{};        ///< The cuFFT plan handle.
      std::size_t mWorkspaceSize{}; ///< The size of the workspace
      std::size_t mSrcElemCount{};  ///< The number of elements in the source buffer
      std::size_t mDstElemCount{};  ///< The number of elements in the destination buffer
  };

  /**
   * @brief Create a cufft mpi plan.
   * @param desc The plan descriptor.
   * @param backendParams The backend parameters.
   * @return The plan.
   */
  [[nodiscard]] AFFT_HEADER_ONLY_INLINE std::unique_ptr<afft::Plan>
  makePlan(const Description& desc, const afft::mpi::cuda::BackendParameters& backendParams)
  {
    const auto& descImpl = desc.get(DescToken::make());

    if (desc.getTargetCount() != 1)
    {
      throw Exception(Error::cufft, "only single target per process is supported");
    }

    switch (descImpl.getPrecision().execution)
    {
    case Precision::f32:
    case Precision::f64:
      break;
    default:
      throw Exception{Error::cufft, "unsupported precision"};
    }

    if (const auto rank = descImpl.getTransformRank(); rank != 2 && rank != 3)
    {
      throw Exception{Error::cufft, "only 2D and 3D transforms are supported"};
    }

    if (const auto howManyRank = descImpl.getTransformHowManyRank(); howManyRank == 0)
    {
      // TODO: check distribution
    }
    else
    {
      throw Exception{Error::cufft, "omitting dimensions is not supported"};
    }

    if (descImpl.getNormalization() != Normalization::none)
    {
      throw Exception{Error::cufft, "normalization is not supported"};
    }

    return std::make_unique<Plan>(desc, backendParams);
  }
} // namespace afft::detail::cufft::mpi

#endif /* AFFT_HEADER_ONLY */

#endif /* AFFT_DETAIL_CUFFT_MPI_HPP */
