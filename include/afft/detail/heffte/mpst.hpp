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

#ifndef AFFT_DETAIL_HEFFTE_MPST_HPP
#define AFFT_DETAIL_HEFFTE_MPST_HPP

#ifndef AFFT_TOP_LEVEL_INCLUDE
# include "../include.hpp"
#endif

#include "Plan.hpp"

namespace afft::detail::heffte::mpst
{
  /**
   * @brief The heffte plan implementation. Common for both cpu and gpu targets.
   * @tparam HeffteBackend The heffte backend.
   * @tparam PrecT The precision type.
   * @tparam fwdCmpl The forward domain complexity.
   */
  template<typename HeffteBackend, typename PrecT, Complexity fwdCmpl>
  class Plan final : public afft::detail::heffte::Plan
  {
    static_assert(std::is_same_v<PrecT, float> || std::is_same_v<PrecT, double>,
                  "The precision type must be either float or double.");

    private:
      /// @brief Alias for the parent class.
      using Parent = afft::detail::heffte::Plan;

      /// @brief Alias for the real type.
      using R = PrecT;

      /// @brief Alias for the complex type.
      using C = std::complex<PrecT>;

      /// @brief Alias for the forward source type.
      using FwdSrcT = std::conditional_t<fwdCmpl == Complexity::complex, C, R>;

      /// @brief Alias for the forward destination type.
      using FwdDstT = C;

      /// @brief Alias for the backward source type.
      using BwdSrcT = C;

      /// @brief Alias for the backward destination type.
      using BwdDstT = std::conditional_t<fwdCmpl == Complexity::complex, C, R>;
    public:
      /// @brief Inherit constructor.
      using Parent::Parent;

      /**
       * @brief Construct the plan.
       * @param desc The plan description.
       * @param heffteParams The heffte parameters.
       */
      template<typename HeffteParamsT>
      Plan(const Desc& desc, const HeffteParamsT& heffteParams)
      : Parent{desc},
        mPlan{makePlan(desc, heffteParams)},
        mBatch{1}, // TODO
        mWorkspaceSize{mBatch * mPlan.size_workspace() * sizeof(C)}
      {
        mDesc.fillDefaultMemoryLayoutStrides();
      }

      /// @brief Default destructor.
      ~Plan() = default;

      /// @brief Inherit assignment operator.
      using Parent::operator=;

      [[nodiscard]] View<std::size_t> getWorkspaceSize() const noexcept override
      {
        return View<std::size_t>{&mWorkspaceSize, 1};
      }

      /**
       * @brief Execute the plan.
       * @param src The source buffers.
       * @param dst The destination buffers.
       * @param execParams The execution parameters.
       */
      auto executeBackendImpl(View<void*> src, View<void*> dst, const afft::mpst::cpu::ExecutionParameters& execParams) override
        -> AFFT_RET_REQUIRES(void, AFFT_PARAM(HeffteBackend == ::heffte::backend::fftw || \
                                              HeffteBackend == ::heffte::backend::mkl))
      {
        executeBackendImplAny(src, dst, execParams.workspace);
      }

      /**
       * @brief Execute the plan.
       * @param src The source buffers.
       * @param dst The destination buffers.
       * @param execParams The execution parameters.
       */
#   if defined(AFFT_ENABLE_CUDA) || defined(AFFT_ENABLE_HIP)
      auto executeBackendImpl(View<void*> src, View<void*> dst, const afft::mpst::gpu::ExecutionParameters& execParams) override
        -> AFFT_RET_REQUIRES(void, AFFT_PARAM(HeffteBackend == ::heffte::backend::cufft || \
                                              HeffteBackend == ::heffte::backend::rocfft))
      {
#     if defined(AFFT_ENABLE_CUDA)
        if (exceParams.stream != cudaStream_t{0})
        {
          throw BackendError{Backend::heffte, "execution can be launched only to the default stream"};
        }
#     elif defined(AFFT_ENABLE_HIP)
        if (exceParams.stream != hipStream_t{0})
        {
          throw BackendError{Backend::heffte, "execution can be launched only to the default stream"};
        }
#     endif

        executeBackendImplAny(src, dst, execParams.workspace);
      }
#   endif
    private:
      /// @brief The heffte plan type.
      using HefftePlan = std::conditional_t<dftType == dft::Type::complexToComplex,
                                            ::heffte::fft3d<HeffteBackend, Index>,
                                            ::heffte::fft3d_r2c<HeffteBackend, Index>>;

      /**
       * @brief The heffte plan factory.
       * @tparam HeffteParamsT The heffte parameters type.
       * @param desc The plan description.
       * @param heffteParams The heffte parameters.
       * @return The heffte plan.
       */
      template<typename HeffteParamsT>
      [[nodiscard]] static HefftePlan makePlan(const Desc& desc, const HeffteParamsT& heffteParams)
      {
        const auto r2cAxis = desc.getTransformAxes().back();

        const auto& memLayout = desc.getMemoryLayout<Distribution::mpst>();

        const Box srcBox = makeBox(desc, memLayout.getSrcStarts(), memLayout.getSrcSizes());
        const Box dstBox = makeBox(desc, memLayout.getDstStarts(), memLayout.getDstSizes());

        MPI_Comm comm{MPI_COMM_NULL};

        switch (desc.getTarget())
        {
        case Target::cpu:
          comm = desc.getArchDesc<Target::cpu, Distribution::mpst>().comm;
          break;
        case Target::gpu:
          comm = desc.getArchDesc<Target::gpu, Distribution::mpst>().comm;
          break;
        default:
          cxx::unreachable();
        }

        const ::heffte::plan_options heffteOptions{heffteParams.useReorder,
                                                   heffteParams.useAllToAll,
                                                   heffteParams.usePencils};

        safeCall([&]
        {
          if constexpr (fwdCmpl == dft::Type::complexToComplex)
          {
            return HefftePlan{srcBox, dstBox, comm, r2cAxis, heffteOptions};
          }
          else
          {
            return HefftePlan{srcBox, dstBox, comm, heffteOptions};
          }
        });
      }

      /**
       * @brief Execute the plan.
       * @param src The source buffers.
       * @param dst The destination buffers.
       * @param workspace The execution workspace.
       */
      template<typename ExecParamsT>
      void executeBackendImplAny(View<void*> src, View<void*> dst, void* workspace)
      {
        safeCall([&]
        {
          const auto scale                = getScale();
          const auto useExternalWorkspace = mDesc.useExternalWorkspace();

          switch (mDesc.getDirection())
          {
          case Direction::forward:
            if (useExternalWorkspace)
            {
              mPlan.forward(mBatch,
                            reinterpret_cast<FwdSrcT*>(src.first()),
                            reinterpret_cast<FwdDstT*>(dst.first()),
                            reinterpret_cast<C*>(execParams.workspace),
                            scale);
            }
            else
            {
              mPlan.forward(mBatch,
                            reinterpret_cast<FwdSrcT*>(src.first()),
                            reinterpret_cast<FwdDstT*>(dst.first()),
                            scale);
            }
            break;
          case Direction::inverse:
            if (useExternalWorkspace)
            {
              mPlan.backward(mBatch,
                             reinterpret_cast<BwdSrcT*>(src.first()),
                             reinterpret_cast<BwdDstT*>(dst.first()),
                             reinterpret_cast<C*>(execParams.workspace),
                             scale);
            }
            else
            {
              mPlan.backward(mBatch,
                             reinterpret_cast<BwdSrcT*>(src.first()),
                             reinterpret_cast<BwdDstT*>(dst.first()),
                             scale);
            }
            break;
          default:
            cxx::unreachable();
          }
        });
      }

      HefftePlan  mPlan;            ///< The heffte plan.
      Index       mBatch{1};        ///< The batch size.
      std::size_t mWorkspaceSize{}; ///< The workspace size.
  };

  template<typename HeffteBackend, typename HeffteParamsT>
  [[nodiscard]] inline std::unique_ptr<heffte::Plan>
  makePlan(const Desc& desc, const HeffteParamsT& heffteParams)
  {
    const auto dftType = desc.getTransformDesc<Transform::dft>().type;

    const auto& memLayout = desc.getMemoryLayout<Distribution::mpst>();

    if (memLayout.hasDefaultSrcMemoryBlock() || memLayout.hasDefaultDstMemoryBlock())
    {
      throw BackendError{Backend::heffte, "only custom memory layout is supported"};
    }

    switch (const auto precision = desc.getPrecision().execution)
    {
    case Precision::_float:
      if (dftType == dft::Type::complexToComplex)
      {
        return std::make_unique<Plan<HeffteBackend, float, Complexity::complex>>(desc, heffteParams);
      }
      else
      {
        return std::make_unique<Plan<HeffteBackend, float, Complexity::real>>(desc, heffteParams);
      }
    case Precision::_double:
      if (dftType == dft::Type::complexToComplex)
      {
        return std::make_unique<Plan<HeffteBackend, double, Complexity::complex>>(desc, heffteParams);
      }
      else
      {
        return std::make_unique<Plan<HeffteBackend, double, Complexity::real>>(desc, heffteParams);
      }
    default:
      throw BackendError{Backend::heffte, "unsupported precision"};
    }
  }

namespace cpu
{
  [[nodiscard]] inline std::unique_ptr<heffte::Plan>
  makePlan(const Desc& desc, const afft::mpst::cpu::heffte::Parameters& heffteParams)
  {
    switch (heffteParams.backend)
    {
    case afft::heffte::Backend::fftw3:
#   ifdef Heffte_ENABLE_FFTW
      mpst::makePlan<::heffte::backend::fftw>(desc, heffteParams);
#   else
      throw BackendError{Backend::heffte, "fftw backend is not enabled"};
#   endif
    case afft::heffte::Backend::mkl:
#   ifdef Heffte_ENABLE_MKL
      mpst::makePlan<::heffte::backend::mkl>(desc, heffteParams);
#   else
      throw BackendError{Backend::heffte, "mkl backend is not enabled"};
#   endif
    default:
      throw BackendError{Backend::heffte, "unsupported backend"};
    }
  }
} // namespace cpu

namespace gpu
{
  [[nodiscard]] inline std::unique_ptr<heffte::Plan>
  makePlan(const Desc& desc, const afft::mpst::gpu::heffte::Parameters& heffteParams)
  {
    switch (heffteParams.backend)
    {
    case afft::heffte::Backend::cufft:
#   ifdef Heffte_ENABLE_CUDA
      mpst::makePlan<::heffte::backend::cufft>(desc, heffteParams); 
#   else
      throw BackendError{Backend::heffte, "cufft backend is not enabled"};
#   endif
    case afft::heffte::Backend::rocfft:
#   ifdef Heffte_ENABLE_ROCM
      mpst::makePlan<::heffte::backend::rocfft>(desc, heffteParams);
#   else
      throw BackendError{Backend::heffte, "rocfft backend is not enabled"};
#   endif
    default:
      throw BackendError{Backend::heffte, "unsupported backend"};
    }
  }
} // namespace gpu
} // namespace afft::detail::heffte::mpst

#endif /* AFFT_DETAIL_HEFFTE_MPST_HPP */
