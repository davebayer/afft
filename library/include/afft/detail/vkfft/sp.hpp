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

#ifndef AFFT_DETAIL_VKFFT_SP_HPP
#define AFFT_DETAIL_VKFFT_SP_HPP

#ifndef AFFT_TOP_LEVEL_INCLUDE
# include "../include.hpp"
#endif

#include "common.hpp"
#include "../../Plan.hpp"

namespace afft::detail::vkfft::sp
{
  /**
   * @brief Create a vkfft sp plan implementation.
   * @param desc Plan description.
   * @param backendParams Backend parameters.
   * @return Plan implementation.
   */
  [[nodiscard]] std::unique_ptr<afft::Plan> makePlan(const Desc& desc, const BackendParameters<MpBackend::none, vkfft::target>& backendParams);
} // namespace afft::detail::vkfft::sp

#ifdef AFFT_HEADER_ONLY

#include "Plan.hpp"

namespace afft::detail::vkfft::sp
{
  /// @brief Alias for the unsigned integer type used by VkFFT
  using UInt = pfUINT;

  /**
   * @class Plan
   * @brief Implementation of the plan for the sp architecture using VkFFT
   */
  class Plan final : public vkfft::Plan
  {
    private:
      /// @brief Alias for the parent class
      using Parent = vkfft::Plan;

    public:
      /// @brief inherit constructors
      using Parent::Parent;

      /**
       * @brief Constructor
       * @param Desc The plan description
       */
      Plan(const Description& desc, const afft::BackendParameters<MpBackend::none, target>& backendParams)
      : Parent{desc, backendParams},
        mTargetData{makeTargetData(desc)}
      {
        Parent::mIsDestructive = (Parent::mDesc.getDirection() == Direction::inverse ||
                                  Parent::mDesc.getPlacement() == Placement::inPlace);

        const auto& memDesc = mDesc.getMemDesc<MemoryLayout::centralized>();

        // mSrcElemCount = memDesc.getSrcElemCount();
        // mDstElemCount = memDesc.getDstElemCount();

        VkFFTConfiguration vkfftConfig{};
        fillConfigTarget(vkfftConfig, mTargetData);
        fillConfigPrecision(vkfftConfig, mDesc.getPrecision());
        fillConfigShape(vkfftConfig, mDesc.getShape());
        fillConfigTransform(vkfftConfig, mDesc);
        fillConfigMemoryLayout(vkfftConfig, mDesc);

        // Disable locale
        vkfftConfig.disableSetLocale = 1;

        // Initialize VkFFT with the configuration
        checkError(initializeVkFFT(&mApp, std::move(vkfftConfig)));
        mInitialized = true;
      }

      /// @brief Destructor
      ~Plan()
      {
        if (mInitialized)
        {
          deleteVkFFT(&mApp);
        }
      }

      /// @brief Inherit assignment operator
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
       * @brief Get the workspace sizes
       * @return The workspace sizes
       */
      [[nodiscard]] View<std::size_t> getExternalWorkspaceSizes() const noexcept override
      {
        return makeScalarView(mWorkspaceSize);
      }

    protected:
    private:
      struct TargetData
      {
#     if AFFT_VKFFT_BACKEND == 1
        CUdevice     device{};
        cudaStream_t stream{};
#     elif AFFT_VKFFT_BACKEND == 2
        hipDevice_t device{};
        hipStream_t stream{};
#     elif AFFT_VKFFT_BACKEND == 3
        cl_context   context{};
        cl_device_id device{};
#     endif
      };

      [[nodiscard]] static TargetData makeTargetData(const Description& desc)
      {
        if (desc.getTargetCount() != 1)
        {
          throw Exception{Error::vkfft, "only one target is supported"};
        }

        TargetData targetData{};

        const auto& descImpl = desc.get(DescToken::make());

#     if AFFT_VKFFT_BACKEND == 1
        const auto& cudaDesc = descImpl.getTargetDesc<Target::cuda>();

        cuda::checkError(cuDeviceGet(&targetData.device, cudaDesc.getDevices()[0]));
#     elif AFFT_VKFFT_BACKEND == 2
        const auto& hipDesc = descImpl.getTargetDesc<Target::hip>();

        hip::checkError(hipGetDevice(&targetData.device, hipDesc.getDevices()[0]));
#     elif AFFT_VKFFT_BACKEND == 3
        const auto& openclDesc = descImpl.getTargetDesc<Target::opencl>();

        targetData.context = openclDesc.context;
        targetData.device  = openclDesc.getDevices()[0];
#     endif
  
        return targetData;
      }

      static void fillConfigTarget([[maybe_unused]] VkFFTConfiguration& vkfftConfig,
                                   [[maybe_unused]] const TargetData& targetData) noexcept
      {
#     if AFFT_VKFFT_BACKEND == 1
        vkfftConfig.device          = const_cast<CUdevice*>(&targetData.device);
        vkfftConfig.stream          = const_cast<cudaStream_t*>(&targetData.stream);
        vkfftConfig.num_streams     = 1;
        vkfftConfig.coalescedMemory = 32;
        vkfftConfig.numSharedBanks  = 32;

        // vkfftConfig.registerBoost              = getRegisterFileSize(Parent::mDevice) / getSharedMemorySize(Parent::mDevice);
        // vkfftConfig.registerBoostNonPow2       = 1;
        // vkfftConfig.registerBoost4Step         = getRegisterFileSize(Parent::mDevice) / getSharedMemorySize(Parent::mDevice);
#     elif AFFT_VKFFT_BACKEND == 2
        vkfftConfig.device          = const_cast<hipDevice_t*>(&targetData.device);
        vkfftConfig.stream          = const_cast<hipStream_t*>(&targetData.stream);
        vkfftConfig.num_streams     = 1;
        vkfftConfig.coalescedMemory = 32;
        vkfftConfig.numSharedBanks  = 32;
#     elif AFFT_VKFFT_BACKEND == 3
        vkfftConfig.device          = &targetData.device;
        vkfftConfig.context         = &targetData.context;
#     endif
      }

      static void fillConfigPrecision(VkFFTConfiguration& vkfftConfig, const PrecisionTriad& prec)
      {
        bool precIsOk{true};

        switch (prec.source)
        {
        case Precision::f16:
          switch (prec.execution)
          {
          case Precision::f16:
            vkfftConfig.halfPrecision = 1;
            break;
          case Precision::f32:
            vkfftConfig.halfPrecisionMemoryOnly = 1;
            break;
          default:
            precIsOk = false;
            break;
          }
          break;
        case Precision::f32:
          switch (prec.execution)
          {
          case Precision::f32:
            break;
          case Precision::f64:
            vkfftConfig.doublePrecisionFloatMemory = 1;
            break;
          default:
            precIsOk = false;
            break;
          }
          break;
        case Precision::f64:
          switch (prec.execution)
          {
          case Precision::f64:
            vkfftConfig.doublePrecision = 1;
            break;
#         ifdef AFFT_VKFFT_HAS_DOUBLE_DOUBLE
          case Precision::f64f64:
            vkfftConfig.quadDoubleDoublePrecisionDoubleMemory = 1;
            break;
#         endif
          default:
            precIsOk = false;
            break;
          }
          break;
#       ifdef AFFT_VKFFT_HAS_DOUBLE_DOUBLE
        case Precision::f64f64:
          switch (prec.execution)
          {
          case Precision::f64f64:
            vkfftConfig.quadDoubleDoublePrecision = 1;
            break;
          default:
            precIsOk = false;
            break;
          }
#       endif
        default:
          precIsOk = false;
          break;
        }

        if (!precIsOk)
        {
          throw Exception{Error::vkfft, "unsupported precision combination"};
        }
      }

      static void fillConfigShape(VkFFTConfiguration& vkfftConfig, View<Size> shape)
      {
        vkfftConfig.FFTdim = safeIntCast<UInt>(shape.size());

        std::transform(shape.rbegin(), shape.rend(), vkfftConfig.size, [](const auto dim)
        {
          return safeIntCast<UInt>(dim);
        });
      }

      static void fillConfigTransform(VkFFTConfiguration& vkfftConfig, const TransformDesc& transformDesc)
      {
        // Set up transform axes
        std::fill_n(vkfftConfig.omitDimension, maxDimCount, UInt{1});
        for (const auto axis : transformDesc.getTransformAxes())
        {
          vkfftConfig.omitDimension[axis] = UInt{0};
        }

        // Set up VkFFT config transform type
        switch (transformDesc.getTransform())
        {
        case Transform::dft:
        {
          const auto& dftDesc = transformDesc.getTransformDesc<Transform::dft>();

          switch (dftDesc.type)
          {
          case dft::Type::realToComplex:
          case dft::Type::complexToReal:
            if (transformDesc.getTransformAxes().back() != (transformDesc.getTransformRank() - 1))
            {
              throw Exception{Error::vkfft, "when performing real data FFT, the last axis cannot be omited"};
            }

            vkfftConfig.performR2C = 1;
            break;
          default:
            break;
          }
          break;
        }
        case Transform::dtt:
        {
          const auto transformAxes = transformDesc.getTransformAxes();
          const auto dttAxisTypes  = transformDesc.getTransformDesc<Transform::dtt>().types;

          for (std::size_t i{}; i < transformDesc.getTransformRank(); ++i)
          {
            // VkFFT uses reverse order of axes
            const auto vkfftAxis = transformAxes[transformDesc.getTransformRank() - i - 1];

            switch (dttAxisTypes[i])
            {
            case dtt::Type::dct1: vkfftConfig.performR2R[vkfftAxis] = 1;  break;
            case dtt::Type::dct2: vkfftConfig.performR2R[vkfftAxis] = 2;  break;
            case dtt::Type::dct3: vkfftConfig.performR2R[vkfftAxis] = 3;  break;
            case dtt::Type::dct4: vkfftConfig.performR2R[vkfftAxis] = 4;  break;
            case dtt::Type::dst1: vkfftConfig.performR2R[vkfftAxis] = 11; break;
            case dtt::Type::dst2: vkfftConfig.performR2R[vkfftAxis] = 12; break;
            case dtt::Type::dst3: vkfftConfig.performR2R[vkfftAxis] = 13; break;
            case dtt::Type::dst4: vkfftConfig.performR2R[vkfftAxis] = 14; break;
            default:
              cxx::unreachable();
            }
          }
          break;
        }
        default:
          throw Exception{Error::vkfft, "unsupported transform type"};
        }

        // Set up VkFFT config normalization
        switch (transformDesc.getNormalization())
        {
        case Normalization::none:
          vkfftConfig.normalize = 0;
          break;
        case Normalization::unitary:
          vkfftConfig.normalize = 1;
          break;
        default:
          throw Exception{Error::vkfft, "unsupported normalization type"};
        }

        // Select which plan to make
        vkfftConfig.makeForwardPlanOnly = (transformDesc.getDirection() != Direction::inverse);
        vkfftConfig.makeInversePlanOnly = (transformDesc.getDirection() != Direction::forward);
      }

      static void fillConfigMemoryLayout(VkFFTConfiguration& vkfftConfig, const Desc& desc)
      {
        // Set up VkFFT config complex format
        const UInt separateComplexComponents = (desc.getComplexFormat() == ComplexFormat::planar) ? 1 : 0;
        vkfftConfig.bufferSeparateComplexComponents       = separateComplexComponents;
        vkfftConfig.inputBufferSeparateComplexComponents  = separateComplexComponents;
        vkfftConfig.outputBufferSeparateComplexComponents = separateComplexComponents;

        // Set up VkFFT config buffer strides
        vkfftConfig.isInputFormatted  = 1;
        vkfftConfig.isOutputFormatted = 1;

        UInt* srcStrides = (desc.getDirection() == Direction::forward) ? vkfftConfig.inputBufferStride : vkfftConfig.bufferStride;
        UInt* dstStrides = (desc.getDirection() == Direction::forward) ? vkfftConfig.bufferStride : vkfftConfig.outputBufferStride;

        const auto& memDesc = desc.getMemDesc<MemoryLayout::centralized>();

        if (memDesc.getSrcStrides().back() != 1)
        {
          throw Exception{Error::vkfft, "source fastest axis stride must be 1"};
        }

        if (memDesc.getDstStrides().back() != 1)
        {
          throw Exception{Error::vkfft, "destination fastest axis stride must be 1"};
        }

        for (std::size_t i{}; i < desc.getShapeRank() - 1; ++i)
        {
          srcStrides[i] = safeIntCast<UInt>(memDesc.getSrcStrides()[desc.getShapeRank() - i - 2]);
          dstStrides[i] = safeIntCast<UInt>(memDesc.getDstStrides()[desc.getShapeRank() - i - 2]);
        }
      }



#   if AFFT_VKFFT_BACKEND == 1
      /**
       * @brief Execute the plan
       * @param src The source buffer
       * @param dst The destination buffer
       * @param execParams The execution parameters
       */
      void executeBackendImpl(View<void*> src, View<void*> dst, const afft::cuda::ExecutionParameters& execParams) override
      {
        VkFFTLaunchParams launchParams{};

        switch (mDesc.getDirection())
        {
        case Direction::forward:
          launchParams.inputBuffer = const_cast<void**>(src.data());
          launchParams.buffer      = const_cast<void**>(dst.data());
          break;
        case Direction::inverse:
          launchParams.buffer       = const_cast<void**>(src.data());
          launchParams.outputBuffer = const_cast<void**>(dst.data());
          break;
        default:
          cxx::unreachable();
        }

        mTargetData.stream = execParams.stream;

        // if (mDesc.useExternalWorkspace())
        // {
        //   launchParams.tempBuffer = const_cast<void**>(&execParams.workspace);
        // }

        checkError(VkFFTAppend(&mApp, getDirection(), &launchParams));
      }
#   elif AFFT_VKFFT_BACKEND == 2
      /**
       * @brief Execute the plan
       * @param src The source buffer
       * @param dst The destination buffer
       * @param execParams The execution parameters
       */
      void executeBackendImpl(View<void*> src, View<void*> dst, const afft::hip::ExecutionParameters& execParams) override
      {
        VkFFTLaunchParams launchParams{};

        switch (mDesc.getDirection())
        {
        case Direction::forward:
          launchParams.inputBuffer = const_cast<void**>(src.data());
          launchParams.buffer      = const_cast<void**>(dst.data());
          break;
        case Direction::backward:
          launchParams.buffer       = const_cast<void**>(src.data());
          launchParams.outputBuffer = const_cast<void**>(dst.data());
          break;
        default:
          cxx::unreachable();
        }

        mTargetData.stream = execParams.stream;

        // if (mDesc.useExternalWorkspace())
        // {
        //   launchParams.tempBuffer = const_cast<void**>(&execParams.workspace);
        // }

        checkError(VkFFTAppend(&mApp, getDirection(), &launchParams));
      }
#   elif AFFT_VKFFT_BACKEND == 3
      /**
       * @brief Execute the plan
       * @param src The source buffer
       * @param dst The destination buffer
       * @param execParams The execution parameters
       */
      void executeBackendImpl(View<void*> src, View<void*> dst, const afft::opencl::ExecutionParameters& execParams) override
      {
        VkFFTLaunchParams launchParams{};

        switch (mDesc.getDirection())
        {
        case Direction::forward:
          launchParams.inputBuffer = reinterpret_cast<cl_mem*>(src.data());
          launchParams.buffer      = reinterpret_cast<cl_mem*>(dst.data());
          break;
        case Direction::backward:
          launchParams.buffer       = reinterpret_cast<cl_mem*>(src.data());
          launchParams.outputBuffer = reinterpret_cast<cl_mem*>(dst.data());
          break;
        default:
          cxx::unreachable();
        }

        launchParams.commandQueue = &execParams.commandQueue;

        // if (mDesc.useExternalWorkspace())
        // {
        //   launchParams.tempBuffer = const_cast<void**>(&execParams.workspace);
        // }

        checkError(VkFFTAppend(&mApp, getDirection(), &launchParams));
      }
#   endif

      TargetData       mTargetData{};       ///< The target data
      VkFFTApplication mApp{};              ///< The VkFFT application
      std::size_t      mWorkspaceSize{};    ///< The size of the workspace
      std::size_t      mSrcElemCount{};     ///< The number of elements in the source buffer
      std::size_t      mDstElemCount{};     ///< The number of elements in the destination buffer
      bool             mInitialized{false}; ///< Whether the vkfft app is initialized
  };

  /**
   * @brief Create a vkfft sp plan implementation.
   * @param desc Plan description.
   * @param backendParams Backend parameters.
   * @return Plan implementation.
   */
  [[nodiscard]] AFFT_HEADER_ONLY_INLINE std::unique_ptr<afft::Plan>
  makePlan(const Description&                                       desc,
           const BackendParameters<MpBackend::none, vkfft::target>& backendParams)
  {
    const auto& descImpl = desc.get(DescToken::make());

    const auto& precision = descImpl.getPrecision();

    if (precision.source != precision.destination)
    {
      throw Exception{Error::vkfft, "source and destination precision must match"};
    }

    if (descImpl.getDirection() == Direction::inverse && !backendParams.allowDestructive)
    {
      throw Exception{Error::vkfft, "inverse transform is always destructive"};
    }

    switch (desc.getTarget())
    {
#   if AFFT_VKFFT_BACKEND == 1
    case Target::cuda:
      return std::make_unique<Plan>(desc, backendParams);
#   elif AFFT_VKFFT_BACKEND == 2
    case Target::hip:
      return std::make_unique<Plan>(desc, backendParams);
#   elif AFFT_VKFFT_BACKEND == 3
    case Target::opencl:
      return std::make_unique<Plan>(desc, backendParams);
#   endif
    default:
      throw Exception{Error::vkfft, "unsupported target"};
    }
  }
} // namespace afft::detail::vkfft::sp

#endif /* AFFT_HEADER_ONLY */

#endif /* AFFT_DETAIL_VKFFT_SP_HPP */
