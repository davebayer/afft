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

#ifndef AFFT_DETAIL_VKFFT_SPST_HPP
#define AFFT_DETAIL_VKFFT_SPST_HPP

#ifndef AFFT_TOP_LEVEL_INCLUDE
# include "../include.hpp"
#endif

#include "../../Plan.hpp"

#ifndef AFFT_DISABLE_GPU

namespace afft::detail::vkfft::spst::gpu
{
  /**
   * @brief Create a vkfft spst gpu plan implementation.
   * @param desc Plan description.
   * @return Plan implementation.
   */
  [[nodiscard]] std::unique_ptr<afft::Plan> makePlan(const Desc& desc);
} // namespace afft::detail::vkfft::spst::gpu

#ifdef AFFT_HEADER_ONLY

#include "Plan.hpp"

namespace afft::detail::vkfft::spst::gpu
{
  /// @brief Alias for the unsigned integer type used by VkFFT
  using UInt = pfUINT;

  /**
   * @class Plan
   * @brief Implementation of the plan for the spst gpu architecture using VkFFT
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
      Plan(const Desc& desc)
      : Parent{desc}
      {
        mDesc.fillDefaultMemoryLayoutStrides();

        const auto& gpuDesc = mDesc.getArchDesc<Target::gpu, Distribution::spst>();

        const auto direction = mDesc.getDirection();
        const auto shapeRank = mDesc.getShapeRank();

        // Set up GPU device variables
#     if defined(AFFT_ENABLE_CUDA)
        cuda::checkError(cuDeviceGet(&mCuDevice, gpuDesc.device));
#     elif defined(AFFT_ENABLE_HIP)
        hip::checkError(hipGetDevice(&mHipDevice, gpuDesc.device));
#     elif defined(AFFT_ENABLE_OPENCL)
        mContext = gpuDesc.context;
        mDevice  = gpuDesc.device;
#     endif

        VkFFTConfiguration vkfftConfig{};

        // Set up VkFFT config shape
        vkfftConfig.FFTdim = safeIntCast<UInt>(shapeRank);

        std::transform(mDesc.getShape().rbegin(), mDesc.getShape().rend(), vkfftConfig.size, [](const auto& dim)
        {
          return safeIntCast<UInt>(dim);
        });

        // Set up VkFFT config GPU device
#    if defined(AFFT_ENABLE_CUDA)
        vkfftConfig.device      = &mCuDevice;
        vkfftConfig.stream      = &mStream;
        vkfftConfig.num_streams = 1;
#    elif defined(AFFT_ENABLE_HIP)
        vkfftConfig.device      = &mHipDevice;
        vkfftConfig.stream      = &mStream;
        vkfftConfig.num_streams = 1;
#    elif defined(AFFT_ENABLE_OPENCL)
        vkfftConfig.device      = &mDevice;
        vkfftConfig.context     = &mContext;
#    endif

        // Set up VkFFT config user temp buffer flag
        vkfftConfig.userTempBuffer = mDesc.useExternalWorkspace();

        // Set up VkFFT config complex format
        const UInt separateComplexComponents = (mDesc.getComplexFormat() == ComplexFormat::planar) ? 1 : 0;
        vkfftConfig.bufferSeparateComplexComponents       = separateComplexComponents;
        vkfftConfig.inputBufferSeparateComplexComponents  = separateComplexComponents;
        vkfftConfig.outputBufferSeparateComplexComponents = separateComplexComponents;

        // Set up VkFFT config GPU memory parameters
#     if defined(AFFT_ENABLE_CUDA)
        vkfftConfig.coalescedMemory = 32;
        vkfftConfig.numSharedBanks  = 32;
#     elif defined(AFFT_ENABLE_HIP)
        vkfftConfig.coalescedMemory = 32; // same for NVIDIA and AMD
        vkfftConfig.numSharedBanks  = 32; // same for NVIDIA and AMD
#     elif defined(AFFT_ENABLE_OPENCL)
        // set by VkFFT internally
#     endif

        // Set up VkFFT config 
        std::fill_n(vkfftConfig.omitDimension, maxDimCount, UInt{1});
        for (const auto axis : mDesc.getTransformAxes())
        {
          vkfftConfig.omitDimension[axis] = UInt{0};
        }

        // Set up VkFFT config precision
        {
          const auto& prec = mDesc.getPrecision();

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
#           ifdef AFFT_VKFFT_HAS_DOUBLE_DOUBLE
            case Precision::f64f64:
              vkfftConfig.quadDoubleDoublePrecisionDoubleMemory = 1;
              break;
#           endif
            default:
              precIsOk = false;
              break;
            }
            break;
#         ifdef AFFT_VKFFT_HAS_DOUBLE_DOUBLE
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
#         endif
          default:
            precIsOk = false;
            break;
          }

          if (!precIsOk)
          {
            throw BackendError{Backend::vkfft, "unsupported precision combination"};
          }
        }

        // Set up VkFFT config transform type
        switch (mDesc.getTransform())
        {
        case Transform::dft:
        {
          const auto& dftDesc = mDesc.getTransformDesc<Transform::dft>();

          switch (dftDesc.type)
          {
          case dft::Type::realToComplex:
          case dft::Type::complexToReal:
            if (mDesc.getTransformAxes().back() != (mDesc.getTransformRank() - 1))
            {
              throw BackendError{Backend::vkfft, "when performing real data FFT, the last axis cannot be omited"};
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
          const auto transformAxes = mDesc.getTransformAxes();
          const auto dttAxisTypes  = mDesc.getTransformDesc<Transform::dtt>().types;

          for (std::size_t i{}; i < mDesc.getTransformRank(); ++i)
          {
            // VkFFT uses reverse order of axes
            const auto vkfftAxis = transformAxes[mDesc.getTransformRank() - i - 1];

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
          throw BackendError{Backend::vkfft, "unsupported transform type"};
        }
        
        // Set up VkFFT config normalization
        switch (mDesc.getNormalization())
        {
        case Normalization::none:
          vkfftConfig.normalize = 0;
          break;
        case Normalization::unitary:
          vkfftConfig.normalize = 1;
          break;
        default:
          throw BackendError{Backend::vkfft, "unsupported normalization type"};
        }

        // Select which plan to make
        vkfftConfig.makeForwardPlanOnly = (direction != Direction::inverse);
        vkfftConfig.makeInversePlanOnly = (direction != Direction::forward);

        // Set up VkFFT config buffer strides
        vkfftConfig.isInputFormatted  = 1;
        vkfftConfig.isOutputFormatted = 1;

        UInt* srcStrides = (direction == Direction::forward) ? vkfftConfig.inputBufferStride : vkfftConfig.bufferStride;
        UInt* dstStrides = (direction == Direction::forward) ? vkfftConfig.bufferStride : vkfftConfig.outputBufferStride;

        const auto& memLayout = mDesc.getMemoryLayout<Distribution::spst>();

        for (std::size_t i{}; i < shapeRank; ++i)
        {
          srcStrides[i] = safeIntCast<UInt>(memLayout.getSrcStrides()[shapeRank - i - 1]);
          dstStrides[i] = safeIntCast<UInt>(memLayout.getDstStrides()[shapeRank - i - 1]);
        }

        // Disable locale
        vkfftConfig.disableSetLocale = 1;

        // vkfftConfig.registerBoost              = getRegisterFileSize(Parent::mDevice) / getSharedMemorySize(Parent::mDevice);
        // vkfftConfig.registerBoostNonPow2       = 1;
        // vkfftConfig.registerBoost4Step         = getRegisterFileSize(Parent::mDevice) / getSharedMemorySize(Parent::mDevice);

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
       * @brief Execute the plan
       * @param src The source buffer
       * @param dst The destination buffer
       */
      void executeBackendImpl(View<void*> src, View<void*> dst, const afft::spst::gpu::ExecutionParameters& execParams) override
      {
        VkFFTLaunchParams launchParams{};

        switch (mDesc.getDirection())
        {
        case Direction::forward:
#       if defined(AFFT_ENABLE_CUDA)
          launchParams.inputBuffer = const_cast<void**>(src.data());
          launchParams.buffer      = const_cast<void**>(dst.data());
#       elif defined(AFFT_ENABLE_HIP)
          launchParams.inputBuffer = const_cast<void**>(src.data());
          launchParams.buffer      = const_cast<void**>(dst.data());
#       elif defined(AFFT_ENABLE_OPENCL)
          launchParams.inputBuffer = reinterpret_cast<cl_mem*>(src.data());
          launchParams.buffer      = reinterpret_cast<cl_mem*>(dst.data());
#       endif
          break;
        case Direction::backward:
#       if defined(AFFT_ENABLE_CUDA)
          launchParams.buffer      = const_cast<void**>(src.data());
          launchParams.inputBuffer = const_cast<void**>(dst.data());
#       elif defined(AFFT_ENABLE_HIP)
          launchParams.buffer      = const_cast<void**>(src.data());
          launchParams.inputBuffer = const_cast<void**>(dst.data());
#       elif defined(AFFT_ENABLE_OPENCL)
          launchParams.buffer      = reinterpret_cast<cl_mem*>(src.data());
          launchParams.inputBuffer = reinterpret_cast<cl_mem*>(dst.data());
#       endif
          break;
        default:
          cxx::unreachable();
        }

#     if defined(AFFT_ENABLE_CUDA)
        mStream = execParams.stream;
#     elif defined(AFFT_ENABLE_HIP)
        mStream = execParams.stream;
#     elif defined(AFFT_ENABLE_OPENCL)
        mQueue  = execParams.commandQueue;
#     endif

        if (mDesc.useExternalWorkspace())
        {
          launchParams.tempBuffer = const_cast<void**>(&execParams.workspace);
        }

        checkError(VkFFTAppend(&mApp, getDirection(), &launchParams));
      }
    protected:
    private:
      VkFFTApplication mApp{};
      bool             mInitialized{false};
#   if defined(AFFT_ENABLE_CUDA)
      CUdevice         mCuDevice{};
      cudaStream_t     mStream{0};
#   elif defined(AFFT_ENABLE_HIP)
      hipDevice_t      mHipDevice{};
      hipStream_t      mStream{0};
#   elif defined(AFFT_ENABLE_OPENCL)
      cl_context       mContext{};
      cl_device_id     mDevice{};
      cl_command_queue mQueue{};
#   endif
  };

  /**
   * @brief Create a vkfft spst gpu plan implementation.
   * @param desc Plan description.
   * @return Plan implementation.
   */
  [[nodiscard]] AFFT_HEADER_ONLY_INLINE std::unique_ptr<afft::Plan> makePlan(const Desc& desc)
  {
    const auto& precision = desc.getPrecision();

    if (precision.source != precision.destination)
    {
      throw BackendError{Backend::vkfft, "source and destination precision must match"};
    }

    return std::make_unique<Plan>(desc);
  }
} // namespace afft::detail::vkfft::spst::gpu

#endif /* AFFT_HEADER_ONLY */

#endif /* AFFT_DISABLE_GPU */

#endif /* AFFT_DETAIL_VKFFT_SPST_HPP */
