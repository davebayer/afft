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

#ifndef AFFT_DETAIL_GPU_VKFFT_PLAN_IMPL_HPP
#define AFFT_DETAIL_GPU_VKFFT_PLAN_IMPL_HPP

#include <algorithm>
#include <cstdint>
#include <memory>
#include <stdexcept>

#include "App.hpp"
#include "error.hpp"
#include "include.hpp"
#include "../../PlanImpl.hpp"

namespace afft::detail::gpu::vkfft
{
  /// @brief VkFFT's unsigned integer type
  using UInt = pfUINT;

  /**
   * @brief VkFFT plan implementation
   */
  class PlanImpl final : public afft::detail::PlanImpl
  {
    private:
      /// @brief Parent class alias
      using Parent = afft::detail::PlanImpl;
    public:
      /// @brief Inherit constructors
      using Parent::Parent;
      
      /**
       * @brief Constructor
       * @param config Configuration
       */
      PlanImpl(const Config& config)
      : Parent(config)
      {
        const auto& gpuConfig = getConfig().getTargetConfig<Target::gpu>();

#     if AFFT_GPU_FRAMEWORK_IS_CUDA
        cuda::ScopedDevice scopedDevice{gpuConfig.device};
        Error::check(cuDeviceGet(&mCuDevice, gpuConfig.device));
#     elif AFFT_GPU_FRAMEWORK_IS_HIP
        hip::ScopedDevice scopedDevice{gpuConfig.device};
        Error::check(hipDeviceGet(&mHipDevice, gpuConfig.device));
#     elif AFFT_GPU_FRAMEWORK_IS_OPENCL
        mContext = gpuConfig.context;
        mDevice  = gpuConfig.device;
#     endif

        VkFFTConfiguration vkfftConfig{};

        vkfftConfig.FFTdim               = safeIntCast<UInt>(getConfig().getShapeRank());

        std::transform(getConfig().getShape().rbegin(),
                       getConfig().getShape().rend(),
                       vkfftConfig.size,
                       [](const auto& dim)
        {
          return safeIntCast<UInt>(dim);
        });

#     if AFFT_GPU_FRAMEWORK_IS_CUDA
        vkfftConfig.device               = &mCuDevice;
        vkfftConfig.stream               = &mStream;
        vkfftConfig.num_streams          = 1;
#     elif AFFT_GPU_FRAMEWORK_IS_HIP
        vkfftConfig.device               = &mHipDevice;
        vkfftConfig.stream               = &mStream;
        vkfftConfig.num_streams          = 1;
#     elif AFFT_GPU_FRAMEWORK_IS_OPENCL
        vkfftConfig.context              = &mContext;
        vkfftConfig.device               = &mDevice;
#     endif

        vkfftConfig.userTempBuffer       = gpuConfig.externalWorkspace;

#     if AFFT_GPU_FRAMEWORK_IS_CUDA
        vkfftConfig.coalescedMemory      = 32;
        vkfftConfig.numSharedBanks       = 32;
#     elif AFFT_GPU_FRAMEWORK_IS_HIP
        vkfftConfig.coalescedMemory      = 32; // same for NVIDIA and AMD
        vkfftConfig.numSharedBanks       = 32; // same for NVIDIA and AMD
#     elif AFFT_GPU_FRAMEWORK_IS_OPENCL
        // TODO: OpenCL
#     endif

        vkfftConfig.inverseReturnToInputBuffer = 1;

        std::fill_n(vkfftConfig.omitDimension, maxDimCount, UInt{1});
        for (const auto axis : getConfig().getTransformAxes())
        {
          vkfftConfig.omitDimension[axis] = UInt{0};
        }

        if (const auto& prec = getConfig().getTransformPrecision(); prec.source != prec.destination)
        {
          throw std::runtime_error("Different source and destination precision is not supported");
        }
        else
        {
          bool ok = true;

          switch (prec.source)
          {
          case Precision::f16: switch (prec.execution)
            {
            case Precision::f16: vkfftConfig.halfPrecision           = 1; break;
            case Precision::f32: vkfftConfig.halfPrecisionMemoryOnly = 1; break;
            default: ok = false; break;
            }
            break;
          case Precision::f32: switch (prec.execution)
            {
            case Precision::f32:                                             break;
            case Precision::f64: vkfftConfig.doublePrecisionFloatMemory = 1; break;
            default: ok = false; break;
            }
            break;
          case Precision::f64: switch (prec.execution)
            {
            case Precision::f64:  vkfftConfig.doublePrecision                       = 1; break;
            case Precision::f128: vkfftConfig.quadDoubleDoublePrecisionDoubleMemory = 1; break;
            default: ok = false; break;
            }
            break;
          case Precision::f128: switch (prec.execution)
            {
            case Precision::f128: vkfftConfig.quadDoubleDoublePrecision = 1; break;
            default: ok = false; break;
            }
            break;
          default: ok = false; break;
          }

          if (!ok)
          {
            throw std::runtime_error("Unsupported precision combination");
          }
        }

        switch (getConfig().getTransform())
        {
        case Transform::dft:
        {
          const auto& dftConfig = getConfig().getTransformConfig<Transform::dft>();

          switch (dftConfig.type)
          {
          case dft::Type::realToComplex:
          case dft::Type::complexToReal:
            if (getConfig().getTransformAxes().back() != 0)
            {
              throw makeException<std::runtime_error>("Vkfft supports only last axis for rfft");
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
          const auto transformAxes = getConfig().getTransformAxes();
          const auto dttAxisTypes  = getConfig().getTransformConfig<Transform::dtt>().axisTypes;

          for (std::size_t i{}; i < getConfig().getTransformRank(); ++i)
          {
            // VkFFT uses reverse order of axes
            const auto vkfftAxis = transformAxes[getConfig().getTransformRank() - i - 1];

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
              throw std::runtime_error("Unknown dtt type");
            }
          }

          break;
        }
        default:
          throw std::runtime_error("Unsupported transform type");
        }
        
        // fixme: normalization for hfft
        switch (getConfig().getCommonParameters().normalize)
        {
        case Normalize::none:
          vkfftConfig.normalize = 0;
          break;
        case Normalize::unitary:
          vkfftConfig.normalize = 1;
          break;
        default:
          throw std::runtime_error("Unsupported normalization type");
        }
        vkfftConfig.makeForwardPlanOnly        = (getConfig().getTransformDirection() != Direction::inverse);
        vkfftConfig.makeInversePlanOnly        = (getConfig().getTransformDirection() != Direction::forward);

        vkfftConfig.isInputFormatted           = 1;
        vkfftConfig.isOutputFormatted          = 1;
        for (std::size_t i{}; i < getConfig().getShapeRank(); ++i)
        {
          vkfftConfig.inputBufferStride[i]  = safeIntCast<UInt>((getConfig().getTransformDirection() == Direction::forward)
                                                               ? getConfig().getSrcStrides()[i]
                                                               : getConfig().getDstStrides()[i]);
          vkfftConfig.outputBufferStride[i] = safeIntCast<UInt>((getConfig().getTransformDirection() == Direction::inverse)
                                                               ? getConfig().getSrcStrides()[i]
                                                               : getConfig().getDstStrides()[i]);
        }

        vkfftConfig.disableSetLocale = 1;

        // vkfftConfig.registerBoost              = getRegisterFileSize(Parent::mDevice) / getSharedMemorySize(Parent::mDevice);
        // vkfftConfig.registerBoostNonPow2       = 1;
        // vkfftConfig.registerBoost4Step         = getRegisterFileSize(Parent::mDevice) / getSharedMemorySize(Parent::mDevice);

        mApp.init(std::move(vkfftConfig));
      }

      /// @brief Destructor
      ~PlanImpl() override = default;

      /// @brief Inherit assignment operator
      using Parent::operator=;

      /**
       * @brief Execute the plan
       * @param src Source buffer
       * @param dst Destination buffer
       * @param execParams Execution parameters
       */
      void executeImpl(ExecParam src, ExecParam dst, const afft::gpu::ExecutionParameters& execParams) override
      {
        if (src.isSplit() || dst.isSplit())
        {
          throw std::runtime_error("vkfft does not support planar complex format");
        }

        int               inverse{};
        VkFFTLaunchParams params{};

        if (getConfig().getTransformDirection() == Direction::forward)
        {
          inverse = -1;

#       if AFFT_GPU_FRAMEWORK_IS_CUDA
          params.inputBuffer = src.data();
          params.buffer      = dst.data();
#       elif AFFT_GPU_FRAMEWORK_IS_HIP
          params.inputBuffer = src.data();
          params.buffer      = dst.data();
#       elif AFFT_GPU_FRAMEWORK_IS_OPENCL
          params.inputBuffer = reinterpret_cast<cl_mem*>(src.data());
          params.buffer      = reinterpret_cast<cl_mem*>(dst.data());
#       endif
        }
        else
        {
          inverse = 1;

#       if AFFT_GPU_FRAMEWORK_IS_CUDA
          params.buffer      = src.data();
          params.inputBuffer = dst.data();
#       elif AFFT_GPU_FRAMEWORK_IS_HIP
          params.buffer      = src.data();
          params.inputBuffer = dst.data();
#       elif AFFT_GPU_FRAMEWORK_IS_OPENCL
          params.buffer      = reinterpret_cast<cl_mem*>(src.data());
          params.inputBuffer = reinterpret_cast<cl_mem*>(dst.data());
#       endif
        }

#     if AFFT_GPU_FRAMEWORK_IS_CUDA
        mStream = execParams.stream;
#     elif AFFT_GPU_FRAMEWORK_IS_HIP
        mStream = execParams.stream;
#     elif AFFT_GPU_FRAMEWORK_IS_OPENCL
        mQueue = execParams.commandQueue;
#     endif

#     if AFFT_GPU_FRAMEWORK_IS_CUDA || AFFT_GPU_FRAMEWORK_IS_HIP
        void*             workspace{};
#     elif AFFT_GPU_FRAMEWORK_IS_OPENCL
        cl_mem            workspace{};
#     endif

        if (getConfig().getTargetConfig<Target::gpu>().externalWorkspace)
        {
          workspace = execParams.workspace;

          params.tempBuffer = &workspace;
        }

        {
#       if AFFT_GPU_FRAMEWORK_IS_CUDA
          cuda::ScopedDevice scopedDevice{getConfig().getTargetConfig<Target::gpu>().device};
#       elif AFFT_GPU_FRAMEWORK_IS_HIP
          hip::ScopedDevice scopedDevice{getConfig().getTargetConfig<Target::gpu>().device};
#       endif

          Error::check(VkFFTAppend(&mApp.get(), inverse, &params));
        }
      }
    protected:
    private:
#   if AFFT_GPU_FRAMEWORK_IS_CUDA
      CUdevice         mCuDevice{};
      cudaStream_t     mStream{0};
#   elif AFFT_GPU_FRAMEWORK_IS_HIP
      hipDevice_t      mHipDevice{};
      hipStream_t      mStream{0};
#   elif AFFT_GPU_FRAMEWORK_IS_OPENCL
      cl_context       mContext{};
      cl_device_id     mDevice{};
      cl_command_queue mQueue{};
#   endif
      App              mApp{};
  };

  /**
   * @brief Create a plan implementation
   * @param config Configuration
   * @return Plan implementation
   */
  [[nodiscard]] std::unique_ptr<PlanImpl> makePlanImpl(const Config& config)
  {
    const auto& commonParams = config.getCommonParameters();

    switch (commonParams.complexFormat)
    {
    case ComplexFormat::interleaved:
      break;
    default:
      throw makeException<std::runtime_error>("vkfft supports only interleaved complex format");
    }

    return std::make_unique<PlanImpl>(config);
  }
} // namespace afft::detail::gpu::vkfft

#endif /* AFFT_DETAIL_GPU_VKFFT_PLAN_IMPL_HPP */
