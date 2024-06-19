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

#ifndef ARCHITECTURE_HPP
#define ARCHITECTURE_HPP

#include <afft/afft.h>
#include <afft/afft.hpp>

#include "common.hpp"
#include "convert.hpp"

template<std::size_t shapeExt>
struct Convert<afft::MemoryBlock<shapeExt>>
  : StructConvertBase<afft::MemoryBlock<shapeExt>, afft_MemoryBlock>
{
  using typename StructConvertBase<afft::MemoryBlock<shapeExt>, afft_MemoryBlock>::CxxType;
  using typename StructConvertBase<afft::MemoryBlock<shapeExt>, afft_MemoryBlock>::CType;

  [[nodiscard]] static constexpr CxxType fromC(const CType& cValue, std::size_t shapeRank)
  {
    if constexpr (shapeExt != afft::dynamicExtent)
    {
      if (shapeRank != shapeExt)
      {
        throw afft_Error_internal;
      }
    }

    CxxType cxxValue{};
    cxxValue.starts  = afft::View<std::size_t, shapeExt>{cValue.starts, shapeRank};
    cxxValue.sizes   = afft::View<std::size_t, shapeExt>{cValue.sizes, shapeRank};
    cxxValue.strides = afft::View<std::size_t, shapeExt>{cValue.strides, shapeRank};

    return cxxValue;
  }

  [[nodiscard]] static constexpr CType toC(const CxxType& cxxValue)
  {
    CType cValue{};
    cValue.starts  = cxxValue.starts.data();
    cValue.sizes   = cxxValue.sizes.data();
    cValue.strides = cxxValue.strides.data();

    return cValue;
  }
};

template<std::size_t shapeExt>
struct Convert<afft::spst::MemoryLayout<shapeExt>>
  : StructConvertBase<afft::spst::MemoryLayout<shapeExt>, afft_spst_MemoryLayout>
{
  using typename StructConvertBase<afft::spst::MemoryLayout<shapeExt>, afft_spst_MemoryLayout>::CxxType;
  using typename StructConvertBase<afft::spst::MemoryLayout<shapeExt>, afft_spst_MemoryLayout>::CType;

  [[nodiscard]] static constexpr CxxType fromC(const CType& cValue, std::size_t shapeRank)
  {
    if constexpr (shapeExt != afft::dynamicExtent)
    {
      if (shapeRank != shapeExt)
      {
        throw afft_Error_internal;
      }
    }

    CxxType cxxValue{};

    if (cValue.srcStrides != nullptr)
    {
      cxxValue.srcStrides = afft::View<std::size_t, shapeExt>{cValue.srcStrides, shapeRank};
    }

    if (cValue.dstStrides != nullptr)
    {
      cxxValue.dstStrides = afft::View<std::size_t, shapeExt>{cValue.dstStrides, shapeRank};
    }

    return cxxValue;
  }

  [[nodiscard]] static constexpr CType toC(const CxxType& cxxValue)
  {
    CType cValue{};
    cValue.srcStrides = cxxValue.srcStrides.data();
    cValue.dstStrides = cxxValue.dstStrides.data();

    return cValue;
  }
};

template<std::size_t shapeExt>
struct Convert<afft::spst::cpu::Parameters<shapeExt>>
  : StructConvertBase<afft::spst::cpu::Parameters<shapeExt>, afft_spst_cpu_Parameters>
{
  using typename StructConvertBase<afft::spst::cpu::Parameters<shapeExt>, afft_spst_cpu_Parameters>::CxxType;
  using typename StructConvertBase<afft::spst::cpu::Parameters<shapeExt>, afft_spst_cpu_Parameters>::CType;

  [[nodiscard]] static constexpr CxxType fromC(const CType& cValue, std::size_t shapeRank)
  {
    if constexpr (shapeExt != afft::dynamicExtent)
    {
      if (shapeRank != shapeExt)
      {
        throw afft_Error_internal;
      }
    }

    CxxType cxxValue{};
    cxxValue.memoryLayout   = Convert<afft::spst::MemoryLayout<shapeExt>>::fromC(cValue.memoryLayout, shapeRank);
    cxxValue.complexFormat  = Convert<afft::ComplexFormat>::fromC(cValue.complexFormat);
    cxxValue.preserveSource = cValue.preserveSource;
    cxxValue.alignment      = Convert<afft::Alignment>::fromC(cValue.alignment);
    cxxValue.threadLimit    = cValue.threadLimit;

    return cxxValue;
  }

  [[nodiscard]] static constexpr CType toC(const CxxType& cxxValue)
  {
    CType cValue{};
    cValue.memoryLayout   = Convert<afft::spst::MemoryLayout<shapeExt>>::toC(cxxValue.memoryLayout);
    cValue.complexFormat  = Convert<afft::ComplexFormat>::toC(cxxValue.complexFormat);
    cValue.preserveSource = cxxValue.preserveSource;
    cValue.alignment      = Convert<afft::Alignment>::toC(cxxValue.alignment);
    cValue.threadLimit    = cxxValue.threadLimit;

    return cValue;
  }
};

template<std::size_t shapeExt>
struct Convert<afft::spst::gpu::Parameters<shapeExt>>
  : StructConvertBase<afft::spst::gpu::Parameters<shapeExt>, afft_spst_gpu_Parameters>
{
  using typename StructConvertBase<afft::spst::gpu::Parameters<shapeExt>, afft_spst_gpu_Parameters>::CxxType;
  using typename StructConvertBase<afft::spst::gpu::Parameters<shapeExt>, afft_spst_gpu_Parameters>::CType;

  [[nodiscard]] static constexpr CxxType fromC(const CType& cValue, std::size_t shapeRank)
  {
    if constexpr (shapeExt != afft::dynamicExtent)
    {
      if (shapeRank != shapeExt)
      {
        throw afft_Error_internal;
      }
    }

    CxxType cxxValue{};
    cxxValue.memoryLayout   = Convert<afft::spst::MemoryLayout<shapeExt>>::fromC(cValue.memoryLayout, shapeRank);
    cxxValue.complexFormat  = Convert<afft::ComplexFormat>::fromC(cValue.complexFormat);
    cxxValue.preserveSource = cValue.preserveSource;
# if AFFT_GPU_BACKEND_IS(CUDA)
    cxxValue.device         = cValue.device;
# elif AFFT_GPU_BACKEND_IS(HIP)
    cxxValue.device         = cValue.device;
# elif AFFT_GPU_BACKEND_IS(OPENCL)
    cxxValue.context        = cValue.context;
    cxxValue.device         = cValue.device;
# endif

    return cxxValue;
  }

  [[nodiscard]] static constexpr CType toC(const CxxType& cxxValue)
  {
    CType cValue{};
    cValue.memoryLayout   = Convert<afft::spst::MemoryLayout<shapeExt>>::toC(cxxValue.memoryLayout);
    cValue.complexFormat  = Convert<afft::ComplexFormat>::toC(cxxValue.complexFormat);
    cValue.preserveSource = cxxValue.preserveSource;
# if AFFT_GPU_BACKEND_IS(CUDA)
    cValue.device         = cxxValue.device;
# elif AFFT_GPU_BACKEND_IS(HIP)
    cValue.device         = cxxValue.device;
# elif AFFT_GPU_BACKEND_IS(OPENCL)
    cValue.context        = cxxValue.context;
    cValue.device         = cxxValue.device;
# endif

    return cValue;
  }
};

template<>
struct Convert<afft::spst::cpu::ExecutionParameters>
  : StructConvertBase<afft::spst::cpu::ExecutionParameters, afft_spst_cpu_ExecutionParameters>
{
  using typename StructConvertBase<afft::spst::cpu::ExecutionParameters, afft_spst_cpu_ExecutionParameters>::CxxType;
  using typename StructConvertBase<afft::spst::cpu::ExecutionParameters, afft_spst_cpu_ExecutionParameters>::CType;

  [[nodiscard]] static constexpr CxxType fromC(const CType&)
  {
    return CxxType{};
  }

  [[nodiscard]] static constexpr CType toC(const CxxType&)
  {
    return CType{};
  }
};

template<>
struct Convert<afft::spst::gpu::ExecutionParameters>
  : StructConvertBase<afft::spst::gpu::ExecutionParameters, afft_spst_gpu_ExecutionParameters>
{
  using typename StructConvertBase<afft::spst::gpu::ExecutionParameters, afft_spst_gpu_ExecutionParameters>::CxxType;
  using typename StructConvertBase<afft::spst::gpu::ExecutionParameters, afft_spst_gpu_ExecutionParameters>::CType;

  [[nodiscard]] static constexpr CxxType fromC([[maybe_unused]] const CType& cValue)
  {
    CxxType cxxValue{};
# if AFFT_GPU_BACKEND_IS(CUDA)
    cxxValue.stream       = cValue.stream;
    cxxValue.workspace    = cValue.workspace;
# elif AFFT_GPU_BACKEND_IS(HIP)
    cxxValue.stream       = cValue.stream;
    cxxValue.workspace    = cValue.workspace;
# elif AFFT_GPU_BACKEND_IS(OPENCL)
    cxxValue.commandQueue = cValue.commandQueue;
    cxxValue.workspace    = cValue.workspace;
# endif

    return cxxValue;
  }

  [[nodiscard]] static constexpr CType toC([[maybe_unused]] const CxxType& cxxValue)
  {
    CType cValue{};
# if AFFT_GPU_BACKEND_IS(CUDA)
    cValue.stream       = cxxValue.stream;
    cValue.workspace    = cxxValue.workspace;
# elif AFFT_GPU_BACKEND_IS(HIP)
    cValue.stream       = cxxValue.stream;
    cValue.workspace    = cxxValue.workspace;
# elif AFFT_GPU_BACKEND_IS(OPENCL)
    cValue.commandQueue = cxxValue.commandQueue;
    cValue.workspace    = cxxValue.workspace;
# endif

    return cValue;
  }
};

template<std::size_t shapeExt>
struct Convert<afft::mpst::MemoryLayout<shapeExt>>
  : StructConvertBase<afft::mpst::MemoryLayout<shapeExt>, afft_mpst_MemoryLayout>
{
  using typename StructConvertBase<afft::mpst::MemoryLayout<shapeExt>, afft_mpst_MemoryLayout>::CxxType;
  using typename StructConvertBase<afft::mpst::MemoryLayout<shapeExt>, afft_mpst_MemoryLayout>::CType;

  [[nodiscard]] static constexpr CxxType fromC(const CType& cValue, std::size_t shapeRank)
  {
    if constexpr (shapeExt != afft::dynamicExtent)
    {
      if (shapeRank != shapeExt)
      {
        throw afft_Error_internal;
      }
    }

    CxxType cxxValue{};
    cxxValue.srcBlock = Convert<afft::MemoryBlock<shapeExt>>::fromC(cValue.srcBlock, shapeRank);
    cxxValue.dstBlock = Convert<afft::MemoryBlock<shapeExt>>::fromC(cValue.dstBlock, shapeRank);
    
    if (cValue.srcAxesOrder != nullptr)
    {
      cxxValue.srcAxesOrder = afft::View<std::size_t, shapeExt>{cValue.srcAxesOrder, shapeRank};
    }

    if (cValue.dstAxesOrder != nullptr)
    {
      cxxValue.dstAxesOrder = afft::View<std::size_t, shapeExt>{cValue.dstAxesOrder, shapeRank};
    }

    return cxxValue;
  }

  [[nodiscard]] static constexpr CType toC(const CxxType& cxxValue)
  {
    CType cValue{};
    cValue.srcBlock     = Convert<afft::MemoryBlock<shapeExt>>::toC(cxxValue.srcBlock);
    cValue.dstBlock     = Convert<afft::MemoryBlock<shapeExt>>::toC(cxxValue.dstBlock);
    cValue.srcAxesOrder = cxxValue.srcAxesOrder.data();
    cValue.dstAxesOrder = cxxValue.dstAxesOrder.data();

    return cValue;
  }
};

template<std::size_t shapeExt>
struct Convert<afft::mpst::cpu::Parameters<shapeExt>>
  : StructConvertBase<afft::mpst::cpu::Parameters<shapeExt>, afft_mpst_cpu_Parameters>
{
  using typename StructConvertBase<afft::mpst::cpu::Parameters<shapeExt>, afft_mpst_cpu_Parameters>::CxxType;
  using typename StructConvertBase<afft::mpst::cpu::Parameters<shapeExt>, afft_mpst_cpu_Parameters>::CType;

  [[nodiscard]] static constexpr CxxType fromC(const CType& cValue, std::size_t shapeRank)
  {
    if constexpr (shapeExt != afft::dynamicExtent)
    {
      if (shapeRank != shapeExt)
      {
        throw afft_Error_internal;
      }
    }

    CxxType cxxValue{};
    cxxValue.memoryLayout   = Convert<afft::mpst::MemoryLayout<shapeExt>>::fromC(cValue.memoryLayout, shapeRank);
    cxxValue.complexFormat  = Convert<afft::ComplexFormat>::fromC(cValue.complexFormat);
    cxxValue.preserveSource = cValue.preserveSource;
    cxxValue.alignment      = Convert<afft::Alignment>::fromC(cValue.alignment);
    cxxValue.threadLimit    = cValue.threadLimit;

    return cxxValue;
  }

  [[nodiscard]] static constexpr CType toC(const CxxType& cxxValue)
  {
    CType cValue{};
    cValue.memoryLayout   = Convert<afft::mpst::MemoryLayout<shapeExt>>::toC(cxxValue.memoryLayout);
    cValue.complexFormat  = Convert<afft::ComplexFormat>::toC(cxxValue.complexFormat);
    cValue.preserveSource = cxxValue.preserveSource;
    cValue.alignment      = Convert<afft::Alignment>::toC(cxxValue.alignment);
    cValue.threadLimit    = cxxValue.threadLimit;

    return cValue;
  }
};

template<std::size_t shapeExt>
struct Convert<afft::mpst::gpu::Parameters<shapeExt>>
  : StructConvertBase<afft::mpst::gpu::Parameters<shapeExt>, afft_mpst_gpu_Parameters>
{
  using typename StructConvertBase<afft::mpst::gpu::Parameters<shapeExt>, afft_mpst_gpu_Parameters>::CxxType;
  using typename StructConvertBase<afft::mpst::gpu::Parameters<shapeExt>, afft_mpst_gpu_Parameters>::CType;

  [[nodiscard]] static constexpr CxxType fromC(const CType& cValue, std::size_t shapeRank)
  {
    if constexpr (shapeExt != afft::dynamicExtent)
    {
      if (shapeRank != shapeExt)
      {
        throw afft_Error_internal;
      }
    }

    CxxType cxxValue{};
    cxxValue.memoryLayout   = Convert<afft::mpst::MemoryLayout<shapeExt>>::fromC(cValue.memoryLayout, shapeRank);
    cxxValue.complexFormat  = Convert<afft::ComplexFormat>::fromC(cValue.complexFormat);
    cxxValue.preserveSource = cValue.preserveSource;
# if AFFT_GPU_BACKEND_IS(CUDA)
    cxxValue.device         = cValue.device;
# elif AFFT_GPU_BACKEND_IS(HIP)
    cxxValue.device         = cValue.device;
# elif AFFT_GPU_BACKEND_IS(OPENCL)
    cxxValue.context        = cValue.context;
    cxxValue.device         = cValue.device;
# endif
  
    return cxxValue;
  }
  
  [[nodiscard]] static constexpr CType toC(const CxxType& cxxValue)
  {
    CType cValue{};
    cValue.memoryLayout   = Convert<afft::mpst::MemoryLayout<shapeExt>>::toC(cxxValue.memoryLayout);
    cValue.complexFormat  = Convert<afft::ComplexFormat>::toC(cxxValue.complexFormat);
    cValue.preserveSource = cxxValue.preserveSource;
# if AFFT_GPU_BACKEND_IS(CUDA)
    cValue.device         = cxxValue.device;
# elif AFFT_GPU_BACKEND_IS(HIP)
    cValue.device         = cxxValue.device;
# elif AFFT_GPU_BACKEND_IS(OPENCL)
    cValue.context        = cxxValue.context;
    cValue.device         = cxxValue.device;
# endif
  
    return cValue;
  }
};

template<>
struct Convert<afft::mpst::cpu::ExecutionParameters>
  : StructConvertBase<afft::mpst::cpu::ExecutionParameters, afft_mpst_cpu_ExecutionParameters>
{
  using typename StructConvertBase<afft::mpst::cpu::ExecutionParameters, afft_mpst_cpu_ExecutionParameters>::CxxType;
  using typename StructConvertBase<afft::mpst::cpu::ExecutionParameters, afft_mpst_cpu_ExecutionParameters>::CType;

  [[nodiscard]] static constexpr CxxType fromC(const CType& cValue)
  {
    CxxType cxxValue{};
    cxxValue.workspace = cValue.workspace;

    return cxxValue;
  }

  [[nodiscard]] static constexpr CType toC(const CxxType& cxxValue)
  {
    CType cValue{};
    cValue.workspace = cxxValue.workspace;

    return cValue;
  }
};

template<>
struct Convert<afft::mpst::gpu::ExecutionParameters>
  : StructConvertBase<afft::mpst::gpu::ExecutionParameters, afft_mpst_gpu_ExecutionParameters>
{
  using typename StructConvertBase<afft::mpst::gpu::ExecutionParameters, afft_mpst_gpu_ExecutionParameters>::CxxType;
  using typename StructConvertBase<afft::mpst::gpu::ExecutionParameters, afft_mpst_gpu_ExecutionParameters>::CType;

  [[nodiscard]] static constexpr CxxType fromC([[maybe_unused]] const CType& cValue)
  {
    CxxType cxxValue{};
# if AFFT_GPU_BACKEND_IS(CUDA)
    cxxValue.stream    = cValue.stream;
    cxxValue.workspace = cValue.workspace;
# elif AFFT_GPU_BACKEND_IS(HIP)
    cxxValue.stream    = cValue.stream;
    cxxValue.workspace = cValue.workspace;
# elif AFFT_GPU_BACKEND_IS(OPENCL)
    cxxValue.commandQueue = cValue.commandQueue;
    cxxValue.workspace    = cValue.workspace;
# endif
  
    return cxxValue;
  }

  [[nodiscard]] static constexpr CType toC([[maybe_unused]] const CxxType& cxxValue)
  {
    CType cValue{};
# if AFFT_GPU_BACKEND_IS(CUDA)
    cValue.stream    = cxxValue.stream;
    cValue.workspace = cxxValue.workspace;
# elif AFFT_GPU_BACKEND_IS(HIP)
    cValue.stream    = cxxValue.stream;
    cValue.workspace = cxxValue.workspace;
# elif AFFT_GPU_BACKEND_IS(OPENCL)
    cValue.commandQueue = cxxValue.commandQueue;
    cValue.workspace    = cxxValue.workspace;
# endif
    
    return cValue;
  }
};

#endif /* ARCHITECTURE_HPP */
