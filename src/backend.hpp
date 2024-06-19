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

#ifndef BACKEND_HPP
#define BACKEND_HPP

#include <afft/afft.h>
#include <afft/afft.hpp>

#include "convert.hpp"

// Backend
template<>
struct Convert<afft::Backend>
  : EnumConvertBase<afft::Backend, afft_Backend, afft_Error_invalidBackend>
{
  static_assert(afft_Backend_clfft     == afft::Backend::clfft);
  static_assert(afft_Backend_cufft     == afft::Backend::cufft);
  static_assert(afft_Backend_fftw3     == afft::Backend::fftw3);
  static_assert(afft_Backend_heffte    == afft::Backend::heffte);
  static_assert(afft_Backend_hipfft    == afft::Backend::hipfft);
  static_assert(afft_Backend_mkl       == afft::Backend::mkl);
  static_assert(afft_Backend_pocketfft == afft::Backend::pocketfft);
  static_assert(afft_Backend_rocfft    == afft::Backend::rocfft);
  static_assert(afft_Backend_vkfft     == afft::Backend::vkfft);
};

static_assert(AFFT_BACKEND_COUNT == afft::backendCount);

// BackendMask
template<>
struct Convert<afft::BackendMask>
  : EnumConvertBase<afft::BackendMask, afft_BackendMask> {};

// SelectStrategy
template<>
struct Convert<afft::SelectStrategy>
  : EnumConvertBase<afft::SelectStrategy, afft_SelectStrategy, afft_Error_invalidSelectStrategy>
{
  static_assert(afft_SelectStrategy_first == afft::SelectStrategy::first);
  static_assert(afft_SelectStrategy_best  == afft::SelectStrategy::best);
};

template<>
struct Convert<afft::spst::gpu::clfft::Parameters>
  : StructConvertBase<afft::spst::gpu::clfft::Parameters, afft_spst_gpu_clfft_Parameters>
{
  /**
   * @brief Convert from C to C++.
   * @param cValue C struct value.
   * @return C++ struct value.
   */
  [[nodiscard]] static constexpr CxxType fromC(const CType& cValue)
  {
    CxxType cxxValue{};
    cxxValue.useFastMath = cValue.useFastMath;

    return cxxValue;
  }

  /**
   * @brief Convert from C++ to C.
   * @param cxxValue C++ struct value.
   * @return C struct value.
   */
  [[nodiscard]] static constexpr CType toC(const CxxType& cxxValue)
  {
    CType cValue{};
    cValue.useFastMath = cxxValue.useFastMath;

    return cValue;
  }
};

// cuFFT WorkspacePolicy
template<>
struct Convert<afft::cufft::WorkspacePolicy>
  : EnumConvertBase<afft::cufft::WorkspacePolicy, afft_cufft_WorkspacePolicy, afft_Error_invalidCufftWorkspacePolicy>
{
  static_assert(afft_cufft_WorkspacePolicy_performance == afft::cufft::WorkspacePolicy::performance);
  static_assert(afft_cufft_WorkspacePolicy_minimal     == afft::cufft::WorkspacePolicy::minimal);
  static_assert(afft_cufft_WorkspacePolicy_user        == afft::cufft::WorkspacePolicy::user);
};

template<>
struct Convert<afft::spst::gpu::cufft::Parameters>
  : StructConvertBase<afft::spst::gpu::cufft::Parameters, afft_spst_gpu_cufft_Parameters>
{
  /**
   * @brief Convert from C to C++.
   * @param cValue C struct value.
   * @return C++ struct value.
   */
  [[nodiscard]] static constexpr CxxType fromC(const CType& cValue)
  {
    CxxType cxxValue{};
    cxxValue.workspacePolicy   = Convert<afft::cufft::WorkspacePolicy>::fromC(cValue.workspacePolicy);
    cxxValue.usePatientJit     = cValue.usePatientJit;
    cxxValue.userWorkspaceSize = cValue.userWorkspaceSize;

    return cxxValue;
  }

  /**
   * @brief Convert from C++ to C.
   * @param cxxValue C++ struct value.
   * @return C struct value.
   */
  [[nodiscard]] static constexpr CType toC(const CxxType& cxxValue)
  {
    CType cValue{};
    cValue.workspacePolicy   = Convert<afft::cufft::WorkspacePolicy>::toC(cxxValue.workspacePolicy);
    cValue.usePatientJit     = cxxValue.usePatientJit;
    cValue.userWorkspaceSize = cxxValue.userWorkspaceSize;

    return cValue;
  }
};

template<>
struct Convert<afft::spmt::gpu::cufft::Parameters>
  : StructConvertBase<afft::spmt::gpu::cufft::Parameters, afft_spmt_gpu_cufft_Parameters>
{
  /**
   * @brief Convert from C to C++.
   * @param cValue C struct value.
   * @return C++ struct value.
   */
  [[nodiscard]] static constexpr CxxType fromC(const CType& cValue)
  {
    CxxType cxxValue{};
    cxxValue.usePatientJit = cValue.usePatientJit;

    return cxxValue;
  }

  /**
   * @brief Convert from C++ to C.
   * @param cxxValue C++ struct value.
   * @return C struct value.
   */
  [[nodiscard]] static constexpr CType toC(const CxxType& cxxValue)
  {
    CType cValue{};
    cValue.usePatientJit = cxxValue.usePatientJit;

    return cValue;
  }
};

template<>
struct Convert<afft::mpst::gpu::cufft::Parameters>
  : StructConvertBase<afft::mpst::gpu::cufft::Parameters, afft_mpst_gpu_cufft_Parameters>
{
  /**
   * @brief Convert from C to C++.
   * @param cValue C struct value.
   * @return C++ struct value.
   */
  [[nodiscard]] static constexpr CxxType fromC(const CType& cValue)
  {
    CxxType cxxValue{};
    cxxValue.usePatientJit = cValue.usePatientJit;

    return cxxValue;
  }

  /**
   * @brief Convert from C++ to C.
   * @param cxxValue C++ struct value.
   * @return C struct value.
   */
  [[nodiscard]] static constexpr CType toC(const CxxType& cxxValue)
  {
    CType cValue{};
    cValue.usePatientJit = cxxValue.usePatientJit;

    return cValue;
  }
};

// FFTW3 PlannerFlag
template<>
struct Convert<afft::fftw3::PlannerFlag>
  : EnumConvertBase<afft::fftw3::PlannerFlag, afft_fftw3_PlannerFlag, afft_Error_invalidFftw3PlannerFlag>
{
  static_assert(afft_fftw3_PlannerFlag_estimate        == afft::fftw3::PlannerFlag::estimate);
  static_assert(afft_fftw3_PlannerFlag_measure         == afft::fftw3::PlannerFlag::measure);
  static_assert(afft_fftw3_PlannerFlag_patient         == afft::fftw3::PlannerFlag::patient);
  static_assert(afft_fftw3_PlannerFlag_exhaustive      == afft::fftw3::PlannerFlag::exhaustive);
  static_assert(afft_fftw3_PlannerFlag_estimatePatient == afft::fftw3::PlannerFlag::estimatePatient);
};

template<>
struct Convert<afft::spst::cpu::fftw3::Parameters>
  : StructConvertBase<afft::spst::cpu::fftw3::Parameters, afft_spst_cpu_fftw3_Parameters>
{
  /**
   * @brief Convert from C to C++.
   * @param cValue C struct value.
   * @return C++ struct value.
   */
  [[nodiscard]] static constexpr CxxType fromC(const CType& cValue)
  {
    CxxType cxxValue{};
    cxxValue.plannerFlag       = Convert<afft::fftw3::PlannerFlag>::fromC(cValue.plannerFlag);
    cxxValue.conserveMemory    = cValue.conserveMemory;
    cxxValue.wisdomOnly        = cValue.wisdomOnly;
    cxxValue.allowLargeGeneric = cValue.allowLargeGeneric;
    cxxValue.allowPruning      = cValue.allowPruning;
    cxxValue.timeLimit         = std::chrono::duration<double>{cValue.timeLimit};

    return cxxValue;
  }

  /**
   * @brief Convert from C++ to C.
   * @param cxxValue C++ struct value.
   * @return C struct value.
   */
  [[nodiscard]] static constexpr CType toC(const CxxType& cxxValue)
  {
    CType cValue{};
    cValue.plannerFlag       = Convert<afft::fftw3::PlannerFlag>::toC(cxxValue.plannerFlag);
    cValue.conserveMemory    = cxxValue.conserveMemory;
    cValue.wisdomOnly        = cxxValue.wisdomOnly;
    cValue.allowLargeGeneric = cxxValue.allowLargeGeneric;
    cValue.allowPruning      = cxxValue.allowPruning;
    cValue.timeLimit         = cxxValue.timeLimit.count();

    return cValue;
  }
};

template<>
struct Convert<afft::mpst::cpu::fftw3::Parameters>
  : StructConvertBase<afft::mpst::cpu::fftw3::Parameters, afft_mpst_cpu_fftw3_Parameters>
{
  /**
   * @brief Convert from C to C++.
   * @param cValue C struct value.
   * @return C++ struct value.
   */
  [[nodiscard]] static constexpr CxxType fromC(const CType& cValue)
  {
    CxxType cxxValue{};
    cxxValue.plannerFlag       = Convert<afft::fftw3::PlannerFlag>::fromC(cValue.plannerFlag);
    cxxValue.conserveMemory    = cValue.conserveMemory;
    cxxValue.wisdomOnly        = cValue.wisdomOnly;
    cxxValue.allowLargeGeneric = cValue.allowLargeGeneric;
    cxxValue.allowPruning      = cValue.allowPruning;
    cxxValue.timeLimit         = std::chrono::duration<double>{cValue.timeLimit};
    cxxValue.blockSize         = cValue.blockSize;

    if (cxxValue.timeLimit.count() < 0.0)
    {
      throw afft_Error_fftw3_invalidTimeLimit;
    }

    if (cxxValue.blockSize == 0)
    {
      throw afft_Error_fftw3_invalidBlockSize;
    }

    return cxxValue;
  }

  /**
   * @brief Convert from C++ to C.
   * @param cxxValue C++ struct value.
   * @return C struct value.
   */
  [[nodiscard]] static constexpr CType toC(const CxxType& cxxValue)
  {
    CType cValue{};
    cValue.plannerFlag       = Convert<afft::fftw3::PlannerFlag>::toC(cxxValue.plannerFlag);
    cValue.conserveMemory    = cxxValue.conserveMemory;
    cValue.wisdomOnly        = cxxValue.wisdomOnly;
    cValue.allowLargeGeneric = cxxValue.allowLargeGeneric;
    cValue.allowPruning      = cxxValue.allowPruning;
    cValue.timeLimit         = cxxValue.timeLimit.count();
    cValue.blockSize         = cxxValue.blockSize;

    if (cxxValue.timeLimit.count() < 0.0)
    {
      throw afft_Error_internal;
    }

    if (cValue.blockSize == 0)
    {
      throw afft_Error_internal;
    }

    return cValue;
  }
};

// HeFFTe cpu backend
template<>
struct Convert<afft::heffte::cpu::Backend>
  : EnumConvertBase<afft::heffte::cpu::Backend, afft_heffte_cpu_Backend, afft_Error_invalidHeffteCpuBackend>
{
  static_assert(afft_heffte_cpu_Backend_fftw3 == afft::heffte::cpu::Backend::fftw3);
  static_assert(afft_heffte_cpu_Backend_mkl   == afft::heffte::cpu::Backend::mkl);
};

// HeFFTe gpu backend
template<>
struct Convert<afft::heffte::gpu::Backend>
  : EnumConvertBase<afft::heffte::gpu::Backend, afft_heffte_gpu_Backend, afft_Error_invalidHeffteGpuBackend>
{
  static_assert(afft_heffte_gpu_Backend_cufft  == afft::heffte::gpu::Backend::cufft);
  static_assert(afft_heffte_gpu_Backend_rocfft == afft::heffte::gpu::Backend::rocfft);
};

template<>
struct Convert<afft::mpst::cpu::heffte::Parameters>
  : StructConvertBase<afft::mpst::cpu::heffte::Parameters, afft_mpst_cpu_heffte_Parameters>
{
  /**
   * @brief Convert from C to C++.
   * @param cValue C struct value.
   * @return C++ struct value.
   */
  [[nodiscard]] static constexpr CxxType fromC(const CType& cValue)
  {
    CxxType cxxValue{};
    cxxValue.backend     = Convert<afft::heffte::cpu::Backend>::fromC(cValue.backend);
    cxxValue.useReorder  = cValue.useReorder;
    cxxValue.useAllToAll = cValue.useAllToAll;
    cxxValue.usePencils  = cValue.usePencils;

    return cxxValue;
  }

  /**
   * @brief Convert from C++ to C.
   * @param cxxValue C++ struct value.
   * @return C struct value.
   */
  [[nodiscard]] static constexpr CType toC(const CxxType& cxxValue)
  {
    CType cValue{};
    cValue.backend     = Convert<afft::heffte::cpu::Backend>::toC(cxxValue.backend);
    cValue.useReorder  = cxxValue.useReorder;
    cValue.useAllToAll = cxxValue.useAllToAll;
    cValue.usePencils  = cxxValue.usePencils;

    return cValue;
  }
};

template<>
struct Convert<afft::mpst::gpu::heffte::Parameters>
  : StructConvertBase<afft::mpst::gpu::heffte::Parameters, afft_mpst_gpu_heffte_Parameters>
{
  /**
   * @brief Convert from C to C++.
   * @param cValue C struct value.
   * @return C++ struct value.
   */
  [[nodiscard]] static constexpr CxxType fromC(const CType& cValue)
  {
    CxxType cxxValue{};
    cxxValue.backend     = Convert<afft::heffte::gpu::Backend>::fromC(cValue.backend);
    cxxValue.useReorder  = cValue.useReorder;
    cxxValue.useAllToAll = cValue.useAllToAll;
    cxxValue.usePencils  = cValue.usePencils;

    return cxxValue;
  }

  /**
   * @brief Convert from C++ to C.
   * @param cxxValue C++ struct value.
   * @return C struct value.
   */
  [[nodiscard]] static constexpr CType toC(const CxxType& cxxValue)
  {
    CType cValue{};
    cValue.backend     = Convert<afft::heffte::gpu::Backend>::toC(cxxValue.backend);
    cValue.useReorder  = cxxValue.useReorder;
    cValue.useAllToAll = cxxValue.useAllToAll;
    cValue.usePencils  = cxxValue.usePencils;

    return cValue;
  }
};

template<>
struct Convert<afft::spst::cpu::BackendParameters>
  : StructConvertBase<afft::spst::cpu::BackendParameters, afft_spst_cpu_BackendParameters>
{
  /**
   * @brief Convert from C to C++.
   * @param cValue C struct value.
   * @return C++ struct value.
   */
  [[nodiscard]] static CxxType fromC(const CType& cValue)
  {
    CxxType cxxValue{};
    cxxValue.strategy = Convert<afft::SelectStrategy>::fromC(cValue.strategy);
    cxxValue.mask     = Convert<afft::BackendMask>::fromC(cValue.mask);
    cxxValue.order    = afft::View<afft::Backend>{reinterpret_cast<const afft::Backend*>(cValue.order), cValue.orderSize};
    cxxValue.fftw3    = Convert<afft::spst::cpu::fftw3::Parameters>::fromC(cValue.fftw3);

    if (cValue.orderSize > 0 && cValue.order == nullptr)
    {
      throw afft_Error_invalidBackendOrder;
    }

    for (const auto backend : cxxValue.order)
    {
      if (!afft::detail::isValid(backend))
      {
        throw afft_Error_invalidBackendOrder;
      }
    }

    return cxxValue;
  }

  /**
   * @brief Convert from C++ to C.
   * @param cxxValue C++ struct value.
   * @return C struct value.
   */
  [[nodiscard]] static CType toC(const CxxType& cxxValue)
  {
    CType cValue{};
    cValue.strategy  = Convert<afft::SelectStrategy>::toC(cxxValue.strategy);
    cValue.mask      = Convert<afft::BackendMask>::toC(cxxValue.mask);
    cValue.orderSize = cxxValue.order.size();
    cValue.order     = reinterpret_cast<const afft_Backend*>(cxxValue.order.data());
    cValue.fftw3     = Convert<afft::spst::cpu::fftw3::Parameters>::toC(cxxValue.fftw3);

    if (cValue.orderSize > 0 && cValue.order == nullptr)
    {
      throw afft_Error_internal;
    }

    for (const auto backend : cxxValue.order)
    {
      if (!afft::detail::isValid(backend))
      {
        throw afft_Error_internal;
      }
    }

    return cValue;
  }
};

template<>
struct Convert<afft::spst::gpu::BackendParameters>
  : StructConvertBase<afft::spst::gpu::BackendParameters, afft_spst_gpu_BackendParameters>
{
  /**
   * @brief Convert from C to C++.
   * @param cValue C struct value.
   * @return C++ struct value.
   */
  [[nodiscard]] static CxxType fromC(const CType& cValue)
  {
    CxxType cxxValue{};
    cxxValue.strategy = Convert<afft::SelectStrategy>::fromC(cValue.strategy);
    cxxValue.mask     = Convert<afft::BackendMask>::fromC(cValue.mask);
    cxxValue.order    = afft::View<afft::Backend>{reinterpret_cast<const afft::Backend*>(cValue.order), cValue.orderSize};
    cxxValue.clfft    = Convert<afft::spst::gpu::clfft::Parameters>::fromC(cValue.clfft);
    cxxValue.cufft    = Convert<afft::spst::gpu::cufft::Parameters>::fromC(cValue.cufft);

    if (cValue.orderSize > 0 && cValue.order == nullptr)
    {
      throw afft_Error_invalidBackendOrder;
    }

    for (const auto backend : cxxValue.order)
    {
      if (!afft::detail::isValid(backend))
      {
        throw afft_Error_invalidBackendOrder;
      }
    }

    return cxxValue;
  }

  /**
   * @brief Convert from C++ to C.
   * @param cxxValue C++ struct value.
   * @return C struct value.
   */
  [[nodiscard]] static CType toC(const CxxType& cxxValue)
  {
    CType cValue{};
    cValue.strategy  = Convert<afft::SelectStrategy>::toC(cxxValue.strategy);
    cValue.mask      = Convert<afft::BackendMask>::toC(cxxValue.mask);
    cValue.orderSize = cxxValue.order.size();
    cValue.order     = reinterpret_cast<const afft_Backend*>(cxxValue.order.data());
    cValue.clfft     = Convert<afft::spst::gpu::clfft::Parameters>::toC(cxxValue.clfft);
    cValue.cufft     = Convert<afft::spst::gpu::cufft::Parameters>::toC(cxxValue.cufft);

    if (cValue.orderSize > 0 && cValue.order == nullptr)
    {
      throw afft_Error_internal;
    }

    for (const auto backend : cxxValue.order)
    {
      if (!afft::detail::isValid(backend))
      {
        throw afft_Error_internal;
      }
    }

    return cValue;
  }
};

template<>
struct Convert<afft::spmt::gpu::BackendParameters>
  : StructConvertBase<afft::spmt::gpu::BackendParameters, afft_spmt_gpu_BackendParameters>
{
  /**
   * @brief Convert from C to C++.
   * @param cValue C struct value.
   * @return C++ struct value.
   */
  [[nodiscard]] static CxxType fromC(const CType& cValue)
  {
    CxxType cxxValue{};
    cxxValue.strategy = Convert<afft::SelectStrategy>::fromC(cValue.strategy);
    cxxValue.mask     = Convert<afft::BackendMask>::fromC(cValue.mask);
    cxxValue.order    = afft::View<afft::Backend>{reinterpret_cast<const afft::Backend*>(cValue.order), cValue.orderSize};
    cxxValue.cufft    = Convert<afft::spmt::gpu::cufft::Parameters>::fromC(cValue.cufft);

    if (cValue.orderSize > 0 && cValue.order == nullptr)
    {
      throw afft_Error_invalidBackendOrder;
    }

    for (const auto backend : cxxValue.order)
    {
      if (!afft::detail::isValid(backend))
      {
        throw afft_Error_invalidBackendOrder;
      }
    }

    return cxxValue;
  }

  /**
   * @brief Convert from C++ to C.
   * @param cxxValue C++ struct value.
   * @return C struct value.
   */
  [[nodiscard]] static CType toC(const CxxType& cxxValue)
  {
    CType cValue{};
    cValue.strategy  = Convert<afft::SelectStrategy>::toC(cxxValue.strategy);
    cValue.mask      = Convert<afft::BackendMask>::toC(cxxValue.mask);
    cValue.orderSize = cxxValue.order.size();
    cValue.order     = reinterpret_cast<const afft_Backend*>(cxxValue.order.data());
    cValue.cufft     = Convert<afft::spmt::gpu::cufft::Parameters>::toC(cxxValue.cufft);

    if (cValue.orderSize > 0 && cValue.order == nullptr)
    {
      throw afft_Error_internal;
    }

    for (const auto backend : cxxValue.order)
    {
      if (!afft::detail::isValid(backend))
      {
        throw afft_Error_internal;
      }
    }

    return cValue;
  }
};

template<>
struct Convert<afft::mpst::cpu::BackendParameters>
  : StructConvertBase<afft::mpst::cpu::BackendParameters, afft_mpst_cpu_BackendParameters>
{
  /**
   * @brief Convert from C to C++.
   * @param cValue C struct value.
   * @return C++ struct value.
   */
  [[nodiscard]] static CxxType fromC(const CType& cValue)
  {
    CxxType cxxValue{};
    cxxValue.strategy = Convert<afft::SelectStrategy>::fromC(cValue.strategy);
    cxxValue.mask     = Convert<afft::BackendMask>::fromC(cValue.mask);
    cxxValue.order    = afft::View<afft::Backend>{reinterpret_cast<const afft::Backend*>(cValue.order), cValue.orderSize};
    cxxValue.fftw3    = Convert<afft::mpst::cpu::fftw3::Parameters>::fromC(cValue.fftw3);
    cxxValue.heffte   = Convert<afft::mpst::cpu::heffte::Parameters>::fromC(cValue.heffte);

    if (cValue.orderSize > 0 && cValue.order == nullptr)
    {
      throw afft_Error_invalidBackendOrder;
    }

    for (const auto backend : cxxValue.order)
    {
      if (!afft::detail::isValid(backend))
      {
        throw afft_Error_invalidBackendOrder;
      }
    }

    return cxxValue;
  }

  /**
   * @brief Convert from C++ to C.
   * @param cxxValue C++ struct value.
   * @return C struct value.
   */
  [[nodiscard]] static CType toC(const CxxType& cxxValue)
  {
    CType cValue{};
    cValue.strategy  = Convert<afft::SelectStrategy>::toC(cxxValue.strategy);
    cValue.mask      = Convert<afft::BackendMask>::toC(cxxValue.mask);
    cValue.orderSize = cxxValue.order.size();
    cValue.order     = reinterpret_cast<const afft_Backend*>(cxxValue.order.data());
    cValue.fftw3     = Convert<afft::mpst::cpu::fftw3::Parameters>::toC(cxxValue.fftw3);
    cValue.heffte    = Convert<afft::mpst::cpu::heffte::Parameters>::toC(cxxValue.heffte);

    if (cValue.orderSize > 0 && cValue.order == nullptr)
    {
      throw afft_Error_internal;
    }

    for (const auto backend : cxxValue.order)
    {
      if (!afft::detail::isValid(backend))
      {
        throw afft_Error_internal;
      }
    }

    return cValue;
  }
};

template<>
struct Convert<afft::mpst::gpu::BackendParameters>
  : StructConvertBase<afft::mpst::gpu::BackendParameters, afft_mpst_gpu_BackendParameters>
{
  /**
   * @brief Convert from C to C++.
   * @param cValue C struct value.
   * @return C++ struct value.
   */
  [[nodiscard]] static CxxType fromC(const CType& cValue)
  {
    CxxType cxxValue{};
    cxxValue.strategy = Convert<afft::SelectStrategy>::fromC(cValue.strategy);
    cxxValue.mask     = Convert<afft::BackendMask>::fromC(cValue.mask);
    cxxValue.order    = afft::View<afft::Backend>{reinterpret_cast<const afft::Backend*>(cValue.order), cValue.orderSize};
    cxxValue.cufft    = Convert<afft::mpst::gpu::cufft::Parameters>::fromC(cValue.cufft);
    cxxValue.heffte   = Convert<afft::mpst::gpu::heffte::Parameters>::fromC(cValue.heffte);

    if (cValue.orderSize > 0 && cValue.order == nullptr)
    {
      throw afft_Error_invalidBackendOrder;
    }

    for (const auto backend : cxxValue.order)
    {
      if (!afft::detail::isValid(backend))
      {
        throw afft_Error_invalidBackendOrder;
      }
    }

    return cxxValue;
  }

  /**
   * @brief Convert from C++ to C.
   * @param cxxValue C++ struct value.
   * @return C struct value.
   */
  [[nodiscard]] static CType toC(const CxxType& cxxValue)
  {
    CType cValue{};
    cValue.strategy  = Convert<afft::SelectStrategy>::toC(cxxValue.strategy);
    cValue.mask      = Convert<afft::BackendMask>::toC(cxxValue.mask);
    cValue.orderSize = cxxValue.order.size();
    cValue.order     = reinterpret_cast<const afft_Backend*>(cxxValue.order.data());
    cValue.cufft     = Convert<afft::mpst::gpu::cufft::Parameters>::toC(cxxValue.cufft);
    cValue.heffte    = Convert<afft::mpst::gpu::heffte::Parameters>::toC(cxxValue.heffte);

    if (cValue.orderSize > 0 && cValue.order == nullptr)
    {
      throw afft_Error_internal;
    }

    for (const auto backend : cxxValue.order)
    {
      if (!afft::detail::isValid(backend))
      {
        throw afft_Error_internal;
      }
    }

    return cValue;
  }
};

#endif /* BACKEND_HPP */
