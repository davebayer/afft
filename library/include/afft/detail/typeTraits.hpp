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

#ifndef AFFT_DETAIL_TYPE_TRAITS_HPP
#define AFFT_DETAIL_TYPE_TRAITS_HPP

#ifndef AFFT_TOP_LEVEL_INCLUDE
# include "include.hpp"
#endif

#include "common.hpp"
#include "../backend.hpp"
#include "../memory.hpp"
#include "../mp.hpp"
#include "../target.hpp"
#include "../transform.hpp"
#include "../type.hpp"

namespace afft::detail
{
  /**
   * @brief TypeProperties helper. Removes const and volatile from Complex template parameter type.
   * @tparam T The type.
   */
  template<typename T>
  struct TypePropertiesHelper
    : TypeProperties<std::remove_cv_t<T>> {};
  
  /// @brief Specialization for std::complex.
  template<typename T>
  struct TypePropertiesHelper<std::complex<T>>
    : TypeProperties<std::complex<std::remove_cv_t<T>>> {};

  /**
   * @brief TransformParameters type for given transform.
   * @tparam transform The transform type.
   */
  template<Transform transform>
  struct TransformParametersSelect;

  /// @brief Specialization for dft transform.
  template<>
  struct TransformParametersSelect<Transform::dft>
  {
    using CType   = afft_dft_Parameters;
    using CxxType = afft::dft::Parameters;
  };

  /// @brief Specialization for dht transform.
  template<>
  struct TransformParametersSelect<Transform::dht>
  {
    using CType   = afft_dht_Parameters;
    using CxxType = afft::dht::Parameters;
  };

  /// @brief Specialization for dtt transform.
  template<>
  struct TransformParametersSelect<Transform::dtt>
  {
    using CType   = afft_dtt_Parameters;
    using CxxType = afft::dtt::Parameters;
  };

  /**
   * @brief Check if the type is C++ TransformParameters.
   * @tparam T The type.
   */
  template<typename T>
  struct IsCxxTransformParameters
    : std::bool_constant<std::is_same_v<cxx::remove_cvref_t<T>, afft::dft::Parameters> ||
                         std::is_same_v<cxx::remove_cvref_t<T>, afft::dht::Parameters> ||
                         std::is_same_v<cxx::remove_cvref_t<T>, afft::dtt::Parameters>> {};

  /**
   * @brief Check if the type is C++ TransformParameters.
   * @tparam T The type.
   */
  template<typename T>
  inline constexpr bool isCxxTransformParameters = IsCxxTransformParameters<T>::value;

  /**
   * @brief Check if the type is C TransformParameters.
   * @tparam T The type.
   */
  template<typename T>
  struct IsCTransformParameters
    : std::bool_constant<std::is_same_v<cxx::remove_cvref_t<T>, afft_dft_Parameters> ||
                         std::is_same_v<cxx::remove_cvref_t<T>, afft_dht_Parameters> ||
                         std::is_same_v<cxx::remove_cvref_t<T>, afft_dtt_Parameters>> {};

  /**
   * @brief Check if the type is C TransformParameters.
   * @tparam T The type.
   */
  template<typename T>
  inline constexpr bool isCTransformParameters = IsCTransformParameters<T>::value;

  /**
   * @brief MpBackendParameters type for given backend.
   * @tparam mpBackend The backend type.
   */
  template<MpBackend mpBackend>
  struct MpBackendParametersSelect;

  /// @brief Specialization for none backend.
  template<>
  struct MpBackendParametersSelect<MpBackend::none>
  {
    using CType   = void;
    using CxxType = afft::SingleProcessParameters;
  };

  /// @brief Specialization for mpi backend.
  template<>
  struct MpBackendParametersSelect<MpBackend::mpi>
  {
    using CType   = afft_mpi_Parameters;
    using CxxType = afft::mpi::Parameters;
  };

  /**
   * @brief Check if the type is C++ MpBackendParameters.
   * @tparam T The type.
   */
  template<typename T>
  struct IsCxxMpBackendParameters
    : std::bool_constant<std::is_same_v<cxx::remove_cvref_t<T>, afft::SingleProcessParameters> ||
                         std::is_same_v<cxx::remove_cvref_t<T>, afft::mpi::Parameters>> {};

  /**
   * @brief Check if the type is Cxx MpBackendParameters.
   * @tparam T The type.
   */
  template<typename T>
  inline constexpr bool isCxxMpBackendParameters = IsCxxMpBackendParameters<T>::value;

  /**
   * @brief Check if the type is C MpBackendParameters.
   * @tparam T The type.
   */
  template<typename T>
  struct IsCMpBackendParameters
    : std::bool_constant<std::is_same_v<cxx::remove_cvref_t<T>, afft_mpi_Parameters>> {};

  /**
   * @brief Check if the type is C MpBackendParameters.
   * @tparam T The type.
   */
  template<typename T>
  inline constexpr bool isCMpBackendParameters = IsCMpBackendParameters<T>::value;

  /**
   * @brief TargetParameters type for given target.
   * @tparam target The target type.
   */
  template<Target target>
  struct TargetParametersSelect;

  /// @brief Specialization for cpu target.
  template<>
  struct TargetParametersSelect<Target::cpu>
  {
    using CType   = afft_cpu_Parameters;
    using CxxType = afft::cpu::Parameters;
  };

  /// @brief Specialization for CUDA target.
  template<>
  struct TargetParametersSelect<Target::cuda>
  {
    using CType   = afft_cuda_Parameters;
    using CxxType = afft::cuda::Parameters;
  };

  /// @brief Specialization for HIP target.
  template<>
  struct TargetParametersSelect<Target::hip>
  {
    using CType   = afft_hip_Parameters;
    using CxxType = afft::hip::Parameters;
  };

  /// @brief Specialization for OpenCL target.
  template<>
  struct TargetParametersSelect<Target::opencl>
  {
    using CType   = afft_opencl_Parameters;
    using CxxType = afft::opencl::Parameters;
  };

  /// @brief Specialization for OpenMP target.
  template<>
  struct TargetParametersSelect<Target::openmp>
  {
    using CType   = afft_openmp_Parameters;
    using CxxType = afft::openmp::Parameters;
  };

  /**
   * @brief Check if the type is C++ TargetParameters.
   * @tparam T The type.
   */
  template<typename T>
  struct IsCxxTargetParameters
    : std::bool_constant<std::is_same_v<cxx::remove_cvref_t<T>, afft::cpu::Parameters> ||
                         std::is_same_v<cxx::remove_cvref_t<T>, afft::cuda::Parameters> ||
                         std::is_same_v<cxx::remove_cvref_t<T>, afft::hip::Parameters> ||
                         std::is_same_v<cxx::remove_cvref_t<T>, afft::opencl::Parameters> ||
                         std::is_same_v<cxx::remove_cvref_t<T>, afft::openmp::Parameters>> {};  

  /**
   * @brief Check if the type is C++ TargetParameters.
   * @tparam T The type.
   */
  template<typename T>
  inline constexpr bool isCxxTargetParameters = IsCxxTargetParameters<T>::value;

  /**
   * @brief Check if the type is C TargetParameters.
   * @tparam T The type.
   */
  template<typename T>
  struct IsCTargetParameters
    : std::bool_constant<std::is_same_v<cxx::remove_cvref_t<T>, afft_cpu_Parameters> ||
                         std::is_same_v<cxx::remove_cvref_t<T>, afft_cuda_Parameters> ||
                         std::is_same_v<cxx::remove_cvref_t<T>, afft_hip_Parameters> ||
                         std::is_same_v<cxx::remove_cvref_t<T>, afft_opencl_Parameters> ||
                         std::is_same_v<cxx::remove_cvref_t<T>, afft_openmp_Parameters>> {};

  /**
   * @brief Check if the type is C TargetParameters.
   * @tparam T The type.
   */
  template<typename T>
  inline constexpr bool isCTargetParameters = IsCTargetParameters<T>::value;

  /**
   * @brief MemoryLayoutParameters type for given memory layout.
   * @tparam memoryLayout The memory layout type.
   */
  template<MemoryLayout memoryLayout>
  struct MemoryLayoutParametersSelect;

  /// @brief Specialization for centralized memory layout.
  template<>
  struct MemoryLayoutParametersSelect<MemoryLayout::centralized>
  {
    using CType   = afft_CentralizedMemoryLayout;
    using CxxType = afft::CentralizedMemoryLayout;
  };

  /// @brief Specialization for distributed memory layout.
  template<>
  struct MemoryLayoutParametersSelect<MemoryLayout::distributed>
  {
    using CType   = afft_DistributedMemoryLayout;
    using CxxType = afft::DistributedMemoryLayout;
  };

  /**
   * @brief Check if the type is C++ MemoryLayoutParameters.
   * @tparam T The type.
   */
  template<typename T>
  struct IsCxxMemoryLayoutParameters
    : std::bool_constant<std::is_same_v<cxx::remove_cvref_t<T>, afft::CentralizedMemoryLayout> ||
                         std::is_same_v<cxx::remove_cvref_t<T>, afft::DistributedMemoryLayout>> {};

  /**
   * @brief Check if the type is C++ TransformParameters.
   * @tparam T The type.
   */
  template<typename T>
  inline constexpr bool isCxxMemoryLayoutParameters = IsCxxMemoryLayoutParameters<T>::value;

  /**
   * @brief Check if the type is C MemoryLayoutParameters.
   * @tparam T The type.
   */
  template<typename T>
  struct IsCMemoryLayoutParameters
    : std::bool_constant<std::is_same_v<cxx::remove_cvref_t<T>, afft_CentralizedMemoryLayout> ||
                         std::is_same_v<cxx::remove_cvref_t<T>, afft_DistributedMemoryLayout>> {};

  /**
   * @brief Check if the type is C MemoryLayoutParameters.
   * @tparam T The type.
   */
  template<typename T>
  inline constexpr bool isCMemoryLayoutParameters = IsCMemoryLayoutParameters<T>::value;

  /**
   * @brief BackendParameters type for given multi-process backend and target.
   * @tparam mpBackend The multi-process backend type.
   * @tparam target The target type.
   */
  template<MpBackend mpBackend, Target target>
  struct BackendParametersSelect;

  /// @brief Specialization for single-process cpu target.
  template<>
  struct BackendParametersSelect<MpBackend::none, Target::cpu>
  {
    using CType   = afft_cpu_BackendParameters;
    using CxxType = afft::cpu::BackendParameters;
  };

  /// @brief Specialization for single-process CUDA target.
  template<>
  struct BackendParametersSelect<MpBackend::none, Target::cuda>
  {
    using CType   = afft_cuda_BackendParameters;
    using CxxType = afft::cuda::BackendParameters;
  };

  /// @brief Specialization for single-process HIP target.
  template<>
  struct BackendParametersSelect<MpBackend::none, Target::hip>
  {
    using CType   = afft_hip_BackendParameters;
    using CxxType = afft::hip::BackendParameters;
  };

  /// @brief Specialization for single-process OpenCL target.
  template<>
  struct BackendParametersSelect<MpBackend::none, Target::opencl>
  {
    using CType   = afft_opencl_BackendParameters;
    using CxxType = afft::opencl::BackendParameters;
  };

  /// @brief Specialization for single-process OpenMP target.
  template<>
  struct BackendParametersSelect<MpBackend::none, Target::openmp>
  {
    using CType   = afft_openmp_BackendParameters;
    using CxxType = afft::openmp::BackendParameters;
  };

  /// @brief Specialization for MPI cpu target.
  template<>
  struct BackendParametersSelect<MpBackend::mpi, Target::cpu>
  {
    using CType   = afft_mpi_cpu_BackendParameters;
    using CxxType = afft::mpi::cpu::BackendParameters;
  };

  /// @brief Specialization for MPI CUDA target.
  template<>
  struct BackendParametersSelect<MpBackend::mpi, Target::cuda>
  {
    using CType   = afft_mpi_cuda_BackendParameters;
    using CxxType = afft::mpi::cuda::BackendParameters;
  };

  /// @brief Specialization for MPI HIP target.
  template<>
  struct BackendParametersSelect<MpBackend::mpi, Target::hip>
  {
    using CType   = afft_mpi_hip_BackendParameters;
    using CxxType = afft::mpi::hip::BackendParameters;
  };

  /**
   * @brief Check if the type is C++ BackendParameters.
   * @tparam T The type.
   */
  template<typename T>
  struct IsCxxBackendParameters
    : std::bool_constant<std::is_same_v<cxx::remove_cvref_t<T>, afft::cpu::BackendParameters> ||
                         std::is_same_v<cxx::remove_cvref_t<T>, afft::cuda::BackendParameters> ||
                         std::is_same_v<cxx::remove_cvref_t<T>, afft::hip::BackendParameters> ||
                         std::is_same_v<cxx::remove_cvref_t<T>, afft::opencl::BackendParameters> ||
                         std::is_same_v<cxx::remove_cvref_t<T>, afft::openmp::BackendParameters> ||
                         std::is_same_v<cxx::remove_cvref_t<T>, afft::mpi::cpu::BackendParameters> ||
                         std::is_same_v<cxx::remove_cvref_t<T>, afft::mpi::cuda::BackendParameters> ||
                         std::is_same_v<cxx::remove_cvref_t<T>, afft::mpi::hip::BackendParameters>> {};

  /**
   * @brief Check if the type is C BackendParameters.
   * @tparam T The type.
   */
  template<typename T>
  struct IsCBackendParameters
    : std::bool_constant<std::is_same_v<cxx::remove_cvref_t<T>, afft_cpu_BackendParameters> ||
                         std::is_same_v<cxx::remove_cvref_t<T>, afft_cuda_BackendParameters> ||
                         std::is_same_v<cxx::remove_cvref_t<T>, afft_hip_BackendParameters> ||
                         std::is_same_v<cxx::remove_cvref_t<T>, afft_opencl_BackendParameters> ||
                         std::is_same_v<cxx::remove_cvref_t<T>, afft_openmp_BackendParameters> ||
                         std::is_same_v<cxx::remove_cvref_t<T>, afft_mpi_cpu_BackendParameters> ||
                         std::is_same_v<cxx::remove_cvref_t<T>, afft_mpi_cuda_BackendParameters> ||
                         std::is_same_v<cxx::remove_cvref_t<T>, afft_mpi_hip_BackendParameters>> {};

  /**
   * @brief Check if the type is C++ BackendParameters.
   * @tparam T The type.
   */
  template<typename T>
  inline constexpr bool isCxxBackendParameters = IsCxxBackendParameters<T>::value;

  /**
   * @brief Check if the type is C BackendParameters.
   * @tparam T The type.
   */
  template<typename T>
  inline constexpr bool isCBackendParameters = IsCBackendParameters<T>::value;

  /**
   * @brief ExecutionParameters type for given architecture.
   * @tparam target The target type.
   */
  template<Target target>
  struct ExecutionParametersSelect;

  /// @brief Specialization for cpu target.
  template<>
  struct ExecutionParametersSelect<Target::cpu>
  {
    using CType   = afft_cpu_ExecutionParameters;
    using CxxType = afft::cpu::ExecutionParameters;
  };

  /// @brief Specialization for CUDA target.
  template<>
  struct ExecutionParametersSelect<Target::cuda>
  {
    using CType   = afft_cuda_ExecutionParameters;
    using CxxType = afft::cuda::ExecutionParameters;
  };

  /// @brief Specialization for HIP target.
  template<>
  struct ExecutionParametersSelect<Target::hip>
  {
    using CType   = afft_hip_ExecutionParameters;
    using CxxType = afft::hip::ExecutionParameters;
  };

  /// @brief Specialization for OpenCL target.
  template<>
  struct ExecutionParametersSelect<Target::opencl>
  {
    using CType   = afft_opencl_ExecutionParameters;
    using CxxType = afft::opencl::ExecutionParameters;
  };

  /// @brief Specialization for OpenMP target.
  template<>
  struct ExecutionParametersSelect<Target::openmp>
  {
    using CType   = afft_openmp_ExecutionParameters;
    using CxxType = afft::openmp::ExecutionParameters;
  };

  /**
   * @brief Check if the type is C++ ExecutionParameters.
   * @tparam T The type.
   */
  template<typename T>
  struct IsCxxExecutionParameters
    : std::bool_constant<std::is_same_v<cxx::remove_cvref_t<T>, afft::cpu::ExecutionParameters> ||
                         std::is_same_v<cxx::remove_cvref_t<T>, afft::cuda::ExecutionParameters> ||
                         std::is_same_v<cxx::remove_cvref_t<T>, afft::hip::ExecutionParameters> ||
                         std::is_same_v<cxx::remove_cvref_t<T>, afft::opencl::ExecutionParameters> ||
                         std::is_same_v<cxx::remove_cvref_t<T>, afft::openmp::ExecutionParameters>> {};

  /**
   * @brief Check if the type is C ExecutionParameters.
   * @tparam T The type.
   */
  template<typename T>
  struct IsCExecutionParameters
    : std::bool_constant<std::is_same_v<cxx::remove_cvref_t<T>, afft_cpu_ExecutionParameters> ||
                         std::is_same_v<cxx::remove_cvref_t<T>, afft_cuda_ExecutionParameters> ||
                         std::is_same_v<cxx::remove_cvref_t<T>, afft_hip_ExecutionParameters> ||
                         std::is_same_v<cxx::remove_cvref_t<T>, afft_opencl_ExecutionParameters> ||
                         std::is_same_v<cxx::remove_cvref_t<T>, afft_openmp_ExecutionParameters>> {};

  /**
   * @brief Check if the type is C++ ExecutionParameters.
   * @tparam T The type.
   */
  template<typename T>
  inline constexpr bool isCxxExecutionParameters = IsCxxExecutionParameters<T>::value;

  /**
   * @brief Check if the type is C ExecutionParameters.
   * @tparam T The type.
   */
  template<typename T>
  inline constexpr bool isCExecutionParameters = IsCExecutionParameters<T>::value;
} // namespace afft::detail

#endif /* AFFT_DETAIL_TYPE_TRAITS_HPP */
