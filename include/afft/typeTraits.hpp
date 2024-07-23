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

#ifndef AFFT_TYPE_TRAITS_HPP
#define AFFT_TYPE_TRAITS_HPP

#ifndef AFFT_TOP_LEVEL_INCLUDE
# include "detail/include.hpp"
#endif

#include "detail/cxx.hpp"
#include "detail/typeTraits.hpp"

AFFT_EXPORT namespace afft
{
  /**
   * @brief Get the precision of the type. There has to be a specialization of TypeProperties for the type.
   * @tparam T The type.
   */
  template<typename T>
  inline constexpr Precision typePrecision = detail::TypePropertiesHelper<std::remove_cv_t<T>>::precision;

  /**
   * @brief Get the complexity of the type. There has to be a specialization of TypeProperties for the type.
   * @tparam T The type.
   */
  template<typename T>
  inline constexpr Complexity typeComplexity = detail::TypePropertiesHelper<std::remove_cv_t<T>>::complexity;

  /**
   * @brief Check if the type is a known type. There has to be a specialization of TypeProperties for the type.
   * @tparam T The type.
   */
  template<typename T>
  inline constexpr bool isKnownType = std::is_base_of_v<KnownTypePropertiesBase, detail::TypePropertiesHelper<std::remove_cv_t<T>>>;

  /**
   * @brief Check if the type is a known real type.
   * @tparam T The type.
   */
  template<typename T>
  inline constexpr bool isRealType = isKnownType<T> && typeComplexity<T> == Complexity::real;

  /**
   * @brief TransformParameters type for given transform.
   * @tparam transform The transform type.
   */
  template<Transform transform>
  using TransformParameters = typename detail::TransformParametersSelect<transform>::CxxType;

  /**
   * @brief Check if the type is TransformParameters.
   * @tparam T The type.
   */
  template<typename T>
  inline constexpr bool isTransformParameters = detail::IsCxxTransformParameters<detail::cxx::remove_cvref_t<T>>::value;

  /**
   * @brief MpBackendParameters type for given backend.
   * @tparam mpBackend The backend type.
   */
  template<MpBackend mpBackend>
  using MpBackendParameters = typename detail::MpBackendParametersSelect<mpBackend>::CxxType;

  /**
   * @brief Check if the type is MpBackendParameters.
   * @tparam T The type.
   */
  template<typename T>
  inline constexpr bool isMpBackendParameters = detail::IsCxxMpBackendParameters<detail::cxx::remove_cvref_t<T>>::value;

  /**
   * @brief TargetParameters type for given architecture.
   * @tparam target The target type.
   */
  template<Target target>
  using TargetParameters = typename detail::TargetParametersSelect<target>::CxxType;

  /**
   * @brief Check if the type is TargetParameters.
   * @tparam T The type.
   */
  template<typename T>
  inline constexpr bool isTargetParameters = detail::IsCxxTargetParameters<detail::cxx::remove_cvref_t<T>>::value;

  /**
   * @brief Memory layout type for given memory layout.
   * @tparam memoryLayout The memory layout type.
   */
  template<MemoryLayout memoryLayout>
  using MemoryLayoutParameters = typename detail::MemoryLayoutParametersSelect<memoryLayout>::CxxType;

  /**
   * @brief Check if the type is memory layout.
   * @tparam T The type.
   */
  template<typename T>
  inline constexpr bool isMemoryLayout = detail::IsCxxMemoryLayoutParameters<detail::cxx::remove_cvref_t<T>>::value;

  /**
   * @brief Backend Parameters type for given architecture.
   * @tparam mpBackend The backend type.
   * @tparam target The target type.
   */
  template<MpBackend mpBackend, Target target>
  using BackendParameters = typename detail::BackendParametersSelect<mpBackend, target>::CxxType;

  /**
   * @brief Is the type BackendParameters.
   * @tparam T The type.
   */
  template<typename T>
  inline constexpr bool isBackendParameters = detail::IsCxxBackendParameters<detail::cxx::remove_cvref_t<T>>::value;

  /**
   * @brief ExecutionParameters type for given architecture.
   * @tparam target The target type.
   * @tparam distrib The distribution type.
   */
  template<Target target>
  using ExecutionParameters = typename detail::ExecutionParametersSelect<target>::CxxType;

  /**
   * @brief Check if the type is ExecutionParameters.
   * @tparam T The type.
   */
  template<typename T>
  inline constexpr bool isExecutionParameters = detail::IsCxxExecutionParameters<detail::cxx::remove_cvref_t<T>>::value;

  namespace c
  {
    /**
     * @brief TransformParameters type for given transform.
     * @tparam transform The transform type.
     */
    template<afft::Transform transform>
    using TransformParameters = typename detail::TransformParametersSelect<transform>::CType;

    /**
     * @brief Check if the type is TransformParameters.
     * @tparam T The type.
     */
    template<typename T>
    inline constexpr bool isTransformParameters = detail::IsCTransformParameters<detail::cxx::remove_cvref_t<T>>::value;

    /**
     * @brief MpBackendParameters type for given backend.
     * @tparam mpBackend The backend type.
     */
    template<afft::MpBackend mpBackend>
    using MpBackendParameters = typename detail::MpBackendParametersSelect<mpBackend>::CType;

    /**
     * @brief Check if the type is MpBackendParameters.
     * @tparam T The type.
     */
    template<typename T>
    inline constexpr bool isMpBackendParameters = detail::IsCMpBackendParameters<detail::cxx::remove_cvref_t<T>>::value;

    /**
     * @brief TargetParameters type for given architecture.
     * @tparam target The target type.
     */
    template<afft::Target target>
    using TargetParameters = typename detail::TargetParametersSelect<target>::CType;

    /**
     * @brief Check if the type is TargetParameters.
     * @tparam T The type.
     */
    template<typename T>
    inline constexpr bool isTargetParameters = detail::IsCTargetParameters<detail::cxx::remove_cvref_t<T>>::value;

    /**
     * @brief Memory layout type for given memory layout.
     * @tparam memoryLayout The memory layout type.
     */
    template<afft::MemoryLayout memoryLayout>
    using MemoryLayoutParameters = typename detail::MemoryLayoutParametersSelect<memoryLayout>::CType;

    /**
     * @brief Check if the type is memory layout.
     * @tparam T The type.
     */
    template<typename T>
    inline constexpr bool isMemoryLayout = detail::IsCMemoryLayoutParameters<detail::cxx::remove_cvref_t<T>>::value;

    /**
     * @brief Backend Parameters type for given architecture.
     * @tparam mpBackend The backend type.
     * @tparam target The target type.
     */
    template<afft::MpBackend mpBackend, afft::Target target>
    using BackendParameters = typename detail::BackendParametersSelect<mpBackend, target>::CType;

    /**
     * @brief Is the type BackendParameters.
     * @tparam T The type.
     */
    template<typename T>
    inline constexpr bool isBackendParameters = detail::IsCBackendParameters<detail::cxx::remove_cvref_t<T>>::value;

    /**
     * @brief ExecutionParameters type for given architecture.
     * @tparam target The target type.
     * @tparam distrib The distribution type.
     */
    template<afft::Target target>
    using ExecutionParameters = typename detail::ExecutionParametersSelect<target>::CType;

    /**
     * @brief Check if the type is ExecutionParameters.
     * @tparam T The type.
     */
    template<typename T>
    inline constexpr bool isExecutionParameters = detail::IsCExecutionParameters<detail::cxx::remove_cvref_t<T>>::value;
  } // namespace c
} // namespace afft

#endif /* AFFT_TYPE_TRAITS_HPP */
