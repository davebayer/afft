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

#ifndef AFFT_DETAIL_FFTW3_LIB_HPP
#define AFFT_DETAIL_FFTW3_LIB_HPP

#ifndef AFFT_TOP_LEVEL_INCLUDE
# include "../include.hpp"
#endif

#include "../common.hpp"

namespace afft::detail::fftw3
{
  /// @brief FFTW3 library for single precision.
  struct FloatLib
  {
# ifdef AFFT_FFTW3_HAS_FLOAT
    using Plan                                     = fftwf_plan;
    using R2RKind                                  = fftwf_r2r_kind;
    using Real                                     = float;
    using Complex                                  = fftwf_complex;
    using IoDim                                    = fftwf_iodim64;

    static constexpr auto initThreads              = fftwf_init_threads;
    static constexpr auto planWithNThreads         = fftwf_plan_with_nthreads;
    static constexpr auto importWisdomFromString   = fftwf_import_wisdom_from_string;

    static constexpr auto planGuruC2C              = fftwf_plan_guru64_dft;
    static constexpr auto planGuruR2C              = fftwf_plan_guru64_dft_r2c;
    static constexpr auto planGuruC2R              = fftwf_plan_guru64_dft_c2r;
    static constexpr auto planGuruR2R              = fftwf_plan_guru64_r2r;
    static constexpr auto planGuruSplitC2C         = fftwf_plan_guru64_split_dft;
    static constexpr auto planGuruSplitR2C         = fftwf_plan_guru64_split_dft_r2c;
    static constexpr auto planGuruSplitC2R         = fftwf_plan_guru64_split_dft_c2r;

    static constexpr auto executeC2C               = fftwf_execute_dft;
    static constexpr auto executeR2C               = fftwf_execute_dft_r2c;
    static constexpr auto executeC2R               = fftwf_execute_dft_c2r;
    static constexpr auto executeR2R               = fftwf_execute_r2r;
    static constexpr auto executeSplitC2C          = fftwf_execute_split_dft;
    static constexpr auto executeSplitR2C          = fftwf_execute_split_dft_r2c;
    static constexpr auto executeSplitC2R          = fftwf_execute_split_dft_c2r;

    static constexpr auto destroyPlan              = fftwf_destroy_plan;

    static constexpr auto cleanUpThreads           = fftwf_cleanup_threads;

    static constexpr auto exportWisdomToFilename   = fftwf_export_wisdom_to_filename;
    static constexpr auto exportWisdomToFile       = fftwf_export_wisdom_to_file;
    static constexpr auto exportWisdomToString     = fftwf_export_wisdom_to_string;
    static constexpr auto importSystemWisdom       = fftwf_import_system_wisdom;
    static constexpr auto importWisdomFromFilename = fftwf_import_wisdom_from_filename;
    static constexpr auto importWisdomFromFile     = fftwf_import_wisdom_from_file;
    static constexpr auto importWisdomFromString   = fftwf_import_wisdom_from_string;
    static constexpr auto forgetWisdom             = fftwf_forget_wisdom;
# endif
  };

  /// @brief FFTW3 library for double precision.
  struct DoubleLib
  {
# ifdef AFFT_FFTW3_HAS_DOUBLE
    using Plan                                     = fftw_plan;
    using R2RKind                                  = fftw_r2r_kind;
    using Real                                     = double;
    using Complex                                  = fftw_complex;
    using IoDim                                    = fftw_iodim64;

    static constexpr auto initThreads              = fftw_init_threads;
    static constexpr auto planWithNThreads         = fftw_plan_with_nthreads;
    static constexpr auto importWisdomFromString   = fftw_import_wisdom_from_string;

    static constexpr auto planGuruC2C              = fftw_plan_guru64_dft;
    static constexpr auto planGuruR2C              = fftw_plan_guru64_dft_r2c;
    static constexpr auto planGuruC2R              = fftw_plan_guru64_dft_c2r;
    static constexpr auto planGuruR2R              = fftw_plan_guru64_r2r;
    static constexpr auto planGuruSplitC2C         = fftw_plan_guru64_split_dft;
    static constexpr auto planGuruSplitR2C         = fftw_plan_guru64_split_dft_r2c;
    static constexpr auto planGuruSplitC2R         = fftw_plan_guru64_split_dft_c2r;

    static constexpr auto executeC2C               = fftw_execute_dft;
    static constexpr auto executeR2C               = fftw_execute_dft_r2c;
    static constexpr auto executeC2R               = fftw_execute_dft_c2r;
    static constexpr auto executeR2R               = fftw_execute_r2r;
    static constexpr auto executeSplitC2C          = fftw_execute_split_dft;
    static constexpr auto executeSplitR2C          = fftw_execute_split_dft_r2c;
    static constexpr auto executeSplitC2R          = fftw_execute_split_dft_c2r;

    static constexpr auto destroyPlan              = fftw_destroy_plan;

    static constexpr auto cleanUpThreads           = fftw_cleanup_threads;

    static constexpr auto exportWisdomToFilename   = fftw_export_wisdom_to_filename;
    static constexpr auto exportWisdomToFile       = fftw_export_wisdom_to_file;
    static constexpr auto exportWisdomToString     = fftw_export_wisdom_to_string;
    static constexpr auto importSystemWisdom       = fftw_import_system_wisdom;
    static constexpr auto importWisdomFromFilename = fftw_import_wisdom_from_filename;
    static constexpr auto importWisdomFromFile     = fftw_import_wisdom_from_file;
    static constexpr auto importWisdomFromString   = fftw_import_wisdom_from_string;
    static constexpr auto forgetWisdom             = fftw_forget_wisdom;
# endif
  };

  /// @brief FFTW3 library for long double precision.
  struct LongDoubleLib
  {
# ifdef AFFT_FFTW3_HAS_LONG
    using Plan                                     = fftwl_plan;
    using R2RKind                                  = fftwl_r2r_kind;
    using Real                                     = long double;
    using Complex                                  = fftwl_complex;
    using IoDim                                    = fftwl_iodim64;

    static constexpr auto initThreads              = fftwl_init_threads;
    static constexpr auto planWithNThreads         = fftwl_plan_with_nthreads;
    static constexpr auto importWisdomFromString   = fftwl_import_wisdom_from_string;

    static constexpr auto planGuruC2C              = fftwl_plan_guru64_dft;
    static constexpr auto planGuruR2C              = fftwl_plan_guru64_dft_r2c;
    static constexpr auto planGuruC2R              = fftwl_plan_guru64_dft_c2r;
    static constexpr auto planGuruR2R              = fftwl_plan_guru64_r2r;
    static constexpr auto planGuruSplitC2C         = fftwl_plan_guru64_split_dft;
    static constexpr auto planGuruSplitR2C         = fftwl_plan_guru64_split_dft_r2c;
    static constexpr auto planGuruSplitC2R         = fftwl_plan_guru64_split_dft_c2r;

    static constexpr auto executeC2C               = fftwl_execute_dft;
    static constexpr auto executeR2C               = fftwl_execute_dft_r2c;
    static constexpr auto executeC2R               = fftwl_execute_dft_c2r;
    static constexpr auto executeR2R               = fftwl_execute_r2r;
    static constexpr auto executeSplitC2C          = fftwl_execute_split_dft;
    static constexpr auto executeSplitR2C          = fftwl_execute_split_dft_r2c;
    static constexpr auto executeSplitC2R          = fftwl_execute_split_dft_c2r;

    static constexpr auto destroyPlan              = fftwl_destroy_plan;

    static constexpr auto cleanUpThreads           = fftwl_cleanup_threads;

    static constexpr auto exportWisdomToFilename   = fftwl_export_wisdom_to_filename;
    static constexpr auto exportWisdomToFile       = fftwl_export_wisdom_to_file;
    static constexpr auto exportWisdomToString     = fftwl_export_wisdom_to_string;
    static constexpr auto importSystemWisdom       = fftwl_import_system_wisdom;
    static constexpr auto importWisdomFromFilename = fftwl_import_wisdom_from_filename;
    static constexpr auto importWisdomFromFile     = fftwl_import_wisdom_from_file;
    static constexpr auto importWisdomFromString   = fftwl_import_wisdom_from_string;
    static constexpr auto forgetWisdom             = fftwl_forget_wisdom;
# endif
  };

  /// @brief FFTW3 library for quadruple precision.
  struct QuadLib
  {
# ifdef AFFT_FFTW3_HAS_QUAD
    using Plan                                     = fftwq_plan;
    using R2RKind                                  = fftwq_r2r_kind;
    using Real                                     = __float128;
    using Complex                                  = fftwq_complex;
    using IoDim                                    = fftwq_iodim64;

    static constexpr auto initThreads              = fftwq_init_threads;
    static constexpr auto planWithNThreads         = fftwq_plan_with_nthreads;
    static constexpr auto importWisdomFromString   = fftwq_import_wisdom_from_string;

    static constexpr auto planGuruC2C              = fftwq_plan_guru64_dft;
    static constexpr auto planGuruR2C              = fftwq_plan_guru64_dft_r2c;
    static constexpr auto planGuruC2R              = fftwq_plan_guru64_dft_c2r;
    static constexpr auto planGuruR2R              = fftwq_plan_guru64_r2r;
    static constexpr auto planGuruSplitC2C         = fftwq_plan_guru64_split_dft;
    static constexpr auto planGuruSplitR2C         = fftwq_plan_guru64_split_dft_r2c;
    static constexpr auto planGuruSplitC2R         = fftwq_plan_guru64_split_dft_c2r;

    static constexpr auto executeC2C               = fftwq_execute_dft;
    static constexpr auto executeR2C               = fftwq_execute_dft_r2c;
    static constexpr auto executeC2R               = fftwq_execute_dft_c2r;
    static constexpr auto executeR2R               = fftwq_execute_r2r;
    static constexpr auto executeSplitC2C          = fftwq_execute_split_dft;
    static constexpr auto executeSplitR2C          = fftwq_execute_split_dft_r2c;
    static constexpr auto executeSplitC2R          = fftwq_execute_split_dft_c2r;

    static constexpr auto destroyPlan              = fftwq_destroy_plan;

    static constexpr auto cleanUpThreads           = fftwq_cleanup_threads;

    static constexpr auto exportWisdomToFilename   = fftwq_export_wisdom_to_filename;
    static constexpr auto exportWisdomToFile       = fftwq_export_wisdom_to_file;
    static constexpr auto exportWisdomToString     = fftwq_export_wisdom_to_string;
    static constexpr auto importSystemWisdom       = fftwq_import_system_wisdom;
    static constexpr auto importWisdomFromFilename = fftwq_import_wisdom_from_filename;
    static constexpr auto importWisdomFromFile     = fftwq_import_wisdom_from_file;
    static constexpr auto importWisdomFromString   = fftwq_import_wisdom_from_string;
    static constexpr auto forgetWisdom             = fftwq_forget_wisdom;
# endif
  };

  struct MpiFloatLib
  {
# if AFFT_MP_BACKEND_IS(MPI) && defined(AFFT_FFTW3_HAS_MPI_FLOAT)
    using DDim                            = fftwf_mpi_ddim;

    static constexpr auto init            = fftwf_mpi_init;

    static constexpr auto planManyC2C     = fftwf_mpi_plan_many_dft;
    static constexpr auto planManyR2C     = fftwf_mpi_plan_many_dft_r2c;
    static constexpr auto planManyC2R     = fftwf_mpi_plan_many_dft_c2r;
    static constexpr auto planManyR2R     = fftwf_mpi_plan_many_r2r;

    static constexpr auto executeC2C      = fftwf_mpi_execute_dft;
    static constexpr auto executeR2C      = fftwf_mpi_execute_dft_r2c;
    static constexpr auto executeC2R      = fftwf_mpi_execute_dft_c2r;
    static constexpr auto executeR2R      = fftwf_mpi_execute_r2r;

    static constexpr auto cleanUp         = fftwf_mpi_cleanup;

    static constexpr auto broadcastWisdom = fftwf_mpi_broadcast_wisdom;
    static constexpr auto gatherWisdom    = fftwf_mpi_gather_wisdom;
# endif
  };

  struct MpiDoubleLib
  {
# if AFFT_MP_BACKEND_IS(MPI) && defined(AFFT_FFTW3_HAS_MPI_DOUBLE)
    using DDim                            = fftw_mpi_ddim;

    static constexpr auto init            = fftw_mpi_init;

    static constexpr auto planManyC2C     = fftw_mpi_plan_many_dft;
    static constexpr auto planManyR2C     = fftw_mpi_plan_many_dft_r2c;
    static constexpr auto planManyC2R     = fftw_mpi_plan_many_dft_c2r;
    static constexpr auto planManyR2R     = fftw_mpi_plan_many_r2r;

    static constexpr auto executeC2C      = fftw_mpi_execute_dft;
    static constexpr auto executeR2C      = fftw_mpi_execute_dft_r2c;
    static constexpr auto executeC2R      = fftw_mpi_execute_dft_c2r;
    static constexpr auto executeR2R      = fftw_mpi_execute_r2r;

    static constexpr auto cleanUp         = fftw_mpi_cleanup;

    static constexpr auto broadcastWisdom = fftw_mpi_broadcast_wisdom;
    static constexpr auto gatherWisdom    = fftw_mpi_gather_wisdom;
# endif
  };

  struct LongMpiLib
  {
# if AFFT_MP_BACKEND_IS(MPI) && defined(AFFT_FFTW3_HAS_MPI_LONG)
    using DDim                            = fftwl_mpi_ddim;

    static constexpr auto init            = fftwl_mpi_init;

    static constexpr auto planManyC2C     = fftwl_mpi_plan_many_dft;
    static constexpr auto planManyR2C     = fftwl_mpi_plan_many_dft_r2c;
    static constexpr auto planManyC2R     = fftwl_mpi_plan_many_dft_c2r;
    static constexpr auto planManyR2R     = fftwl_mpi_plan_many_r2r;

    static constexpr auto executeC2C      = fftwl_mpi_execute_dft;
    static constexpr auto executeR2C      = fftwl_mpi_execute_dft_r2c;
    static constexpr auto executeC2R      = fftwl_mpi_execute_dft_c2r;
    static constexpr auto executeR2R      = fftwl_mpi_execute_r2r;

    static constexpr auto cleanUp         = fftwl_mpi_cleanup;

    static constexpr auto broadcastWisdom = fftwl_mpi_broadcast_wisdom;
    static constexpr auto gatherWisdom    = fftwl_mpi_gather_wisdom;
# endif
  };

  /**
   * @brief Selects the FFTW3 library for the given precision.
   * @tparam prec Precision.
   */
  template<Precision prec>
  struct LibSelect
  {
    using Type = void;
  }

#ifdef AFFT_FFTW3_HAS_FLOAT
  /// @brief Specialization of LibSelect for Precision::f32.
  template<>
  struct LibSelect<Precision::f32>
  {
    using Type = FloatLib;
  };
#endif

  /// @brief Specialization of LibSelect for Precision::f64.
  template<>
  struct LibSelect<Precision::f64>
  {
#if defined(AFFT_FFTW3_HAS_DOUBLE)
    using Type = DoubleLib;
#elif defined(AFFT_FFTW3_HAS_LONG)
    using Type = std::conditional_t<(typePrecision<long double> == Precision::f64), LongDoubleLib, void>;
#else
    using Type = void;
#endif
  };

#ifdef AFFT_FFTW3_HAS_LONG
  /// @brief Specialization of LibSelect for Precision::f80.
  template<>
  struct LibSelect<Precision::f80>
  {
    using Type = std::conditional_t<(typePrecision<long double> == Precision::f80), LongDoubleLib, void>;
  };

  /// @brief Specialization of LibSelect for Precision::f64f64.
  template<>
  struct LibSelect<Precision::f64f64>
  {
    using Type = std::conditional_t<(typePrecision<long double> == Precision::f64f64), LongDoubleLib, void>;
  };
#endif

  /// @brief Specialization of LibSelect for Precision::f128. Prefer long double if possible.
  template<>
  struct LibSelect<Precision::f128>
  {
# ifdef AFFT_FFTW3_HAS_QUAD
#   ifdef AFFT_FFTW3_HAS_LONG
    using Type = std::conditional_t<(typePrecision<long double> == Precision::f128), LongDoubleLib, QuadLib>;
#   else
    using Type = QuadLib;
#   endif
# else
#   ifdef AFFT_FFTW3_HAS_LONG
    using Type = std::conditional_t<(typePrecision<long double> == Precision::f128), LongDoubleLib, void>;
#   else
    using Type = void;
#   endif
# endif
  };

  /**
   * @brief Selects the MPI FFTW3 library for the given precision.
   * @tparam prec Precision.
   */
  template<Precision prec>
  struct MpiLibSelect
  {
    using Type = void;
  };

  /// @brief Specialization of MpiLibSelect for Precision::f32.
#if AFFT_MP_BACKEND_IS(MPI) && defined(AFFT_FFTW3_HAS_MPI_FLOAT)
  template<>
  struct MpiLibSelect<Precision::f32>
  {
    using Type = MpiFloatLib;
  };
#endif

  /// @brief Specialization of MpiLibSelect for Precision::f64.
  template<>
  struct MpiLibSelect<Precision::f64>
  {
#if AFFT_MP_BACKEND_IS(MPI) && defined(AFFT_FFTW3_HAS_MPI_DOUBLE)
    using Type = MpiDoubleLib;
#elif AFFT_MP_BACKEND_IS(MPI) && defined(AFFT_FFTW3_HAS_MPI_LONG)
    using Type = std::conditional_t<(typePrecision<long double> == Precision::f64), LongMpiLib, void>;
#else
    using Type = void;
#endif
  };

#if AFFT_MP_BACKEND_IS(MPI) && defined(AFFT_FFTW3_HAS_MPI_LONG)
  /// @brief Specialization of MpiLibSelect for Precision::f80.
  template<>
  struct MpiLibSelect<Precision::f80>
  {
    using Type = std::conditional_t<(typePrecision<long double> == Precision::f80), LongMpiLib, void>;
  };

  /// @brief Specialization of MpiLibSelect for Precision::f64f64.
  template<>
  struct MpiLibSelect<Precision::f64f64>
  {
    using Type = std::conditional_t<(typePrecision<long double> == Precision::f64f64), LongMpiLib, void>;
  };

  /// @brief Specialization of MpiLibSelect for Precision::f128.
  template<>
  struct MpiLibSelect<Precision::f128>
  {
    using Type = std::conditional_t<(typePrecision<long double> == Precision::f128), LongMpiLib, void>;
  };
#endif

  /**
   * @brief Determines if the FFTW3 library is available for the given precision.
   * @tparam prec Precision.
   */
  template<Precision prec>
  inline constexpr bool hasPrecision = !std::is_void_v<typename LibSelect<prec>::Type>;

  /**
   * @brief Determines if the MPI FFTW3 library is available for the given precision.
   * @tparam prec Precision.
   */
  template<Precision prec>
  inline constexpr bool hasMpiPrecision = !std::is_void_v<typename MpiLibSelect<prec>::Type>;

  /**
   * @brief FFTW3 library type alias for the given precision.
   * @tparam prec Precision.
   */
  template<Precision prec>
  using Lib = std::enable_if_t<hasPrecision<prec>, typename LibSelect<prec>::Type>;

  /**
   * @brief MPI FFTW3 library type alias for the given precision.
   * @tparam prec Precision.
   */
  template<Precision prec>
  using MpiLib = std::enable_if_t<hasMpiPrecision<prec>, typename MpiLibSelect<prec>::Type>;
} // namespace afft::detail::fftw3

#endif /* AFFT_DETAIL_FFTW3_LIB_HPP */
