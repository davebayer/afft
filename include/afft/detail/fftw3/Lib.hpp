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

#include <fftw3.h>
#if AFFT_MP_BACKEND_IS(MPI)
# include <fftw3-mpi.h>
#endif

#include "../common.hpp"

namespace afft::detail::fftw3
{
  /**
   * @brief FFTW3 library precision-specific types and functions.
   * @tparam prec Precision.
   */
  template<Precision prec>
  struct Lib;

  /// @brief Library specialization for single precision.
  template<>
  struct Lib<Precision::f32>
  {
    using Plan                                     = fftwf_plan;
    using R2RKind                                  = fftwf_r2r_kind;
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

# if AFFT_MP_BACKEND_IS(MPI)
    using MpiDDim                                  = fftwf_mpi_ddim;

    static constexpr auto mpiInit                  = fftwf_mpi_init;

    static constexpr auto mpiPlanManyC2C           = fftwf_mpi_plan_many_dft;
    static constexpr auto mpiPlanManyR2C           = fftwf_mpi_plan_many_dft_r2c;
    static constexpr auto mpiPlanManyC2R           = fftwf_mpi_plan_many_dft_c2r;
    static constexpr auto mpiPlanManyR2R           = fftwf_mpi_plan_many_r2r;

    static constexpr auto mpiExecuteC2C            = fftwf_mpi_execute_dft;
    static constexpr auto mpiExecuteR2C            = fftwf_mpi_execute_dft_r2c;
    static constexpr auto mpiExecuteC2R            = fftwf_mpi_execute_dft_c2r;
    static constexpr auto mpiExecuteR2R            = fftwf_mpi_execute_r2r;

    static constexpr auto mpiCleanUp               = fftwf_mpi_cleanup;
# endif
  };

  /// @brief Library specialization for double precision.
  template<>
  struct Lib<Precision::f64>
  {
    using Plan                                     = fftw_plan;
    using R2RKind                                  = fftw_r2r_kind;
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

# if AFFT_MP_BACKEND_IS(MPI)
    using MpiDDim                                  = fftw_mpi_ddim;

    static constexpr auto mpiInit                  = fftw_mpi_init;

    static constexpr auto mpiPlanManyC2C           = fftw_mpi_plan_many_dft;
    static constexpr auto mpiPlanManyR2C           = fftw_mpi_plan_many_dft_r2c;
    static constexpr auto mpiPlanManyC2R           = fftw_mpi_plan_many_dft_c2r;
    static constexpr auto mpiPlanManyR2R           = fftw_mpi_plan_many_r2r;

    static constexpr auto mpiExecuteC2C            = fftw_mpi_execute_dft;
    static constexpr auto mpiExecuteR2C            = fftw_mpi_execute_dft_r2c;
    static constexpr auto mpiExecuteC2R            = fftw_mpi_execute_dft_c2r;
    static constexpr auto mpiExecuteR2R            = fftw_mpi_execute_r2r;

    static constexpr auto mpiCleanUp               = fftw_mpi_cleanup;
# endif
  };

#if defined(AFFT_HAS_F80) && defined(AFFT_CPU_FFTW3_LONG_FOUND)
  /// @brief Library specialization for long float precision.
  template<>
  struct Lib<Precision::f80>
  {
    using Plan                                     = fftwl_plan;
    using R2RKind                                  = fftwl_r2r_kind;
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

# if AFFT_MP_BACKEND_IS(MPI)
    using MpiDDim                                  = fftwl_mpi_ddim;

    static constexpr auto mpiInit                  = fftwl_mpi_init;

    static constexpr auto mpiPlanManyC2C           = fftwl_mpi_plan_many_dft;
    static constexpr auto mpiPlanManyR2C           = fftwl_mpi_plan_many_dft_r2c;
    static constexpr auto mpiPlanManyC2R           = fftwl_mpi_plan_many_dft_c2r;
    static constexpr auto mpiPlanManyR2R           = fftwl_mpi_plan_many_r2r;

    static constexpr auto mpiExecuteC2C            = fftwl_mpi_execute_dft;
    static constexpr auto mpiExecuteR2C            = fftwl_mpi_execute_dft_r2c;
    static constexpr auto mpiExecuteC2R            = fftwl_mpi_execute_dft_c2r;
    static constexpr auto mpiExecuteR2R            = fftwl_mpi_execute_r2r;

    static constexpr auto mpiCleanUp               = fftwl_mpi_cleanup;
# endif
  };
#endif

#if defined(AFFT_HAS_F128) && defined(AFFT_CPU_FFTW3_QUAD_FOUND)
  /// @brief Library specialization for quad precision.
  template<>
  struct Lib<Precision::f128>
  {
    using Plan                                     = fftwq_plan;
    using R2RKind                                  = fftwq_r2r_kind;
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
  };
#endif
} // namespace afft::detail::fftw3

#endif /* AFFT_DETAIL_FFTW3_LIB_HPP */
