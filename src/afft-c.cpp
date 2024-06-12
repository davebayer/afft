#include <afft/afft.h>
#include <afft/afft.hpp>

#define checkPlan(plan) do { \
    if (plan == nullptr) \
    { \
      return afft_Error_invalidPlan; \
    } \
  } while (0)


template<typename CE, typename CxxE>
[[nodiscard]] constexpr bool isSameEnumValue(const CE cValue, const CxxE cxxValue) noexcept
{
  static_assert(std::is_enum_v<CE>, "CE must be an enum type");
  static_assert(std::is_enum_v<CxxE>, "CxxT must be an enum type");

  return cValue == afft::detail::cxx::to_underlying(cxxValue);
}

template<typename CT, typename CxxE, afft_Error _error>
struct EnumMappingBase
{
  static_assert(std::is_enum_v<CxxE>, "E must be an enum type");
  static_assert(std::is_same_v<CT, std::underlying_type_t<CxxE>>, "CT must be the underlying type of E");
  
  using CxxType = CxxE;
  using CType   = CT;

  static constexpr afft_Error error = _error;
};

template<typename CxxE>
struct EnumMapping;

template<typename CxxE>
constexpr auto convertFromC(const typename EnumMapping<CxxE>::CType cValue)
  -> AFFT_RET_REQUIRES(typename EnumMapping<CxxE>::CxxType, std::is_enum_v<CxxE>)
{
  const auto cxxValue = static_cast<typename EnumMapping<CxxE>::CxxType>(cValue);

  if constexpr (EnumMapping<CxxE>::error != afft_Error_success)
  {
    if (!afft::detail::isValid(cxxValue))
    {
      throw EnumMapping<CxxE>::error;
    }
  }

  return cxxValue;
}

template<typename CxxE>
constexpr auto convertToC(const CxxE cxxValue)
  -> AFFT_RET_REQUIRES(typename EnumMapping<CxxE>::CType, std::is_enum_v<CxxE>)
{
  if constexpr (EnumMapping<CxxE>::error != afft_Error_success)
  {
    if (!afft::detail::isValid(cxxValue))
    {
      throw EnumMapping<CxxE>::error;
    }
  }

  return static_cast<typename EnumMapping<CxxE>::CType>(cxxValue);
}

/**********************************************************************************************************************/
// Common
/**********************************************************************************************************************/
// Precision
template<>
struct EnumMapping<afft::Precision> : EnumMappingBase<afft_Precision, afft::Precision, afft_Error_invalidPrecision>
{
  static_assert(isSameEnumValue(afft_Precision_bf16,         afft::Precision::bf16));
  static_assert(isSameEnumValue(afft_Precision_f16,          afft::Precision::f16));
  static_assert(isSameEnumValue(afft_Precision_f32,          afft::Precision::f32));
  static_assert(isSameEnumValue(afft_Precision_f64,          afft::Precision::f64));
  static_assert(isSameEnumValue(afft_Precision_f80,          afft::Precision::f80));
  static_assert(isSameEnumValue(afft_Precision_f64f64,       afft::Precision::f64f64));
  static_assert(isSameEnumValue(afft_Precision_f128,         afft::Precision::f128));

  static_assert(isSameEnumValue(afft_Precision_float,        afft::Precision::_float));
  static_assert(isSameEnumValue(afft_Precision_double,       afft::Precision::_double));
  static_assert(isSameEnumValue(afft_Precision_longDouble,   afft::Precision::_longDouble));
  static_assert(isSameEnumValue(afft_Precision_doubleDouble, afft::Precision::_doubleDouble));
  static_assert(isSameEnumValue(afft_Precision_quad,         afft::Precision::_quad));
};

// Alignment
template<>
struct EnumMapping<afft::Alignment> : EnumMappingBase<afft_Alignment, afft::Alignment, afft_Error_invalidAlignment>
{
  static_assert(isSameEnumValue(afft_Alignment_simd128,  afft::Alignment::simd128));
  static_assert(isSameEnumValue(afft_Alignment_simd256,  afft::Alignment::simd256));
  static_assert(isSameEnumValue(afft_Alignment_simd512,  afft::Alignment::simd512));
  static_assert(isSameEnumValue(afft_Alignment_simd1024, afft::Alignment::simd1024));
  static_assert(isSameEnumValue(afft_Alignment_simd2048, afft::Alignment::simd2048));

  static_assert(isSameEnumValue(afft_Alignment_sse,      afft::Alignment::sse));
  static_assert(isSameEnumValue(afft_Alignment_sse2,     afft::Alignment::sse2));
  static_assert(isSameEnumValue(afft_Alignment_sse3,     afft::Alignment::sse3));
  static_assert(isSameEnumValue(afft_Alignment_sse4,     afft::Alignment::sse4));
  static_assert(isSameEnumValue(afft_Alignment_sse4_1,   afft::Alignment::sse4_1));
  static_assert(isSameEnumValue(afft_Alignment_sse4_2,   afft::Alignment::sse4_2));
  static_assert(isSameEnumValue(afft_Alignment_avx,      afft::Alignment::avx));
  static_assert(isSameEnumValue(afft_Alignment_avx2,     afft::Alignment::avx2));
  static_assert(isSameEnumValue(afft_Alignment_avx512,   afft::Alignment::avx512));
  static_assert(isSameEnumValue(afft_Alignment_neon,     afft::Alignment::neon));
  static_assert(isSameEnumValue(afft_Alignment_sve,      afft::Alignment::sve));
};

// Complexity
template<>
struct EnumMapping<afft::Complexity> : EnumMappingBase<afft_Complexity, afft::Complexity, afft_Error_invalidComplexity>
{
  static_assert(isSameEnumValue(afft_Complexity_real,    afft::Complexity::real));
  static_assert(isSameEnumValue(afft_Complexity_complex, afft::Complexity::complex));
};

// ComplexFormat
template<>
struct EnumMapping<afft::ComplexFormat> : EnumMappingBase<afft_ComplexFormat, afft::ComplexFormat, afft_Error_invalidComplexFormat>
{
  static_assert(isSameEnumValue(afft_ComplexFormat_interleaved, afft::ComplexFormat::interleaved));
  static_assert(isSameEnumValue(afft_ComplexFormat_planar,      afft::ComplexFormat::planar));
};

// Direction
template<>
struct EnumMapping<afft::Direction> : EnumMappingBase<afft_Direction, afft::Direction, afft_Error_invalidDirection>
{
  static_assert(isSameEnumValue(afft_Direction_forward,  afft::Direction::forward));
  static_assert(isSameEnumValue(afft_Direction_inverse, afft::Direction::inverse));
};

// Placement
template<>
struct EnumMapping<afft::Placement> : EnumMappingBase<afft_Placement, afft::Placement, afft_Error_invalidPlacement>
{
  static_assert(isSameEnumValue(afft_Placement_inPlace,    afft::Placement::inPlace));
  static_assert(isSameEnumValue(afft_Placement_outOfPlace, afft::Placement::outOfPlace));
};

// Transform
template<>
struct EnumMapping<afft::Transform> : EnumMappingBase<afft_Transform, afft::Transform, afft_Error_invalidTransform>
{
  static_assert(isSameEnumValue(afft_Transform_dft, afft::Transform::dft));
  static_assert(isSameEnumValue(afft_Transform_dht, afft::Transform::dht));
  static_assert(isSameEnumValue(afft_Transform_dtt, afft::Transform::dtt));
};

// Target
template<>
struct EnumMapping<afft::Target> : EnumMappingBase<afft_Target, afft::Target, afft_Error_invalidTarget>
{
  static_assert(isSameEnumValue(afft_Target_cpu, afft::Target::cpu));
  static_assert(isSameEnumValue(afft_Target_gpu, afft::Target::gpu));
};

// Distribution
template<>
struct EnumMapping<afft::Distribution> : EnumMappingBase<afft_Distribution, afft::Distribution, afft_Error_invalidDistribution>
{
  static_assert(isSameEnumValue(afft_Distribution_spst, afft::Distribution::spst));
  static_assert(isSameEnumValue(afft_Distribution_spmt, afft::Distribution::spmt));
  static_assert(isSameEnumValue(afft_Distribution_mpst, afft::Distribution::mpst));
};

// Normalization
template<>
struct EnumMapping<afft::Normalization> : EnumMappingBase<afft_Normalization, afft::Normalization, afft_Error_invalidNormalization>
{
  static_assert(isSameEnumValue(afft_Normalization_none,       afft::Normalization::none));
  static_assert(isSameEnumValue(afft_Normalization_orthogonal, afft::Normalization::orthogonal));
  static_assert(isSameEnumValue(afft_Normalization_unitary,    afft::Normalization::unitary));
};

static constexpr afft::PrecisionTriad convertFromC(const afft_PrecisionTriad& cPrec)
{
  afft::PrecisionTriad cxxPrec{};
  cxxPrec.execution   = convertFromC<afft::Precision>(cPrec.execution);
  cxxPrec.source      = convertFromC<afft::Precision>(cPrec.source);
  cxxPrec.destination = convertFromC<afft::Precision>(cPrec.destination);

  return cxxPrec;
}

static constexpr afft_PrecisionTriad convertToC(const afft::PrecisionTriad& cxxPrec)
{
  afft_PrecisionTriad cPrec{};
  cPrec.execution   = convertToC(cxxPrec.execution);
  cPrec.source      = convertToC(cxxPrec.source);
  cPrec.destination = convertToC(cxxPrec.destination);

  return cPrec;
}

/**********************************************************************************************************************/
// Backend
/**********************************************************************************************************************/
// Backend
template<>
struct EnumMapping<afft::Backend> : EnumMappingBase<afft_Backend, afft::Backend, afft_Error_invalidBackend>
{
  static_assert(isSameEnumValue(afft_Backend_clfft,     afft::Backend::clfft));
  static_assert(isSameEnumValue(afft_Backend_cufft,     afft::Backend::cufft));
  static_assert(isSameEnumValue(afft_Backend_fftw3,     afft::Backend::fftw3));
  static_assert(isSameEnumValue(afft_Backend_heffte,    afft::Backend::heffte));
  static_assert(isSameEnumValue(afft_Backend_hipfft,    afft::Backend::hipfft));
  static_assert(isSameEnumValue(afft_Backend_mkl,       afft::Backend::mkl));
  static_assert(isSameEnumValue(afft_Backend_pocketfft, afft::Backend::pocketfft));
  static_assert(isSameEnumValue(afft_Backend_rocfft,    afft::Backend::rocfft));
  static_assert(isSameEnumValue(afft_Backend_vkfft,     afft::Backend::vkfft));
};

// BackendMask
template<>
struct EnumMapping<afft::BackendMask> : EnumMappingBase<afft_BackendMask, afft::BackendMask, afft_Error_success> {};

// SelectStrategy
template<>
struct EnumMapping<afft::SelectStrategy> : EnumMappingBase<afft_SelectStrategy, afft::SelectStrategy, afft_Error_invalidSelectStrategy>
{
  static_assert(isSameEnumValue(afft_SelectStrategy_first, afft::SelectStrategy::first));
  static_assert(isSameEnumValue(afft_SelectStrategy_best,  afft::SelectStrategy::best));
};

static constexpr afft::spst::gpu::clfft::Parameters convertFromC(const afft_spst_gpu_clfft_Parameters& cParams) noexcept
{
  afft::spst::gpu::clfft::Parameters cxxParams{};
  cxxParams.useFastMath = cParams.useFastMath;

  return cxxParams;
}

static constexpr afft_spst_gpu_clfft_Parameters convertToC(const afft::spst::gpu::clfft::Parameters& cxxParams) noexcept
{
  afft_spst_gpu_clfft_Parameters cParams{};
  cParams.useFastMath = cxxParams.useFastMath;

  return cParams;
}

// cuFFT WorkspacePolicy
template<>
struct EnumMapping<afft::cufft::WorkspacePolicy>
  : EnumMappingBase<afft_cufft_WorkspacePolicy, afft::cufft::WorkspacePolicy, afft_Error_invalidCufftWorkspacePolicy>
{
  static_assert(isSameEnumValue(afft_cufft_WorkspacePolicy_performance, afft::cufft::WorkspacePolicy::performance));
  static_assert(isSameEnumValue(afft_cufft_WorkspacePolicy_minimal,    afft::cufft::WorkspacePolicy::minimal));
  static_assert(isSameEnumValue(afft_cufft_WorkspacePolicy_user,       afft::cufft::WorkspacePolicy::user));
};

static constexpr afft::spst::gpu::cufft::Parameters convertFromC(const afft_spst_gpu_cufft_Parameters& cParams)
{
  afft::spst::gpu::cufft::Parameters cxxParams{};
  cxxParams.workspacePolicy   = convertFromC<afft::cufft::WorkspacePolicy>(cParams.workspacePolicy);
  cxxParams.usePatientJit     = cParams.usePatientJit;
  cxxParams.userWorkspaceSize = cParams.userWorkspaceSize;

  return cxxParams;
}

static constexpr afft_spst_gpu_cufft_Parameters convertToC(const afft::spst::gpu::cufft::Parameters& cxxParams)
{
  afft_spst_gpu_cufft_Parameters cParams{};
  cParams.workspacePolicy   = convertToC(cxxParams.workspacePolicy);
  cParams.usePatientJit     = cxxParams.usePatientJit;
  cParams.userWorkspaceSize = cxxParams.userWorkspaceSize;

  return cParams;
}

static constexpr afft::spmt::gpu::cufft::Parameters convertFromC(const afft_spmt_gpu_cufft_Parameters& cParams) noexcept
{
  afft::spmt::gpu::cufft::Parameters cxxParams{};
  cxxParams.usePatientJit = cParams.usePatientJit;

  return cxxParams;
}

static constexpr afft_spmt_gpu_cufft_Parameters convertToC(const afft::spmt::gpu::cufft::Parameters& cxxParams) noexcept
{
  afft_spmt_gpu_cufft_Parameters cParams{};
  cParams.usePatientJit = cxxParams.usePatientJit;

  return cParams;
}

static constexpr afft::mpst::gpu::cufft::Parameters convertFromC(const afft_mpst_gpu_cufft_Parameters& cParams) noexcept
{
  afft::mpst::gpu::cufft::Parameters cxxParams{};
  cxxParams.usePatientJit = cParams.usePatientJit;

  return cxxParams;
}

static constexpr afft_mpst_gpu_cufft_Parameters convertToC(const afft::mpst::gpu::cufft::Parameters& cxxParams) noexcept
{
  afft_mpst_gpu_cufft_Parameters cParams{};
  cParams.usePatientJit = cxxParams.usePatientJit;

  return cParams;
}

// FFTW3 PlannerFlag
template<>
struct EnumMapping<afft::fftw3::PlannerFlag>
  : EnumMappingBase<afft_fftw3_PlannerFlag, afft::fftw3::PlannerFlag, afft_Error_invalidFftw3PlannerFlag>
{
  static_assert(isSameEnumValue(afft_fftw3_PlannerFlag_estimate,         afft::fftw3::PlannerFlag::estimate));
  static_assert(isSameEnumValue(afft_fftw3_PlannerFlag_measure,          afft::fftw3::PlannerFlag::measure));
  static_assert(isSameEnumValue(afft_fftw3_PlannerFlag_patient,          afft::fftw3::PlannerFlag::patient));
  static_assert(isSameEnumValue(afft_fftw3_PlannerFlag_exhaustive,       afft::fftw3::PlannerFlag::exhaustive));
  static_assert(isSameEnumValue(afft_fftw3_PlannerFlag_estimatePatient,  afft::fftw3::PlannerFlag::estimatePatient));
};

static constexpr afft::spst::cpu::fftw3::Parameters convertFromC(const afft_spst_cpu_fftw3_Parameters& cParams)
{
  afft::spst::cpu::fftw3::Parameters cxxParams{};
  cxxParams.plannerFlag       = convertFromC<afft::fftw3::PlannerFlag>(cParams.plannerFlag);
  cxxParams.conserveMemory    = cParams.conserveMemory;
  cxxParams.wisdomOnly        = cParams.wisdomOnly;
  cxxParams.allowLargeGeneric = cParams.allowLargeGeneric;
  cxxParams.allowPruning      = cParams.allowPruning;
  cxxParams.timeLimit         = std::chrono::duration<double>{cParams.timeLimit};

  return cxxParams;
}

static constexpr afft_spst_cpu_fftw3_Parameters convertToC(const afft::spst::cpu::fftw3::Parameters& cxxParams)
{
  afft_spst_cpu_fftw3_Parameters cParams{};
  cParams.plannerFlag       = convertToC(cxxParams.plannerFlag);
  cParams.conserveMemory    = cxxParams.conserveMemory;
  cParams.wisdomOnly        = cxxParams.wisdomOnly;
  cParams.allowLargeGeneric = cxxParams.allowLargeGeneric;
  cParams.allowPruning      = cxxParams.allowPruning;
  cParams.timeLimit         = cxxParams.timeLimit.count();

  return cParams;
}

static constexpr afft::mpst::cpu::fftw3::Parameters convertFromC(const afft_mpst_cpu_fftw3_Parameters& cParams)
{
  afft::mpst::cpu::fftw3::Parameters cxxParams{};
  cxxParams.plannerFlag       = convertFromC<afft::fftw3::PlannerFlag>(cParams.plannerFlag);
  cxxParams.conserveMemory    = cParams.conserveMemory;
  cxxParams.wisdomOnly        = cParams.wisdomOnly;
  cxxParams.allowLargeGeneric = cParams.allowLargeGeneric;
  cxxParams.allowPruning      = cParams.allowPruning;
  cxxParams.timeLimit         = std::chrono::duration<double>{cParams.timeLimit};
  cxxParams.blockSize         = cParams.blockSize;

  return cxxParams;
}

static constexpr afft_mpst_cpu_fftw3_Parameters convertToC(const afft::mpst::cpu::fftw3::Parameters& cxxParams)
{
  afft_mpst_cpu_fftw3_Parameters cParams{};
  cParams.plannerFlag       = convertToC(cxxParams.plannerFlag);
  cParams.conserveMemory    = cxxParams.conserveMemory;
  cParams.wisdomOnly        = cxxParams.wisdomOnly;
  cParams.allowLargeGeneric = cxxParams.allowLargeGeneric;
  cParams.allowPruning      = cxxParams.allowPruning;
  cParams.timeLimit         = cxxParams.timeLimit.count();
  cParams.blockSize         = cxxParams.blockSize;

  return cParams;
}

// HeFFTe CPU Backend
template<>
struct EnumMapping<afft::heffte::cpu::Backend>
  : EnumMappingBase<afft_heffte_cpu_Backend, afft::heffte::cpu::Backend, afft_Error_invalidHeffteCpuBackend>
{
  static_assert(isSameEnumValue(afft_heffte_cpu_Backend_fftw3, afft::heffte::cpu::Backend::fftw3));
  static_assert(isSameEnumValue(afft_heffte_cpu_Backend_mkl,   afft::heffte::cpu::Backend::mkl));
};

// HeFFTe GPU Backend
template<>
struct EnumMapping<afft::heffte::gpu::Backend>
  : EnumMappingBase<afft_heffte_gpu_Backend, afft::heffte::gpu::Backend, afft_Error_invalidHeffteGpuBackend>
{
  static_assert(isSameEnumValue(afft_heffte_gpu_Backend_cufft,  afft::heffte::gpu::Backend::cufft));
  static_assert(isSameEnumValue(afft_heffte_gpu_Backend_rocfft, afft::heffte::gpu::Backend::rocfft));
};

static constexpr afft::mpst::cpu::heffte::Parameters convertFromC(const afft_mpst_cpu_heffte_Parameters& cParams)
{
  afft::mpst::cpu::heffte::Parameters cxxParams{};
  cxxParams.backend     = convertFromC<afft::heffte::cpu::Backend>(cParams.backend);
  cxxParams.useReorder  = cParams.useReorder;
  cxxParams.useAllToAll = cParams.useAllToAll;
  cxxParams.usePencils  = cParams.usePencils;

  return cxxParams;
}

static constexpr afft_mpst_cpu_heffte_Parameters convertToC(const afft::mpst::cpu::heffte::Parameters& cxxParams)
{
  afft_mpst_cpu_heffte_Parameters cParams{};
  cParams.backend     = convertToC(cxxParams.backend);
  cParams.useReorder  = cxxParams.useReorder;
  cParams.useAllToAll = cxxParams.useAllToAll;
  cParams.usePencils  = cxxParams.usePencils;

  return cParams;
}

static constexpr afft::mpst::gpu::heffte::Parameters convertFromC(const afft_mpst_gpu_heffte_Parameters& cParams)
{
  afft::mpst::gpu::heffte::Parameters cxxParams{};
  cxxParams.backend     = convertFromC<afft::heffte::gpu::Backend>(cParams.backend);
  cxxParams.useReorder  = cParams.useReorder;
  cxxParams.useAllToAll = cParams.useAllToAll;
  cxxParams.usePencils  = cParams.usePencils;

  return cxxParams;
}

static constexpr afft_mpst_gpu_heffte_Parameters convertToC(const afft::mpst::gpu::heffte::Parameters& cxxParams)
{
  afft_mpst_gpu_heffte_Parameters cParams{};
  cParams.backend     = convertToC(cxxParams.backend);
  cParams.useReorder  = cxxParams.useReorder;
  cParams.useAllToAll = cxxParams.useAllToAll;
  cParams.usePencils  = cxxParams.usePencils;

  return cParams;
}

static afft::spst::cpu::BackendParameters convertFromC(const afft_spst_cpu_BackendParameters& cParams)
{
  afft::spst::cpu::BackendParameters cxxParams{};
  cxxParams.strategy = convertFromC<afft::SelectStrategy>(cParams.strategy);
  cxxParams.mask     = convertFromC<afft::BackendMask>(cParams.mask);
  cxxParams.order    = afft::View<afft::Backend>{reinterpret_cast<const afft::Backend*>(cParams.order), cParams.orderSize};
  cxxParams.fftw3    = convertFromC(cParams.fftw3);

  return cxxParams;
}

static afft_spst_cpu_BackendParameters convertToC(const afft::spst::cpu::BackendParameters& cxxParams)
{
  afft_spst_cpu_BackendParameters cParams{};
  cParams.strategy  = convertToC(cxxParams.strategy);
  cParams.mask      = convertToC(cxxParams.mask);
  cParams.orderSize = cxxParams.order.size();
  cParams.order     = reinterpret_cast<const afft_Backend*>(cxxParams.order.data());
  cParams.fftw3     = convertToC(cxxParams.fftw3);

  return cParams;
}

static afft::spst::gpu::BackendParameters convertFromC(const afft_spst_gpu_BackendParameters& cParams)
{
  afft::spst::gpu::BackendParameters cxxParams{};
  cxxParams.strategy = convertFromC<afft::SelectStrategy>(cParams.strategy);
  cxxParams.mask     = convertFromC<afft::BackendMask>(cParams.mask);
  cxxParams.order    = afft::View<afft::Backend>{reinterpret_cast<const afft::Backend*>(cParams.order), cParams.orderSize};
  cxxParams.clfft    = convertFromC(cParams.clfft);
  cxxParams.cufft    = convertFromC(cParams.cufft);

  return cxxParams;
}

static afft_spst_gpu_BackendParameters convertToC(const afft::spst::gpu::BackendParameters& cxxParams)
{
  afft_spst_gpu_BackendParameters cParams{};
  cParams.strategy  = convertToC(cxxParams.strategy);
  cParams.mask      = convertToC(cxxParams.mask);
  cParams.orderSize = cxxParams.order.size();
  cParams.order     = reinterpret_cast<const afft_Backend*>(cxxParams.order.data());
  cParams.clfft     = convertToC(cxxParams.clfft);
  cParams.cufft     = convertToC(cxxParams.cufft);

  return cParams;
}

static afft::spmt::gpu::BackendParameters convertFromC(const afft_spmt_gpu_BackendParameters& cParams)
{
  afft::spmt::gpu::BackendParameters cxxParams{};
  cxxParams.strategy = convertFromC<afft::SelectStrategy>(cParams.strategy);
  cxxParams.mask     = convertFromC<afft::BackendMask>(cParams.mask);
  cxxParams.order    = afft::View<afft::Backend>{reinterpret_cast<const afft::Backend*>(cParams.order), cParams.orderSize};
  cxxParams.cufft    = convertFromC(cParams.cufft);

  return cxxParams;
}

static afft_spmt_gpu_BackendParameters convertToC(const afft::spmt::gpu::BackendParameters& cxxParams)
{
  afft_spmt_gpu_BackendParameters cParams{};
  cParams.strategy  = convertToC(cxxParams.strategy);
  cParams.mask      = convertToC(cxxParams.mask);
  cParams.orderSize = cxxParams.order.size();
  cParams.order     = reinterpret_cast<const afft_Backend*>(cxxParams.order.data());
  cParams.cufft     = convertToC(cxxParams.cufft);

  return cParams;
}

static afft::mpst::cpu::BackendParameters convertFromC(const afft_mpst_cpu_BackendParameters& cParams)
{
  afft::mpst::cpu::BackendParameters cxxParams{};
  cxxParams.strategy = convertFromC<afft::SelectStrategy>(cParams.strategy);
  cxxParams.mask     = convertFromC<afft::BackendMask>(cParams.mask);
  cxxParams.order    = afft::View<afft::Backend>{reinterpret_cast<const afft::Backend*>(cParams.order), cParams.orderSize};
  cxxParams.fftw3    = convertFromC(cParams.fftw3);
  cxxParams.heffte   = convertFromC(cParams.heffte);

  return cxxParams;
}

static afft_mpst_cpu_BackendParameters convertToC(const afft::mpst::cpu::BackendParameters& cxxParams)
{
  afft_mpst_cpu_BackendParameters cParams{};
  cParams.strategy  = convertToC(cxxParams.strategy);
  cParams.mask      = convertToC(cxxParams.mask);
  cParams.orderSize = cxxParams.order.size();
  cParams.order     = reinterpret_cast<const afft_Backend*>(cxxParams.order.data());
  cParams.fftw3     = convertToC(cxxParams.fftw3);
  cParams.heffte    = convertToC(cxxParams.heffte);

  return cParams;
}

static afft::mpst::gpu::BackendParameters convertFromC(const afft_mpst_gpu_BackendParameters& cParams)
{
  afft::mpst::gpu::BackendParameters cxxParams{};
  cxxParams.strategy = convertFromC<afft::SelectStrategy>(cParams.strategy);
  cxxParams.mask     = convertFromC<afft::BackendMask>(cParams.mask);
  cxxParams.order    = afft::View<afft::Backend>{reinterpret_cast<const afft::Backend*>(cParams.order), cParams.orderSize};
  cxxParams.cufft    = convertFromC(cParams.cufft);
  cxxParams.heffte   = convertFromC(cParams.heffte);

  return cxxParams;
}

static afft_mpst_gpu_BackendParameters convertToC(const afft::mpst::gpu::BackendParameters& cxxParams)
{
  afft_mpst_gpu_BackendParameters cParams{};
  cParams.strategy  = convertToC(cxxParams.strategy);
  cParams.mask      = convertToC(cxxParams.mask);
  cParams.orderSize = cxxParams.order.size();
  cParams.order     = reinterpret_cast<const afft_Backend*>(cxxParams.order.data());
  cParams.cufft     = convertToC(cxxParams.cufft);
  cParams.heffte    = convertToC(cxxParams.heffte);

  return cParams;
}

// TODO:
/// @brief Feedback structure
// typedef struct
// {
//   afft_Backend backend;      ///< Backend
//   const char*  message;      ///< Message from the backend
//   double       measuredTime; ///< Measured time in seconds
// } afft_Feedback;

/**
 * @brief Get the name of the backend.
 * @param backend Backend.
 * @return Name of the backend.
 */
extern "C" const char* getBackendName(afft_Backend backend)
{
  return {};
}

/**********************************************************************************************************************/
// Distribution
/**********************************************************************************************************************/

static constexpr afft::MemoryBlock<> convertFromC(const afft_MemoryBlock cMemBlock, std::size_t shapeRank)
{
  afft::MemoryBlock<> cxxMemBlock{};
  if (cMemBlock.starts != nullptr)
  {
    cxxMemBlock.starts = afft::View<size_t>{cMemBlock.starts, shapeRank};
  }
  if (cMemBlock.sizes != nullptr)
  {
    cxxMemBlock.sizes = afft::View<size_t>{cMemBlock.sizes, shapeRank};
  }
  if (cMemBlock.strides != nullptr)
  {
    cxxMemBlock.strides = afft::View<size_t>{cMemBlock.strides, shapeRank};
  }

  return cxxMemBlock;
}

static constexpr afft_MemoryBlock convertToC(const afft::MemoryBlock<>& cxxMemBlock)
{
  afft_MemoryBlock cMemBlock{};
  cMemBlock.starts  = cxxMemBlock.starts.data();
  cMemBlock.sizes   = cxxMemBlock.sizes.data();
  cMemBlock.strides = cxxMemBlock.strides.data();

  return cMemBlock;
}

static constexpr afft::spst::MemoryLayout<> convertFromC(const afft_spst_MemoryLayout cMemLayout, std::size_t shapeRank)
{
  afft::spst::MemoryLayout<> cxxMemLayout{};
  if (cMemLayout.srcStrides != nullptr)
  {
    cxxMemLayout.srcStrides = afft::View<size_t>{cMemLayout.srcStrides, shapeRank};
  }
  if (cMemLayout.dstStrides != nullptr)
  {
    cxxMemLayout.dstStrides = afft::View<size_t>{cMemLayout.dstStrides, shapeRank};
  }

  return cxxMemLayout;
}

// TODO: mpst MemoryLayout

static constexpr afft_spst_MemoryLayout convertToC(const afft::spst::MemoryLayout<>& cxxMemLayout)
{
  afft_spst_MemoryLayout cMemLayout{};
  cMemLayout.srcStrides = cxxMemLayout.srcStrides.data();
  cMemLayout.dstStrides = cxxMemLayout.dstStrides.data();

  return cMemLayout;
}

static constexpr afft::mpst::MemoryLayout<> convertFromC(const afft_mpst_MemoryLayout cMemLayout, std::size_t shapeRank)
{
  afft::mpst::MemoryLayout<> cxxMemLayout{};
  cxxMemLayout.srcBlock = convertFromC(cMemLayout.srcBlock, shapeRank);
  cxxMemLayout.dstBlock = convertFromC(cMemLayout.dstBlock, shapeRank);
  if (cMemLayout.srcAxesOrder != nullptr)
  {
    cxxMemLayout.srcAxesOrder = afft::View<size_t>{cMemLayout.srcAxesOrder, shapeRank};
  }
  if (cMemLayout.dstAxesOrder != nullptr)
  {
    cxxMemLayout.dstAxesOrder = afft::View<size_t>{cMemLayout.dstAxesOrder, shapeRank};
  }

  return cxxMemLayout;
}

static constexpr afft_mpst_MemoryLayout convertToC(const afft::mpst::MemoryLayout<>& cxxMemLayout)
{
  afft_mpst_MemoryLayout cMemLayout{};
  cMemLayout.srcBlock     = convertToC(cxxMemLayout.srcBlock);
  cMemLayout.dstBlock     = convertToC(cxxMemLayout.dstBlock);
  cMemLayout.srcAxesOrder = cxxMemLayout.srcAxesOrder.data();
  cMemLayout.dstAxesOrder = cxxMemLayout.dstAxesOrder.data();

  return cMemLayout;
}

/**********************************************************************************************************************/
// Transforms
/**********************************************************************************************************************/
// DFT
template<>
struct EnumMapping<afft::dft::Type>
  : EnumMappingBase<afft_dft_Type, afft::dft::Type, afft_Error_invalidDftType>
{
  static_assert(isSameEnumValue(afft_dft_Type_complexToComplex, afft::dft::Type::complexToComplex));
  static_assert(isSameEnumValue(afft_dft_Type_realToComplex,    afft::dft::Type::realToComplex));
  static_assert(isSameEnumValue(afft_dft_Type_complexToReal,    afft::dft::Type::complexToReal));

  static_assert(isSameEnumValue(afft_dft_Type_c2c,              afft::dft::Type::c2c));
  static_assert(isSameEnumValue(afft_dft_Type_r2c,              afft::dft::Type::r2c));
  static_assert(isSameEnumValue(afft_dft_Type_c2r,              afft::dft::Type::c2r));
};

static constexpr afft::dft::Parameters<> convertFromC(const afft_dft_Parameters& cParams)
{
  afft::dft::Parameters<> cxxParams{};
  cxxParams.direction     = convertFromC<afft::Direction>(cParams.direction);
  cxxParams.precision     = convertFromC(cParams.precision);
  cxxParams.shape         = afft::View<size_t>{cParams.shape, cParams.shapeRank};
  cxxParams.axes          = afft::View<size_t>{cParams.axes, cParams.axesRank};
  cxxParams.normalization = convertFromC<afft::Normalization>(cParams.normalization);
  cxxParams.placement     = convertFromC<afft::Placement>(cParams.placement);
  cxxParams.type          = convertFromC<afft::dft::Type>(cParams.type);

  return cxxParams;
}

static afft_dft_Parameters convertToC(const afft::dft::Parameters<>& cxxParams)
{
  afft_dft_Parameters cParams{};
  cParams.direction     = convertToC(cxxParams.direction);
  cParams.precision     = convertToC(cxxParams.precision);
  cParams.shapeRank     = cxxParams.shape.size();
  cParams.shape         = cxxParams.shape.data();
  cParams.axesRank      = cxxParams.axes.size();
  cParams.axes          = cxxParams.axes.data();
  cParams.normalization = convertToC(cxxParams.normalization);
  cParams.placement     = convertToC(cxxParams.placement);
  cParams.type          = convertToC(cxxParams.type);

  return cParams;
}

// DHT
template<>
struct EnumMapping<afft::dht::Type>
  : EnumMappingBase<afft_dht_Type, afft::dht::Type, afft_Error_invalidDhtType>
{
  static_assert(isSameEnumValue(afft_dht_Type_separable, afft::dht::Type::separable));
};

static constexpr afft::dht::Parameters<> convertFromC(const afft_dht_Parameters& cParams)
{
  afft::dht::Parameters<> cxxParams{};
  cxxParams.direction     = convertFromC<afft::Direction>(cParams.direction);
  cxxParams.precision     = convertFromC(cParams.precision);
  cxxParams.shape         = afft::View<size_t>{cParams.shape, cParams.shapeRank};
  cxxParams.axes          = afft::View<size_t>{cParams.axes, cParams.axesRank};
  cxxParams.normalization = convertFromC<afft::Normalization>(cParams.normalization);
  cxxParams.placement     = convertFromC<afft::Placement>(cParams.placement);
  cxxParams.type          = convertFromC<afft::dht::Type>(cParams.type);

  return cxxParams;
}

static afft_dht_Parameters convertToC(const afft::dht::Parameters<>& cxxParams)
{
  afft_dht_Parameters cParams{};
  cParams.direction     = convertToC(cxxParams.direction);
  cParams.precision     = convertToC(cxxParams.precision);
  cParams.shapeRank     = cxxParams.shape.size();
  cParams.shape         = cxxParams.shape.data();
  cParams.axesRank      = cxxParams.axes.size();
  cParams.axes          = cxxParams.axes.data();
  cParams.normalization = convertToC(cxxParams.normalization);
  cParams.placement     = convertToC(cxxParams.placement);
  cParams.type          = convertToC(cxxParams.type);

  return cParams;
}

// DTT
template<>
struct EnumMapping<afft::dtt::Type>
  : EnumMappingBase<afft_dtt_Type, afft::dtt::Type, afft_Error_invalidDttType>
{
  static_assert(isSameEnumValue(afft_dtt_Type_dct1, afft::dtt::Type::dct1));
  static_assert(isSameEnumValue(afft_dtt_Type_dct2, afft::dtt::Type::dct2));
  static_assert(isSameEnumValue(afft_dtt_Type_dct3, afft::dtt::Type::dct3));
  static_assert(isSameEnumValue(afft_dtt_Type_dct4, afft::dtt::Type::dct4));
  static_assert(isSameEnumValue(afft_dtt_Type_dst1, afft::dtt::Type::dst1));
  static_assert(isSameEnumValue(afft_dtt_Type_dst2, afft::dtt::Type::dst2));
  static_assert(isSameEnumValue(afft_dtt_Type_dst3, afft::dtt::Type::dst3));
  static_assert(isSameEnumValue(afft_dtt_Type_dst4, afft::dtt::Type::dst4));

  static_assert(isSameEnumValue(afft_dtt_Type_dct,  afft::dtt::Type::dct));
  static_assert(isSameEnumValue(afft_dtt_Type_dst,  afft::dtt::Type::dst));
};

static afft::dtt::Parameters<> convertFromC(const afft_dtt_Parameters& cParams)
{
  afft::dtt::Parameters<> cxxParams{};
  cxxParams.direction     = convertFromC<afft::Direction>(cParams.direction);
  cxxParams.precision     = convertFromC(cParams.precision);
  cxxParams.shape         = afft::View<size_t>{cParams.shape, cParams.shapeRank};
  cxxParams.axes          = afft::View<size_t>{cParams.axes, cParams.axesRank};
  cxxParams.normalization = convertFromC<afft::Normalization>(cParams.normalization);
  cxxParams.placement     = convertFromC<afft::Placement>(cParams.placement);
  cxxParams.types         = afft::View<afft::dtt::Type>{reinterpret_cast<const afft::dtt::Type*>(cParams.types), cParams.axesRank};

  return cxxParams;
}

static afft_dtt_Parameters convertToC(const afft::dtt::Parameters<>& cxxParams)
{
  afft_dtt_Parameters cParams{};
  cParams.direction     = convertToC(cxxParams.direction);
  cParams.precision     = convertToC(cxxParams.precision);
  cParams.shapeRank     = cxxParams.shape.size();
  cParams.shape         = cxxParams.shape.data();
  cParams.axesRank      = cxxParams.axes.size();
  cParams.axes          = cxxParams.axes.data();
  cParams.normalization = convertToC(cxxParams.normalization);
  cParams.types         = reinterpret_cast<const afft_dtt_Type*>(cxxParams.types.data());

  return cParams;
}

/**********************************************************************************************************************/
// Architecture
/**********************************************************************************************************************/
static constexpr afft::spst::cpu::Parameters<> convertFromC(const afft_spst_cpu_Parameters& cParams, std::size_t shapeRank)
{
  afft::spst::cpu::Parameters cxxParams{};
  cxxParams.memoryLayout   = convertFromC(cParams.memoryLayout, shapeRank);
  cxxParams.complexFormat  = convertFromC<afft::ComplexFormat>(cParams.complexFormat);
  cxxParams.preserveSource = cParams.preserveSource;
  cxxParams.alignment      = convertFromC<afft::Alignment>(cParams.alignment);
  cxxParams.threadLimit    = cParams.threadLimit;

  return cxxParams;
}

static constexpr afft_spst_cpu_Parameters convertToC(const afft::spst::cpu::Parameters<>& cxxParams)
{
  afft_spst_cpu_Parameters cParams{};
  cParams.memoryLayout   = convertToC(cxxParams.memoryLayout);
  cParams.complexFormat  = convertToC(cxxParams.complexFormat);
  cParams.preserveSource = cxxParams.preserveSource;
  cParams.alignment      = convertToC(cxxParams.alignment);
  cParams.threadLimit    = cxxParams.threadLimit;

  return cParams;
}

static afft::spst::gpu::Parameters<> convertFromC(const afft_spst_gpu_Parameters& cParams, std::size_t shapeRank)
{
  afft::spst::gpu::Parameters<> cxxParams;
  cxxParams.memoryLayout         = convertFromC(cParams.memoryLayout, shapeRank);
  cxxParams.complexFormat        = convertFromC<afft::ComplexFormat>(cParams.complexFormat);
  cxxParams.preserveSource       = cParams.preserveSource;
  cxxParams.useExternalWorkspace = cParams.useExternalWorkspace;
# if AFFT_GPU_BACKEND_IS(CUDA)
  cxxParams.device               = cParams.device;
# elif AFFT_GPU_BACKEND_IS(HIP)
  cxxParams.device               = cParams.device;
# elif AFFT_GPU_BACKEND_IS(OPENCL)
  cxxParams.context              = cParams.context;
  cxxParams.device               = cParams.device;
# endif

  return cxxParams;
}

static constexpr afft_spst_gpu_Parameters convertToC(const afft::spst::gpu::Parameters<>& cxxParams)
{
  afft_spst_gpu_Parameters cParams{};
  cParams.memoryLayout         = convertToC(cxxParams.memoryLayout);
  cParams.complexFormat        = convertToC(cxxParams.complexFormat);
  cParams.preserveSource       = cxxParams.preserveSource;
  cParams.useExternalWorkspace = cxxParams.useExternalWorkspace;
#if AFFT_GPU_BACKEND_IS(CUDA)
  cParams.device               = cxxParams.device;
#elif AFFT_GPU_BACKEND_IS(HIP)
  cParams.device               = cxxParams.device;
#elif AFFT_GPU_BACKEND_IS(OPENCL)
  cParams.context              = cxxParams.context;
  cParams.device               = cxxParams.device;
#endif

  return cParams;
}

// TODO: mpst gpu Parameters

static constexpr afft::mpst::cpu::Parameters<> convertFromC(const afft_mpst_cpu_Parameters& cParams, std::size_t shapeRank)
{
  afft::mpst::cpu::Parameters cxxParams{};
  cxxParams.memoryLayout         = convertFromC(cParams.memoryLayout, shapeRank);
  cxxParams.complexFormat        = convertFromC<afft::ComplexFormat>(cParams.complexFormat);
  cxxParams.preserveSource       = cParams.preserveSource;
  cxxParams.useExternalWorkspace = cParams.useExternalWorkspace;
#if AFFT_MP_BACKEND_IS(MPI)
  cxxParams.communicator         = cParams.communicator;
#endif
  cxxParams.alignment            = convertFromC<afft::Alignment>(cParams.alignment);
  cxxParams.threadLimit          = cParams.threadLimit;

  return cxxParams;
}

static constexpr afft_mpst_cpu_Parameters convertToC(const afft::mpst::cpu::Parameters<>& cxxParams)
{
  afft_mpst_cpu_Parameters cParams{};
  cParams.memoryLayout         = convertToC(cxxParams.memoryLayout);
  cParams.complexFormat        = convertToC(cxxParams.complexFormat);
  cParams.preserveSource       = cxxParams.preserveSource;
  cParams.useExternalWorkspace = cxxParams.useExternalWorkspace;
#if AFFT_MP_BACKEND_IS(MPI)
  cParams.communicator         = cxxParams.communicator;
#endif
  cParams.alignment            = convertToC(cxxParams.alignment);
  cParams.threadLimit          = cxxParams.threadLimit;

  return cParams;
}

static afft::mpst::gpu::Parameters<> convertFromC(const afft_mpst_gpu_Parameters& cParams, std::size_t shapeRank)
{
  afft::mpst::gpu::Parameters cxxParams{};
  cxxParams.memoryLayout         = convertFromC(cParams.memoryLayout, shapeRank);
  cxxParams.complexFormat        = convertFromC<afft::ComplexFormat>(cParams.complexFormat);
  cxxParams.preserveSource       = cParams.preserveSource;
  cxxParams.useExternalWorkspace = cParams.useExternalWorkspace;
#if AFFT_MP_BACKEND_IS(MPI)
  cxxParams.communicator         = cParams.communicator;
#endif
#if AFFT_GPU_BACKEND_IS(CUDA)
  cxxParams.device               = cParams.device;
#elif AFFT_GPU_BACKEND_IS(HIP)
  cxxParams.device               = cParams.device;
#elif AFFT_GPU_BACKEND_IS(OPENCL)
  cxxParams.context              = cParams.context;
  cxxParams.device               = cParams.device;
#endif

  return cxxParams;
}

static constexpr afft_mpst_gpu_Parameters convertToC(const afft::mpst::gpu::Parameters<>& cxxParams)
{
  afft_mpst_gpu_Parameters cParams{};
  cParams.memoryLayout         = convertToC(cxxParams.memoryLayout);
  cParams.complexFormat        = convertToC(cxxParams.complexFormat);
  cParams.preserveSource       = cxxParams.preserveSource;
  cParams.useExternalWorkspace = cxxParams.useExternalWorkspace;
#if AFFT_MP_BACKEND_IS(MPI)
  cParams.communicator         = cxxParams.communicator;
#endif
#if AFFT_GPU_BACKEND_IS(CUDA)
  cParams.device               = cxxParams.device;
#elif AFFT_GPU_BACKEND_IS(HIP)
  cParams.device               = cxxParams.device;
#elif AFFT_GPU_BACKEND_IS(OPENCL)
  cParams.context              = cxxParams.context;
  cParams.device               = cxxParams.device;
#endif
  
  return cParams;
}

static constexpr afft::spst::cpu::ExecutionParameters convertFromC(const afft_spst_cpu_ExecutionParameters&)
{
  afft::spst::cpu::ExecutionParameters cxxParams{};

  return cxxParams;
}

static afft_spst_cpu_ExecutionParameters convertToC(const afft::spst::cpu::ExecutionParameters&)
{
  afft_spst_cpu_ExecutionParameters cParams{};

  return cParams;
}

static constexpr afft::spst::gpu::ExecutionParameters convertFromC(const afft_spst_gpu_ExecutionParameters& cParams)
{
  afft::spst::gpu::ExecutionParameters cxxParams{};
#if AFFT_GPU_BACKEND_IS(CUDA)
  cxxParams.stream    = cParams.stream;
  cxxParams.workspace = cParams.workspace;
#elif AFFT_GPU_BACKEND_IS(HIP)
  cxxParams.stream    = cParams.stream;
  cxxParams.workspace = cParams.workspace;
#elif AFFT_GPU_BACKEND_IS(OPENCL)
  cxxParams.queue     = cParams.queue;
  cxxParams.workspace = cParams.workspace;
#endif

  return cxxParams;
}

static afft_spst_gpu_ExecutionParameters convertToC(const afft::spst::gpu::ExecutionParameters& cxxParams)
{
  afft_spst_gpu_ExecutionParameters cParams{};
#if AFFT_GPU_BACKEND_IS(CUDA)
  cParams.stream    = cxxParams.stream;
  cParams.workspace = cxxParams.workspace;
#elif AFFT_GPU_BACKEND_IS(HIP)
  cParams.stream    = cxxParams.stream;
  cParams.workspace = cxxParams.workspace;
#elif AFFT_GPU_BACKEND_IS(OPENCL)
  cParams.queue     = cxxParams.queue;
  cParams.workspace = cxxParams.workspace;
#endif
  
  return cParams;
}

// TODO: mpst gpu ExecutionParameters

static constexpr afft::mpst::cpu::ExecutionParameters convertFromC(const afft_mpst_cpu_ExecutionParameters& cParams)
{
  afft::mpst::cpu::ExecutionParameters cxxParams{};
  cxxParams.workspace = cParams.workspace;

  return cxxParams;
}

static afft_mpst_cpu_ExecutionParameters convertToC(const afft::mpst::cpu::ExecutionParameters& cxxParams)
{
  afft_mpst_cpu_ExecutionParameters cParams{};
  cParams.workspace = cxxParams.workspace;

  return cParams;
}

static constexpr afft::mpst::gpu::ExecutionParameters convertFromC(const afft_mpst_gpu_ExecutionParameters& cParams)
{
  afft::mpst::gpu::ExecutionParameters cxxParams{};
#if AFFT_GPU_BACKEND_IS(CUDA)
  cxxParams.stream    = cParams.stream;
  cxxParams.workspace = cParams.workspace;
#elif AFFT_GPU_BACKEND_IS(HIP)
  cxxParams.stream    = cParams.stream;
  cxxParams.workspace = cParams.workspace;
#elif AFFT_GPU_BACKEND_IS(OPENCL)
  cxxParams.queue     = cParams.queue;
  cxxParams.workspace = cParams.workspace;
#endif
  
  return cxxParams;
}

/**********************************************************************************************************************/
// Initialization
/**********************************************************************************************************************/

/// @brief Initialize the library
extern "C" afft_Error afft_init()
try
{
  afft::init();
  return afft_Error_success;
}
catch (...)
{
  return afft_Error_internal;
}

/// @brief Finalize the library
extern "C" afft_Error afft_finalize()
try
{
  afft::finalize();
  return afft_Error_success;
}
catch (...)
{
  return afft_Error_internal;
}

/**********************************************************************************************************************/
// Plan
/**********************************************************************************************************************/
template<typename TransformParamsT, typename ArchParamsT, typename BackendParamsT>
static afft_Error makePlanImpl(const TransformParamsT& transformParams,
                               const ArchParamsT&      archParams,
                               const BackendParamsT&   backendParams,
                               afft_Plan**             planPtr)
try
{
  if (planPtr == nullptr)
  {
    return afft_Error_invalidArgument;
  }

  *planPtr = reinterpret_cast<afft_Plan*>(afft::makePlan(transformParams, archParams, backendParams).release());
  return afft_Error_success;
}
catch (...)
{
  return afft_Error_internal;
}

/**
 * @brief Make a plan object implementation. Internal use only.
 * @param transformParams Transform parameters.
 * @param archParams Architecture parameters.
 * @param planPtr Pointer to the plan object pointer.
 * @return Error code.
 */
extern "C" afft_Error _afft_makePlan(afft_TransformParameters    transformParams,
                                     afft_ArchitectureParameters archParams,
                                     afft_Plan**                 planPtr)
try
{
  auto makePlanImpl1 = [&](const auto& tParams)
  {
    const auto shapeRank = tParams.shape.size();

    switch (archParams.target)
    {
    case afft_Target_cpu:
      switch (archParams.distribution)
      {
      case afft_Distribution_spst:
        return makePlanImpl(tParams, convertFromC(archParams.spstCpu, shapeRank), afft::spst::cpu::BackendParameters{}, planPtr);
      case afft_Distribution_mpst:
        return makePlanImpl(tParams, convertFromC(archParams.mpstCpu, shapeRank), afft::mpst::cpu::BackendParameters{}, planPtr);
      default:
        return afft_Error_invalidDistribution;
      }
    case afft_Target_gpu:
      switch (archParams.distribution)
      {
      case afft_Distribution_spst:
        return makePlanImpl(tParams, convertFromC(archParams.spstGpu, shapeRank), afft::spst::gpu::BackendParameters{}, planPtr);
      // case afft_Distribution_spmt:
      //   return makePlanImpl(tParams, convertFromC(archParams.spmtGpu, shapeRank), afft::spmt::gpu::BackendParameters{}, planPtr);
      case afft_Distribution_mpst:
        return makePlanImpl(tParams, convertFromC(archParams.mpstGpu, shapeRank), afft::mpst::gpu::BackendParameters{}, planPtr);
      default:
        return afft_Error_invalidDistribution;
      }
    default:
      return afft_Error_invalidTarget;
    }
  };

  switch (transformParams.transform)
  {
  case afft_Transform_dft:
    return makePlanImpl1(convertFromC(transformParams.dft));
  case afft_Transform_dht:
    return makePlanImpl1(convertFromC(transformParams.dht));
  case afft_Transform_dtt:
    return makePlanImpl1(convertFromC(transformParams.dtt));
  default:
    return afft_Error_invalidTransform;
  }
}
catch (...)
{
  return afft_Error_internal;
}

/**
 * @brief Make a plan object with backend parameters implementation. Internal use only.
 * @param transformParams Transform parameters.
 * @param archParams Architecture parameters.
 * @param backendParams Backend parameters.
 * @param planPtr Pointer to the plan object pointer.
 * @return Error code.
 */
extern "C" afft_Error _afft_makePlanWithBackendParameters(afft_TransformParameters    transformParams,
                                                          afft_ArchitectureParameters archParams,
                                                          afft_BackendParameters      backendParams,
                                                          afft_Plan**                 planPtr)
try
{
  auto makePlanImpl1 = [&](const auto& tParams)
  {
    if (archParams.target != backendParams.target || archParams.distribution != backendParams.distribution)
    {
      return afft_Error_invalidArgument;
    }

    const auto shapeRank = tParams.shape.size();

    switch (archParams.target)
    {
    case afft_Target_cpu:
      switch (archParams.distribution)
      {
      case afft_Distribution_spst:
        return makePlanImpl(tParams, convertFromC(archParams.spstCpu, shapeRank), convertFromC(backendParams.spstCpu), planPtr);
      case afft_Distribution_mpst:
        return makePlanImpl(tParams, convertFromC(archParams.mpstCpu, shapeRank), convertFromC(backendParams.mpstCpu), planPtr);
      default:
        return afft_Error_invalidDistribution;
      }
    case afft_Target_gpu:
      switch (archParams.distribution)
      {
      case afft_Distribution_spst:
        return makePlanImpl(tParams, convertFromC(archParams.spstGpu, shapeRank), convertFromC(backendParams.spstGpu), planPtr);
      // case afft_Distribution_spmt:
      //   return makePlanImpl(tParams, convertFromC(archParams.spmtGpu, shapeRank), convertFromC(backendParams.spmtGpu), planPtr);
      case afft_Distribution_mpst:
        return makePlanImpl(tParams, convertFromC(archParams.mpstGpu, shapeRank), convertFromC(backendParams.mpstGpu), planPtr);
      default:
        return afft_Error_invalidDistribution;
      }
    default:
      return afft_Error_invalidTarget;
    }
  };

  switch (transformParams.transform)
  {
  case afft_Transform_dft:
    return makePlanImpl1(convertFromC(transformParams.dft));
  case afft_Transform_dht:
    return makePlanImpl1(convertFromC(transformParams.dht));
  case afft_Transform_dtt:
    return makePlanImpl1(convertFromC(transformParams.dtt));
  default:
    return afft_Error_invalidTransform;
  }
}
catch (...)
{
  return afft_Error_internal;
}

/**
 * @brief Get the plan transform.
 * @param plan Plan object.
 * @param transform Pointer to the transform variable.
 * @return Error code.
 */
extern "C" afft_Error afft_Plan_getTransform(const afft_Plan* plan, afft_Transform* transform)
try
{
  checkPlan(plan);

  if (transform == nullptr)
  {
    return afft_Error_invalidArgument;
  }

  *transform = convertToC(reinterpret_cast<const afft::Plan*>(plan)->getTransform());

  return afft_Error_success;
}
catch (...)
{
  return afft_Error_internal;
}

/**
 * @brief Get the plan target.
 * @param plan Plan object.
 * @param target Pointer to the target variable.
 * @return Error code.
 */
extern "C" afft_Error afft_Plan_getTarget(const afft_Plan* plan, afft_Target* target)
try
{
  checkPlan(plan);

  if (target == nullptr)
  {
    return afft_Error_invalidArgument;
  }

  *target = convertToC(reinterpret_cast<const afft::Plan*>(plan)->getTarget());

  return afft_Error_success;
}
catch (...)
{
  return afft_Error_internal;
}

/**
 * @brief Get the target count.
 * @param plan Plan object.
 * @param targetCount Pointer to the target count variable.
 * @return Error code.
 */
extern "C" afft_Error afft_Plan_getTargetCount(const afft_Plan* plan, size_t* targetCount)
try
{
  checkPlan(plan);

  if (targetCount == nullptr)
  {
    return afft_Error_invalidArgument;
  }

  *targetCount = reinterpret_cast<const afft::Plan*>(plan)->getTargetCount();

  return afft_Error_success;
}
catch (...)
{
  return afft_Error_internal;
}

/**
 * @brief Get the plan distribution.
 * @param plan Plan object.
 * @param distribution Pointer to the distribution variable.
 * @return Error code.
 */
extern "C" afft_Error afft_Plan_getDistribution(const afft_Plan* plan, afft_Distribution* distribution)
try
{
  checkPlan(plan);

  if (distribution == nullptr)
  {
    return afft_Error_invalidArgument;
  }

  *distribution = convertToC(reinterpret_cast<const afft::Plan*>(plan)->getDistribution());

  return afft_Error_success;
}
catch (...)
{
  return afft_Error_internal;
}

/**
 * @brief Get the plan backend.
 * @param plan Plan object.
 * @param backend Pointer to the backend variable.
 * @return Error code.
 */
extern "C" afft_Error afft_Plan_getBackend(const afft_Plan* plan, afft_Backend* backend)
try
{
  checkPlan(plan);

  if (backend == nullptr)
  {
    return afft_Error_invalidArgument;
  }

  *backend = convertToC(reinterpret_cast<const afft::Plan*>(plan)->getBackend());

  return afft_Error_success;
}
catch (...)
{
  return afft_Error_internal;
}

/**
 * @brief Get the plan workspace size.
 * @param plan Plan object.
 * @param workspaceSize Pointer to the workspace array the same size as number of targets.
 * @return Error code.
 */
extern "C" afft_Error afft_Plan_getWorkspaceSize(const afft_Plan* plan, size_t* workspaceSize)
try
{
  checkPlan(plan);

  if (workspaceSize == nullptr)
  {
    return afft_Error_invalidArgument;
  }

  const auto workspaceSizes = reinterpret_cast<const afft::Plan*>(plan)->getWorkspaceSize();
  std::copy(workspaceSizes.begin(), workspaceSizes.end(), workspaceSize);

  return afft_Error_success;
}
catch (...)
{
  return afft_Error_internal;
}

// extern "C" afft_Error afft_Plan_getTransformParameters(afft_Plan* plan, void* transformParams)
// try
// {
//   if (plan == nullptr || transformParams == nullptr)
//   {
//     return afft_Error_invalidArgument;
//   }

//   afft::Plan* cxxPlan = static_cast<afft::Plan*>(plan);

//   switch (cxxPlan->getTransform())
//   {
//   case afft::Transform::dft:
//     const auto dftParams = static_cast<afft_dft_Parameters*>(transformParams);
//     convertToC(cxxPlan->getDftParameters(), *dftParams);
//     break;
//   case afft::Transform::dht:
//     const auto dhtParams = static_cast<afft_dht_Parameters*>(transformParams);
//     convertToC(cxxPlan->getDhtParameters(), *dhtParams);
//     break;
//   case afft::Transform::dtt:
//     const auto dttParams = static_cast<afft_dtt_Parameters*>(transformParams);
//     convertToC(cxxPlan->getDttParameters(), *dttParams);
//     break;
//   default:
//     return afft_Error_internal;
//   }

//   return afft_Error_success;
// }
// catch (...)
// {
//   return afft_Error_internal;
// }

/**
 * @brief Execute a plan implementation. Internal use only.
 * @param plan Plan object.
 * @param src Source data.
 * @param dst Destination data.
 * @return Error code.
 */
extern "C" afft_Error _afft_Plan_execute(afft_Plan* plan, afft_ExecutionBuffers src, afft_ExecutionBuffers dst)
try
{
  return afft_Error_success;
}
catch (...)
{
  return afft_Error_internal;
}

/**
 * @brief Execute a plan with execution parameters implementation. Internal use only.
 * @param plan Plan object.
 * @param src Source data.
 * @param dst Destination data.
 * @param execParams Execution parameters.
 * @return Error code.
 */
extern "C" afft_Error _afft_Plan_executeWithParameters(afft_Plan*               plan,
                                                       afft_ExecutionBuffers    src,
                                                       afft_ExecutionBuffers    dst,
                                                       afft_ExecutionParameters execParams)
try
{
  return afft_Error_success;
}
catch (...)
{
  return afft_Error_internal;
}

/**
 * @brief Destroy a plan object.
 * @param plan Plan object.
 */
extern "C" void afft_Plan_destroy(afft_Plan* plan)
{
  ::operator delete(reinterpret_cast<afft::Plan*>(plan), std::nothrow);
}

/**********************************************************************************************************************/
// Allocations
/**********************************************************************************************************************/

/**
 * @brief Allocate aligned memory.
 * @param sizeInBytes Size of the memory block in bytes.
 * @param alignment Alignment of the memory block.
 * @return Pointer to the allocated memory block or NULL if the allocation failed.
 */
extern "C" void* afft_cpu_alignedAlloc(size_t sizeInBytes, afft_Alignment alignment)
{
  return ::operator new[](sizeInBytes, static_cast<std::align_val_t>(alignment), std::nothrow);
}

/**
 * @brief Free aligned memory.
 * @param ptr Pointer to the memory block.
 */
extern "C" void afft_cpu_alignedFree(void* ptr, afft_Alignment alignment)
{
  ::operator delete[](ptr, static_cast<std::align_val_t>(alignment), std::nothrow);
}

#if AFFT_GPU_BACKEND_IS(CUDA) || AFFT_GPU_BACKEND_IS(HIP)
/**
 * @brief Allocate unified memory.
 * @param sizeInBytes Size of the memory block in bytes.
 * @return Pointer to the allocated memory block or NULL if the allocation failed.
 */
extern "C" void* afft_gpu_unifiedAlloc(size_t sizeInBytes)
{
  void* ptr;

#if AFFT_GPU_BACKEND_IS(CUDA)
  if (cudaMallocManaged(&ptr, sizeInBytes) == cudaSuccess)
  {
    return ptr;
  }
#elif AFFT_GPU_BACKEND_IS(HIP)
  if (hipMallocManaged(&ptr, sizeInBytes) == hipSuccess)
  {
    return ptr;
  }
#endif
  
  return nullptr;
}

/**
 * @brief Free unified memory.
 * @param ptr Pointer to the memory block.
 */
extern "C" void afft_gpu_unifiedFree(void* ptr)
{
#if AFFT_GPU_BACKEND_IS(CUDA)
  cudaFree(ptr);
#elif AFFT_GPU_BACKEND_IS(HIP)
  hipFree(ptr);
#endif
}
#elif AFFT_GPU_BACKEND_IS(OPENCL)
/**
 * @brief Allocate unified memory.
 * @param context OpenCL context.
 * @param sizeInBytes Size of the memory block in bytes.
 * @return Pointer to the allocated memory block or NULL if the allocation failed.
 */
extern "C" void* afft_gpu_unifiedAlloc(cl_context context, size_t sizeInBytes);

/**
 * @brief Free unified memory.
 * @param context OpenCL context.
 * @param ptr Pointer to the memory block.
 */
extern "C" void afft_gpu_unifiedFree(cl_context context, void* ptr);
#endif

/**********************************************************************************************************************/
// Utilities
/**********************************************************************************************************************/

/**
 * @brief Make strides from the shape.
 * @param rank Rank of the shape.
 * @param shape Shape of the array.
 * @param strides Strides of the array.
 * @return Error code.
 */
extern "C" afft_Error afft_makeStrides(size_t rank, const size_t* shape, size_t* strides)
try
{
  return afft_Error_success;
}
catch (...)
{
  return afft_Error_internal;
}

/**
 * @brief Make transposed strides.
 * @param rank Rank of the shape.
 * @param resultShape Shape of the result array.
 * @param orgAxesOrder Order of the original axes.
 * @param strides Strides of the array.
 * @return Error code.
 */
extern "C" afft_Error afft_makeTransposedStrides(size_t        rank,
                                                 const size_t* resultShape,
                                                 const size_t* orgAxesOrder,
                                                 size_t*       strides)
try
{
  return afft_Error_success;
}
catch (...)
{
  return afft_Error_internal;
}
