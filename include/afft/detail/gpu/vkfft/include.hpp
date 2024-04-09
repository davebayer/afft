#ifndef AFFT_DETAIL_GPU_VKFFT_INCLUDE_HPP
#define AFFT_DETAIL_GPU_VKFFT_INCLUDE_HPP

#include "../../../gpu.hpp"

// check if AFFT has been included before including vkFFT
#ifdef VKFFT_H
# error "AFFT and vkFFT cannot be included together in the same translation unit"
#endif

// push the current value of VKFFT_BACKEND
#pragma push_macro("VKFFT_BACKEND")
#undef VKFFT_BACKEND

// push the current value of VKFFT_MAX_FFT_DIMENSIONS
#pragma push_macro("VKFFT_MAX_FFT_DIMENSIONS")
#undef VKFFT_MAX_FFT_DIMENSIONS

// define VKFFT_BACKEND based on the current GPU backend
#if AFFT_GPU_BACKEND_IS_CUDA
# define VKFFT_BACKEND 1
#elif AFFT_GPU_BACKEND_IS_HIP
# define VKFFT_BACKEND 2
#elif AFFT_GPU_BACKEND_IS_OPENCL
# define VKFFT_BACKEND 3
#else
# error "vkFFT backend is only supported with CUDA or HIP"
#endif

// define VKFFT_MAX_FFT_DIMENSIONS based on the maximum number of dimensions
#define VKFFT_MAX_FFT_DIMENSIONS AFFT_MAX_DIM_COUNT

// include the vkFFT header
#include <vkFFT.h>

// restore the original value of VKFFT_BACKEND
#pragma pop_macro("VKFFT_BACKEND")

// restore the original value of VKFFT_MAX_FFT_DIMENSIONS
#pragma pop_macro("VKFFT_MAX_FFT_DIMENSIONS")

#endif /* AFFT_DETAIL_GPU_VKFFT_INCLUDE_HPP */
