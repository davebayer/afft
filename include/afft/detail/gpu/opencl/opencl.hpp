#ifndef AFFT_DETAIL_GPU_OPENCL_OPENCL_HPP
#define AFFT_DETAIL_GPU_OPENCL_OPENCL_HPP

#include "error.hpp"
#include "include.hpp"
#include "init.hpp"

namespace afft::detail::gpu::opencl
{
  static_assert(std::is_pointer_v<cl_mem>, "AFFT relies on cl_mem being a pointer type");
} // namespace afft::detail::gpu::opencl

#endif /* AFFT_DETAIL_GPU_OPENCL_OPENCL_HPP */
