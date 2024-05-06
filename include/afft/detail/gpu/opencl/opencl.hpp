#ifndef AFFT_DETAIL_GPU_OPENCL_OPENCL_HPP
#define AFFT_DETAIL_GPU_OPENCL_OPENCL_HPP

#include <memory>

#include "error.hpp"
#include "include.hpp"
#include "init.hpp"

namespace afft::detail::gpu::opencl
{
  static_assert(std::is_pointer_v<cl_mem>, "afft relies on cl_mem being a pointer type");

  /**
   * @brief Deleter for cl_mem objects
   */
  struct MemDeleter
  {
    /**
     * @brief Deleter for cl_mem objects
     * @param mem cl_mem object to delete
     */
    void operator()(cl_mem mem) const
    {
      Error::check(clReleaseMemObject(mem));
    }
  };

  /**
   * @brief Create a buffer from a pointer
   * @tparam T Type of the buffer
   * @param context OpenCL context
   * @param svmBuffer Pointer to the buffer
   * @param size Size of the buffer
   * @return std::unique_ptr<std::remove_pointer_t<cl_mem>, MemDeleter> Unique pointer to the buffer
   */
  template<typename T>
  std::unique_ptr<std::remove_pointer_t<cl_mem>, MemDeleter> makeBufferFromPtr(cl_context context, T* svmBuffer, std::size_t size)
  {
    cl_int       error{};
    cl_mem_flags flags  = CL_MEM_USE_HOST_PTR | ((std::is_const_v<T>) ? CL_MEM_READ_ONLY : CL_MEM_READ_WRITE);
    cl_mem       buffer = clCreateBuffer(context, flags, size, svmBuffer, &error);
    
    Error::check(buffer);

    return std::unique_ptr<std::remove_pointer_t<cl_mem>, MemDeleter>{buffer};
  }

  /**
   * @brief Check if a buffer is read-only
   * @param buffer Buffer to check
   * @return true if the buffer is read-only, false otherwise
   */
  inline bool isReadOnlyBuffer(cl_mem buffer)
  {
    cl_mem_flags flags{};
    Error::check(clGetMemObjectInfo(buffer, CL_MEM_FLAGS, sizeof(flags), &flags, nullptr));

    return flags & CL_MEM_READ_ONLY;
  }

  /**
   * @brief Check if a buffer is read-write
   * @param buffer Buffer to check
   * @return true if the buffer is read-write, false otherwise
   */
  inline bool isReadWrityBuffer(cl_mem buffer)
  {
    cl_mem_flags flags{};
    Error::check(clGetMemObjectInfo(buffer, CL_MEM_FLAGS, sizeof(flags), &flags, nullptr));

    return flags & CL_MEM_READ_WRITE;
  }
} // namespace afft::detail::gpu::opencl

#endif /* AFFT_DETAIL_GPU_OPENCL_OPENCL_HPP */
