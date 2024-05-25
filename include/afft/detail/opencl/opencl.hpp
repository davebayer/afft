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

#ifndef AFFT_DETAIL_OPENCL_OPENCL_HPP
#define AFFT_DETAIL_OPENCL_OPENCL_HPP

#ifndef AFFT_TOP_LEVEL_INCLUDE
# include "../include.hpp"
#endif

#include "error.hpp"
#include "init.hpp"

namespace afft::detail::opencl
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
  std::unique_ptr<std::remove_pointer_t<cl_mem>, MemDeleter>
  makeBufferFromPtr(cl_context context, T* svmBuffer, std::size_t size)
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
} // namespace afft::detail::opencl

#endif /* AFFT_DETAIL_OPENCL_OPENCL_HPP */
