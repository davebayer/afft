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

#ifndef AFFT_CUFFT_HPP
#define AFFT_CUFFT_HPP

#ifndef AFFT_TOP_LEVEL_INCLUDE
# include "detail/include.hpp"
#endif

namespace afft::cufft
{
  /// @brief cuFFT Workspace policy
  enum class WorkspacePolicy : std::uint8_t
  {
    performance, ///< Use the workspace for performance.
    minimal,     ///< Use the minimal workspace.
    user,        ///< Use the user-defined workspace size.
  };

inline namespace spst
{
  /// @brief cuFFT initialization parameters
  struct InitParameters
  {
    WorkspacePolicy workspacePolicy{WorkspacePolicy::performance}; ///< Workspace policy.
    std::size_t     userWorkspaceSize{};                           ///< Workspace size in bytes when using user-defined workspace policy.
    bool            usePatientJIT{true};                           ///< Use patient JIT compilation. Supported when using cuFFT 11.2 or later.
  };
} // inline namespace spst

namespace spmt
{
  /// @brief cuFFT initialization parameters
  struct InitParameters
  {
    bool usePatientJIT{true}; ///< Use patient JIT compilation. Supported when using cuFFT 11.2 or later.
  };
} // namespace spmt

namespace mpst
{
  /// @brief cuFFT initialization parameters
  struct InitParameters
  {
    bool usePatientJIT{true}; ///< Use patient JIT compilation. Supported when using cuFFT 11.2 or later.
  };
} // namespace mpst
} // namespace afft::cufft

#endif /* AFFT_CUFFT_HPP */
