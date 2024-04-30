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

#ifndef AFFT_3RD_PARTY_HPP
#define AFFT_3RD_PARTY_HPP

// Set up namespaces for fmtlib
#define FMT_START_NAMESPACE namespace afft::fmt {
#define FMT_END_NAMESPACE }

// Include fmtlib
#include <fmt/format.h>

// Undefine macros to avoid conflicts with other libraries
#undef FMT_START_NAMESPACE
#undef FMT_END_NAMESPACE

// Set up namespaces for mdspan
#define MDSPAN_IMPL_STANDARD_NAMESPACE afft::mdspan
// #define MDSPAN_IMPL_PROPOSED_NAMESPACE

// Include mdspan
#include <mdspan/mdspan.hpp>

// Undefine macros to avoid conflicts with other libraries
#undef MDSPAN_IMPL_STANDARD_NAMESPACE
// #undef MDSPAN_IMPL_PROPOSED_NAMESPACE

#endif /* AFFT_3RD_PARTY_HPP */
