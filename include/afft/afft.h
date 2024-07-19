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

#ifndef AFFT_H
#define AFFT_H

// Include the version header.
#include "afft-version.h"

// Include the config header.
#include "detail/config.h"

// Include only once in the top-level header.
#include "detail/include.h"
#define AFFT_TOP_LEVEL_INCLUDE

// Include all public headers.
#include "backend.h"
#include "common.h"
#include "error.h"
#include "fftw3.h"
#include "init.h"
#include "memory.h"
#include "mp.h"
#include "Plan.h"
// #include "PlanCache.h"
#include "target.h"
#include "transform.h"
#include "type.h"
#include "utils.h"
#include "version.h"

#endif /* AFFT_H */
