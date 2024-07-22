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

#ifndef AFFT_FFTW3_H
#define AFFT_FFTW3_H

#ifndef AFFT_TOP_LEVEL_INCLUDE
# include "detail/include.h"
#endif

#include "error.h"
#include "type.h"

#ifdef __cplusplus
extern "C"
{
#endif

/**
 * @brief Does the FFTW3 library support the given precision?
 * @param prec Precision of the FFTW3 library.
 */
bool afft_fftw3_isSupportedPrecision(afft_Precision prec);

/**
 * @brief Export FFTW3 wisdom to a file.
 * @param prec Precision of the FFTW3 library.
 * @param filename Name of the file to export the wisdom to.
 * @param errDetails Error details.
 * @return Error code.
 */
afft_Error afft_fftw3_exportWisdomToFilename(afft_Precision prec, const char* filename, afft_ErrorDetails* errDetails);

/**
 * @brief Export FFTW3 wisdom to a file.
 * @param prec Precision of the FFTW3 library.
 * @param file File to export the wisdom to.
 * @param errDetails Error details.
 * @return Error code.
 */
afft_Error afft_fftw3_exportWisdomToFile(afft_Precision prec, FILE* file, afft_ErrorDetails* errDetails);

/**
 * @brief Export FFTW3 wisdom to a string. The string must be freed by the caller using free().
 * @param prec Precision of the FFTW3 library.
 * @param str String to export the wisdom to.
 * @param errDetails Error details.
 * @return Error code.
 */
afft_Error afft_fftw3_exportWisdomToString(afft_Precision prec, char** str, afft_ErrorDetails* errDetails);

/**
 * @brief Import FFTW3 system wisdom.
 * @param prec Precision of the FFTW3 library.
 * @param errDetails Error details.
 * @return Error code.
 */
afft_Error afft_fftw3_importSystemWisdom(afft_Precision prec, afft_ErrorDetails* errDetails);

/**
 * @brief Import FFTW3 wisdom from a file.
 * @param prec Precision of the FFTW3 library.
 * @param filename Name of the file to import the wisdom from.
 * @param errDetails Error details.
 * @return Error code.
 */
afft_Error afft_fftw3_importWisdomFromFilename(afft_Precision prec, const char* filename, afft_ErrorDetails* errDetails);

/**
 * @brief Import FFTW3 wisdom from a file.
 * @param prec Precision of the FFTW3 library.
 * @param file File to import the wisdom from.
 * @param errDetails Error details.
 * @return Error code.
 */
afft_Error afft_fftw3_importWisdomFromFile(afft_Precision prec, FILE* file, afft_ErrorDetails* errDetails);

/**
 * @brief Import FFTW3 wisdom from a string.
 * @param prec Precision of the FFTW3 library.
 * @param str String to import the wisdom from.
 * @param errDetails Error details.
 * @return Error code.
 */
afft_Error afft_fftw3_importWisdomFromString(afft_Precision prec, const char* str, afft_ErrorDetails* errDetails);

/**
 * @brief Forget FFTW3 wisdom.
 * @param prec Precision of the FFTW3 library.
 * @param errDetails Error details.
 * @return Error code.
 */
afft_Error afft_fftw3_forgetWisdom(afft_Precision prec, afft_ErrorDetails* errDetails);

#ifdef AFFT_ENABLE_MPI
/**
 * @brief Broadcast FFTW3 wisdom.
 * @param prec Precision of the FFTW3 library.
 * @param comm MPI communicator.
 * @param errDetails Error details.
 * @return Error code.
 */
afft_Error afft_fftw3_mpi_broadcastWisdom(afft_Precision prec, MPI_Comm comm, afft_ErrorDetails* errDetails);

/**
 * @brief Gather FFTW3 wisdom.
 * @param prec Precision of the FFTW3 library.
 * @param comm MPI communicator.
 * @param errDetails Error details.
 * @return Error code.
 */
afft_Error afft_fftw3_mpi_gatherWisdom(afft_Precision prec, MPI_Comm comm, afft_ErrorDetails* errDetails);

#endif /* AFFT_ENABLE_MPI */

#ifdef __cplusplus
}
#endif

#endif /* AFFT_FFTW3_H */
