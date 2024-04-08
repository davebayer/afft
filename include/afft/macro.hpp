#ifndef AFFT_MACRO_HPP
#define AFFT_MACRO_HPP

#include "detail/macro.hpp"

/// @brief Macro for identity function
#define AFFT_IDENTITY(...) __VA_ARGS__

/// @brief Macro for emty delimiter
#define AFFT_DELIM_EMPTY  AFFT_DETAIL_EMPTY
/// @brief Macro for comma delimiter
#define AFFT_DELIM_COMMA  AFFT_DETAIL_COMMA

/**
 * @brief Macro for applying a macro to each variadic argument
 * @param macro Macro to apply
 * @param ... Variadic arguments
 * @return Macro applied to each variadic argument
 */
#define AFFT_FOR_EACH(macro, ...) \
  AFFT_FOR_EACH_WITH_DELIM(macro, AFFT_DELIM_EMPTY, __VA_ARGS__)

/**
 * @brief Macro for applying a macro to each variadic argument with a delimiter
 * @param macro Macro to apply
 * @param delimMacro Delimiter macro
 * @param ... Variadic arguments
 * @return Macro applied to each variadic argument with a delimiter
 */
#define AFFT_FOR_EACH_WITH_DELIM(macro, delimMacro, ...) \
  AFFT_DETAIL_FOR_EACH_WITH_DELIM(macro, delimMacro, __VA_ARGS__)

/// @brief Macro for bit-wise OR on variadic arguments
#define AFFT_BITOR(...) AFFT_DETAIL_BITOR(__VA_ARGS__)

#endif /* AFFT_MACRO_HPP */
