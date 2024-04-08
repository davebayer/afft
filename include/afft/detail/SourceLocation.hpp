#ifndef AFFT_DETAIL_SOURCE_LOCATION_HPP
#define AFFT_DETAIL_SOURCE_LOCATION_HPP

#include <cstdint>
#include <version>

#ifdef __cpp_lib_source_location
# include <source_location>
#endif

namespace afft::detail
{
#ifndef __cpp_lib_source_location
  class SourceLocation
  {
    public:
      /// @brief Default constructor
      constexpr SourceLocation() = default;
      /**
       * @brief Constructor
       * @param fileName File name
       * @param funcName Function name
       * @param line Line number
       */
      constexpr SourceLocation(const char* fileName, const char* funcName, std::uint_least32_t line) noexcept
      : mFileName{fileName}, mFuncName{funcName}, mLine{line}
      {}
      /// @brief Copy constructor
      constexpr SourceLocation(const SourceLocation&) = default;
      /// @brief Move constructor
      constexpr SourceLocation(SourceLocation&&) = default;
      /// @brief Destructor
      constexpr ~SourceLocation() = default;
      /// @brief Copy assignment operator
      constexpr SourceLocation& operator=(const SourceLocation&) = default;
      /// @brief Move assignment operator
      constexpr SourceLocation& operator=(SourceLocation&&) = default;
      /// @brief Get file name
      constexpr const char* file_name() const noexcept
      {
        return (mFileName != nullptr) ? mFileName : "";
      }
      /// @brief Get function name
      constexpr const char* function_name() const noexcept
      {
        return (mFuncName != nullptr) ? mFuncName : "";
      }
      /// @brief Get line number
      constexpr std::uint_least32_t line() const noexcept
      {
        return mLine;
      }
      /// @brief Get column number, always returns 0
      constexpr std::uint_least32_t column() const noexcept
      {
        return {};
      }
    private:
      const char*         mFileName; ///< File name
      const char*         mFuncName; ///< Function name
      std::uint_least32_t mLine;     ///< Line number
  };

# define AFFT_DETAIL_SOURCE_LOCATION_CURRENT afft::detail::SourceLocation{__FILE__, __func__, __LINE__}
#else
  using SourceLocation = std::source_location;

# define AFFT_DETAIL_SOURCE_LOCATION_CURRENT std::source_location::current()
#endif
} // namespace afft::detail

#endif /* AFFT_DETAIL_SOURCE_LOCATION_HPP */
