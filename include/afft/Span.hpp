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

#ifndef AFFT_SPAN_HPP
#define AFFT_SPAN_HPP

#ifndef AFFT_TOP_LEVEL_INCLUDE
# include "detail/include.hpp"
#endif

AFFT_EXPORT namespace afft
{
#if defined(AFFT_CXX_HAS_VERSION) && defined(__cpp_lib_span) && (__cpp_lib_span >= 202002L)
  /// @brief The dynamic extent value for Span.
  inline constexpr std::size_t dynamicExtent = std::dynamic_extent;

  /// @brief The Span type.
  template<typename T, std::size_t extent = dynamicExtent>
  using Span = std::span<T, extent>;
#else
  /// @brief The dynamic extent value for Span.
  inline constexpr std::size_t dynamicExtent = afft::thirdparty::span::dynamic_extent;

  /// @brief The Span type.
  template<typename T, std::size_t extent = dynamicExtent>
  using Span = afft::thirdparty::span::span<T, extent>;
#endif

// namespace detail
// {
//   template<typename T, std::size_t extent>
//   class SpanBase
//   {
//   public:
//     constexpr SpanBase() noexcept = default;

//     constexpr SpanBase(std::size_t) noexcept
//     {}

//     [[nodiscard]] constexpr std::size_t size() const noexcept
//     {
//       return size;
//     }
//   private:
//     static constexpr std::size_t size = extent;
//   };

//   template<typename T>
//   class SpanBase<T, dynamicExtent>
//   {
//   public:
//     constexpr SpanBase() noexcept = default;

//     constexpr SpanBase(std::size_t size) noexcept
//     : mSize(size)
//     {}

//     [[nodiscard]] constexpr std::size_t size() const noexcept
//     {
//       return mSize;
//     }
//   protected:
    
//   private:
//     std::size_t mSize{};
//   };
// } // namespace detail

//   template<typename T, std::size_t _extent = dynamicExtent>
//   class Span : private detail::SpanBase<T, _extent>
//   {
//     static_assert(std::is_object_v<T>, "T must be an object type.");
//     static_assert(sizeof(T) != 0, "T must not be an incomplete type.");
//     static_assert(!std::is_abstract_v<T>, "T must not be an abstract class.");

//     private:
//       using Parent = detail::SpanBase<T, _extent>;
//     public:
//       using element_type           = T;
//       using value_type             = std::remove_cv_t<T>;
//       using size_type              = std::size_t;
//       using difference_type        = std::ptrdiff_t;
//       using pointer                = T*;
//       using const_pointer          = const T*;
//       using reference              = T&;
//       using const_reference        = const T&;
//       using iterator               = pointer;
//       using const_iterator         = const_pointer;
//       using reverse_iterator       = std::reverse_iterator<iterator>;
//       using const_reverse_iterator = std::reverse_iterator<const_iterator>;

//       static constexpr std::size_t extent = _extent;

//       template<std::size_t e, std::enable_if_t<(e == dynamicExtent || e == extent), int> = 0>
//       constexpr Span() noexcept
//       {}

//       constexpr Span(pointer first, size_type count) noexcept
//       : Parent(count), mData(first)
//       {
//         if (count != dynamicExtent && count != extent)
//         {
//           throw std::logic_error("Invalid span count.");
//         }
//       }


//     private:
//       T* mData{};
//   };
}

#endif /* AFFT_SPAN_HPP */
