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

#ifndef AFFT_DETAIL_MP_DESC_HPP
#define AFFT_DETAIL_MP_DESC_HPP

#ifndef AFFT_TOP_LEVEL_INCLUDE
# include "include.hpp"
#endif

#include "validate.hpp"
#include "../mp.hpp"

namespace afft::detail
{
  /// @brief Single-process description (empty)
  struct SingleProcessDesc
  {
    /**
     * @brief Equality operator
     * @param[in] lhs Left-hand side
     * @param[in] rhs Right-hand side
     * @return True if equal, false otherwise
     */
    [[nodiscard]] constexpr friend bool operator==(const SingleProcessDesc&, const SingleProcessDesc&) noexcept
    {
      return true;
    }

    /**
     * @brief Inequality operator
     * @param[in] lhs Left-hand side
     * @param[in] rhs Right-hand side
     * @return True if not equal, false otherwise
     */
    [[nodiscard]] constexpr friend bool operator!=(const SingleProcessDesc&, const SingleProcessDesc&) noexcept
    {
      return false;
    }
  };

  /// @brief MPI description
  struct MpiDesc
  {
# ifdef AFFT_ENABLE_MPI
    MPI_Comm comm{}; ///< MPI communicator
# endif
  
    /**
     * @brief Equality operator
     * @param[in] lhs Left-hand side
     * @param[in] rhs Right-hand side
     * @return True if equal, false otherwise
     */
    [[nodiscard]] constexpr friend bool operator==([[maybe_unused]] const MpiDesc& lhs,
                                                   [[maybe_unused]] const MpiDesc& rhs) noexcept
    {
#   ifdef AFFT_ENABLE_MPI
      return lhs.comm == rhs.comm;
#   else
      return true;
#   endif
    }

    /**
     * @brief Inequality operator
     * @param[in] lhs Left-hand side
     * @param[in] rhs Right-hand side
     * @return True if not equal, false otherwise
     */
    [[nodiscard]] constexpr friend bool operator!=(const MpiDesc& lhs, const MpiDesc& rhs) noexcept
    {
      return !(lhs == rhs);
    }
  };

  /// @brief Multi-process description
  class MpDesc
  {
    public:
      /// @brief Default constructor is deleted
      MpDesc()
      : mMpVariant{SingleProcessDesc{}}
      {}

      /**
       * @brief Constructor from multi-process parameters
       * @tparam MpParamsT Multi-process parameters type
       * @param mpParams Multi-process parameters
       */
      template<typename MpParamsT>
      explicit constexpr MpDesc(const MpParamsT& mpParams)
      : mMpVariant{makeMpVariant(mpParams)}
      {
        static_assert(isCxxMpBackendParameters<MpParamsT> || isCMpBackendParameters<MpParamsT>,
                      "invalid multi-process parameters");
      }      

      /// @brief Copy constructor is default
      MpDesc(const MpDesc&) = default;

      /// @brief Move constructor is default
      MpDesc(MpDesc&&) = default;

      /// @brief Destructor is default
      ~MpDesc() = default;

      /// @brief Copy assignment operator is default
      MpDesc& operator=(const MpDesc&) = default;

      /// @brief Move assignment operator is default
      MpDesc& operator=(MpDesc&&) = default;

      /**
       * @brief Get the multi-process backend
       * @return Multi-process backend
       */
      [[nodiscard]] constexpr MpBackend getMpBackend() const
      {
        switch (mMpVariant.index())
        {
          case 0:
            return MpBackend::none;
          case 1:
            return MpBackend::mpi;
          default:
            throw std::runtime_error("invalid mp backend variant index");
        }
      }

      /**
       * @brief Get the multi-process description for the given multi-process backend
       * @tparam mpBackend Multi-process backend
       * @return Multi-process description
       */
      template<MpBackend mpBackend>
      [[nodiscard]] constexpr const auto& getMpDesc() const
      {
        static_assert(isValid(mpBackend), "Invalid multi-process backend");

        if constexpr (mpBackend == MpBackend::none)
        {
          return std::get<SingleProcessDesc>(mMpVariant);
        }
        else if constexpr (mpBackend == MpBackend::mpi)
        {
          return std::get<MpiDesc>(mMpVariant);
        }
      }

      /**
       * @brief Get the C++ multi-process parameters for the given multi-process backend
       * @tparam mpBackend Multi-process backend
       * @return Multi-process parameters
       */
      template<MpBackend mpBackend>
      [[nodiscard]] constexpr MpBackendParameters<mpBackend> getCxxMpParameters() const
      {
        static_assert(isValid(mpBackend), "Invalid multi-process backend");

        if constexpr (mpBackend == MpBackend::none)
        {
          return SingleProcessParameters{};
        }
        else if constexpr (mpBackend == MpBackend::mpi)
        {
          return afft::mpi::Parameters{getMpDesc<mpBackend>().comm};
        }
      }

      /**
       * @brief Get the C multi-process parameters for the given multi-process backend
       * @tparam mpBackend Multi-process backend
       * @return Multi-process parameters
       */
      template<MpBackend mpBackend>
      [[nodiscard]] constexpr typename MpBackendParametersSelect<mpBackend>::CType getCMpParameters() const
      {
        static_assert(isValid(mpBackend) && mpBackend != MpBackend::none, "invalid multi-process backend");

        if constexpr (mpBackend == MpBackend::mpi)
        {
          return afft_mpi_Parameters{getMpDesc<mpBackend>().comm};
        }
      }

      /**
       * @brief Equality operator
       * @param[in] lhs Left-hand side
       * @param[in] rhs Right-hand side
       * @return True if equal, false otherwise
       */
      [[nodiscard]] constexpr friend bool operator==(const MpDesc& lhs, const MpDesc& rhs)
      {
        return lhs.mMpVariant == rhs.mMpVariant;
      }

      /**
       * @brief Inequality operator
       * @param[in] lhs Left-hand side
       * @param[in] rhs Right-hand side
       * @return True if not equal, false otherwise
       */
      [[nodiscard]] constexpr friend bool operator!=(const MpDesc& lhs, const MpDesc& rhs)
      {
        return !(lhs == rhs);
      }

    private:
      /// @brief Multi-process variant type
      using MpVariant = std::variant<SingleProcessDesc, MpiDesc>;

      /**
       * @brief Make a multi-process variant
       * @param[in] spParams Single-process parameters
       * @return Multi-process variant
       */
      [[nodiscard]] static constexpr MpVariant makeMpVariant(const SingleProcessParameters&)
      {
        return SingleProcessDesc{};
      }

#   ifdef AFFT_ENABLE_MPI
      /**
       * @brief Make a multi-process variant
       * @param[in] mpiParams MPI parameters
       * @return Multi-process variant
       */
      [[nodiscard]] static constexpr MpVariant makeMpVariant(const afft::mpi::Parameters& mpiParams)
      {
        if (mpiParams.comm == nullptr || mpiParams.comm == MPI_COMM_NULL)
        {
          throw Exception{Error::invalidArgument, "invalid MPI communicator"};
        }

        return MpiDesc{mpiParams.comm};
      }
#   endif

#   ifdef AFFT_ENABLE_MPI
      /**
       * @brief Make a multi-process variant
       * @param[in] mpiParams MPI parameters
       * @return Multi-process variant
       */
      [[nodiscard]] static constexpr MpVariant makeMpVariant(const afft_mpi_Parameters& mpiParams)
      {
        if (mpiParams.comm == nullptr || mpiParams.comm == MPI_COMM_NULL)
        {
          throw Exception{Error::invalidArgument, "invalid MPI communicator"};
        }

        return MpiDesc{mpiParams.comm};
      }
#   endif

      MpVariant mMpVariant; ///< Multi-process variant
  };
} // namespace afft::detail

#endif /* AFFT_DETAIL_MP_DESC_HPP */
