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

#ifndef AFFT_DETAIL_MEM_DESC_HPP
#define AFFT_DETAIL_MEM_DESC_HPP

#ifndef AFFT_TOP_LEVEL_INCLUDE
# include "include.hpp"
#endif

#include "common.hpp"
#include "MpDesc.hpp"
#include "TargetDesc.hpp"
#include "TransformDesc.hpp"
#include "validate.hpp"
#include "../memory.hpp"
#include "../typeTraits.hpp"
#include "../utils.hpp"

namespace afft::detail
{
  /**
   * @breif NEmbed and stride structure.
   * @tparam T Type of the result elements.
   */
  template<typename T>
  struct NEmbedAndStride
  {
    static_assert(std::is_integral_v<T>, "T must be an integral type");

    MaxDimBuffer<T> nEmbed; ///< NEmbed.
    T               stride; ///< Stride.
  };

  /**
   * @brief Makes nEmbed and stride from shape and strides.
   * @tparam T Type of the result elements.
   * @param shape Shape.
   * @param axes Axes.
   * @param strides Strides.
   * @return NEmbed and stride or std::nullopt if the shape and strides do not have the NEmbed and stride format.
   */
  template<typename T = Size>
  [[nodiscard]] constexpr NEmbedAndStride<T>
  makeNEmbedAndStride(const Size*       shape,
                      const std::size_t shapeRank,
                      const Axis*       axes,
                      const std::size_t axesRank,
                      const Size*       strides)
  {
    NEmbedAndStride<T> nEmbedAndStride{};
    nEmbedAndStride.stride = strides[shapeRank - 1];

    for (std::size_t i = axesRank - 1; i > 0; --i)
    {
      auto [quot, rem] = div(strides[axes[i - 1]], strides[axes[i]]);

      if (quot < shape[axes[i]] || rem != 0)
      {
        throw Exception{Error::invalidArgument, "strides cannot be converted to nEmbed and stride"};
      }

      nEmbedAndStride.nEmbed.data[i] = safeIntCast<T>(quot);
    }

    nEmbedAndStride.nEmbed.data[0] = safeIntCast<T>(shape[axes[0]]);

    return nEmbedAndStride;
  }

  /// @brief Centralized memory layout descriptor
  class CentralMemDesc
  {
    public:
      /// @brief Default constructor (default)
      CentralMemDesc() = default;

      /**
       * @brief Constructor.
       * @param[in] srcStrides Source strides.
       * @param[in] dstStrides Destination strides.
       * @param[in] transformDesc Transform descriptor.
       * @param[in] mpDesc MPI descriptor.
       * @param[in] targetDesc Target descriptor.
       */
      CentralMemDesc(const Size*          srcStrides,
                     const Size*          dstStrides,
                     const TransformDesc& transformDesc,
                     const MpDesc&        mpDesc,
                     const TargetDesc&    targetDesc)
      : mShapeRank{transformDesc.getShapeRank()},
        mHasDefaultSrcStrides{srcStrides == nullptr},
        mHasDefaultDstStrides{dstStrides == nullptr}
      {
        if (mpDesc.getMpBackend() != MpBackend::none || targetDesc.getTargetCount() != 1)
        {
          throw Exception{Error::invalidArgument, "Centralized memory layout can be used only with single-process application on sigle target"};
        }

        if (mHasDefaultSrcStrides)
        {
          auto srcShape = transformDesc.getSrcShape();

          if (transformDesc.getTransform() == Transform::dft && transformDesc.getPlacement() == Placement::inPlace)
          {
            if (transformDesc.getTransformDesc<Transform::dft>().type == dft::Type::realToComplex)
            {
              srcShape[mShapeRank - 1] = (srcShape[mShapeRank - 1] / 2 + 1) * 2;
            }
          }

          makeStrides(mShapeRank, srcShape.data, mSrcStrides.data);
        }
        else
        {
          std::copy_n(srcStrides, mShapeRank, mSrcStrides.data);
        }

        if (mHasDefaultDstStrides)
        {
          auto dstShape = transformDesc.getDstShape();

          if (transformDesc.getTransform() == Transform::dft && transformDesc.getPlacement() == Placement::inPlace)
          {
            if (transformDesc.getTransformDesc<Transform::dft>().type == dft::Type::complexToReal)
            {
              dstShape[mShapeRank - 1] = (dstShape[mShapeRank - 1] / 2 + 1) * 2;
            }
          }

          makeStrides(mShapeRank, dstShape.data, mDstStrides.data);
        }
        else
        {
          std::copy_n(dstStrides, mShapeRank, mDstStrides.data);
        }
      }

      /// @brief Copy constructor (default)
      CentralMemDesc(const CentralMemDesc&) = default;

      /// @brief Move constructor (default)
      CentralMemDesc(CentralMemDesc&&) = default;

      /// @brief Destructor (default)
      ~CentralMemDesc() = default;

      /// @brief Copy assignment operator (default)
      CentralMemDesc& operator=(const CentralMemDesc&) = default;

      /// @brief Move assignment operator (default)
      CentralMemDesc& operator=(CentralMemDesc&&) = default;

      /**
       * @brief Get the source strides.
       * @return Source strides.
       */
      [[nodiscard]] constexpr const Size* getSrcStrides() const noexcept
      {
        return mSrcStrides.data;
      }

      /**
       * @brief Get the destination strides.
       * @return Destination strides.
       */
      [[nodiscard]] constexpr const Size* getDstStrides() const noexcept
      {
        return mDstStrides.data;
      }

      /**
       * @brief Check if the source strides are default.
       * @return True if the source strides are default, false otherwise.
       */
      [[nodiscard]] constexpr bool hasDefaultSrcStrides() const noexcept
      {
        return mHasDefaultSrcStrides;
      }

      /**
       * @brief Check if the destination strides are default.
       * @return True if the destination strides are default, false otherwise.
       */
      [[nodiscard]] constexpr bool hasDefaultDstStrides() const noexcept
      {
        return mHasDefaultDstStrides;
      }

      /**
       * @brief Equality operator.
       * @param[in] lhs Left-hand side.
       * @param[in] rhs Right-hand side.
       * @return True if the memory descriptors are equal, false otherwise.
       */
      [[nodiscard]] friend bool operator==(const CentralMemDesc& lhs, const CentralMemDesc& rhs) noexcept
      {
        const auto lhsSrcStrides = lhs.getSrcStrides();
        const auto rhsSrcStrides = rhs.getSrcStrides();

        const auto lhsDstStrides = lhs.getDstStrides();
        const auto rhsDstStrides = rhs.getDstStrides();

        return std::equal(lhsSrcStrides, lhsSrcStrides + lhs.mShapeRank, rhsSrcStrides, rhsSrcStrides + rhs.mShapeRank) &&
               std::equal(lhsDstStrides, lhsDstStrides + lhs.mShapeRank, rhsDstStrides, rhsDstStrides + rhs.mShapeRank);
      }

      /**
       * @brief Inequality operator.
       * @param[in] lhs Left-hand side.
       * @param[in] rhs Right-hand side.
       * @return True if the memory descriptors are not equal, false otherwise.
       */
      [[nodiscard]] friend bool operator!=(const CentralMemDesc& lhs, const CentralMemDesc& rhs) noexcept
      {
        return !(lhs == rhs);
      }

    private:
      std::size_t        mShapeRank{};            ///< Shape rank.
      MaxDimBuffer<Size> mSrcStrides{};           ///< Source strides.
      MaxDimBuffer<Size> mDstStrides{};           ///< Destination strides.
      bool               mHasDefaultSrcStrides{}; ///< Has default source strides.
      bool               mHasDefaultDstStrides{}; ///< Has default destination strides.
  };

  /// @brief Distributed memory layout descriptor.
  class DistribMemDesc
  {
    public:
      /// @brief Default constructor.
      DistribMemDesc() = default;

      /**
       * @brief Constructor.
       * @param[in] memLayout Memory layout.
       * @param[in] transformDesc Transform descriptor.
       * @param[in] mpDesc MPI descriptor.
       * @param[in] targetDesc Target descriptor.
       */
      DistribMemDesc(const std::size_t    srcDistribAxesRank,
                     const Axis*          srcDistribAxes,
                     const Size* const*   srcStarts,
                     const Size* const*   srcSizes,
                     const Size* const*   srcStrides,
                     const Axis*          srcAxesOrder,
                     const std::size_t    dstDistribAxesRank,
                     const Axis*          dstDistribAxes,
                     const Size* const*   dstStarts,
                     const Size* const*   dstSizes,
                     const Size* const*   dstStrides,
                     const Axis*          dstAxesOrder,
                     const TransformDesc& transformDesc,
                     const MpDesc&        ,
                     const TargetDesc&    targetDesc)
      : mShapeRank{transformDesc.getShapeRank()},
        mTargetCount{targetDesc.getTargetCount()},
        mData{std::make_unique<Size[]>(mShapeRank * mTargetCount * DataPos::_count)},
        mDataPtrs{std::make_unique<Size*[]>(mTargetCount * DataPos::_count)},
        mSrcDistribAxesRank{srcDistribAxesRank},
        mDstDistribAxesRank{dstDistribAxesRank}
      {
        // const auto srcShape = transformDesc.getSrcShape();
        // const auto dstShape = transformDesc.getDstShape();

        updateDataPtrs();

        if (srcStarts != nullptr && srcSizes != nullptr)
        {
          for (std::size_t i{}; i < mTargetCount; ++i)
          {
            if (srcStarts[i] == nullptr)
            {
              throw Exception{Error::invalidArgument, "invalid source starts"};
            }

            if (srcSizes[i] == nullptr)
            {
              throw Exception{Error::invalidArgument, "invalid source sizes"};
            }

            std::copy_n(srcStarts[i], mShapeRank, getSrcStarts()[i]);
            std::copy_n(srcSizes[i], mShapeRank, getSrcSizes()[i]);
          }
        }
        else if (srcStarts == nullptr && srcSizes == nullptr)
        {
          mHasDefaultSrcBlocks = true;
        }
        else
        {
          throw Exception{Error::invalidArgument, "either both source starts and sizes must be provided or none"};
        }

        if (srcStrides != nullptr)
        {
          if (std::any_of(srcStrides, srcStrides + mTargetCount, IsNullPtr{}))
          {
            throw Exception{Error::invalidArgument, "invalid source strides"};
          }

          for (std::size_t i{}; i < mTargetCount; ++i)
          {
            std::copy_n(srcStrides[i], mShapeRank, getSrcStrides()[i]);
          }
        }
        else
        {
          mHasDefaultSrcStrides = true;
        }

        if (mSrcDistribAxesRank > 0)
        {
          if (srcDistribAxes == nullptr)
          {
            throw Exception{Error::invalidArgument, "invalid source distributed axes"};
          }

          std::copy_n(srcDistribAxes, mSrcDistribAxesRank, mSrcDistribAxes.data);
        }

        if (srcAxesOrder != nullptr)
        {
          std::copy_n(srcAxesOrder, mShapeRank, mSrcAxesOrder.data);
        }
        else
        {
          mHasDefaultSrcAxesOrder = true;
          std::iota(mSrcAxesOrder.data, mSrcAxesOrder.data + mShapeRank, 0);
        }

        if (dstStarts != nullptr && dstSizes != nullptr)
        {
          for (std::size_t i{}; i < mTargetCount; ++i)
          {
            if (dstStarts[i] == nullptr)
            {
              throw Exception{Error::invalidArgument, "invalid destination starts"};
            }

            if (dstSizes[i] == nullptr)
            {
              throw Exception{Error::invalidArgument, "invalid destination sizes"};
            }

            std::copy_n(dstStarts[i], mShapeRank, getDstStarts()[i]);
            std::copy_n(dstSizes[i], mShapeRank, getDstSizes()[i]);
          }
        }
        else if (dstStarts == nullptr && dstSizes == nullptr)
        {
          mHasDefaultDstBlocks = true;
        }
        else
        {
          throw Exception{Error::invalidArgument, "either both destination starts and sizes must be provided or none"};
        }

        if (dstStrides != nullptr)
        {
          if (std::any_of(dstStrides, dstStrides + mTargetCount, IsNullPtr{}))
          {
            throw Exception{Error::invalidArgument, "invalid destination strides"};
          }

          for (std::size_t i{}; i < mTargetCount; ++i)
          {
            std::copy_n(dstStrides[i], mShapeRank, getDstStrides()[i]);
          }
        }
        else
        {
          mHasDefaultDstStrides = true;
        }

        if (mDstDistribAxesRank > 0)
        {
          if (srcDistribAxes == nullptr)
          {
            throw Exception{Error::invalidArgument, "invalid destination distributed axes"};
          }

          std::copy_n(dstDistribAxes, mDstDistribAxesRank, mDstDistribAxes.data);
        }

        if (dstAxesOrder != nullptr)
        {
          std::copy_n(dstAxesOrder, mShapeRank, mDstAxesOrder.data);
        }
        else
        {
          mHasDefaultDstAxesOrder = true;
          std::iota(mDstAxesOrder.data, mDstAxesOrder.data + mShapeRank, 0);
        }
      }

      /**
       * @brief Copy constructor.
       * @param other Other.
       */
      DistribMemDesc(const DistribMemDesc& other)
      : mShapeRank{other.mShapeRank},
        mTargetCount{other.mTargetCount},
        mData{std::make_unique<Size[]>(mShapeRank * mTargetCount * DataPos::_count)},
        mDataPtrs{std::make_unique<Size*[]>(mTargetCount * DataPos::_count)},
        mSrcDistribAxes{other.mSrcDistribAxes},
        mDstDistribAxes{other.mDstDistribAxes},
        mSrcAxesOrder{other.mSrcAxesOrder},
        mDstAxesOrder{other.mDstAxesOrder}
      {
        std::copy_n(other.mData.get(), mShapeRank * mTargetCount * DataPos::_count, mData.get());
        updateDataPtrs();
      }

      /// @brief Move constructor.
      DistribMemDesc(DistribMemDesc&&) = default;

      /// @brief Destructor.
      ~DistribMemDesc() = default;

      /**
       * @brief Copy assignment operator.
       * @param other Other.
       * @return Reference to this.
       */
      DistribMemDesc& operator=(const DistribMemDesc& other)
      {
        if (this != std::addressof(other))
        {
          if (mShapeRank != other.mShapeRank || mTargetCount != other.mTargetCount)
          {
            mData     = std::make_unique<Size[]>(other.mShapeRank * other.mTargetCount * DataPos::_count);
            mDataPtrs = std::make_unique<Size*[]>(other.mTargetCount * DataPos::_count);
          }

          mShapeRank      = other.mShapeRank;
          mTargetCount    = other.mTargetCount;
          mSrcDistribAxes = other.mSrcDistribAxes;
          mDstDistribAxes = other.mDstDistribAxes;
          mSrcAxesOrder   = other.mSrcAxesOrder;
          mDstAxesOrder   = other.mDstAxesOrder;

          std::copy_n(other.mData.get(), mShapeRank * mTargetCount * DataPos::_count, mData.get());
          updateDataPtrs();
        }

        return *this;
      }

      /// @brief Move assignment operator.
      DistribMemDesc& operator=(DistribMemDesc&&) = default;

      /**
       * @brief Get the source starts.
       * @return Source starts.
       */
      [[nodiscard]] const Size* const* getSrcStarts() const noexcept
      {
        return mDataPtrs.get() + DataPos::srcStarts * mTargetCount;
      }

      /**
       * @brief Get the source starts.
       * @return Source starts.
       */
      [[nodiscard]] Size* const* getSrcStarts() noexcept
      {
        return mDataPtrs.get() + DataPos::srcStarts * mTargetCount;
      }

      /**
       * @brief Get the source sizes.
       * @return Source sizes.
       */
      [[nodiscard]] const Size* const* getSrcSizes() const noexcept
      {
        return mDataPtrs.get() + DataPos::srcSizes * mTargetCount;
      }

      /**
       * @brief Get the source sizes.
       * @return Source sizes.
       */
      [[nodiscard]] Size* const* getSrcSizes() noexcept
      {
        return mDataPtrs.get() + DataPos::srcSizes * mTargetCount;
      }

      /**
       * @brief Get the source strides.
       * @return Source strides.
       */
      [[nodiscard]] const Size* const* getSrcStrides() const noexcept
      {
        return mDataPtrs.get() + DataPos::srcStrides * mTargetCount;
      }

      /**
       * @brief Get the source strides.
       * @return Source strides.
       */
      [[nodiscard]] Size* const* getSrcStrides() noexcept
      {
        return mDataPtrs.get() + DataPos::srcStrides * mTargetCount;
      }

      /**
       * @brief Get the source distributed axes.
       * @return Source distributed axes.
       */
      [[nodiscard]] const Axis* getSrcDistribAxes() const noexcept
      {
        return mSrcDistribAxes.data;
      }

      /**
       * @brief Set the source distributed axes.
       * @param srcDistribAxes Source distributed axes.
       * @return Source distributed axes.
       */
      void setSrcDistribAxes(const Axis* srcDistribAxes, const std::size_t srcDistribAxesRank) noexcept
      {
        mSrcDistribAxesRank = srcDistribAxesRank;
        std::copy_n(srcDistribAxes, srcDistribAxesRank, mSrcDistribAxes.data);
      }

      /**
       * @brief Get the source axes order.
       * @return Source axes order.
       */
      [[nodiscard]] const Axis* getSrcAxesOrder() const noexcept
      {
        return mSrcAxesOrder.data;
      }

      /**
       * @brief Get the destination starts.
       * @return Destination starts.
       */
      [[nodiscard]] const Size* const* getDstStarts() const noexcept
      {
        return mDataPtrs.get() + DataPos::dstStarts * mTargetCount;
      }
      
      /**
       * @brief Get the destination starts.
       * @return Destination starts.
       */
      [[nodiscard]] Size* const* getDstStarts() noexcept
      {
        return mDataPtrs.get() + DataPos::dstStarts * mTargetCount;
      }

      /**
       * @brief Get the destination sizes.
       * @return Destination sizes.
       */
      [[nodiscard]] const Size* const* getDstSizes() const noexcept
      {
        return mDataPtrs.get() + DataPos::dstSizes * mTargetCount;
      }

      /**
       * @brief Get the destination sizes.
       * @return Destination sizes.
       */
      [[nodiscard]] Size* const* getDstSizes() noexcept
      {
        return mDataPtrs.get() + DataPos::dstSizes * mTargetCount;
      }

      /**
       * @brief Get the destination strides.
       * @return Destination strides.
       */
      [[nodiscard]] const Size* const* getDstStrides() const noexcept
      {
        return mDataPtrs.get() + DataPos::dstStrides * mTargetCount;
      }

      /**
       * @brief Get the destination strides.
       * @return Destination strides.
       */
      [[nodiscard]] Size* const* getDstStrides() noexcept
      {
        return mDataPtrs.get() + DataPos::dstStrides * mTargetCount;
      }

      /**
       * @brief Get the destination distributed axes.
       * @return Destination distributed axes.
       */
      [[nodiscard]] constexpr const Axis* getDstDistribAxes() const noexcept
      {
        return mDstDistribAxes.data;
      }

      /**
       * @brief Set the destination distributed axes.
       * @param dstDistribAxes Destination distributed axes.
       * @return Destination distributed axes.
       */
      void setDstDistribAxes(const Axis* dstDistribAxes, const std::size_t dstDistribAxesRank) noexcept
      {
        mDstDistribAxesRank = dstDistribAxesRank;
        std::copy_n(dstDistribAxes, dstDistribAxesRank, mDstDistribAxes.data);
      }

      /**
       * @brief Get the destination axes order.
       * @return Destination axes order.
       */
      [[nodiscard]] constexpr const Axis* getDstAxesOrder() const noexcept
      {
        return mDstAxesOrder.data;
      }

      /**
       * @brief Get the destination axes order.
       * @return Destination axes order.
       */
      [[nodiscard]] constexpr const Axis* getDstAxesOrder() noexcept
      {
        return mDstAxesOrder.data;
      }

      /**
       * @brief Check if the source starts and sizes are default.
       * @return True if the source starts and sizes are default, false otherwise.
       */
      [[nodiscard]] constexpr bool hasDefaultSrcBlocks() const noexcept
      {
        return mHasDefaultSrcBlocks;
      }

      /**
       * @brief Check if the destination starts and sizes are default.
       * @return True if the destination starts and sizes are default, false otherwise.
       */
      [[nodiscard]] constexpr bool hasDefaultDstBlocks() const noexcept
      {
        return mHasDefaultDstBlocks;
      }

      /**
       * @brief Check if the source strides are default.
       * @return True if the source strides are default, false otherwise.
       */
      [[nodiscard]] constexpr bool hasDefaultSrcStrides() const noexcept
      {
        return mHasDefaultSrcStrides;
      }

      /**
       * @brief Check if the destination strides are default.
       * @return True if the destination strides are default, false otherwise.
       */
      [[nodiscard]] constexpr bool hasDefaultDstStrides() const noexcept
      {
        return mHasDefaultDstStrides;
      }

      /**
       * @brief Check if the source distributed axes are default.
       * @return True if the source distributed axes are default, false otherwise.
       */
      [[nodiscard]] constexpr bool hasDefaultSrcDistribAxes() const noexcept
      {
        return mSrcDistribAxesRank == 0;
      }

      /**
       * @brief Check if the destination distributed axes are default.
       * @return True if the destination distributed axes are default, false otherwise.
       */
      [[nodiscard]] constexpr bool hasDefaultDstDistribAxes() const noexcept
      {
        return mDstDistribAxesRank == 0;
      }

      /**
       * @brief Check if the source axes order is default.
       * @return True if the source axes order is default, false otherwise.
       */
      [[nodiscard]] constexpr bool hasDefaultSrcAxesOrder() const noexcept
      {
        return mHasDefaultSrcAxesOrder;
      }

      /**
       * @brief Check if the destination axes order is default.
       * @return True if the destination axes order is default, false otherwise.
       */
      [[nodiscard]] constexpr bool hasDefaultDstAxesOrder() const noexcept
      {
        return mHasDefaultDstAxesOrder;
      }

      /**
       * @brief Equality operator.
       * @param lhs Left-hand side.
       * @param rhs Right-hand side.
       * @return True if the memory descriptors are equal, false otherwise.
       */
      [[nodiscard]] friend bool operator==(const DistribMemDesc& lhs, const DistribMemDesc& rhs) noexcept
      {
        const auto shapeRank     = lhs.mShapeRank;
        const auto targetCount   = lhs.mTargetCount;
        const auto dataElemCount = shapeRank * targetCount * DataPos::_count;

        return (lhs.mShapeRank == rhs.mShapeRank) &&
               (lhs.mTargetCount == rhs.mTargetCount) &&
               std::equal(lhs.mSrcDistribAxes.data, lhs.mSrcDistribAxes.data + shapeRank, rhs.mSrcDistribAxes.data) &&
               std::equal(lhs.mDstDistribAxes.data, lhs.mDstDistribAxes.data + shapeRank, rhs.mDstDistribAxes.data) &&
               std::equal(lhs.mSrcAxesOrder.data, lhs.mSrcAxesOrder.data + shapeRank, rhs.mSrcAxesOrder.data) &&
               std::equal(lhs.mDstAxesOrder.data, lhs.mDstAxesOrder.data + shapeRank, rhs.mDstAxesOrder.data) &&
               std::equal(lhs.mData.get(), lhs.mData.get() + dataElemCount, rhs.mData.get());
      }

      /**
       * @brief Inequality operator.
       * @param lhs Left-hand side.
       * @param rhs Right-hand side.
       * @return True if the memory descriptors are not equal, false otherwise.
       */
      [[nodiscard]] friend bool operator!=(const DistribMemDesc& lhs, const DistribMemDesc& rhs) noexcept
      {
        return !(lhs == rhs);
      }

    private:
      /// @brief Data position names, replaces enum class to allow implict cast to std::size_t.
      struct DataPos
      {
        enum : std::size_t
        {
          srcStarts,  ///< Source starts.
          srcSizes,   ///< Source sizes.
          srcStrides, ///< Source strides.
          dstStarts,  ///< Destination starts.
          dstSizes,   ///< Destination sizes.
          dstStrides, ///< Destination strides.
          _count,     ///< Count of the enumeration.
        };
      };

      /// @brief Update data pointers.
      void updateDataPtrs()
      {
        for (std::size_t i{}; i < DataPos::_count; ++i)
        {
          for (std::size_t j{}; j < mTargetCount; ++j)
          {
            mDataPtrs[i * mTargetCount + j] = mData.get() + (i * mTargetCount + j) * mShapeRank;
          }
        }
      }

      std::size_t              mShapeRank{};                ///< Shape rank.
      std::size_t              mTargetCount{};              ///< Target count.
      std::unique_ptr<Size[]>  mData{};                     ///< Data.
      std::unique_ptr<Size*[]> mDataPtrs{};                 ///< Pointers.
      std::size_t              mSrcDistribAxesRank{};       ///< Source distributed axes rank.
      MaxDimBuffer<Axis>       mSrcDistribAxes{};           ///< Source distributed axes.
      std::size_t              mDstDistribAxesRank{};       ///< Destination distributed axes rank.
      MaxDimBuffer<Axis>       mDstDistribAxes{};           ///< Destination distributed axes.
      MaxDimBuffer<Axis>       mSrcAxesOrder{};             ///< Source axes order.
      MaxDimBuffer<Axis>       mDstAxesOrder{};             ///< Destination axes order.
      bool                     mHasDefaultSrcBlocks{};      ///< Has default source blocks.
      bool                     mHasDefaultDstBlocks{};      ///< Has default destination blocks.
      bool                     mHasDefaultSrcStrides{};     ///< Has default source strides.
      bool                     mHasDefaultDstStrides{};     ///< Has default destination strides;
      bool                     mHasDefaultSrcAxesOrder{};   ///< Has default source axes order.
      bool                     mHasDefaultDstAxesOrder{};   ///< Has default destination axes order;
  };

  /// @brief Memory descriptor.
  class MemDesc
  {
    public:
      /// @brief Default constructor is deleted.
      MemDesc() = delete;

      /**
       * @brief Construct a memory descriptor.
       * @tparam MemoryLayoutT Memory layout type.
       * @param[in] memLayout     Memory layout.
       * @param[in] transformDesc Transform descriptor.
       * @param[in] mpDesc        MP descriptor.
       * @param[in] targetDesc    Target descriptor.
       */
      AFFT_TEMPL_REQUIRES(typename MemoryLayoutT,
                          (isCxxMemoryLayoutParameters<MemoryLayoutT> ||
                           isCMemoryLayoutParameters<MemoryLayoutT>))
      MemDesc(const MemoryLayoutT& memLayout,
              const TransformDesc& transformDesc,
              const MpDesc&        mpDesc,
              const TargetDesc&    targetDesc)
      : mAlignment{static_cast<afft::Alignment>(memLayout.alignment)},
        mComplexFormat{static_cast<afft::ComplexFormat>(memLayout.complexFormat)},
        mMemVariant{makeMemVariant(memLayout, transformDesc, mpDesc, targetDesc)}
      {}

      /// @brief Copy constructor is default.
      MemDesc(const MemDesc&) = default;

      /// @brief Move constructor is default.
      MemDesc(MemDesc&&) = default;

      /// @brief Destructor is default.
      ~MemDesc() = default;

      /// @brief Copy assignment operator is default.
      MemDesc& operator=(const MemDesc&) = default;

      /// @brief Move assignment operator is default.
      MemDesc& operator=(MemDesc&&) = default;

      /**
       * @brief Get the memory layout.
       * @return Memory layout.
       */
      [[nodiscard]] constexpr MemoryLayout getMemoryLayout() const
      {
        switch (mMemVariant.index())
        {
          case 0:
            return MemoryLayout::centralized;
          case 1:
            return MemoryLayout::distributed;
          default:
            throw std::runtime_error{"Invalid memory variant index"};
        }
      }

      /**
       * @brief Get the memory descriptor for the given memory layout.
       * @tparam memoryLayout Memory layout.
       * @return Memory descriptor.
       */
      template<MemoryLayout memoryLayout>
      [[nodiscard]] constexpr const auto& getMemDesc() const
      {
        static_assert(isValid(memoryLayout), "invalid memory layout");

        if constexpr (memoryLayout == MemoryLayout::centralized)
        {
          return std::get<CentralMemDesc>(mMemVariant);
        }
        else if constexpr (memoryLayout == MemoryLayout::distributed)
        {
          return std::get<DistribMemDesc>(mMemVariant);
        }

        cxx::unreachable();
      }

      /**
       * @brief Get complex format.
       * @return Complex format.
       */
      [[nodiscard]] constexpr ComplexFormat getComplexFormat() const noexcept
      {
        return mComplexFormat;
      }

      /**
       * @brief Get the memory alignment.
       * @return Memory alignment.
       */
      [[nodiscard]] constexpr Alignment getAlignment() const noexcept
      {
        return mAlignment;
      }

      /**
       * @brief Reconstruct the C++ memory layout parameters.
       * @tparam memoryLayout Memory layout.
       * @return Memory layout parameters.
       */
      template<MemoryLayout memoryLayout>
      [[nodiscard]] constexpr MemoryLayoutParameters<memoryLayout> getCxxMemoryLayoutParameters() const noexcept
      {
        static_assert(isValid(memoryLayout), "invalid memory layout");

        MemoryLayoutParameters<memoryLayout> memLayout{};
        memLayout.alignment     = getAlignment();
        memLayout.complexFormat = getComplexFormat();

        if constexpr (memoryLayout == MemoryLayout::centralized)
        {
          const CentralMemDesc& memDesc = std::get<CentralMemDesc>(mMemVariant);
          memLayout.srcStrides = memDesc.getSrcStrides();
          memLayout.dstStrides = memDesc.getDstStrides();
        }
        else if constexpr (memoryLayout == MemoryLayout::distributed)
        {
          const DistribMemDesc& memDesc = std::get<DistribMemDesc>(mMemVariant);
          memLayout.srcStarts      = memDesc.getSrcStarts();
          memLayout.srcSizes       = memDesc.getSrcSizes();
          memLayout.srcStrides     = memDesc.getSrcStrides();
          memLayout.srcDistribAxes = memDesc.getSrcDistribAxes();
          memLayout.srcAxesOrder   = memDesc.getSrcAxesOrder();
          memLayout.dstStarts      = memDesc.getDstStarts();
          memLayout.dstSizes       = memDesc.getDstSizes();
          memLayout.dstStrides     = memDesc.getDstStrides();
          memLayout.dstDistribAxes = memDesc.getDstDistribAxes();
          memLayout.dstAxesOrder   = memDesc.getDstAxesOrder();
        }

        return memLayout;
      }

      /**
       * @brief Reconstruct the C memory layout parameters.
       * @tparam memoryLayout Memory layout.
       * @return Memory layout parameters.
       */
      template<MemoryLayout memoryLayout>
      [[nodiscard]] constexpr typename MemoryLayoutParametersSelect<memoryLayout>::CType
      getCMemoryLayoutParameters() const noexcept
      {
        static_assert(isValid(memoryLayout), "invalid memory layout");

        typename MemoryLayoutParametersSelect<memoryLayout>::CType memLayout{};
        memLayout.alignment     = static_cast<afft_Alignment>(getAlignment());
        memLayout.complexFormat = static_cast<afft_ComplexFormat>(getComplexFormat());

        if constexpr (memoryLayout == MemoryLayout::centralized)
        {
          const CentralMemDesc& memDesc = std::get<CentralMemDesc>(mMemVariant);
          memLayout.srcStrides = memDesc.getSrcStrides();
          memLayout.dstStrides = memDesc.getDstStrides();
        }
        else if constexpr (memoryLayout == MemoryLayout::distributed)
        {
          const DistribMemDesc& memDesc = std::get<DistribMemDesc>(mMemVariant);
          memLayout.srcStarts      = memDesc.getSrcStarts();
          memLayout.srcSizes       = memDesc.getSrcSizes();
          memLayout.srcStrides     = memDesc.getSrcStrides();
          memLayout.srcDistribAxes = memDesc.getSrcDistribAxes();
          memLayout.srcAxesOrder   = memDesc.getSrcAxesOrder();
          memLayout.dstStarts      = memDesc.getDstStarts();
          memLayout.dstSizes       = memDesc.getDstSizes();
          memLayout.dstStrides     = memDesc.getDstStrides();
          memLayout.dstDistribAxes = memDesc.getDstDistribAxes();
          memLayout.dstAxesOrder   = memDesc.getDstAxesOrder();
        }

        return memLayout;
      }

      /**
       * @brief Equality operator.
       * @param[in] lhs Left-hand side.
       * @param[in] rhs Right-hand side.
       * @return True if equal, false otherwise.
       */
      [[nodiscard]] friend bool operator==(const MemDesc& lhs, const MemDesc& rhs)
      {
        return lhs.mAlignment == rhs.mAlignment &&
               lhs.mComplexFormat == rhs.mComplexFormat &&
               lhs.mMemVariant == rhs.mMemVariant;
      }

      /**
       * @brief Inequality operator.
       * @param[in] lhs Left-hand side.
       * @param[in] rhs Right-hand side.
       * @return True if not equal, false otherwise.
       */
      [[nodiscard]] friend bool operator!=(const MemDesc& lhs, const MemDesc& rhs)
      {
        return !(lhs == rhs);
      }

    private:
      /// @brief Memory variant type.
      using MemVariant = std::variant<CentralMemDesc, DistribMemDesc>;

      /**
       * @brief Make a memory variant.
       * @param[in] memLayout     Memory layout.
       * @param[in] transformDesc Transform descriptor.
       * @param[in] mpDesc        MP descriptor.
       * @param[in] targetDesc    Target descriptor.
       * @return Memory variant.
       */
      [[nodiscard]] static CentralMemDesc makeMemVariant(const CentralizedMemoryLayout& memLayout,
                                                         const TransformDesc&           transformDesc,
                                                         const MpDesc&                  mpDesc,
                                                         const TargetDesc&              targetDesc)
      {
        return CentralMemDesc{memLayout.srcStrides,
                              memLayout.dstStrides,
                              transformDesc,
                              mpDesc,
                              targetDesc};
      }

      /**
       * @brief Make a memory variant.
       * @param[in] memLayout     Memory layout.
       * @param[in] transformDesc Transform descriptor.
       * @param[in] mpDesc        MP descriptor.
       * @param[in] targetDesc    Target descriptor.
       * @return Memory variant.
       */
      [[nodiscard]] static DistribMemDesc makeMemVariant(const DistributedMemoryLayout& memLayout,
                                                         const TransformDesc&           transformDesc,
                                                         const MpDesc&                  mpDesc,
                                                         const TargetDesc&              targetDesc)
      {
        const auto targetCount = targetDesc.getTargetCount();

        if (memLayout.srcStarts != nullptr)
        {
          if (std::any_of(memLayout.srcStarts, memLayout.srcStarts + targetCount, IsNullPtr{}))
          {
            throw Exception{Error::invalidArgument, "invalid source starts"};
          }
        }

        if (memLayout.srcSizes != nullptr)
        {
          if (std::any_of(memLayout.srcSizes, memLayout.srcSizes + targetCount, IsNullPtr{}))
          {
            throw Exception{Error::invalidArgument, "invalid source sizes"};
          }
        }

        if (memLayout.srcStrides != nullptr)
        {
          if (std::any_of(memLayout.srcStrides, memLayout.srcStrides + targetCount, IsNullPtr{}))
          {
            throw Exception{Error::invalidArgument, "invalid source strides"};
          }
        }

        if (memLayout.srcDistribAxes == nullptr)
        {
          throw Exception{Error::invalidArgument, "invalid source distributed axes"};
        }

        if (memLayout.srcAxesOrder != nullptr)
        {
          // todo: check if the source axes order is valid
        }
        
        if (memLayout.dstStarts != nullptr)
        {
          if (std::any_of(memLayout.dstStarts, memLayout.dstStarts + targetCount, IsNullPtr{}))
          {
            throw Exception{Error::invalidArgument, "invalid destination starts"};
          }
        }

        if (memLayout.dstSizes != nullptr)
        {
          if (std::any_of(memLayout.dstSizes, memLayout.dstSizes + targetCount, IsNullPtr{}))
          {
            throw Exception{Error::invalidArgument, "invalid destination sizes"};
          }
        }

        if (memLayout.dstStrides != nullptr)
        {
          if (std::any_of(memLayout.dstStrides, memLayout.dstStrides + targetCount, IsNullPtr{}))
          {
            throw Exception{Error::invalidArgument, "invalid destination strides"};
          }
        }

        if (memLayout.dstDistribAxes == nullptr)
        {
          throw Exception{Error::invalidArgument, "invalid destination distributed axes"};
        }

        if (memLayout.dstAxesOrder != nullptr)
        {
          // todo: check if the destination axes order is valid
        }

        return DistribMemDesc{memLayout.srcDistribAxesRank,
                              memLayout.srcDistribAxes,
                              memLayout.srcStarts,
                              memLayout.srcSizes,
                              memLayout.srcStrides,
                              memLayout.srcAxesOrder,
                              memLayout.dstDistribAxesRank,
                              memLayout.dstDistribAxes,
                              memLayout.dstStarts,
                              memLayout.dstSizes,
                              memLayout.dstStrides,
                              memLayout.dstAxesOrder,
                              transformDesc,
                              mpDesc,
                              targetDesc};
      }

      /**
       * @brief Make a memory variant.
       * @param[in] memLayout     Memory layout.
       * @param[in] transformDesc Transform descriptor.
       * @param[in] mpDesc        MP descriptor.
       * @param[in] targetDesc    Target descriptor.
       * @return Memory variant.
       */
      [[nodiscard]] static CentralMemDesc makeMemVariant(const afft_CentralizedMemoryLayout& memLayout,
                                                         const TransformDesc&                transformDesc,
                                                         const MpDesc&                       mpDesc,
                                                         const TargetDesc&                   targetDesc)
      {
        return CentralMemDesc{memLayout.srcStrides, memLayout.dstStrides, transformDesc, mpDesc, targetDesc};
      }

      /**
       * @brief Make a memory variant.
       * @param[in] memLayout     Memory layout.
       * @param[in] transformDesc Transform descriptor.
       * @param[in] mpDesc        MP descriptor.
       * @param[in] targetDesc    Target descriptor.
       * @return Memory variant.
       */
      [[nodiscard]] static DistribMemDesc makeMemVariant(const afft_DistributedMemoryLayout& memLayout,
                                                         const TransformDesc&                transformDesc,
                                                         const MpDesc&                       mpDesc,
                                                         const TargetDesc&                   targetDesc)
      {
        DistributedMemoryLayout cxxDistributedMemLayout{};
        cxxDistributedMemLayout.srcDistribAxesRank = memLayout.srcDistribAxesRank;
        cxxDistributedMemLayout.srcDistribAxes     = memLayout.srcDistribAxes;
        cxxDistributedMemLayout.srcStarts          = memLayout.srcStarts;
        cxxDistributedMemLayout.srcSizes           = memLayout.srcSizes;
        cxxDistributedMemLayout.srcStrides         = memLayout.srcStrides;
        cxxDistributedMemLayout.srcAxesOrder       = memLayout.srcAxesOrder;
        cxxDistributedMemLayout.dstDistribAxesRank = memLayout.dstDistribAxesRank;
        cxxDistributedMemLayout.dstDistribAxes     = memLayout.dstDistribAxes;
        cxxDistributedMemLayout.dstStarts          = memLayout.dstStarts;
        cxxDistributedMemLayout.dstSizes           = memLayout.dstSizes;
        cxxDistributedMemLayout.dstStrides         = memLayout.dstStrides;
        cxxDistributedMemLayout.dstAxesOrder       = memLayout.dstAxesOrder;

        return makeMemVariant(cxxDistributedMemLayout, transformDesc, mpDesc, targetDesc);
      }

      Alignment     mAlignment{};     ///< Memory alignment.
      ComplexFormat mComplexFormat{}; ///< Complex format.
      MemVariant    mMemVariant;      ///< Memory variant.
  };
} // namespace afft::detail

#endif /* AFFT_DETAIL_MEM_DESC_HPP */
