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

#ifndef AFFT_DETAIL_ARCH_DESC_HPP
#define AFFT_DETAIL_ARCH_DESC_HPP

#ifndef AFFT_TOP_LEVEL_INCLUDE
# include "include.hpp"
#endif

#include "common.hpp"
#include "../typeTraits.hpp"

namespace afft::detail
{
  /// @brief Memory layout for spst targets.
  class SpstMemoryLayout
  {
    public:
      /// @brief Default constructor.
      SpstMemoryLayout() = default;

      /**
       * @brief Constructor.
       * @tparam shapeExt Extent of the shape.
       * @param shapeRank Shape rank.
       * @param memLayout Memory layout.
       */
      template<std::size_t shapeExt>
      SpstMemoryLayout(std::size_t shapeRank, const afft::spst::MemoryLayout<shapeExt>& memLayout)
      : mShapeRank{shapeRank},
        mHasDefaultSrcStrides{memLayout.srcStrides.empty()},
        mHasDefaultDstStrides{memLayout.dstStrides.empty()}
      {
        if (!mHasDefaultSrcStrides)
        {
          if (memLayout.srcStrides.size() != shapeRank)
          {
            throw std::invalid_argument("invalid source strides size");
          }

          std::copy(memLayout.srcStrides.begin(), memLayout.srcStrides.end(), mSrcStrides.begin());
        }

        if (!mHasDefaultDstStrides)
        {
          if (memLayout.dstStrides.size() != shapeRank)
          {
            throw std::invalid_argument("invalid destination strides size");
          }

          std::copy(memLayout.dstStrides.begin(), memLayout.dstStrides.end(), mDstStrides.begin());
        }
      }

      /// @brief Copy constructor.
      SpstMemoryLayout(const SpstMemoryLayout&) = default;

      /// @brief Move constructor.
      SpstMemoryLayout(SpstMemoryLayout&&) = default;

      /// @brief Destructor.
      ~SpstMemoryLayout() = default;

      /// @brief Copy assignment operator.
      SpstMemoryLayout& operator=(const SpstMemoryLayout&) = default;

      /// @brief Move assignment operator.
      SpstMemoryLayout& operator=(SpstMemoryLayout&&) = default;

      /**
       * @brief Check if the memory layout has default source strides.
       * @return True if the memory layout has default source strides, false otherwise.
       */
      [[nodiscard]] constexpr bool hasDefaultSrcStrides() const noexcept
      {
        return mHasDefaultSrcStrides;
      }

      /**
       * @brief Check if the memory layout has default destination strides.
       * @return True if the memory layout has default destination strides, false otherwise.
       */
      [[nodiscard]] constexpr bool hasDefaultDstStrides() const noexcept
      {
        return mHasDefaultDstStrides;
      }

      /**
       * @brief Get the source strides.
       * @return Source strides.
       */
      [[nodiscard]] constexpr View<std::size_t> getSrcStrides() const noexcept
      {
        return View<std::size_t>{mSrcStrides.data(), mShapeRank};
      }

      /**
       * @brief Get the source strides.
       * @tparam I Integral type.
       * @return Source strides.
       */
      template<typename I>
      [[nodiscard]] constexpr MaxDimArray<I> getSrcStrides() const
      {
        return mSrcStrides.cast<I>();
      }

      /**
       * @brief Get the writable source strides. Should be used carefully.
       * @return Source strides.
       */
      [[nodiscard]] constexpr Span<std::size_t> getSrcStridesWritable() noexcept
      {
        return Span<std::size_t>{mSrcStrides.data(), mShapeRank};
      }

      /**
       * @brief Get the destination strides.
       * @return Destination strides.
       */
      [[nodiscard]] constexpr View<std::size_t> getDstStrides() const noexcept
      {
        return View<std::size_t>{mDstStrides.data(), mShapeRank};
      }

      /**
       * @brief Get the destination strides.
       * @tparam I Integral type.
       * @return Destination strides.
       */
      template<typename I>
      [[nodiscard]] constexpr MaxDimArray<I> getDstStrides() const
      {
        return mDstStrides.cast<I>();
      }

      /**
       * @brief Get the writable destination strides. Should be used carefully.
       * @return Destination strides.
       */
      [[nodiscard]] constexpr Span<std::size_t> getDstStridesWritable() noexcept
      {
        return Span<std::size_t>{mDstStrides.data(), mShapeRank};
      }

      /**
       * @brief Get the memory layout.
       * @return Memory layout.
       */
      [[nodiscard]] constexpr afft::spst::MemoryLayout<> getView() const noexcept
      {
        afft::spst::MemoryLayout<> memLayout;
        memLayout.srcStrides = getSrcStrides();
        memLayout.dstStrides = getDstStrides();
        return memLayout;
      }

    private:
      std::size_t              mShapeRank{};                ///< Shape rank.
      MaxDimArray<std::size_t> mSrcStrides{};               ///< Source strides.
      MaxDimArray<std::size_t> mDstStrides{};               ///< Destination strides.
      bool                     mHasDefaultSrcStrides{true}; ///< Has default source strides.
      bool                     mHasDefaultDstStrides{true}; ///< Has default destination strides.
  };

  /// @brief Memory layout for spmt targets.
  class SpmtMemoryLayout
  {
    public:
      /// @brief Default constructor.
      SpmtMemoryLayout() = default;

      /**
       * @brief Constructor.
       * @tparam shapeExt Extent of the shape.
       * @param shapeRank Shape rank.
       * @param targetCount Number of targets.
       * @param memLayout Memory layout.
       */
      template<std::size_t shapeExt>
      SpmtMemoryLayout(std::size_t shapeRank, std::size_t targetCount, const afft::spmt::MemoryLayout<shapeExt>& memLayout)
      : mShapeRank{shapeRank},
        mData(targetCount),
        mBlockViews(targetCount)
      {
        if (const auto& srcBlocks = memLayout.srcBlocks; srcBlocks.empty())
        {
          mHasDefaultSrcMemoryBlocks = true;
        }
        else if (srcBlocks.size() == targetCount)
        {
          mHasDefaultSrcMemoryBlocks = false;

          for (std::size_t i{}; i < targetCount; ++i)
          {
            if (srcBlocks[i].starts.size() == shapeRank)
            {
              std::copy(srcBlocks[i].starts.begin(), srcBlocks[i].starts.end(), mData[i].mSrcStarts.begin());
            }
            else
            {
              throw std::invalid_argument("Invalid source memory blocks starts size");
            }

            if (srcBlocks[i].sizes.size() == shapeRank)
            {
              std::copy(srcBlocks[i].sizes.begin(), srcBlocks[i].sizes.end(), mData[i].mSrcSizes.begin());
            }
            else
            {
              throw std::invalid_argument("Invalid source memory blocks sizes size");
            }

            if (srcBlocks[i].strides.size() == shapeRank)
            {
              std::copy(srcBlocks[i].strides.begin(), srcBlocks[i].strides.end(), mData[i].mSrcStrides.begin());
            }
            else if (srcBlocks[i].strides.empty())
            {
              mData[i].mHasDefaultSrcStrides = true;
            }
            else
            {
              throw std::invalid_argument("Invalid source memory blocks strides size");
            }
          }
        }
        else
        {
          throw std::invalid_argument("Invalid source memory blocks size");
        }

        if (const auto& dstBlocks = memLayout.dstBlocks; dstBlocks.empty())
        {
          mHasDefaultDstMemoryBlocks = true;
        }
        else if (dstBlocks.size() == targetCount)
        {
          mHasDefaultDstMemoryBlocks = false;

          for (std::size_t i{}; i < targetCount; ++i)
          {
            if (dstBlocks[i].starts.size() == shapeRank)
            {
              std::copy(dstBlocks[i].starts.begin(), dstBlocks[i].starts.end(), mData[i].mDstStarts.begin());
            }
            else
            {
              throw std::invalid_argument("Invalid destination memory blocks starts size");
            }

            if (dstBlocks[i].sizes.size() == shapeRank)
            {
              std::copy(dstBlocks[i].sizes.begin(), dstBlocks[i].sizes.end(), mData[i].mDstSizes.begin());
            }
            else
            {
              throw std::invalid_argument("Invalid destination memory blocks sizes size");
            }

            if (dstBlocks[i].strides.size() == shapeRank)
            {
              std::copy(dstBlocks[i].strides.begin(), dstBlocks[i].strides.end(), mData[i].mDstStrides.begin());
            }
            else if (dstBlocks[i].strides.empty())
            {
              mData[i].mHasDefaultDstStrides = true;
            }
            else
            {
              throw std::invalid_argument("Invalid destination memory blocks strides size");
            }
          }
        }
        else
        {
          throw std::invalid_argument("Invalid destination memory blocks size");
        }

        if (const auto& srcAxesOrder = memLayout.srcAxesOrder; srcAxesOrder.empty())
        {
          mHasDefaultSrcAxesOrder = true;
          std::iota(mSrcAxesOrder.begin(), mSrcAxesOrder.end(), 0);
        }
        else if (srcAxesOrder.size() == shapeRank)
        {
          mHasDefaultSrcAxesOrder = false;
          std::copy(srcAxesOrder.begin(), srcAxesOrder.end(), mSrcAxesOrder.begin());
        }
        else
        {
          throw std::invalid_argument("Invalid source axes order size");
        }

        if (const auto& dstAxesOrder = memLayout.dstAxesOrder; dstAxesOrder.empty())
        {
          mHasDefaultDstAxesOrder = true;
          std::iota(mDstAxesOrder.begin(), mDstAxesOrder.end(), 0);
        }
        else if (dstAxesOrder.size() == shapeRank)
        {
          mHasDefaultDstAxesOrder = false;
          std::copy(dstAxesOrder.begin(), dstAxesOrder.end(), mDstAxesOrder.begin());
        }
        else
        {
          throw std::invalid_argument("Invalid destination axes order size");
        }

        for (std::size_t i{}; i < targetCount; ++i)
        {
          mBlockViews[i] = MemoryBlock<>{View<std::size_t>{mData[i].mSrcStarts.data(), shapeRank},
                                         View<std::size_t>{mData[i].mSrcSizes.data(), shapeRank},
                                         View<std::size_t>{mData[i].mSrcStrides.data(), shapeRank}};

          mBlockViews[i + targetCount] = MemoryBlock<>{View<std::size_t>{mData[i].mDstStarts.data(), shapeRank},
                                                       View<std::size_t>{mData[i].mDstSizes.data(), shapeRank},
                                                       View<std::size_t>{mData[i].mDstStrides.data(), shapeRank}};
        }
      }

      /// @brief Copy constructor.
      SpmtMemoryLayout(const SpmtMemoryLayout&) = default;

      /// @brief Move constructor.
      SpmtMemoryLayout(SpmtMemoryLayout&&) = default;

      /// @brief Destructor.
      ~SpmtMemoryLayout() = default;

      /// @brief Copy assignment operator.
      SpmtMemoryLayout& operator=(const SpmtMemoryLayout&) = default;

      /// @brief Move assignment operator.
      SpmtMemoryLayout& operator=(SpmtMemoryLayout&&) = default;

      /**
       * @brief Check if the target has default source memory blocks.
       * @return True if the target has default source memory blocks, false otherwise.
       */
      [[nodiscard]] constexpr bool hasDefaultSrcMemoryBlocks() const noexcept
      {
        return mHasDefaultSrcMemoryBlocks;
      }

      /**
       * @brief Check if the target has default destination memory blocks.
       * @return True if the target has default destination memory blocks, false otherwise.
       */
      [[nodiscard]] constexpr bool hasDefaultDstMemoryBlocks() const noexcept
      {
        return mHasDefaultDstMemoryBlocks;
      }

      /**
       * @brief Check if the target has default source axes order.
       * @return True if the target has default source axes order, false otherwise.
       */
      [[nodiscard]] constexpr bool hasDefaultSrcAxesOrder() const noexcept
      {
        return mHasDefaultSrcAxesOrder;
      }

      /**
       * @brief Check if the target has default destination axes order.
       * @return True if the target has default destination axes order, false otherwise.
       */
      [[nodiscard]] constexpr bool hasDefaultDstAxesOrder() const noexcept
      {
        return mHasDefaultDstAxesOrder;
      }

      /**
       * @brief Check if the target has default source strides.
       * @param targetIndex Target index.
       * @return True if the target has default source strides, false otherwise.
       */
      [[nodiscard]] bool hasDefaultSrcStrides(std::size_t targetIndex) const
      {
        return mData.at(targetIndex).mHasDefaultSrcStrides;
      }

      /**
       * @brief Check if the target has default destination strides.
       * @param targetIndex Target index.
       * @return True if the target has default destination strides, false otherwise.
       */
      [[nodiscard]] bool hasDefaultDstStrides(std::size_t targetIndex) const
      {
        return mData.at(targetIndex).mHasDefaultDstStrides;
      }

      /**
       * @brief Get the source starts.
       * @param targetIndex Target index.
       * @return Source starts.
       */
      [[nodiscard]] View<std::size_t> getSrcStarts(std::size_t targetIndex) const
      {
        return View<std::size_t>{mData.at(targetIndex).mSrcStarts.data(), mShapeRank};
      }

      /**
       * @brief Get the source sizes.
       * @param targetIndex Target index.
       * @return Source sizes.
       */
      [[nodiscard]] View<std::size_t> getSrcSizes(std::size_t targetIndex) const
      {
        return View<std::size_t>{mData.at(targetIndex).mSrcSizes.data(), mShapeRank};
      }

      /**
       * @brief Get the source strides.
       * @param targetIndex Target index.
       * @return Source strides.
       */
      [[nodiscard]] View<std::size_t> getSrcStrides(std::size_t targetIndex) const
      {
        return View<std::size_t>{mData.at(targetIndex).mSrcStrides.data(), mShapeRank};
      }

      /**
       * @brief Get the destination starts.
       * @param targetIndex Target index.
       * @return Destination starts.
       */
      [[nodiscard]] View<std::size_t> getDstStarts(std::size_t targetIndex) const
      {
        return View<std::size_t>{mData.at(targetIndex).mDstStarts.data(), mShapeRank};
      }

      /**
       * @brief Get the destination sizes.
       * @param targetIndex Target index.
       * @return Destination sizes.
       */
      [[nodiscard]] View<std::size_t> getDstSizes(std::size_t targetIndex) const
      {
        return View<std::size_t>{mData.at(targetIndex).mDstSizes.data(), mShapeRank};
      }

      /**
       * @brief Get the destination strides.
       * @param targetIndex Target index.
       * @return Destination strides.
       */
      [[nodiscard]] View<std::size_t> getDstStrides(std::size_t targetIndex) const
      {
        return View<std::size_t>{mData.at(targetIndex).mDstStrides.data(), mShapeRank};
      }

      /**
       * @brief Get the source starts.
       * @tparam I Integral type.
       * @param targetIndex Target index.
       * @return Source starts.
       */
      template<typename I>
      [[nodiscard]] MaxDimArray<I> getSrcStarts(std::size_t targetIndex) const
      {
        static_assert(std::is_integral_v<I>, "I must be an integral type");

        return mData.at(targetIndex).mSrcStarts.cast<I>();
      }

      /**
       * @brief Get the source sizes.
       * @tparam I Integral type.
       * @param targetIndex Target index.
       * @return Source sizes.
       */
      template<typename I>
      [[nodiscard]] MaxDimArray<I> getSrcSizes(std::size_t targetIndex) const
      {
        static_assert(std::is_integral_v<I>, "I must be an integral type");

        return mData.at(targetIndex).mSrcSizes.cast<I>();
      }

      /**
       * @brief Get the source strides.
       * @tparam I Integral type.
       * @param targetIndex Target index.
       * @return Source strides.
       */
      template<typename I>
      [[nodiscard]] MaxDimArray<I> getSrcStrides(std::size_t targetIndex) const
      {
        static_assert(std::is_integral_v<I>, "I must be an integral type");

        return mData.at(targetIndex).mSrcStrides.cast<I>();
      }

      /**
       * @brief Get the destination starts.
       * @tparam I Integral type.
       * @param targetIndex Target index.
       * @return Destination starts.
       */
      template<typename I>
      [[nodiscard]] MaxDimArray<I> getDstStarts(std::size_t targetIndex) const
      {
        static_assert(std::is_integral_v<I>, "I must be an integral type");

        return mData.at(targetIndex).mDstStarts.cast<I>();
      }

      /**
       * @brief Get the destination sizes.
       * @tparam I Integral type.
       * @param targetIndex Target index.
       * @return Destination sizes.
       */
      template<typename I>
      [[nodiscard]] MaxDimArray<I> getDstSizes(std::size_t targetIndex) const
      {
        static_assert(std::is_integral_v<I>, "I must be an integral type");

        return mData.at(targetIndex).mDstSizes.cast<I>();
      }

      /**
       * @brief Get the destination strides.
       * @tparam I Integral type.
       * @param targetIndex Target index.
       * @return Destination strides.
       */
      template<typename I>
      [[nodiscard]] MaxDimArray<I> getDstStrides(std::size_t targetIndex) const
      {
        static_assert(std::is_integral_v<I>, "I must be an integral type");

        return mData.at(targetIndex).mDstStrides.cast<I>();
      }

      /**
       * @brief Get the writable source starts. Should be used carefully.
       * @param targetIndex Target index.
       * @return Source starts.
       */
      [[nodiscard]] Span<std::size_t> getSrcStartsWritable(std::size_t targetIndex) noexcept
      {
        return Span<std::size_t>{mData.at(targetIndex).mSrcStarts.data(), mShapeRank};
      }

      /**
       * @brief Get the writable source sizes. Should be used carefully.
       * @param targetIndex Target index.
       * @return Source sizes.
       */
      [[nodiscard]] Span<std::size_t> getSrcSizesWritable(std::size_t targetIndex) noexcept
      {
        return Span<std::size_t>{mData.at(targetIndex).mSrcSizes.data(), mShapeRank};
      }

      /**
       * @brief Get the writable source strides. Should be used carefully.
       * @param targetIndex Target index.
       * @return Source strides.
       */
      [[nodiscard]] Span<std::size_t> getSrcStridesWritable(std::size_t targetIndex) noexcept
      {
        return Span<std::size_t>{mData.at(targetIndex).mSrcStrides.data(), mShapeRank};
      }

      /**
       * @brief Get the writable destination starts. Should be used carefully.
       * @param targetIndex Target index.
       * @return Destination starts.
       */
      [[nodiscard]] Span<std::size_t> getDstStartsWritable(std::size_t targetIndex) noexcept
      {
        return Span<std::size_t>{mData.at(targetIndex).mDstStarts.data(), mShapeRank};
      }

      /**
       * @brief Get the writable destination sizes. Should be used carefully.
       * @param targetIndex Target index.
       * @return Destination sizes.
       */
      [[nodiscard]] Span<std::size_t> getDstSizesWritable(std::size_t targetIndex) noexcept
      {
        return Span<std::size_t>{mData.at(targetIndex).mDstSizes.data(), mShapeRank};
      }

      /**
       * @brief Get the writable destination strides. Should be used carefully.
       * @param targetIndex Target index.
       * @return Destination strides.
       */
      [[nodiscard]] Span<std::size_t> getDstStridesWritable(std::size_t targetIndex) noexcept
      {
        return Span<std::size_t>{mData.at(targetIndex).mDstStrides.data(), mShapeRank};
      }

      /**
       * @brief Get the source axes order.
       * @return Source axes order.
       */
      [[nodiscard]] View<std::size_t> getSrcAxesOrder() const
      {
        return View<std::size_t>{mSrcAxesOrder.data(), mShapeRank};
      }

      /**
       * @brief Get the destination axes order.
       * @return Destination axes order.
       */
      [[nodiscard]] View<std::size_t> getDstAxesOrder() const
      {
        return View<std::size_t>{mDstAxesOrder.data(), mShapeRank};
      }

      /**
       * @brief Get the source axes order.
       * @tparam I Integral type.
       * @return Source axes order.
       */
      [[nodiscard]] afft::spmt::MemoryLayout<> getView() const
      {
        afft::spmt::MemoryLayout<> memLayout;
        memLayout.srcBlocks    = View<MemoryBlock<>>{mBlockViews.data(), mBlockViews.size() / 2};
        memLayout.dstBlocks    = View<MemoryBlock<>>{mBlockViews.data() + mBlockViews.size() / 2, mBlockViews.size() / 2};
        memLayout.srcAxesOrder = getSrcAxesOrder();
        memLayout.dstAxesOrder = getDstAxesOrder();
        return memLayout;
      }
    private:
      /// @brief Per target data.
      struct PerTargetData
      {
        MaxDimArray<std::size_t> mSrcStarts{};            ///< Source starts.
        MaxDimArray<std::size_t> mSrcSizes{};             ///< Source sizes.
        MaxDimArray<std::size_t> mSrcStrides{};           ///< Source strides.
        MaxDimArray<std::size_t> mDstStarts{};            ///< Destination starts.
        MaxDimArray<std::size_t> mDstSizes{};             ///< Destination sizes.
        MaxDimArray<std::size_t> mDstStrides{};           ///< Destination strides.
        bool                     mHasDefaultSrcStrides{}; ///< Has default source strides.
        bool                     mHasDefaultDstStrides{}; ///< Has default destination strides.
      };

      std::size_t                mShapeRank{};                 ///< Shape rank.
      std::vector<PerTargetData> mData{};                      ///< Data.
      std::vector<MemoryBlock<>> mBlockViews{};                ///< Block views.
      MaxDimArray<std::size_t>   mSrcAxesOrder{};              ///< Source axes order.
      MaxDimArray<std::size_t>   mDstAxesOrder{};              ///< Destination axes order;
      bool                       mHasDefaultSrcMemoryBlocks{}; ///< Has default memory layout.
      bool                       mHasDefaultDstMemoryBlocks{}; ///< Has default memory layout.
      bool                       mHasDefaultSrcAxesOrder{};    ///< Has default source axes order.
      bool                       mHasDefaultDstAxesOrder{};    ///< Has default destination axes order.
  };

  /// @brief Memory layout for mpst targets.
  class MpstMemoryLayout
  {
    public:
      /// @brief Default constructor.
      MpstMemoryLayout() = default;

      /**
       * @brief Constructor.
       * @tparam shapeExt Extent of the shape.
       * @param shapeRank Shape rank.
       * @param memLayout Memory layout.
       */
      template<std::size_t shapeExt>
      MpstMemoryLayout(std::size_t shapeRank, const afft::mpst::MemoryLayout<shapeExt>& memLayout)
      : mShapeRank{shapeRank},
        mHasDefaultSrcStrides{memLayout.srcBlock.starts.empty() && memLayout.srcBlock.sizes.empty()},
        mHasDefaultDstStrides{memLayout.dstBlock.starts.empty() && memLayout.dstBlock.sizes.empty()},
        mHasDefaultSrcAxesOrder{memLayout.srcAxesOrder.empty()},
        mHasDefaultDstStrides{memLayout.dstBlock.strides.empty()},
        mHasDefaultSrcAxesOrder{memLayout.srcAxesOrder.empty()},
        mHasDefaultDstAxesOrder{memLayout.dstAxesOrder.empty()}
      {
        if (memLayout.srcBlock.starts.size() == shapeRank)
        {
          std::copy(memLayout.srcBlock.starts.begin(), memLayout.srcBlock.starts.end(), mSrcStarts.begin());
        }
        else
        {
          throw std::invalid_argument("Invalid source memory block starts size");
        }

        if (memLayout.srcBlock.sizes.size() == shapeRank)
        {
          std::copy(memLayout.srcBlock.sizes.begin(), memLayout.srcBlock.sizes.end(), mSrcSizes.begin());
        }
        else
        {
          throw std::invalid_argument("Invalid source memory block sizes size");
        }

        if (const auto& srcStrides = memLayout.srcBlock.strides; srcStrides.empty())
        {
          mHasDefaultSrcStrides = true;
        }
        else if (srcStrides.size() == shapeRank)
        {
          mHasDefaultSrcStrides = false;
          std::copy(srcStrides.begin(), srcStrides.end(), mSrcStrides.begin());
        }
        else
        {
          throw std::invalid_argument("Invalid source memory block strides size");
        }

        if (memLayout.dstBlock.starts.size() == shapeRank)
        {
          std::copy(memLayout.dstBlock.starts.begin(), memLayout.dstBlock.starts.end(), mDstStarts.begin());
        }
        else
        {
          throw std::invalid_argument("Invalid destination memory block starts size");
        }

        if (memLayout.dstBlock.sizes.size() == shapeRank)
        {
          std::copy(memLayout.dstBlock.sizes.begin(), memLayout.dstBlock.sizes.end(), mDstSizes.begin());
        }
        else
        {
          throw std::invalid_argument("Invalid destination memory block sizes size");
        }

        if (const auto& dstStrides = memLayout.dstBlock.strides; dstStrides.empty())
        {
          mHasDefaultDstStrides = true;
          std::iota(mDstStrides.begin(), mDstStrides.end(), 0);
        }
        else if (dstStrides.size() == shapeRank)
        {
          mHasDefaultDstStrides = false;
          std::copy(dstStrides.begin(), dstStrides.end(), mDstStrides.begin());
        }
        else
        {
          throw std::invalid_argument("Invalid destination memory block strides size");
        }

        if (const auto& srcAxesOrder = memLayout.srcAxesOrder; srcAxesOrder.empty())
        {
          mHasDefaultSrcAxesOrder = true;
          std::iota(mSrcAxesOrder.begin(), mSrcAxesOrder.end(), 0);
        }
        else if (srcAxesOrder.size() == shapeRank)
        {
          mHasDefaultSrcAxesOrder = false;
          std::copy(srcAxesOrder.begin(), srcAxesOrder.end(), mSrcAxesOrder.begin());
        }
        else
        {
          throw std::invalid_argument("Invalid source axes order size");
        }

        if (const auto& dstAxesOrder = memLayout.dstAxesOrder; dstAxesOrder.empty())
        {
          mHasDefaultDstAxesOrder = true;
          std::iota(mDstAxesOrder.begin(), mDstAxesOrder.end(), 0);
        }
        else if (dstAxesOrder.size() == shapeRank)
        {
          mHasDefaultDstAxesOrder = false;
          std::copy(dstAxesOrder.begin(), dstAxesOrder.end(), mDstAxesOrder.begin());
        }
        else
        {
          throw std::invalid_argument("Invalid destination axes order size");
        }
      }

      /// @brief Copy constructor.
      MpstMemoryLayout(const MpstMemoryLayout&) = default;

      /// @brief Move constructor.
      MpstMemoryLayout(MpstMemoryLayout&&) = default;

      /// @brief Destructor.
      ~MpstMemoryLayout() = default;

      /// @brief Copy assignment operator.
      MpstMemoryLayout& operator=(const MpstMemoryLayout&) = default;

      /// @brief Move assignment operator.
      MpstMemoryLayout& operator=(MpstMemoryLayout&&) = default;

      /**
       * @brief Check if the memory layout has default memory block.
       * @return True if the memory layout has default memory block, false otherwise.
       */
      [[nodiscard]] constexpr bool hasDefaultSrcMemoryBlock() const noexcept
      {
        return mHasDefaultSrcMemoryBlocks;
      }

      /**
       * @brief Check if the memory layout has default memory block.
       * @return True if the memory layout has default memory block, false otherwise.
       */
      [[nodiscard]] constexpr bool hasDefaultDstMemoryBlock() const noexcept
      {
        return mHasDefaultDstMemoryBlocks;
      }

      /**
       * @brief Check if the memory layout has default source strides.
       * @return True if the memory layout has default source strides, false otherwise.
       */
      [[nodiscard]] constexpr bool hasDefaultSrcStrides() const noexcept
      {
        return mHasDefaultSrcStrides;
      }

      /**
       * @brief Check if the memory layout has default destination strides.
       * @return True if the memory layout has default destination strides, false otherwise.
       */
      [[nodiscard]] constexpr bool hasDefaultDstStrides() const noexcept
      {
        return mHasDefaultDstStrides;
      }

      /**
       * @brief Check if the memory layout has default source axes order.
       * @return True if the memory layout has default source axes order, false otherwise.
       */
      [[nodiscard]] constexpr bool hasDefaultSrcAxesOrder() const noexcept
      {
        return mHasDefaultSrcAxesOrder;
      }

      /**
       * @brief Check if the memory layout has default destination axes order.
       * @return True if the memory layout has default destination axes order, false otherwise.
       */
      [[nodiscard]] constexpr bool hasDefaultDstAxesOrder() const noexcept
      {
        return mHasDefaultDstAxesOrder;
      }

      /**
       * @brief Get the source starts.
       * @return Source starts.
       */
      [[nodiscard]] constexpr View<std::size_t> getSrcStarts() const noexcept
      {
        return View<std::size_t>{mSrcStarts.data(), mShapeRank};
      }

      /**
       * @brief Get the source sizes.
       * @return Source sizes.
       */
      [[nodiscard]] constexpr View<std::size_t> getSrcSizes() const noexcept
      {
        return View<std::size_t>{mSrcSizes.data(), mShapeRank};
      }

      /**
       * @brief Get the source strides.
       * @return Source strides.
       */
      [[nodiscard]] constexpr View<std::size_t> getSrcStrides() const noexcept
      {
        return View<std::size_t>{mSrcStrides.data(), mShapeRank};
      }

      /**
       * @brief Get the destination starts.
       * @return Destination starts.
       */
      [[nodiscard]] constexpr View<std::size_t> getDstStarts() const noexcept
      {
        return View<std::size_t>{mDstStarts.data(), mShapeRank};
      }

      /**
       * @brief Get the destination sizes.
       * @return Destination sizes.
       */
      [[nodiscard]] constexpr View<std::size_t> getDstSizes() const noexcept
      {
        return View<std::size_t>{mDstSizes.data(), mShapeRank};
      }

      /**
       * @brief Get the destination strides.
       * @return Destination strides.
       */
      [[nodiscard]] constexpr View<std::size_t> getDstStrides() const noexcept
      {
        return View<std::size_t>{mDstStrides.data(), mShapeRank};
      }

      /**
       * @brief Get the source axes order.
       * @return Source axes order.
       */
      [[nodiscard]] constexpr View<std::size_t> getSrcAxesOrder() const noexcept
      {
        return View<std::size_t>{mSrcAxesOrder.data(), mShapeRank};
      }

      /**
       * @brief Get the destination axes order.
       * @return Destination axes order.
       */
      [[nodiscard]] constexpr View<std::size_t> getDstAxesOrder() const noexcept
      {
        return View<std::size_t>{mDstAxesOrder.data(), mShapeRank};
      }

      /**
       * @brief Get the source starts.
       * @tparam I Integral type.
       * @return Source starts.
       */
      template<typename I>
      [[nodiscard]] constexpr MaxDimArray<I> getSrcStarts() const
      {
        return mSrcStarts.cast<I>();
      }

      /**
       * @brief Get the source sizes.
       * @tparam I Integral type.
       * @return Source sizes.
       */
      template<typename I>
      [[nodiscard]] constexpr MaxDimArray<I> getSrcSizes() const
      {
        return mSrcSizes.cast<I>();
      }

      /**
       * @brief Get the source strides.
       * @tparam I Integral type.
       * @return Source strides.
       */
      template<typename I>
      [[nodiscard]] constexpr MaxDimArray<I> getSrcStrides() const
      {
        return mSrcStrides.cast<I>();
      }

      /**
       * @brief Get the destination starts.
       * @tparam I Integral type.
       * @return Destination starts.
       */
      template<typename I>
      [[nodiscard]] constexpr MaxDimArray<I> getDstStarts() const
      {
        return mDstStarts.cast<I>();
      }

      /**
       * @brief Get the destination sizes.
       * @tparam I Integral type.
       * @return Destination sizes.
       */
      template<typename I>
      [[nodiscard]] constexpr MaxDimArray<I> getDstSizes() const
      {
        return mDstSizes.cast<I>();
      }

      /**
       * @brief Get the destination strides.
       * @tparam I Integral type.
       * @return Destination strides.
       */
      template<typename I>
      [[nodiscard]] constexpr MaxDimArray<I> getDstStrides() const
      {
        return mDstStrides.cast<I>();
      }

      /**
       * @brief Get the writable source starts. Should be used carefully.
       * @return Source starts.
       */
      [[nodiscard]] constexpr Span<std::size_t> getSrcStartsWritable() noexcept
      {
        return Span<std::size_t>{mSrcStarts.data(), mShapeRank};
      }

      /**
       * @brief Get the writable source sizes. Should be used carefully.
       * @return Source sizes.
       */
      [[nodiscard]] constexpr Span<std::size_t> getSrcSizesWritable() noexcept
      {
        return Span<std::size_t>{mSrcSizes.data(), mShapeRank};
      }

      /**
       * @brief Get the writable source strides. Should be used carefully.
       * @return Source strides.
       */
      [[nodiscard]] constexpr Span<std::size_t> getSrcStridesWritable() noexcept
      {
        return Span<std::size_t>{mSrcStrides.data(), mShapeRank};
      }

      /**
       * @brief Get the writable destination starts. Should be used carefully.
       * @return Destination starts.
       */
      [[nodiscard]] constexpr Span<std::size_t> getDstStartsWritable() noexcept
      {
        return Span<std::size_t>{mDstStarts.data(), mShapeRank};
      }

      /**
       * @brief Get the writable destination sizes. Should be used carefully.
       * @return Destination sizes.
       */
      [[nodiscard]] constexpr Span<std::size_t> getDstSizesWritable() noexcept
      {
        return Span<std::size_t>{mDstSizes.data(), mShapeRank};
      }

      /**
       * @brief Get the writable destination strides. Should be used carefully.
       * @return Destination strides.
       */
      [[nodiscard]] constexpr Span<std::size_t> getDstStridesWritable() noexcept
      {
        return Span<std::size_t>{mDstStrides.data(), mShapeRank};
      }

      /**
       * @brief Get the source axes order.
       * @return Source axes order.
       */
      [[nodiscard]] afft::mpst::MemoryLayout<> getView() const
      {
        afft::mpst::MemoryLayout<> memLayout;
        memLayout.srcBlock     = MemoryBlock<>{getSrcStarts(), getSrcSizes(), getSrcStrides()};
        memLayout.dstBlock     = MemoryBlock<>{getDstStarts(), getDstSizes(), getDstStrides()};
        memLayout.srcAxesOrder = getSrcAxesOrder();
        memLayout.dstAxesOrder = getDstAxesOrder();
        return memLayout;
      }
    private:
      std::size_t              mShapeRank{};                 ///< Shape rank.
      MaxDimArray<std::size_t> mSrcStarts{};                 ///< Source starts.
      MaxDimArray<std::size_t> mSrcSizes{};                  ///< Source sizes.
      MaxDimArray<std::size_t> mSrcStrides{};                ///< Source strides.
      MaxDimArray<std::size_t> mDstStarts{};                 ///< Destination starts.
      MaxDimArray<std::size_t> mDstSizes{};                  ///< Destination sizes.
      MaxDimArray<std::size_t> mDstStrides{};                ///< Destination strides.
      MaxDimArray<std::size_t> mSrcAxesOrder{};              ///< Source axes order.
      MaxDimArray<std::size_t> mDstAxesOrder{};              ///< Destination axes order;
      bool                     mHasDefaultSrcMemoryBlocks{}; ///< Has default memory layout.
      bool                     mHasDefaultDstMemoryBlocks{}; ///< Has default memory layout.
      bool                     mHasDefaultSrcStrides{};      ///< Has default source strides.
      bool                     mHasDefaultDstStrides{};      ///< Has default destination strides.
      bool                     mHasDefaultSrcAxesOrder{};    ///< Has default source axes order.
      bool                     mHasDefaultDstAxesOrder{};    ///< Has default destination axes order.
  };

  /// @brief Describes the spst cpu target.
  struct SpstCpuDesc
  {
    SpstMemoryLayout memoryLayout{}; ///< Memory layout.
    Alignment        alignment{};    ///< Alignment.
    unsigned         threadLimit{};  ///< Thread limit.
  };

  /// @brief Describes the spst gpu target.
  struct SpstGpuDesc
  {
    SpstMemoryLayout memoryLayout{}; ///< Memory layout.
# if AFFT_GPU_BACKEND_IS(CUDA)
    int              device{};       ///< CUDA device.
# elif AFFT_GPU_BACKEND_IS(HIP)
    int              device{};       ///< HIP device.
# elif AFFT_GPU_BACKEND_IS(OPENCL)
    cl_context       context{};      ///< OpenCL context.
    cl_device_id     device{};       ///< OpenCL device.
# endif
  };

  /// @brief Describes the spmt gpu architecture.
  struct SpmtGpuDesc
  {
    SpmtMemoryLayout          memoryLayout{}; ///< Memory layout.
# if AFFT_GPU_BACKEND_IS(CUDA)
    std::vector<int>          devices{};      ///< CUDA devices.
# elif AFFT_GPU_BACKEND_IS(HIP)
    std::vector<int>          devices{};      ///< HIP devices.
# elif AFFT_GPU_BACKEND_IS(OPENCL)
    cl_context                context{};      ///< OpenCL context.
    std::vector<cl_device_id> devices{};      ///< OpenCL devices.
# endif
  };

  /// @brief Describes the mpst cpu architecture.
  struct MpstCpuDesc
  {
    MpstMemoryLayout memoryLayout{}; ///< Memory layout.
# if AFFT_MP_BACKEND_IS(MPI)
    MPI_Comm         comm{};         ///< MPI communicator.
# endif
    Alignment        alignment{};    ///< Alignment.
    unsigned         threadLimit{};  ///< Thread limit.
  };

  /// @brief Describes the mpst gpu architecture.
  struct MpstGpuDesc
  {
    MpstMemoryLayout memoryLayout{}; ///< Memory layout.
# if AFFT_MP_BACKEND_IS(MPI)
    MPI_Comm         comm{};         ///< MPI communicator.
# endif
# if AFFT_GPU_BACKEND_IS(CUDA)
    int              device{};       ///< CUDA device.
# elif AFFT_GPU_BACKEND_IS(HIP)
    int              device{};       ///< HIP device.
# elif AFFT_GPU_BACKEND_IS(OPENCL)
    cl_context       context{};      ///< OpenCL context.
    cl_device_id     device{};       ///< OpenCL device.
# endif
  };

  /// @brief Architecture description.
  class ArchDesc
  {
    public:
      /// @brief Default constructor is deleted.
      ArchDesc() = delete;

      /**
       * @brief Constructor.
       * @tparam ArchParamsT Architecture parameters type.
       * @param archParams Architecture parameters.
       * @param shapeRank Shape rank.
       */
      template<typename ArchParamsT>
      ArchDesc(const ArchParamsT& archParams, std::size_t shapeRank)
      : mComplexFormat{validateAndReturn(archParams.complexFormat)},
        mPreserveSource{archParams.preserveSource},
        mUseExternalWorkspace{archParams.useExternalWorkspace},
        mArchVariant{makeArchVariant(archParams, shapeRank)}
      {
        static_assert(isArchitectureParameters<ArchParamsT>, "Invalid architecture parameters");
      }

      /// @brief Copy constructor.
      ArchDesc(const ArchDesc&) = default;

      /// @brief Move constructor.
      ArchDesc(ArchDesc&&) = default;

      /// @brief Destructor.
      ~ArchDesc() = default;
      
      /// @brief Copy assignment operator.
      ArchDesc& operator=(const ArchDesc&) = default;

      /// @brief Move assignment operator.
      ArchDesc& operator=(ArchDesc&&) = default;

      /**
       * @brief Get the target.
       * @return Target.
       */
      [[nodiscard]] constexpr Target getTarget() const
      {
        switch (mArchVariant.index())
        {
        case ArchVariantIdx::spstCpu:
        case ArchVariantIdx::mpstCpu:
          return Target::cpu;
        case ArchVariantIdx::spstGpu:
        case ArchVariantIdx::spmtGpu:
        case ArchVariantIdx::mpstGpu:
          return Target::gpu;
        default:
          cxx::unreachable();
        }
      }

      /**
       * @brief Get the target count.
       * @return Target count.
       */
      [[nodiscard]] constexpr std::size_t getTargetCount() const
      {
        switch (mArchVariant.index())
        {
        case ArchVariantIdx::spstCpu:
        case ArchVariantIdx::spstGpu:
        case ArchVariantIdx::mpstCpu:
        case ArchVariantIdx::mpstGpu:
          return 1;
        case ArchVariantIdx::spmtGpu:
#       if AFFT_GPU_BACKEND_IS(CUDA) || AFFT_GPU_BACKEND_IS(HIP)
          return std::get<SpmtGpuDesc>(mArchVariant).devices.size();
#       else
          return 0;
#       endif
        default:
          cxx::unreachable();
        }
      }

      /**
       * @brief Get the distribution.
       * @return Distribution.
       */
      [[nodiscard]] constexpr Distribution getDistribution() const
      {
        switch (mArchVariant.index())
        {
        case ArchVariantIdx::spstCpu:
        case ArchVariantIdx::spstGpu:
          return Distribution::spst;
        case ArchVariantIdx::spmtGpu:
          return Distribution::spmt;
        case ArchVariantIdx::mpstCpu:
        case ArchVariantIdx::mpstGpu:
          return Distribution::mpst;
        default:
          cxx::unreachable();
        }
      }

      /**
       * @brief Get the architecture description.
       * @tparam target Target.
       * @tparam distrib Distribution.
       * @return Architecture description.
       */
      template<Target target, Distribution distrib>
      [[nodiscard]] constexpr auto& getArchDesc()
      {
        static_assert(isValid(target), "Invalid target");
        static_assert(isValid(distrib), "Invalid distribution");
        static_assert(!(target == Target::cpu && distrib == Distribution::spmt), "Invalid distribution for cpu target");

        if constexpr (target == Target::cpu)
        {
          if constexpr (distrib == Distribution::spst)
          {
            return std::get<ArchVariantIdx::spstCpu>(mArchVariant);
          }
          else if constexpr (distrib == Distribution::mpst)
          {
            return std::get<ArchVariantIdx::mpstCpu>(mArchVariant);
          }
        }
        else if constexpr (target == Target::gpu)
        {
          if constexpr (distrib == Distribution::spst)
          {
            return std::get<ArchVariantIdx::spstGpu>(mArchVariant);
          }
          else if constexpr (distrib == Distribution::spmt)
          {
            return std::get<ArchVariantIdx::spmtGpu>(mArchVariant);
          }
          else if constexpr (distrib == Distribution::mpst)
          {
            return std::get<ArchVariantIdx::mpstGpu>(mArchVariant);
          }
        }

        cxx::unreachable();
      }

      /**
       * @brief Get the architecture description.
       * @tparam target Target.
       * @tparam distrib Distribution.
       * @return Architecture description.
       */
      template<Target target, Distribution distrib>
      [[nodiscard]] constexpr const auto& getArchDesc() const
      {
        static_assert(isValid(target), "Invalid target");
        static_assert(isValid(distrib), "Invalid distribution");
        static_assert(!(target == Target::cpu && distrib == Distribution::spmt), "Invalid distribution for cpu target");

        if constexpr (target == Target::cpu)
        {
          if constexpr (distrib == Distribution::spst)
          {
            return std::get<ArchVariantIdx::spstCpu>(mArchVariant);
          }
          else if constexpr (distrib == Distribution::mpst)
          {
            return std::get<ArchVariantIdx::mpstCpu>(mArchVariant);
          }
        }
        else if constexpr (target == Target::gpu)
        {
          if constexpr (distrib == Distribution::spst)
          {
            return std::get<ArchVariantIdx::spstGpu>(mArchVariant);
          }
          else if constexpr (distrib == Distribution::spmt)
          {
            return std::get<ArchVariantIdx::spmtGpu>(mArchVariant);
          }
          else if constexpr (distrib == Distribution::mpst)
          {
            return std::get<ArchVariantIdx::mpstGpu>(mArchVariant);
          }
        }

        cxx::unreachable();
      }

      /// @brief Get the external workspace flag.
      [[nodiscard]] constexpr bool useExternalWorkspace() const noexcept
      {
        return mUseExternalWorkspace;
      }

      /// @brief Get the complex format.
      [[nodiscard]] constexpr ComplexFormat getComplexFormat() const noexcept
      {
        return mComplexFormat;
      }

      /// @brief Get the preserve source flag.
      [[nodiscard]] constexpr bool getPreserveSource() const noexcept
      {
        return mPreserveSource;
      }

      /**
       * @brief Reconstruction of the architecture parameters.
       * @tparam target Target.
       * @tparam distrib Distribution.
       * @return Architecture parameters.
       */
      template<Target target, Distribution distrib>
      [[nodiscard]] ArchitectureParameters<target, distrib> getArchitectureParameters() const
      {
        ArchitectureParameters<target, distrib> params{};
        params.complexFormat        = mComplexFormat;
        params.preserveSource       = mPreserveSource;
        params.useExternalWorkspace = mUseExternalWorkspace;

        if constexpr (target == Target::cpu)
        {
          if constexpr (distrib == Distribution::spst)
          {
            const auto& desc = getArchDesc<Target::cpu, Distribution::spst>();
            params.memoryLayout = desc.memoryLayout.getView();
            params.alignment    = desc.alignment;
            params.threadLimit  = desc.threadLimit;
          }
          else if constexpr (distrib == Distribution::mpst)
          {
            const auto& desc = getArchDesc<Target::cpu, Distribution::mpst>();
            params.memoryLayout = desc.memoryLayout.getView();
#         if AFFT_MP_BACKEND_IS(MPI)
            params.communicator = desc.comm;
#         endif
            params.alignment    = desc.alignment;
            params.threadLimit  = desc.threadLimit;
          }
        }
        else if constexpr (target == Target::gpu)
        {
          if constexpr (distrib == Distribution::spst)
          {
            const auto& desc = getArchDesc<Target::gpu, Distribution::spst>();
            params.memoryLayout = desc.memoryLayout.getView();
#         if AFFT_GPU_BACKEND_IS(CUDA)
            params.device = desc.device;
#         elif AFFT_GPU_BACKEND_IS(HIP)
            params.device = desc.device;
#         elif AFFT_GPU_BACKEND_IS(OPENCL)
            params.context = desc.context;
            params.device  = desc.device;
#         endif
          }
          else if constexpr (distrib == Distribution::spmt)
          {
            const auto& desc = getArchDesc<Target::gpu, Distribution::spmt>();
            params.memoryLayout = desc.memoryLayout.getView();
#         if AFFT_GPU_BACKEND_IS(CUDA)
            params.devices = desc.devices;
#         elif AFFT_GPU_BACKEND_IS(HIP)
            params.devices = desc.devices;
#         endif
          }
          else if constexpr (distrib == Distribution::mpst)
          {
            const auto& desc = getArchDesc<Target::gpu, Distribution::mpst>();
            params.memoryLayout = desc.memoryLayout.getView();
#         if AFFT_MP_BACKEND_IS(MPI)
            params.communicator = desc.comm;
#         endif
#         if AFFT_GPU_BACKEND_IS(CUDA)
            params.device = desc.device;
#         elif AFFT_GPU_BACKEND_IS(HIP)
            params.device = desc.device;
#         elif AFFT_GPU_BACKEND_IS(OPENCL)
            params.context = desc.context;
            params.device  = desc.device;
#         endif
          }
        }
        
        return params;
      }

      /**
       * @brief Get the memory layout.
       * @tparam distrib Distribution.
       * @return Memory layout.
       */
      template<Distribution distrib>
      [[nodiscard]] constexpr auto& getMemoryLayout()
      {
        static_assert(isValid(distrib), "Invalid distribution");

        switch (getTarget())
        {
        case Target::cpu:
          if constexpr (distrib != Distribution::spmt)
          {
            return getArchDesc<Target::cpu, distrib>().memoryLayout;
          }
          else
          {
            throw std::invalid_argument{"invalid distribution for cpu target"};
          }
        case Target::gpu:
          return getArchDesc<Target::gpu, distrib>().memoryLayout;
        default:
          cxx::unreachable();
        }
      }

      /**
       * @brief Get the memory layout.
       * @tparam distrib Distribution.
       * @return Memory layout.
       */
      template<Distribution distrib>
      [[nodiscard]] constexpr const auto& getMemoryLayout() const
      {
        static_assert(isValid(distrib), "Invalid distribution");

        switch (getTarget())
        {
        case Target::cpu:
          if constexpr (distrib != Distribution::spmt)
          {
            return getArchDesc<Target::cpu, distrib>().memoryLayout;
          }
          else
          {
            throw std::invalid_argument{"invalid distribution for cpu target"};
          }
        case Target::gpu:
          return getArchDesc<Target::gpu, distrib>().memoryLayout;
        default:
          cxx::unreachable();
        }
      }

    private:
      /// @brief Architecture variant.
      using ArchVariant = std::variant<SpstCpuDesc,
                                       SpstGpuDesc,
                                       SpmtGpuDesc,
                                       MpstCpuDesc,
                                       MpstGpuDesc>;

      /// @brief Architecture variant indices.
      struct ArchVariantIdx
      {
        static constexpr std::size_t spstCpu = 0;
        static constexpr std::size_t spstGpu = 1;
        static constexpr std::size_t spmtGpu = 2;
        static constexpr std::size_t mpstCpu = 3;
        static constexpr std::size_t mpstGpu = 4;
      };

      /**
       * @brief Make architecture variant.
       * @tparam shapeExt Extent of the shape.
       * @param params Architecture parameters.
       * @return Architecture variant.
       */
      template<std::size_t shapeExt>
      [[nodiscard]] static SpstCpuDesc
      makeArchVariant(const spst::cpu::Parameters<shapeExt>& params, std::size_t shapeRank)
      {
        SpstCpuDesc desc{};
        desc.memoryLayout = SpstMemoryLayout{shapeRank, params.memoryLayout};
        desc.alignment    = params.alignment;
        desc.threadLimit  = params.threadLimit;

        return desc;
      }

      /**
       * @brief Make architecture variant.
       * @tparam shapeExt Extent of the shape.
       * @param params Architecture parameters.
       * @return Architecture variant.
       */
      template<std::size_t shapeExt>
      [[nodiscard]] static SpstGpuDesc
      makeArchVariant(const spst::gpu::Parameters<shapeExt>& params, std::size_t shapeRank)
      {
        SpstGpuDesc desc{};
        desc.memoryLayout = SpstMemoryLayout{shapeRank, params.memoryLayout};
#     if AFFT_GPU_BACKEND_IS(CUDA)
        if (!cuda::isValidDevice(params.device))
        {
          throw std::invalid_argument{"invalid CUDA device"};
        }
        desc.device = params.device;
#     elif AFFT_GPU_BACKEND_IS(HIP)
        if (!hip::isValidDevice(params.device))
        {
          throw std::invalid_argument{"invalid CUDA device"};
        }
        desc.device = params.device;
#     elif AFFT_GPU_BACKEND_IS(OPENCL)
        if (!opencl::isValidContext(params.context))
        {
          throw std::invalid_argument{"invalid CUDA device"};
        }
        desc.context = params.context;

        if (!opencl::isValidDevice(params.device))
        {
          throw std::invalid_argument{"invalid CUDA device"};
        }
        desc.device = params.device;
#     endif

        return desc;
      }

      /**
       * @brief Make architecture variant.
       * @tparam shapeExt Extent of the shape.
       * @param params Architecture parameters.
       * @return Architecture variant.
       */
      template<std::size_t shapeExt>
      [[nodiscard]] static SpmtGpuDesc
      makeArchVariant(const spmt::gpu::Parameters<shapeExt>& params, std::size_t shapeRank)
      {
        const std::size_t targetCount = params.devices.size();

        SpmtGpuDesc desc{};
        desc.memoryLayout = SpmtMemoryLayout{shapeRank, params.devices.size(), params.memoryLayout};
#     if AFFT_GPU_BACKEND_IS(CUDA)
        desc.devices.resize(targetCount);
        std::copy(params.devices.begin(), params.devices.end(), desc.devices);
#     elif AFFT_GPU_BACKEND_IS(HIP)
        desc.devices.resize(targetCount);
        std::copy(params.devices.begin(), params.devices.end(), desc.devices);
#     endif

        return desc;
      }

      /**
       * @brief Make architecture variant.
       * @tparam shapeExt Extent of the shape.
       * @param params Architecture parameters.
       * @return Architecture variant.
       */
      template<std::size_t shapeExt>
      [[nodiscard]] static MpstCpuDesc
      makeArchVariant(const mpst::cpu::Parameters<shapeExt>& params, std::size_t shapeRank)
      {
        MpstCpuDesc desc{};
        desc.memoryLayout = MpstMemoryLayout{shapeRank, params.memoryLayout};
#     if AFFT_MP_BACKEND_IS(MPI)
        if (!mpi::isValidComm(params.comm))
        {
          throw std::invalid_argument{"invalid MPI communicator"};
        }
        desc.comm         = params.comm;
#     endif
        desc.alignment    = params.alignment;
        desc.threadLimit  = params.threadLimit;

        return desc;
      }

      /**
       * @brief Make architecture variant.
       * @tparam shapeExt Extent of the shape.
       * @param params Architecture parameters.
       * @return Architecture variant.
       */
      template<std::size_t shapeExt>
      [[nodiscard]] static MpstGpuDesc
      makeArchVariant(const mpst::gpu::Parameters<shapeExt>& params, std::size_t shapeRank)
      {
        MpstGpuDesc desc{};
        desc.memoryLayout = MpstMemoryLayout{shapeRank, params.memoryLayout};
#     if AFFT_MP_BACKEND_IS(MPI)
        if (!mpi::isValidComm(params.comm))
        {
          throw std::invalid_argument{"invalid MPI communicator"};
        }
        desc.comm         = params.comm;
#     endif
#     if AFFT_GPU_BACKEND_IS(CUDA)
        if (!cuda::isValidDevice(params.device))
        {
          throw std::invalid_argument{"invalid CUDA device"};
        }
        desc.device = params.device;
#     elif AFFT_GPU_BACKEND_IS(HIP)
        if (!hip::isValidDevice(params.device))
        {
          throw std::invalid_argument{"invalid CUDA device"};
        }
        desc.device = params.device;
#     elif AFFT_GPU_BACKEND_IS(OPENCL)
        if (!opencl::isValidContext(params.context))
        {
          throw std::invalid_argument{"invalid CUDA device"};
        }
        desc.context = params.context;

        if (!opencl::isValidDevice(params.device))
        {
          throw std::invalid_argument{"invalid CUDA device"};
        }
        desc.device = params.device;
#     endif

        return desc;
      }

      ComplexFormat mComplexFormat{};        ///< Complex format.
      bool          mPreserveSource{};       ///< Preserve source.
      bool          mUseExternalWorkspace{}; ///< Use external workspace.
      ArchVariant   mArchVariant;            ///< Architecture variant.
  };
} // namespace afft::detail

#endif /* AFFT_DETAIL_ARCH_DESC_HPP */
