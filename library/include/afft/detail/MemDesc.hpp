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
      [[nodiscard]] constexpr View<Size> getSrcStrides() const noexcept
      {
        return View<Size>{mSrcStrides.data, mShapeRank};
      }

      /**
       * @brief Get the destination strides.
       * @return Destination strides.
       */
      [[nodiscard]] constexpr View<Size> getDstStrides() const noexcept
      {
        return View<Size>{mDstStrides.data, mShapeRank};
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

        return std::equal(lhsSrcStrides.begin(), lhsSrcStrides.end(), rhsSrcStrides.begin(), rhsSrcStrides.end()) &&
               std::equal(lhsDstStrides.begin(), lhsDstStrides.end(), rhsDstStrides.begin(), rhsDstStrides.end());
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

  struct MemBlockDesc
  {
    MaxDimBuffer<Size> starts{};  ///< Starts.
    MaxDimBuffer<Size> sizes{};   ///< Sizes.
    MaxDimBuffer<Size> strides{}; ///< Strides.
  };

  class DistribMemDesc
  {
    public:
      DistribMemDesc() = default;

      DistribMemDesc(const DistributedMemoryLayout& memLayout,
                     const TransformDesc&           transformDesc,
                     const MpDesc&                  ,
                     const TargetDesc&              targetDesc)
      : mShapeRank{transformDesc.getShapeRank()},
        mTargetCount{targetDesc.getTargetCount()},
        mMemBlockDescs{std::make_unique<MemBlockDesc[]>(mTargetCount * 2)},
        mMemoryBlocks{std::make_unique<MemoryBlock[]>(mTargetCount * 2)},
        mHasDefaultSrcAxesOrder{memLayout.srcAxesOrder == nullptr},
        mHasDefaultDstAxesOrder{memLayout.dstAxesOrder == nullptr},
        mHasDefaultSrcMemoryBlocks{memLayout.srcBlocks == nullptr},
        mHasDefaultDstMemoryBlocks{memLayout.dstBlocks == nullptr}
      {
        
      }

      DistribMemDesc(const afft_DistributedMemoryLayout& memLayout,
                     const TransformDesc&                transformDesc,
                     const MpDesc&                       ,
                     const TargetDesc&                   targetDesc)
      : mShapeRank{transformDesc.getShapeRank()},
        mTargetCount{targetDesc.getTargetCount()},
        mMemBlockDescs{std::make_unique<MemBlockDesc[]>(mTargetCount * 2)},
        mMemoryBlocks{std::make_unique<MemoryBlock[]>(mTargetCount * 2)},
        mHasDefaultSrcAxesOrder{memLayout.srcAxesOrder == nullptr},
        mHasDefaultDstAxesOrder{memLayout.dstAxesOrder == nullptr},
        mHasDefaultSrcMemoryBlocks{memLayout.srcBlocks == nullptr},
        mHasDefaultDstMemoryBlocks{memLayout.dstBlocks == nullptr}
      {
        
      }

      DistribMemDesc(const DistribMemDesc& other)
      : mShapeRank{other.mShapeRank},
        mTargetCount{other.mTargetCount},
        mMemBlockDescs{std::make_unique<MemBlockDesc[]>(mTargetCount * 2)},
        mMemoryBlocks{std::make_unique<MemoryBlock[]>(mTargetCount * 2)},
        mSrcDistribAxes{other.mSrcDistribAxes},
        mDstDistribAxes{other.mDstDistribAxes},
        mSrcAxesOrder{other.mSrcAxesOrder},
        mDstAxesOrder{other.mDstAxesOrder},
        mHasDefaultSrcAxesOrder{other.mHasDefaultSrcAxesOrder},
        mHasDefaultDstAxesOrder{other.mHasDefaultDstAxesOrder},
        mHasDefaultSrcMemoryBlocks{other.mHasDefaultSrcMemoryBlocks},
        mHasDefaultDstMemoryBlocks{other.mHasDefaultDstMemoryBlocks},
        mHasDefaultSrcStrides{other.mHasDefaultSrcStrides},
        mHasDefaultDstStrides{other.mHasDefaultDstStrides}
      {
        std::copy_n(other.mMemBlockDescs.get(), mTargetCount * 2, mMemBlockDescs.get());
        
        // TODO: set memory blocks
      }

      DistribMemDesc(DistribMemDesc&&) = default;

      ~DistribMemDesc() = default;

      DistribMemDesc& operator=(const DistribMemDesc& other)
      {
        if (this != &other)
        {
          const auto oldTargetCount = mTargetCount;

          mShapeRank   = other.mShapeRank;
          mTargetCount = other.mTargetCount;
          if (oldTargetCount < mTargetCount)
          {
            mMemBlockDescs = std::make_unique<MemBlockDesc[]>(mTargetCount * 2);
            mMemoryBlocks  = std::make_unique<MemoryBlock[]>(mTargetCount * 2);
          }
          std::copy_n(other.mMemBlockDescs.get(), mTargetCount * 2, mMemBlockDescs.get());
          mSrcDistribAxes            = other.mSrcDistribAxes;
          mDstDistribAxes            = other.mDstDistribAxes;
          mSrcAxesOrder              = other.mSrcAxesOrder;
          mDstAxesOrder              = other.mDstAxesOrder;
          mHasDefaultSrcAxesOrder    = other.mHasDefaultSrcAxesOrder;
          mHasDefaultDstAxesOrder    = other.mHasDefaultDstAxesOrder;
          mHasDefaultSrcMemoryBlocks = other.mHasDefaultSrcMemoryBlocks;
          mHasDefaultDstMemoryBlocks = other.mHasDefaultDstMemoryBlocks;
          mHasDefaultSrcStrides      = other.mHasDefaultSrcStrides;
          mHasDefaultDstStrides      = other.mHasDefaultDstStrides;

          // TODO: set memory blocks
        }

        return *this;
      }

      DistribMemDesc& operator=(DistribMemDesc&&) = default;

      [[nodiscard]] View<MemoryBlock> getSrcMemoryBlocks() const noexcept
      {
        return View<MemoryBlock>{mMemoryBlocks.get(), mShapeRank};
      }

      [[nodiscard]] const MemoryBlock& getSrcMemoryBlock(std::size_t targetIndex) const
      {
        if (targetIndex >= mTargetCount)
        {
          throw std::out_of_range{"The target index is out of range"};
        }

        return mMemoryBlocks[targetIndex];
      }

      [[nodiscard]] View<MemoryBlock> getDstMemoryBlocks() const noexcept
      {
        return View<MemoryBlock>{mMemoryBlocks.get() + mTargetCount, mShapeRank};
      }

      [[nodiscard]] const MemoryBlock& getDstMemoryBlock(std::size_t targetIndex) const
      {
        if (targetIndex >= mTargetCount)
        {
          throw std::out_of_range{"The target index is out of range"};
        }

        return mMemoryBlocks[mTargetCount + targetIndex];
      }
    
      [[nodiscard]] constexpr View<Axis> getSrcDistribAxes() const noexcept
      {
        return View<Axis>{mSrcDistribAxes.data, mShapeRank};
      }

      [[nodiscard]] constexpr View<Axis> getDstDistribAxes() const noexcept
      {
        return View<Axis>{mDstDistribAxes.data, mShapeRank};
      }

      [[nodiscard]] constexpr View<Axis> getSrcAxesOrder() const noexcept
      {
        return View<Axis>{mSrcAxesOrder.data, mShapeRank};
      }

      [[nodiscard]] constexpr View<Axis> getDstAxesOrder() const noexcept
      {
        return View<Axis>{mDstAxesOrder.data, mShapeRank};
      }

      [[nodiscard]] constexpr bool hasDefaultSrcAxesOrder() const noexcept
      {
        return mHasDefaultSrcAxesOrder;
      }

      [[nodiscard]] constexpr bool hasDefaultDstAxesOrder() const noexcept
      {
        return mHasDefaultDstAxesOrder;
      }

      [[nodiscard]] constexpr bool hasDefaultSrcMemoryBlocks() const noexcept
      {
        return mHasDefaultSrcMemoryBlocks;
      }

      [[nodiscard]] constexpr bool hasDefaultDstMemoryBlocks() const noexcept
      {
        return mHasDefaultDstMemoryBlocks;
      }

      [[nodiscard]] constexpr bool hasDefaultSrcStrides() const noexcept
      {
        return mHasDefaultSrcStrides;
      }

      [[nodiscard]] constexpr bool hasDefaultDstStrides() const noexcept
      {
        return mHasDefaultDstStrides;
      }

      [[nodiscard]] friend bool operator==(const DistribMemDesc& lhs, const DistribMemDesc& rhs) noexcept
      {
        // TODO: fixme
        return true;
      //   return lhs.mShapeRank == rhs.mShapeRank &&
      //          lhs.mTargetCount == rhs.mTargetCount &&
      //          (lhs.hasDefaultSrcAxesOrder() ||
      //            rhs.hasDefaultSrcAxesOrder() ||
      //            std::equal(lhs.getSrcAxesOrder().begin(), lhs.getSrcAxesOrder().end(), rhs.getSrcAxesOrder().begin())) &&
      //          (lhs.hasDefaultDstAxesOrder() ||
      //            rhs.hasDefaultDstAxesOrder() ||
      //            std::equal(lhs.getDstAxesOrder().begin(), lhs.getDstAxesOrder().end(), rhs.getDstAxesOrder().begin())) &&
      //          (lhs.hasDefaultSrcMemoryBlocks() ||
      //            rhs.hasDefaultSrcMemoryBlocks() ||
      //            std::equal(lhs.getSrcMemoryBlocks().begin(), lhs.getSrcMemoryBlocks().end(), rhs.getSrcMemoryBlocks().begin())) &&
      //          (lhs.hasDefaultDstMemoryBlocks() ||
      //            rhs.hasDefaultDstMemoryBlocks() ||
      //            std::equal(lhs.getDstMemoryBlocks().begin(), lhs.getDstMemoryBlocks().end(), rhs.getDstMemoryBlocks().begin()));
      }

    private:
      std::size_t                     mShapeRank{};                 ///< Shape rank.
      std::size_t                     mTargetCount{};               ///< Target count.
      std::unique_ptr<MemBlockDesc[]> mMemBlockDescs{};             ///< Source and destination memory block descriptors.
      std::unique_ptr<MemoryBlock[]>  mMemoryBlocks{};              ///< View over memory blocks.
      MaxDimBuffer<Axis>              mSrcDistribAxes{};            ///< Source distributed axes.
      MaxDimBuffer<Axis>              mDstDistribAxes{};            ///< Destination distributed axes.
      MaxDimBuffer<Axis>              mSrcAxesOrder{};              ///< Source axes order.
      MaxDimBuffer<Axis>              mDstAxesOrder{};              ///< Destination axes order.
      bool                            mHasDefaultSrcAxesOrder{};    ///< Has default source axes order.
      bool                            mHasDefaultDstAxesOrder{};    ///< Has default destination axes order.
      bool                            mHasDefaultSrcMemoryBlocks{}; ///< Has default source memory blocks.
      bool                            mHasDefaultDstMemoryBlocks{}; ///< Has default destination memory blocks.
      bool                            mHasDefaultSrcStrides{};      ///< Has default source strides.
      bool                            mHasDefaultDstStrides{};      ///< Has default destination strides.
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

      /**
       * @brief Construct a memory descriptor.
       * @param[in] memLayoutVariant Memory layout variant.
       * @param[in] transformDesc    Transform descriptor.
       * @param[in] mpDesc           MP descriptor.
       * @param[in] targetDesc       Target descriptor.
       */
      MemDesc(const MemoryLayoutParametersVariant& memLayoutVariant,
              const TransformDesc&                 transformDesc,
              const MpDesc&                        mpDesc,
              const TargetDesc&                    targetDesc)
      : MemDesc([&]()
          {
            if (std::holds_alternative<std::monostate>(memLayoutVariant))
            {
              if (mpDesc.getMpBackend() == MpBackend::none || targetDesc.getTargetCount() == 1)
              {
                return MemDesc{CentralizedMemoryLayout{}, transformDesc, mpDesc, targetDesc};
              }
              else
              {
                return MemDesc{DistributedMemoryLayout{}, transformDesc, mpDesc, targetDesc};
              }
            }
            else if (std::holds_alternative<CentralizedMemoryLayout>(memLayoutVariant))
            {
              return MemDesc{std::get<CentralizedMemoryLayout>(memLayoutVariant), transformDesc, mpDesc, targetDesc};
            }
            else if (std::holds_alternative<DistributedMemoryLayout>(memLayoutVariant))
            {
              return MemDesc{std::get<DistributedMemoryLayout>(memLayoutVariant), transformDesc, mpDesc, targetDesc};
            }
            else
            {
              throw Exception{Error::invalidArgument, "invalid memory layout variant"};
            }
          }())
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

          if (!memDesc.hasDefaultSrcMemoryBlocks())
          {
            memLayout.srcMemoryBlocks = memDesc.getSrcMemoryBlocks();
          }
          if (!memDesc.hasDefaultDstMemoryBlocks())
          {
            memLayout.dstMemoryBlocks = memDesc.getDstMemoryBlocks();
          }
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
      [[nodiscard]] static DistribMemDesc makeMemVariant(const DistributedMemoryLayout& memLayout,
                                                         const TransformDesc&           transformDesc,
                                                         const MpDesc&                  mpDesc,
                                                         const TargetDesc&              targetDesc)
      {
        return DistribMemDesc{memLayout, transformDesc, mpDesc, targetDesc};
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
        return DistribMemDesc{memLayout, transformDesc, mpDesc, targetDesc};
      }

      Alignment     mAlignment{};     ///< Memory alignment.
      ComplexFormat mComplexFormat{}; ///< Complex format.
      MemVariant    mMemVariant;      ///< Memory variant.
  };
} // namespace afft::detail

#endif /* AFFT_DETAIL_MEM_DESC_HPP */
