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

#ifndef AFFT_PLAN_CACHE_HPP
#define AFFT_PLAN_CACHE_HPP

#ifndef AFFT_TOP_LEVEL_INCLUDE
# include "detail/include.hpp"
#endif

#include "common.hpp"
#include "error.hpp"
#include "Plan.hpp"
#include "detail/Desc.hpp"

AFFT_EXPORT namespace afft
{
  class PlanCache
  {
    private:
      using ListValue = std::shared_ptr<Plan>;                      ///< The value type of the cache.
      using List      = std::list<ListValue>;                       ///< The list type of the cache.
      using ListIter  = List::iterator;                             ///< The list iterator type of the cache.

      using MapKey    = std::reference_wrapper<const detail::Desc>;              ///< The key type of the cache.
      using MapValue  = ListIter;                                                ///< The value type of the map.
      using MapHash   = struct MapHashImpl;                                      ///< The hash function of the map.
      using MapEqual  = std::equal_to<detail::Desc>;                             ///< The equality function of the map.
      using Map       = std::unordered_map<MapKey, MapValue, MapHash, MapEqual>; ///< The map type of the cache.
      using MapIter   = Map::iterator;                                           ///< The map iterator type of the cache;

      struct MapHashImpl
      {
        [[nodiscard]] std::size_t operator()(MapKey key) const noexcept
        {
          return std::hash<detail::Desc>{}(key.get());
        }
      };
    public:
      // Forward declarations
      class Iterator;

      using value_type             = Plan*;                                 ///< The value type of the cache.
      using size_type              = std::size_t;                           ///< The size type of the cache.
      using difference_type        = std::ptrdiff_t;                        ///< The difference type of the cache.
      using reference	             = value_type&;                           ///< The reference type of the cache.
      using const_reference        = const value_type&;                     ///< The const reference type of the cache.
      using pointer                = value_type*;                           ///< The pointer type of the cache.            
      using const_pointer          = const value_type*;                     ///< The const pointer type of the cache.
      using iterator               = Iterator;                              ///< The iterator type of the cache.
      using const_iterator         = const Iterator;                        ///< The const iterator type of the cache.
      using reverse_iterator       = std::reverse_iterator<iterator>;       ///< The reverse iterator type of the cache.
      using const_reverse_iterator = std::reverse_iterator<const_iterator>; ///< The const reverse iterator type of the cache.

      class Iterator
      {
      public:
        using iterator_category = std::forward_iterator_tag;
        using value_type        = Plan;
        using difference_type   = std::ptrdiff_t;
        using pointer           = Plan*;
        using reference         = Plan&;

        Iterator() = default;

        Iterator(const Iterator&) = default;

        Iterator(Iterator&&) = default;

        ~Iterator() = default;

        Iterator& operator=(const Iterator&) = default;

        Iterator& operator=(Iterator&&) = default;

        reference operator*() const
        {
          return **mIter;
        }

        pointer operator->() const
        {
          return mIter->get();
        }

        Iterator& operator++()
        {
          ++mIter;
          return *this;
        }

        Iterator operator++(int)
        {
          auto temp = *this;
          ++mIter;
          return temp;
        }

        friend bool operator==(const Iterator& lhs, const Iterator& rhs)
        {
          return lhs.mIter == rhs.mIter;
        }

        friend bool operator!=(const Iterator& lhs, const Iterator& rhs)
        {
          return lhs.mIter != rhs.mIter;
        }

      protected:
      private:
        friend class PlanCache;

        using ListIter = PlanCache::ListIter;

        explicit iterator(ListIter iter)
        : mIter{iter}
        {}

        ListIter mIter{};
    };

      /// @brief The default maximum size of the cache.
      static constexpr std::size_t defaultMaxSize = std::numeric_limits<std::size_t>::max();

      /// @brief Constructs a new plan cache with the default maximum size.
      PlanCache() = default;

      /**
       * @brief Constructs a new plan cache with the specified maximum size.
       * @param maxSize The maximum number of plans that the cache can hold.
       */
      PlanCache(std::size_t maxSize)
      : mMaxSize{checkMaxSize(maxSize)}
      {}

      /**
       * @brief Constructs a new plan cache with the default maximum size and a list of plans.
       * @param plans A list of plans to insert into the cache.
       */
      PlanCache(std::initializer_list<std::shared_ptr<Plan>> plans)
      : PlanCache{defaultMaxSize, plans}
      {}

      /**
       * @brief Constructs a new plan cache with the specified maximum size and a list of plans.
       * @param maxSize The maximum number of plans that the cache can hold.
       * @param plans A list of plans to insert into the cache.
       */
      PlanCache(std::size_t maxSize, std::initializer_list<std::shared_ptr<Plan>> plans)
      : PlanCache{maxSize}
      {
        for (auto&& plan : plans)
        {
          insert(std::move(plan));
        }
      }
      
      /// @brief Copy constructor is deleted.
      PlanCache(const PlanCache&) = delete;

      /// @brief Move constructor is defaulted.
      PlanCache(PlanCache&&) = default;

      /// @brief Destructor is defaulted.
      ~PlanCache() = default;

      /// @brief Copy assignment operator is deleted.
      PlanCache& operator=(const PlanCache&) = delete;

      /// @brief Move assignment operator is defaulted.
      PlanCache& operator=(PlanCache&&) = default;

      /**
       * @brief Is the cache empty?
       * @return True if the cache is empty, otherwise false.
       */
      [[nodiscard]] bool isEmpty() const noexcept
      {
        return mList.empty();
      }

      /**
       * @brief Get the number of plans in the cache.
       * @return The number of plans in the cache.
       */
      [[nodiscard]] std::size_t getSize() const noexcept
      {
        return mList.size();
      }

      /**
       * @brief Get the maximum number of plans that the cache can hold.
       * @return The maximum number of plans that the cache can hold.
       */
      [[nodiscard]] std::size_t getMaxSize() const noexcept
      {
        return mMaxSize;
      }

      /**
       * @brief Set the maximum number of plans that the cache can hold.
       * @param maxSize The maximum number of plans that the cache can hold.
       */
      void setMaxSize(std::size_t maxSize)
      {
        mMaxSize = checkMaxSize(maxSize);

        while (mList.size() > mMaxSize)
        {
          mMap.erase(mList.back()->getDesc());
          mList.pop_back();
        }
      }

      /**
       * @brief Clear the cache.
       */
      void clear() noexcept
      {
        mMap.clear();
        mList.clear();
      }

      /**
       * @brief Insert a plan into the cache.
       * @param plan The plan to insert.
       * @return The inserted plan.
       */
      Plan& insert(std::shared_ptr<Plan> plan)
      {
        // Check if the value is null
        if (!plan)
        {
          throw Exception{Error::invalidArgument, "Cannot insert a null plan into the cache"};
        }

        // Check if the plan has a centralized memory layout
        if (plan->getMemoryLayout() != MemoryLayout::centralized)
        {
          throw Exception{Error::invalidArgument, "Only plans with centralized memory layout can be inserted into the cache"};
        }

        // Check if the capacity has been reached
        if (mList.size() >= mMaxSize)
        {
          // Remove the last element from the list and the map
          mMap.erase(mList.back()->getDesc());
          mList.pop_back();
        }

        // Insert the new element at the front of the list
        mList.emplace_front(std::move(plan));

        // Insert the new element into the map
        auto [it, inserted] = mMap.emplace(mList.front()->getDesc(), mList.begin());

        if (!inserted)
        {
          throw std::runtime_error{"Failed to insert plan into cache"};
        }

        return *it->second->get();
      }

      /**
       * @brief Release a plan from the cache that matches the specified parameters.
       * @param transformParams The parameters of the transform.
       * @param targetParams The parameters of the target.
       * @param memoryLayout The memory layout of the plan.
       * @return The released plan.
       */
      template<typename TransformParamsT, typename TargetParamsT>
      [[nodiscard]] std::shared_ptr<Plan> release(const TransformParamsT&        transformParams,
                                                  const TargetParamsT&           targetParams,
                                                  const CentralizedMemoryLayout& memoryLayout = {})
      {
        static_assert(isTransformParameters<TransformParamsT>, "invalid transform parameters type");
        static_assert(isTargetParameters<TargetParamsT>, "invalid target parameters type");

        const auto desc = detail::Desc{transformParams, targetParams, memoryLayout};

        if (auto mapIter = mMap.find(desc); mapIter != mMap.end())
        {
          auto plan = std::move(*mapIter->second);

          mList.erase(mapIter->second);
          mMap.erase(mapIter);

          return plan;
        }

        throw std::runtime_error{"plan not found in cache"};
      }

      /**
       * @brief Erase a plan from the cache that matches the specified parameters.
       * @param transformParams The parameters of the transform.
       * @param targetParams The parameters of the target.
       * @param memoryLayout The memory layout of the plan.
       */
      template<typename TransformParamsT, typename TargetParamsT>
      void erase(const TransformParamsT&        transformParams,
                 const TargetParamsT&           targetParams,
                 const CentralizedMemoryLayout& memoryLayout = {})
      {
        static_assert(isTransformParameters<TransformParamsT>, "invalid transform parameters type");
        static_assert(isTargetParameters<TargetParamsT>, "invalid target parameters type");

        [[maybe_unused]] auto plan = release(transformParams, targetParams, memoryLayout);
      }

      /**
       * @brief Swap the contents of this cache with another cache.
       * @param other The other cache to swap with.
       */
      void swap(PlanCache& other) noexcept
      {
        mMap.swap(other.mMap);
        mList.swap(other.mList);
        std::swap(mMaxSize, other.mMaxSize);
      }

      /**
       * @brief Merge the contents of this cache with another cache.
       * @param other The other cache to merge with.
       */
      void merge(PlanCache& other)
      {
        if (this == &other)
        {
          return;
        }

        if (getSize() + other.getSize() > mMaxSize)
        {
          throw std::runtime_error{"cannot merge plan caches because the maximum size would be exceeded"};
        }

        for (auto& plan : other.mList)
        {
          insert(std::move(plan));
        }

        other.clear();
      }

      /**
       * @brief Find a plan in the cache that matches the specified parameters.
       * @param transformParams The parameters of the transform.
       * @param targetParams The parameters of the target.
       * @param memoryLayout The memory layout of the plan.
       * @return The plan if found.
       */
      template<typename TransformParamsT, typename TargetParamsT>
      [[nodiscard]] iterator find(const TransformParamsT&        transformParams,
                                  const TargetParamsT&           targetParams,
                                  const CentralizedMemoryLayout& memoryLayout = {}) const
      {
        static_assert(isTransformParameters<TransformParamsT>, "invalid transform parameters type");
        static_assert(isTargetParameters<TargetParamsT>, "invalid target parameters type");

        const auto desc = detail::Desc{transformParams, targetParams, memoryLayout};

        if (auto mapIter = mMap.find(desc); mapIter != mMap.end())
        {
          auto listIter = mapIter->second;

          if (listIter != mList.begin())
          {
            mList.splice(mList.begin(), mList, std::next(listIter));
          }

          return iterator{begin()};
        }

        return iterator{end()};
      }

      /**
       * @brief Get an iterator to the first plan in the cache.
       * @return An iterator to the first plan in the cache.
       */
      [[nodiscard]] iterator begin() noexcept
      {
        return iterator{mList.begin()};
      }

      /**
       * @brief Get an iterator to the end of the cache.
       * @return An iterator to the end of the cache.
       */
      [[nodiscard]] iterator end() noexcept
      {
        return iterator{mList.end()};
      }

    protected:
    private:
      

      /**
       * @brief Checks if the maximum size of the cache is valid.
       * @param maxSize The maximum size of the cache.
       * @return The maximum size of the cache.
       */
      static constexpr std::size_t checkMaxSize(std::size_t maxSize)
      {
        if (maxSize == 0)
        {
          throw Exception{Error::invalidArgument, "The maximum size of the cache must be greater than zero"};
        }

        return maxSize;
      }

      Map         mMap{};                   ///< The map of the cache.
      List        mList{};                  ///< The list of the cache.
      std::size_t mMaxSize{defaultMaxSize}; ///< The maximum size of the cache.
  };  
} // namespace afft

#endif /* AFFT_PLAN_CACHE_HPP */
