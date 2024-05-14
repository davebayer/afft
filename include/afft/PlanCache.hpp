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
#include "Plan.hpp"
#include "detail/Config.hpp"
#include "detail/PlanImpl.hpp"

namespace afft
{
  class PlanCache
  {
    public:
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
      PlanCache(std::initializer_list<Plan> plans)
      : PlanCache{defaultMaxSize, plans}
      {}

      /**
       * @brief Constructs a new plan cache with the specified maximum size and a list of plans.
       * @param maxSize The maximum number of plans that the cache can hold.
       * @param plans A list of plans to insert into the cache.
       */
      PlanCache(std::size_t maxSize, std::initializer_list<Plan> plans)
      : PlanCache(maxSize)
      {
        for (const auto& plan : plans)
        {
          insert(plan.mImpl);
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
      [[nodiscard]] bool empty() const noexcept
      {
        return mList.empty();
      }

      /**
       * @brief Get the number of plans in the cache.
       * @return The number of plans in the cache.
       */
      [[nodiscard]] std::size_t size() const noexcept
      {
        return mList.size();
      }

      /**
       * @brief Get the maximum number of plans that the cache can hold.
       * @return The maximum number of plans that the cache can hold.
       */
      [[nodiscard]] std::size_t maxSize() const noexcept
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
          mMap.erase(mList.back()->getConfig());
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
       * @param plan The plan to insert into the cache.
       */
      void insert(const Plan& plan)
      {
        insert(plan.mImpl);
      }

      /**
       * @brief Erase a plan from the cache that matches the specified parameters.
       * @param params The parameters of the transform.
       * @param targetParams The parameters of the target transform.
       */
      template<typename TransformParametersT, typename TargetParametersT>
      void erase(const TransformParametersT& params, const TargetParametersT& targetParams)
      {
        static_assert(isTransformParameters<TransformParametersT>, "Invalid transform parameters type");
        static_assert(isTargetParameters<TargetParametersT>, "Invalid target parameters type");

        const auto config = detail::Config{params, targetParams};

        if (auto mapIter = mMap.find(config); mapIter != mMap.end())
        {
          mList.erase(mapIter->second);
          mMap.erase(mapIter);
        }
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

        if (size() + other.size() > mMaxSize)
        {
          throw std::runtime_error{"Cannot merge caches because the maximum size would be exceeded"};
        }

        for (auto& plan : other.mList)
        {
          insert(std::move(plan));
        }

        other.clear();
      }

      /**
       * @brief Finds a plan in the cache that matches the specified parameters.
       * @param params The parameters of the transform.
       * @param targetParams The parameters of the target transform.
       * @return The plan that matches the specified parameters.
       */
      template<typename TransformParametersT, typename TargetParametersT>
      [[nodiscard]] std::optional<Plan> find(const TransformParametersT& params, const TargetParametersT& targetParams)
      {
        static_assert(isTransformParameters<TransformParametersT>, "Invalid transform parameters type");
        static_assert(isTargetParameters<TargetParametersT>, "Invalid target parameters type");

        std::optional<Plan> plan{};

        const auto config = detail::Config{params, targetParams};

        if (auto mapIter = mMap.find(config); mapIter != mMap.end())
        {
          plan = Plan{*(mapIter->second)};
        }

        return plan;
      }
      
    protected:
    private:
      using Key      = std::reference_wrapper<const detail::Config>;         ///< The key type of the cache.
      using Value    = std::shared_ptr<detail::PlanImpl>;                    ///< The value type of the cache.

      using List     = std::list<std::shared_ptr<detail::PlanImpl>>;         ///< The list type of the cache.
      using ListIter = List::iterator;                                       ///< The list iterator type of the cache.

      using MapValue = ListIter;                                             ///< The value type of the map.
      struct MapHash                                                         ///< The hash function of the map.
      {
        [[nodiscard]] std::size_t operator()(Key key) const noexcept
        {
          return std::hash<detail::Config>{}(key.get());
        }
      };
      using MapEqual = std::equal_to<detail::Config>;                        ///< The equality function of the map.

      using Map      = std::unordered_map<Key, MapValue, MapHash, MapEqual>; ///< The map type of the cache.
      using MapIter  = Map::iterator;                                        ///< The map iterator type of the cache.

      /**
       * @brief Checks if the maximum size of the cache is valid.
       * @param maxSize The maximum size of the cache.
       * @return The maximum size of the cache.
       */
      static constexpr std::size_t checkMaxSize(std::size_t maxSize)
      {
        if (maxSize == 0)
        {
          throw std::invalid_argument{"The maximum size of the cache must be greater than zero"};
        }

        return maxSize;
      }

      /**
       * @brief Inserts a new element into the cache.
       * @param value The value of the new element.
       * @return An iterator to the newly inserted element.
       */
      ListIter insert(Value value)
      {
        // Check if the capacity has been reached
        if (mList.size() >= mMaxSize)
        {
          // Remove the last element from the list and the map
          mMap.erase(mList.back()->getConfig());
          mList.pop_back();
        }

        // Insert the new element at the front of the list
        mList.emplace_front(std::move(value));

        // Insert the new element into the map
        auto [it, inserted] = mMap.emplace(mList.front()->getConfig(), mList.begin());

        if (!inserted)
        {
          throw std::runtime_error{"Failed to insert plan into cache"};
        }

        return it->second;
      }

      Map         mMap{};                   ///< The map of the cache.
      List        mList{};                  ///< The list of the cache.
      std::size_t mMaxSize{defaultMaxSize}; ///< The maximum size of the cache.
  };
} // namespace afft

#endif /* AFFT_PLAN_CACHE_HPP */
