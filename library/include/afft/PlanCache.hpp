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
#include "Description.hpp"
#include "error.hpp"
#include "Plan.hpp"
#include "detail/Desc.hpp"

AFFT_EXPORT namespace afft
{
  /// @brief Plan cache.
  class PlanCache
  {
    private:
      using ListValue     = std::shared_ptr<Plan>;                                   ///< The value type of the cache.
      using List          = std::list<ListValue>;                                    ///< The list type of the cache.
      using ListIter      = List::iterator;                                          ///< The list iterator type of the cache.
      using ListConstIter = List::const_iterator;                                    ///< The list const iterator type of the cache.

      using MapKey        = std::reference_wrapper<const Description>;               ///< The key type of the cache.
      using MapValue      = ListIter;                                                ///< The value type of the map.
      struct MapHash                                                                 ///< The hash function of the map.
      {
        [[nodiscard]] std::size_t operator()(MapKey key) const noexcept
        {
          return std::hash<Description>{}(key);
        }
      };
      using MapEqual      = std::equal_to<Description>;                              ///< The equality function of the map.
      using Map           = std::unordered_map<MapKey, MapValue, MapHash, MapEqual>; ///< The map type of the cache.
      using MapIter       = Map::iterator;                                           ///< The map iterator type of the cache;
      using MapConstIter  = Map::const_iterator;                                     ///< The map const iterator type of the cache;
      
    public:
      using value_type             = List::value_type;             ///< The value type of the cache.
      using size_type              = List::size_type;              ///< The size type of the cache.
      using difference_type        = List::difference_type;        ///< The difference type of the cache.
      using reference	             = List::reference;              ///< The reference type of the cache.
      using const_reference        = List::const_reference;        ///< The const reference type of the cache.
      using pointer                = List::pointer;                ///< The pointer type of the cache.            
      using const_pointer          = List::const_pointer;          ///< The const pointer type of the cache.
      using iterator               = List::iterator;               ///< The iterator type of the cache.
      using const_iterator         = List::const_iterator;         ///< The const iterator type of the cache.
      using reverse_iterator       = List::reverse_iterator;       ///< The reverse iterator type of the cache.
      using const_reverse_iterator = List::const_reverse_iterator; ///< The const reverse iterator type of the cache.

      /// @brief The default maximum size of the cache.
      static constexpr size_type defaultMaxSize = std::numeric_limits<size_type>::max();

      /// @brief Constructs a new plan cache with the default maximum size.
      PlanCache() = default;

      /**
       * @brief Constructs a new plan cache with the specified maximum size.
       * @param maxSize The maximum number of plans that the cache can hold.
       */
      PlanCache(size_type maxSize)
      : mMaxSize{checkMaxSize(maxSize)}
      {}

      /**
       * @brief Constructs a new plan cache with the default maximum size and a list of plans.
       * @param plans A list of plans to insert into the cache.
       */
      PlanCache(std::initializer_list<value_type> plans)
      : PlanCache{defaultMaxSize, plans}
      {}

      /**
       * @brief Constructs a new plan cache with the specified maximum size and a list of plans.
       * @param maxSize The maximum number of plans that the cache can hold.
       * @param plans A list of plans to insert into the cache.
       */
      PlanCache(size_type maxSize, std::initializer_list<value_type> plans)
      : PlanCache{maxSize}
      {
        for (auto& plan : plans)
        {
          insert(plan);
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
      [[nodiscard]] size_type getSize() const noexcept
      {
        return mList.size();
      }

      /**
       * @brief Get the maximum number of plans that the cache can hold.
       * @return The maximum number of plans that the cache can hold.
       */
      [[nodiscard]] size_type getMaxSize() const noexcept
      {
        return mMaxSize;
      }

      /**
       * @brief Set the maximum number of plans that the cache can hold.
       * @param maxSize The maximum number of plans that the cache can hold.
       */
      void setMaxSize(size_type maxSize)
      {
        mMaxSize = checkMaxSize(maxSize);

        while (mList.size() > mMaxSize)
        {
          mMap.erase(mList.back()->getDescription());
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
      iterator insert(value_type plan)
      {
        // Check if the value is null
        if (!plan)
        {
          throw Exception{Error::invalidArgument, "Cannot insert a null plan into the cache"};
        }

        // Check if the plan has a centralized memory layout
        if (plan->getDescription().getMemoryLayout() != MemoryLayout::centralized)
        {
          throw Exception{Error::invalidArgument, "Only plans with centralized memory layout can be inserted into the cache"};
        }

        // Check if the capacity has been reached
        if (mList.size() >= mMaxSize)
        {
          // Remove the last element from the list and the map
          mMap.erase(mList.back()->getDescription());
          mList.pop_back();
        }

        // Insert the new element at the front of the list
        mList.emplace_front(std::move(plan));

        // Insert the new element into the map
        auto [it, inserted] = mMap.emplace(mList.front()->getDescription(), mList.begin());

        if (!inserted)
        {
          throw std::runtime_error{"Failed to insert plan into cache"};
        }

        return it->second;
      }

      /**
       * @brief Erase a plan from the cache that matches the plan description.
       * @param desc The description of the plan to erase.
       */
      void erase(const Description& desc);

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
       * @brief Find a plan in the cache that matches the plan description.
       * @param desc The description of the plan to find.
       * @return The plan if found.
       */
      [[nodiscard]] iterator find(const Description& desc)
      {
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

      /**
       * @brief Get a const iterator to the first plan in the cache.
       * @return A const iterator to the first plan in the cache.
       */
      [[nodiscard]] const_iterator begin() const noexcept
      {
        return cbegin();
      }

      /**
       * @brief Get a const iterator to the end of the cache.
       * @return A const iterator to the end of the cache.
       */
      [[nodiscard]] const_iterator end() const noexcept
      {
        return cend();
      }

      /**
       * @brief Get a const iterator to the first plan in the cache.
       * @return A const iterator to the first plan in the cache.
       */
      [[nodiscard]] const_iterator cbegin() const noexcept
      {
        return const_iterator{mList.cbegin()};
      }

      /**
       * @brief Get a const iterator to the end of the cache.
       * @return A const iterator to the end of the cache.
       */
      [[nodiscard]] const_iterator cend() const noexcept
      {
        return const_iterator{mList.cend()};
      }

      /**
       * @brief Get a reverse iterator to the first plan in the cache.
       * @return A reverse iterator to the first plan in the cache.
       */
      [[nodiscard]] reverse_iterator rbegin() noexcept
      {
        return reverse_iterator{mList.rbegin()};
      }

      /**
       * @brief Get a reverse iterator to the end of the cache.
       * @return A reverse iterator to the end of the cache.
       */
      [[nodiscard]] reverse_iterator rend() noexcept
      {
        return reverse_iterator{mList.rend()};
      }

      /**
       * @brief Get a const reverse iterator to the first plan in the cache.
       * @return A const reverse iterator to the first plan in the cache.
       */
      [[nodiscard]] const_reverse_iterator rbegin() const noexcept
      {
        return crbegin();
      }

      /**
       * @brief Get a const reverse iterator to the end of the cache.
       * @return A const reverse iterator to the end of the cache.
       */
      [[nodiscard]] const_reverse_iterator rend() const noexcept
      {
        return crend();
      }

      /**
       * @brief Get a const reverse iterator to the first plan in the cache.
       * @return A const reverse iterator to the first plan in the cache.
       */
      [[nodiscard]] const_reverse_iterator crbegin() const noexcept
      {
        return const_reverse_iterator{mList.crbegin()};
      }

      /**
       * @brief Get a const reverse iterator to the end of the cache.
       * @return A const reverse iterator to the end of the cache.
       */
      [[nodiscard]] const_reverse_iterator crend() const noexcept
      {
        return const_reverse_iterator{mList.crend()};
      }
    protected:
    private:
      /**
       * @brief Checks if the maximum size of the cache is valid.
       * @param maxSize The maximum size of the cache.
       * @return The maximum size of the cache.
       */
      static constexpr size_type checkMaxSize(size_type maxSize)
      {
        if (maxSize == 0)
        {
          throw Exception{Error::invalidArgument, "The maximum size of the cache must be greater than zero"};
        }

        return maxSize;
      }

      Map       mMap{};                   ///< The map of the cache.
      List      mList{};                  ///< The list of the cache.
      size_type mMaxSize{defaultMaxSize}; ///< The maximum size of the cache.
  };  
} // namespace afft

#endif /* AFFT_PLAN_CACHE_HPP */
