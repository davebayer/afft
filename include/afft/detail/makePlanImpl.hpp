#ifndef AFFT_DETAIL_MAKE_PLAN_IMPL_HPP
#define AFFT_DETAIL_MAKE_PLAN_IMPL_HPP

#include <memory>
#include <utility>

#include "common.hpp"
#include "PlanImpl.hpp"
#include "gpu/makePlanImpl.hpp"
#include "cpu/makePlanImpl.hpp"

namespace afft::detail
{
  /**
   * @brief Make a plan implementation.
   * @tparam target Target.
   * @tparam Args Argument types.
   * @param args Arguments.
   * @return Plan implementation.
   */
  template<Target target, typename... Args>
  std::shared_ptr<PlanImpl> makePlanImpl(Args&&... args)
  {
    if constexpr (target == Target::cpu)
    {
      return cpu::makePlanImpl(std::forward<Args>(args)...);
    }
    else
    {
      return gpu::makePlanImpl(std::forward<Args>(args)...);
    }
  }
} // namespace afft::detail

#endif /* AFFT_DETAIL_MAKE_PLAN_IMPL_HPP */
