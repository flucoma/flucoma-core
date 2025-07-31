#pragma once

#include <cassert>
#include <cstddef>
#include <limits>
#include <type_traits>

namespace fluid {
using index = std::ptrdiff_t;

inline std::make_unsigned_t<index> asUnsigned(index i)
{
  assert(i >= 0);
  return static_cast<std::make_unsigned_t<index>>(i);
}

inline index asSigned(std::make_unsigned_t<index> s)
{
  assert(s <= std::make_unsigned_t<index>(std::numeric_limits<index>::max()));
  return static_cast<index>(s);
}
} // namespace fluid
