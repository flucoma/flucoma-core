#pragma once

#include "ParameterTypes.hpp"
#include "Result.hpp"

#include <algorithm>
#include <cmath>
#include <sstream>
namespace fluid {
namespace client {




/// Predicates

auto makeOdd = [](auto a) { return [=] { return a % 2 ? a - 1 : a; }; };
auto makePower2 = [](auto a) {
  return [=] {
    int exp;
    std::frexp(a, &exp);
    return 1 << (exp - 1);
  };
};

namespace impl {

template <typename T> struct MinImpl {
  constexpr MinImpl(const T m) : value(m) {}
  const T value;
  template <size_t N, typename U, typename Tuple> constexpr void clamp(U &x, Tuple params, Result* r) {
    U oldX = x;
    x = std::max<U>(x, value);
    if(r && oldX != x)
    {
      r->set(Result::Status::kWarning);
      r->addMessage(std::get<N>(params).first.name());
      r->addMessage(" value, "); r->addMessage(oldX);r->addMessage(", below absolute minimum ");r->addMessage(x);
    }
  }
};

template <typename T> struct MaxImpl {
  constexpr MaxImpl(const T m) : value(m) {}
  const T value;
  template <size_t N, typename U, typename Tuple> constexpr void clamp(U &x, Tuple params, Result* r) {
    
    U oldX = x;
    x = std::min<U>(x, value);
    if(r && oldX != x)
    {
      r->set(Result::Status::kWarning);
      r->addMessage(std::get<N>(params).first.name());
      r->addMessage(" value ("); r->addMessage(oldX);r->addMessage(") above absolute maximum (");r->addMessage(x);r->addMessage(')');
    }
  }
};

template <int... Is> struct LowerLimitImpl {
  template <size_t N, typename T, typename Tuple> void clamp(T &v, Tuple params, Result* r) {
    
    T oldV = v;
    
    v = std::max<T>({v, std::get<Is>(params).first.get()...});
    
    if(r && oldV != v)
    {
      r->set(Result::Status::kWarning);
      std::array<T,sizeof...(Is)> constraintValues {std::get<Is>(params).first.get()...};
      size_t minPos = std::distance(constraintValues.begin(), std::min_element(constraintValues.begin(), constraintValues.end()));
      std::array<const char*,sizeof...(Is)> constraintNames {std::get<Is>(params).first.name()...};
      r->addMessage(std::get<N>(params).first.name());
      r->addMessage(" value ("); r->addMessage(oldV);r->addMessage(") below parameter ");r->addMessage(constraintNames[minPos]);
      r->addMessage(" ("); r->addMessage(v);r->addMessage(')');
    }
  }
};

template <int... Is> struct UpperLimitImpl {
  template <size_t N, typename T, typename Tuple> void clamp(T &v, Tuple params, Result* r) {
    
    T oldV = v;
    
    v = std::min<T>({v, std::get<Is>(params).first.get()...});
    
    if(r && oldV != v)
    {
      r->set(Result::Status::kWarning);
      std::array<T,sizeof...(Is)> constraintValues {std::get<Is>(params).first.get()...};
      size_t maxPos = std::distance(constraintValues.begin(), std::max_element(constraintValues.begin(), constraintValues.end()));
      std::array<const char*,sizeof...(Is)> constraintNames  {std::get<Is>(params).first.name()...};
      r->addMessage(std::get<N>(params).first.name());
      r->addMessage(" value, "); r->addMessage(oldV);r->addMessage(", above parameter ");r->addMessage(constraintNames[maxPos]);
      r->addMessage(" ("); r->addMessage(v);r->addMessage(')');
    }
  }
};

} // namespace impl

template <typename T> auto constexpr Min(const T x) {
  return impl::MinImpl<T>(x);
};

template <typename T> auto constexpr Max(const T x) {
  return impl::MaxImpl<T>(x);
}

template <int... Is> auto constexpr LowerLimit() {
  return impl::LowerLimitImpl<Is...>{};
}

template <int... Is> auto constexpr UpperLimit() {
  return impl::UpperLimitImpl<Is...>{};
}

struct PowerOfTwo {
  template <size_t N, typename Tuple> constexpr LongUnderlyingType clamp(LongUnderlyingType x, Tuple params, Result* r) {
    
    int exp = 0;
    double base = std::frexp(x, &exp);
    LongUnderlyingType res =  base > 0.5 ? (1 << exp) : (1 << (exp - 1));
    
    if(r && res != x)
    {
      r->set(Result::Status::kWarning);
      r->addMessage(std::get<N>(params).first.name());
      r->addMessage(" value ("); r->addMessage(x);r->addMessage(") adjusted to power of two (");r->addMessage(res);r->addMessage(')');
    }
    return res;
  }
};

} // namespace client
} // namespace fluid

