#pragma once

//#include "ParameterDescriptor.hpp"
//#include "ParameterDescriptorList.hpp"
//#include "ParameterInstance.hpp"

#include <algorithm>
#include <cmath>
#include <sstream>
namespace fluid {
namespace client {

class ConstraintResult {
public:
  ConstraintResult(const bool ok, const char *errorStr) noexcept
      : mOk(ok), mErrorStr(errorStr) {}
  operator bool() const noexcept { return mOk; }
  const char *message() const noexcept { return mErrorStr; }

private:
  bool mOk;
  const char *mErrorStr;
};

/// Predicates


auto makeOdd = [](auto a) { return [=] { return a % 2 ? a - 1 : a; }; };
auto makePower2 = [](auto a) {
  return [=] {
    int exp;
    std::frexp(a, &exp);
    return 1 << (exp - 1);
  };
};

//
// template <typename F,typename... Ts> struct EnabledBy {
//  EnabledBy(std::tuple<Ts...> args, F fun) : mArgs(args){}
//  template <typename T> static ConstraintResult check(T x, Ts... y) {
//    return f(x, y...);
//  }

// private:
//  F f;
//  std::tuple<Ts...> mArgs;
//};

//template <typename F, std::size_t... Is> struct Constraint {
//
//  template <typename T, typename Tuple> static void invoke(T &x, Tuple y) {
//    F(x, std::get<Is>(y)...);
//  }
//};


namespace impl {

template <typename T> struct MinImpl {
  constexpr MinImpl(const T m) : value(m) {}
  const T value;
  template<typename U, typename Tuple>
  constexpr U clamp(U x, Tuple) { return std::max<U>(x,value);}
};

template <typename T> struct MaxImpl {
  constexpr MaxImpl(const T m) : value(m) {}
  const T value;
  template<typename U, typename Tuple>
  constexpr U clamp(U x, Tuple) { return std::min<U>(x,value);}
};

template <int... Is> struct LowerLimitImpl {
  template<typename T, typename Tuple>
  void clamp(T& v, Tuple params)
  {
    v =  std::max<T>({v, std::get<Is>(params).first.get()...});
  }
};

template <int... Is> struct UpperLimitImpl {
  template<typename T, typename Tuple>
  void clamp(T& v, Tuple params)
  {
    v = std::min<T>({v, std::get<Is>(params).first.get()...});
  }
};

} // namespace impl


template <typename T>
auto constexpr Min(const T x){
  return impl::MinImpl<T>(x);
};

template <typename T> auto constexpr Max(const T x) {
  return impl::MaxImpl<T>(x);
}

template <int...Is>
auto constexpr LowerLimit() {
  return impl::LowerLimitImpl<Is...>{};
}

template <int...Is>
auto constexpr UpperLimit() {
  return impl::UpperLimitImpl<Is...>{};
}

struct PowerOfTwo {
  template<typename Tuple>
  constexpr long clamp(long x, Tuple) {
    int exp=0;
    double r = std::frexp(x, &exp);
    return r > 0.5 ? (1 << exp) : (1 << (exp - 1));
  }
};

} // namespace client
} // namespace fluid
