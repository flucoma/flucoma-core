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

// auto LessThanEqual = [](auto a, auto b) { return [&a, &b] { return a <= b; };
// }; auto LessThan = [](auto a, auto b) { return [&a, &b] { return a < b; }; };
// auto GreaterThanEqual = [](auto a, auto b) {
//  return [&a, &b] { return a >= b; };
//};
// auto GreaterThan = [](auto a, auto b) { return [&a, &b] { return a > b; }; };
//
// auto isOdd = [](auto a) { return [=] { a % 2 == 1; }; };
// auto isPowerOf2 = [](auto a) { return [=] {
//    int exp;
//    return std::frexp(a,&exp) == 0.5;
//  };
//};

auto makeOdd = [](auto a) { return [=] { return a % 2 ? a - 1 : a; }; };
auto makePower2 = [](auto a) {
  return [=] {
    int exp;
    std::frexp(a, &exp);
    return 1 << (exp - 1);
  };
};

// auto isGreaterEqual =
//    [](auto &a, auto &... rest) -> ConstraintResult {
//  bool success = true;
//  std::stringstream msg;
//
//  (void)std::initializer_list<int> { ([], 0)... }
//}

auto minOf = [](auto &a, auto &... rest) {
  return [&] {
    return std::min<decltype(a)>(
        {a, rest...}, [](auto &a, auto &b) { return a.get() < b.get(); });
  };
};

auto maxOf = [](auto &a, auto &... rest) {
//  return [&] {
    return std::max<decltype(a)>(
        {a, rest...}, [](auto &a, auto &b) { return a.get() > b.get(); });
//  };
};
//
//template <typename F,typename... Ts> struct EnabledBy {
//  EnabledBy(std::tuple<Ts...> args, F fun) : mArgs(args){}
//  template <typename T> static ConstraintResult check(T x, Ts... y) {
//    return f(x, y...);
//  }

//private:
//  F f;
//  std::tuple<Ts...> mArgs;
//};


template<typename F, std::size_t...Is>
struct Constraint
{

  template<typename T, typename Tuple>
  static void invoke(T& x, Tuple y)
  {
    F(x, std::get<Is>(y)...);
  }

};




template <typename... Ts> struct LowerLimit {
  constexpr LowerLimit(std::tuple<Ts...> args) : mArgs(args) {}
  template <typename T> static ConstraintResult check(T x, Ts... y) {
    using CR = ConstraintResult;
    auto a = maxOf(x, y...);
    return x == a ? CR(true, "") : CR(false, (std::stringstream() << x.name() << " must be greater than " << a.name()).str());
  }
  template <typename T> static void clamp(T x, Ts... y) {
    x.set((maxOf(x, y...)).get());
  }
  
  std::tuple<Ts...> mArgs;
  
};

} // namespace client
} // namespace fluid

