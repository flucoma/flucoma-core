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

auto maxOf = [](const auto a, const auto ... rest) {
  //  return [&] {
  return std::max<decltype(a)>(
      {a, rest...});//, [](auto a, auto &b) { return a > b.get(); });
  //  };
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

template <typename F, std::size_t... Is> struct Constraint {

  template <typename T, typename Tuple> static void invoke(T &x, Tuple y) {
    F(x, std::get<Is>(y)...);
  }
};
namespace impl {
template <typename T> struct MinImpl {
  constexpr MinImpl(const T m) : value(m) {}
  const T value;
  template<typename Tuple>
  constexpr T clamp(T x, Tuple) { return std::max<T>(x,value);}
};

template <typename T> struct MaxImpl {
  constexpr MaxImpl(const T m) : value(m) {}
  const T value;
  template<typename Tuple>
  constexpr T clamp(T x, Tuple) { return std::min<T>(x,value);}
};




//template<typename T>
//struct TupleToSequence;
template<typename...Ts>
void TupleToSequence()//<std::tuple<Ts...>>
{
  
  using type = std::index_sequence_for<Ts...>;
};



template <int... Is> struct LowerLimitImpl {
 
//  template <typename T> static ConstraintResult check(T x, Ts... y) {
//    using CR = ConstraintResult;
//    auto a = maxOf(x, y...);
//    return x == a
//               ? CR(true, "")
//               : CR(false, (std::stringstream()
//                            << x.name() << " must be greater than " << a.name())
//                               .str());
//  }
//  template <typename T> static void clamp(T x, Ts... y) {
//    x.set((maxOf(x, y...)).get());
//  }
//
  
  template<typename T, typename Tuple>
  T clamp(T& v, Tuple params)
  {
    T res =  maxOf(v,std::get<Is>(params).first.get()...);
    std::cout << v << ' ' << res <<'\n';
    return res; 
  }
//
//  std::tuple<Ts...> mArgs;
};
} // namespace impl

template <typename T> auto constexpr Min(const T x) {
  return impl::MinImpl<T>(x);
}

struct PowerOfTwo {
  template<typename Tuple>
  constexpr long clamp(long x, Tuple) {
    int exp=0;
    double r = std::frexp(x, &exp);
    return r > 0.5 ? (1 << exp) : (1 << (exp - 1));
  }
};

template <int...Is>
auto constexpr LowerLimit() {
  return impl::LowerLimitImpl<Is...>{};
}

//template <typename... Ts>
//auto constexpr LowerLimit(const Ts... args) {
//  return impl::LowerLimitImpl<std::integral_constant<int,Ts::args>...>(std::tuple<std::integral_constant<int,Ts::args>...>{});
//}

} // namespace client
} // namespace fluid
