#pragma once

#include <clients/common/BufferAdaptor.hpp>
#include <tuple>
#include <utility>
#include <vector>
namespace fluid {
namespace client {

enum class TypeTag { kFloat, kLong, kBuffer, kEnum, kFloatArray, kLongArray, kBufferArray };

using FloatUnderlyingType          = double;
using LongUnderlyingType           = intptr_t; // signed int equal to pointer size, k thx
using EnumUnderlyingType           = intptr_t;
using BufferUnderlyingType         = std::unique_ptr<BufferAdaptor>;
using FloatArrayUnderlyingType     = std::vector<FloatUnderlyingType>;
using LongArrayUnderlyingType      = std::vector<LongUnderlyingType>;
using BufferArrayUnderlyingType    = std::vector<BufferUnderlyingType>;
using MagnitudePairsUnderlyingType = std::vector<std::pair<double, double>>;

template<bool b>
struct Fixed
{
  static bool constexpr value{b};
};

struct ParamTypeBase
{
  constexpr ParamTypeBase(const char *n, const char *display)
      : name(n)
      , displayName(display)
  {}
  const char *name;
  const char *displayName;
};

struct FloatT : ParamTypeBase
{
  static constexpr TypeTag typeTag = TypeTag::kFloat;
  using type                       = FloatUnderlyingType;
  constexpr FloatT(const char *name, const char *displayName, const type defaultVal)
      : ParamTypeBase(name, displayName)
      , defaultValue(defaultVal)
  {}
  const std::size_t fixedSize = 1;
  const type        defaultValue;
};

struct LongT : ParamTypeBase
{
  static constexpr TypeTag typeTag = TypeTag::kLong;
  using type                       = LongUnderlyingType;
  constexpr LongT(const char *name, const char *displayName, const type defaultVal)
      : ParamTypeBase(name, displayName)
      , defaultValue(defaultVal)
  {}
  const std::size_t fixedSize = 1;
  const type        defaultValue;
};

struct BufferT : ParamTypeBase
{
  static constexpr TypeTag typeTag = TypeTag::kBuffer;
  using type                       = BufferUnderlyingType;
  constexpr BufferT(const char *name, const char *displayName)
      : ParamTypeBase(name, displayName)
  {}
  const std::size_t    fixedSize = 1;
  const std::nullptr_t defaultValue{nullptr};
}; // no non-relational conditions for buffer?

struct EnumT : ParamTypeBase
{
  static constexpr TypeTag typeTag = TypeTag::kEnum;
  using type                       = EnumUnderlyingType;
  template <std::size_t... N>
  constexpr EnumT(const char *name, const char *displayName, type defaultVal, const char (&... string)[N])
      : strings{string...}
      , ParamTypeBase(name, displayName)
      , fixedSize(1)
      , numOptions(sizeof...(N))
      , defaultValue(defaultVal)
  {
    static_assert(sizeof...(N) > 0, "Fluid Param: No enum strings supplied!");
    static_assert(sizeof...(N) <= 16, "Fluid Param: : Maximum 16 things in an Enum param");
  }
  const char *strings[16]; // unilateral descision klaxon: if you have more than 16 things in an Enum, you need to rethink
  const std::size_t fixedSize;
  const std::size_t numOptions;
  const type        defaultValue;
};

struct FloatArrayT : ParamTypeBase
{
  static constexpr TypeTag typeTag = TypeTag::kFloatArray;
  using type                       = FloatArrayUnderlyingType;

  template <std::size_t N>
  FloatArrayT(const char *name, const char *displayName, type::value_type (&defaultValues)[N])
      : ParamTypeBase(name, displayName)
  {}
  const std::size_t fixedSize;
};

struct LongArrayT : ParamTypeBase
{
  static constexpr TypeTag typeTag = TypeTag::kLongArray;
  using type                       = LongArrayUnderlyingType;
  template <std::size_t N>
  LongArrayT(const char *name, const char *displayName, type::value_type (&defaultValues)[N])
      : ParamTypeBase(name, displayName)
  {}
  const std::size_t fixedSize;
};

struct BufferArrayT : ParamTypeBase
{
  static constexpr TypeTag typeTag = TypeTag::kBufferArray;
  using type                       = BufferArrayUnderlyingType;
  BufferArrayT(const char *name, const char *displayName, const size_t size)
      : ParamTypeBase(name, displayName)
      , fixedSize(size)
  {}
  const std::size_t fixedSize;
};

// Pair of frequency amplitude pairs for HPSS threshold
struct FloatPairsArrayT : ParamTypeBase
{
  //  static constexpr TypeTa
  using type = std::vector<std::pair<FloatUnderlyingType, FloatUnderlyingType>>;

  constexpr FloatPairsArrayT(const char *name, const char *displayName)
      : ParamTypeBase(name, displayName)
  {}
  const std::size_t fixedSize{2};
  static constexpr std::initializer_list<std::pair<double, double>> defaultValue{{0.0, 1.0}, {1.0, 1.0}};
};


// My name's the C++ linker, and I'm a bit of a knob (fixed in C++17)
constexpr std::initializer_list<std::pair<double, double>> FloatPairsArrayT::defaultValue;

template <typename T, typename Fixed, typename... Constraints> using ParamSpec = std::tuple<T, std::tuple<Constraints...>,Fixed>;

template <typename IsFixed = Fixed<false>, typename...Constraints>
constexpr ParamSpec<FloatT,IsFixed, Constraints...> FloatParam(const char *name, const char *displayName, const FloatT::type defaultValue,
                                                       Constraints... c)
{
  return {FloatT(name, displayName, defaultValue), std::make_tuple(c...), IsFixed{}};
}

template <typename IsFixed = Fixed<false>, typename... Constraints>
constexpr ParamSpec<LongT,IsFixed, Constraints...> LongParam(const char *name, const char *displayName,
                                                     const LongT::type defaultValue, Constraints... c)
{
  return {LongT(name, displayName, defaultValue), std::make_tuple(c...),IsFixed{}};
}

template <typename IsFixed = Fixed<false>,typename... Constraints>
constexpr ParamSpec<BufferT,IsFixed, Constraints...> BufferParam(const char *name, const char *displayName, const Constraints... c)
{
  return {BufferT(name, displayName), std::make_tuple(c...),IsFixed{}};
}

template <typename IsFixed = Fixed<false>,size_t... N>
constexpr ParamSpec<EnumT,IsFixed> EnumParam(const char *name, const char *displayName, const EnumT::type defaultVal,
                                     const char (&... strings)[N])
{
  return {EnumT(name, displayName, defaultVal, strings...), std::make_tuple(),IsFixed{}};
}

template <typename IsFixed = Fixed<false>,size_t N, typename... Constraints>
constexpr ParamSpec<FloatArrayT,IsFixed, Constraints...> FloatArrayParam(const char *name, const char *displayName,
                                                                 FloatArrayT::type::value_type (&defaultValues)[N],
                                                                 Constraints... c)
{
  return {FloatArrayT(name, displayName, defaultValues), std::make_tuple(c...),IsFixed{}};
}

template <typename IsFixed = Fixed<false>,size_t N, typename... Constraints>
constexpr ParamSpec<LongArrayT,IsFixed, Constraints...> LongArrayParam(const char *name, const char *displayName,
                                                               LongArrayT::type::value_type (&defaultValues)[N],
                                                               const Constraints... c)
{
  return {LongArrayT(name, displayName, defaultValues), std::make_tuple(c...),IsFixed{}};
}

template <typename IsFixed = Fixed<false>,typename... Constraints>
constexpr ParamSpec<BufferArrayT,IsFixed, Constraints...> BufferArrayParam(const char *name, const char *displayName,
                                                                   const Constraints... c)
{
  return {BufferArrayT(name, displayName,0), std::make_tuple(c...),IsFixed{}};
}

template <typename IsFixed = Fixed<false>,typename... Constraints>
constexpr ParamSpec<FloatPairsArrayT,IsFixed, Constraints...> FloatPairsArrayParam(const char *name, const char *displayName,
                                                                           const Constraints... c)
{
  return {FloatPairsArrayT(name, displayName), std::make_tuple(c...),IsFixed{}};
}

template <typename T> std::ostream &operator<<(std::ostream &o, const std::unique_ptr<T> &p) { return o << p.get(); }

template <typename T, typename U> std::ostream &operator<<(std::ostream &o, const std::unique_ptr<T, U> &p)
{
  return o << p.get();
}

namespace impl {
template <typename T> class ParameterValueBase
{
public:
  using ParameterType = T;
  using type          = typename T::type;

  ParameterValueBase(const T descriptor, type&& v): mDescriptor(descriptor), mValue(std::move(v))
  {}

  ParameterValueBase(const T descriptor) : mDescriptor(descriptor), mValue(mDescriptor.defaultValue)
  {}

  bool        enabled() const noexcept { return true; }
  bool        changed() const noexcept { return mChanged; }
  const char *name() const noexcept { return mDescriptor.name; }
  const T     descriptor() const noexcept { return mDescriptor; }

  type &get() noexcept{ return mValue;}

  void  set(type &&value)
  {
    mValue = std::move(value);
    mChanged = true;
  }

private:
  const T mDescriptor;

protected:
  bool mChanged{false};
  type mValue;
};
} // namespace impl

template <typename T> class ParameterValue : public impl::ParameterValueBase<T>
{
public:
  using type          =  typename impl::ParameterValueBase<T>::type;
  ParameterValue(const T descriptor): impl::ParameterValueBase<T>(descriptor)
  {}
  
  ParameterValue(const T descriptor,type&& value):
    impl::ParameterValueBase<T>(descriptor,std::move(value))
  {
  }
};

} // namespace client
} // namespace fluid

