#pragma once

#include <clients/common/BufferAdaptor.hpp>
#include <tuple>
#include <utility>
#include <vector>
namespace fluid {
namespace client {

enum class TypeTag { kFloat, kLong, kBuffer, kEnum, kFloatArray, kLongArray, kBufferArray };

using FloatUnderlyingType       = double;
using LongUnderlyingType        = intptr_t; // signed int equal to pointer size, k thx
using EnumUnderlyingType        = intptr_t;
using BufferUnderlyingType      = std::unique_ptr<BufferAdaptor>;
using FloatArrayUnderlyingType  = std::vector<FloatUnderlyingType>;
using LongArrayUnderlyingType   = std::vector<LongUnderlyingType>;
using BufferArrayUnderlyingType = std::vector<BufferUnderlyingType>;
using MagnitudePairsUnderlyingType = std::vector<std::pair<double, double>>;

struct ParamTypeBase
{
  constexpr ParamTypeBase(const char *n, const char* display)
      : name(n), displayName(display)
  {}
  const char *name;
  const char *displayName;
};

struct FloatT : ParamTypeBase
{
  static constexpr TypeTag typeTag = TypeTag::kFloat;
  using type                       = FloatUnderlyingType;
  constexpr FloatT(const char *name, const char *displayName, type defaultVal)
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
      : ParamTypeBase(name,displayName)
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
      , ParamTypeBase(name,displayName)
      , fixedSize(sizeof...(N))
      , defaultValue(defaultVal)
  {
    static_assert(sizeof...(N) > 0, "Fluid Param: No enum strings supplied!");
    static_assert(sizeof...(N) <= 16, "Fluid Param: : Maximum 16 things in an Enum param");
  }
  const char *      strings[16]; // unilateral descision klaxon: if you have more than 16 things in an Enum, you need to rethink
  const std::size_t fixedSize;
  const type        defaultValue;
};

struct FloatArrayT : ParamTypeBase
{
  static constexpr TypeTag typeTag = TypeTag::kFloatArray;
  using type                       = FloatArrayUnderlyingType;

  template <std::size_t N>
  FloatArrayT(const char *name, const char *displayName, type::value_type (&defaultValues)[N])
      : ParamTypeBase(name,displayName)
  {}
  const std::size_t fixedSize;
};

struct LongArrayT : ParamTypeBase
{
  static constexpr TypeTag typeTag = TypeTag::kLongArray;
  using type                       = LongArrayUnderlyingType;
  template <std::size_t N>
  LongArrayT(const char *name, const char *displayName, type::value_type (&defaultValues)[N])
      : ParamTypeBase(name,displayName)
  {}
  const std::size_t fixedSize;
};

struct BufferArrayT : ParamTypeBase
{
  static constexpr TypeTag typeTag = TypeTag::kBufferArray;
  using type                       = BufferArrayUnderlyingType;
  BufferArrayT(const char *name, const char *displayName, const size_t size = 0)
      : ParamTypeBase(name,displayName)
      , fixedSize(size)
  {}
  const std::size_t fixedSize;
};

//Pair of frequency amplitude pairs for HPSS threshold
struct FloatPairsArrayT: ParamTypeBase
{
//  static constexpr TypeTa
  using type = std::vector<std::pair<FloatUnderlyingType, FloatUnderlyingType>>;
  
  
  constexpr FloatPairsArrayT(const char* name, const char *displayName)
      : ParamTypeBase(name, displayName)
  {}
  const std::size_t fixedSize{2};
  static constexpr std::initializer_list<std::pair<double,double>> defaultValue{{0.0,1.0},{1.0,1.0}};
};

//My name's the C++ linker, and I'm a bit of a knob (fixed in C++17)
constexpr std::initializer_list<std::pair<double,double>> FloatPairsArrayT::defaultValue;

template <typename T, typename... Constraints> using ParamSpec = std::pair<T, std::tuple<Constraints...>>;

template <typename... Constraints>
constexpr ParamSpec<FloatT, Constraints...> FloatParam(const char *name, const char *displayName, FloatT::type defaultValue,
                                                       Constraints... c)
{
  return {FloatT(name, displayName, defaultValue), std::make_tuple(c...)};
}

template <typename... Constraints>
constexpr ParamSpec<LongT, Constraints...> LongParam(const char *name, const char *displayName,
                                                     const LongT::type defaultValue, Constraints... c)
{
  return {LongT(name, displayName, defaultValue), std::make_tuple(c...)};
}

template <typename... Constraints>
constexpr ParamSpec<BufferT, Constraints...> BufferParam(const char *name, const char *displayName, const Constraints... c)
{
  return {BufferT(name, displayName), std::make_tuple(c...)};
}

template <size_t... N>
constexpr ParamSpec<EnumT> EnumParam(const char *name, const char *displayName, const EnumT::type defaultVal,
                                     const char (&... strings)[N])
{
  return {EnumT(name, displayName, defaultVal, strings...), std::make_tuple()};
}

template <size_t N, typename... Constraints>
constexpr ParamSpec<FloatArrayT, Constraints...> FloatArrayParam(const char *name, const char *displayName,
                                                                 FloatArrayT::type::value_type (&defaultValues)[N],
                                                                 Constraints... c)
{
  return {FloatArrayT(name, displayName, defaultValues), std::make_tuple(c...)};
}

template <size_t N, typename... Constraints>
constexpr ParamSpec<LongArrayT, Constraints...>
LongArrayParam(const char *name, const char *displayName, LongArrayT::type::value_type (&defaultValues)[N], const Constraints... c)
{
  return {LongArrayT(name, displayName, defaultValues), std::make_tuple(c...)};
}

template <typename... Constraints>
constexpr ParamSpec<BufferArrayT, Constraints...> BufferArrayParam(const char *name, const char *displayName,
                                                                   const Constraints... c)
{
  return {BufferArrayT(name, displayName), std::make_tuple(c...)};
}

template <typename... Constraints>
constexpr ParamSpec<FloatPairsArrayT, Constraints...> FloatPairsArrayParam(const char *name, const char *displayName,
                                                                   const Constraints... c)
{
  return {FloatPairsArrayT(name, displayName), std::make_tuple(c...)};
}



template<typename T>
std::ostream& operator <<(std::ostream& o, const std::unique_ptr<T>& p)
{
  return o << p.get();
}

template<typename T, typename U>
std::ostream& operator <<(std::ostream& o, const std::unique_ptr<T,U>& p)
{
  return o << p.get();
}


namespace impl {
template <typename T> class ParameterValueBase
{
public:
  using ParameterType = T;
  using type          = typename T::type;

  ParameterValueBase(const T descriptor, type v)
      : mDescriptor(descriptor)
      , mValue(v)
  {}

  ParameterValueBase(const T descriptor)
      : mDescriptor(descriptor)
      , mValue(mDescriptor.defaultValue)
  {} //std::cout << mDescriptor.name << " " << mValue <<'\n'; }

  bool        enabled() const noexcept { return true; }
  bool        changed() const noexcept { return mChanged; }
  const char* name() const noexcept { return mDescriptor.name; }
  const T     descriptor() const noexcept { return mDescriptor; }

private:
  const T mDescriptor;
  
protected:
  bool mChanged{false};
  type    mValue;

};
} // namespace impl

template <typename T> class ParameterValue : public impl::ParameterValueBase<T>
{
public:
  using ParameterType = T;
  using type          = typename T::type;
  ParameterValue(const T descriptor)
      : impl::ParameterValueBase<T>(descriptor)
  {}//std::cout << descriptor.name << " " << get() <<'\n';  }
  type &get() noexcept { return impl::ParameterValueBase<T>::mValue; }
  void  set(type &&value)
  {
    std::swap(impl::ParameterValueBase<T>::mValue,value);
    impl::ParameterValueBase<T>::mChanged = true;
  }
};

//template <> class ParameterValue<BufferT> : public impl::ParameterValueBase<BufferT>
//{
//public:
//  using type = typename BufferT::type;
//  ParameterValue(const BufferT descriptor)
//      : impl::ParameterValueBase<BufferT>(descriptor)
//  {}
//  auto &get() const noexcept { return mValue; }
//  void  set(type &value)
//  {
//    swap(mValue, value);
//    mChanged = true;
//  }
//
//private:
//  type mValue;
//};

} // namespace client
} // namespace fluid
