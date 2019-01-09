#pragma once

#include <clients/common/BufferAdaptor.hpp>
#include <vector>
#include <utility>
#include <tuple>
namespace fluid {
namespace client {
  

enum class TypeTag { kFloat, kLong, kBuffer, kEnum, kFloatArray, kLongArray, kBufferArray};

using FloatUnderlyingType = double;
using LongUnderlyingType = intptr_t; //signed int equal to pointer size, k thx
using EnumUnderlyingType = intptr_t;
using BufferUnderlyingType = std::unique_ptr<BufferAdaptor>;
using FloatArrayUnderlyingType = std::vector<FloatUnderlyingType>;
using LongArrayUnderlyingType = std::vector<LongUnderlyingType>;
using BufferArrayUnderlyingType = std::vector<BufferUnderlyingType>;

struct ParamTypeBase {
  constexpr ParamTypeBase(const char* n): name(n){}
  const char* name;
};


struct FloatT: ParamTypeBase{
  static constexpr TypeTag typeTag  = TypeTag::kFloat;
  using type = FloatUnderlyingType;
  constexpr FloatT(const char* name, const char* displayName, type defaultVal): ParamTypeBase(name),defaultValue(defaultVal)  {}
  const std::size_t fixedSize = 1;
  const type defaultValue;
};

struct LongT:ParamTypeBase{
  static constexpr TypeTag typeTag = TypeTag::kLong;
  using type = LongUnderlyingType;
  constexpr LongT(const char* name, const char* displayName, const type defaultVal): ParamTypeBase(name), defaultValue(defaultVal) {}
  const std::size_t fixedSize = 1;
  const type defaultValue;
};

struct BufferT: ParamTypeBase{
  static constexpr TypeTag typeTag = TypeTag::kBuffer;
  using type = BufferUnderlyingType;
  constexpr BufferT(const char* name, const char* displayName): ParamTypeBase(name){}
  const std::size_t fixedSize = 1;
}; // no non-relational conditions for buffer?

struct EnumT: ParamTypeBase{
  static constexpr TypeTag typeTag = TypeTag::kEnum;
  using type = EnumUnderlyingType;
  template<std::size_t...N>
  EnumT(const char* name, const char* displayName, const char* (&...string)[N]):ParamTypeBase(name),fixedSize(sizeof...(N))
  {
    static_assert(sizeof...(N) > 0, "Fluid Param: No enum strings supplied!");
    static_assert(sizeof...(N) <= 16, "Fluid Param: : Maximum 16 things in an Enum param"  );
    strings[sizeof...(N)]  = {string...};
  }
  const char* strings[16];//unilateral descision klaxon: if you have more than 16 things in an Enum, you need to rethink
  const std::size_t fixedSize;
};

struct FloatArrayT: ParamTypeBase{
  static constexpr TypeTag typeTag = TypeTag::kFloatArray;
  using type = FloatArrayUnderlyingType;
  
  template<std::size_t N>
  FloatArrayT(const char* name, const char* displayName, type::value_type (&defaultValues) [N]):ParamTypeBase(name){}
  const std::size_t fixedSize;
};

struct LongArrayT: ParamTypeBase{
  static constexpr TypeTag typeTag = TypeTag::kLongArray;
  using type = LongArrayUnderlyingType;
  template<std::size_t N>
  LongArrayT(const char* name, const char* displayName, type::value_type (&defaultValues) [N]):ParamTypeBase(name){}
  const std::size_t fixedSize;
};

struct BufferArrayT: ParamTypeBase{
  static constexpr TypeTag typeTag = TypeTag::kBufferArray;
  using type = BufferArrayUnderlyingType;
  BufferArrayT(const char* name, const char* displayName, const size_t size = 0):ParamTypeBase(name),fixedSize(size){}
  const std::size_t fixedSize;
};
  
template <typename T, typename...Constraints>
using ParamSpec =  std::pair<T,std::tuple<Constraints...>>;

template<typename...Constraints>
constexpr ParamSpec<FloatT, Constraints...>FloatParam(const char* name, const char* displayName, FloatT::type defaultValue,  Constraints...c)
{
  return {FloatT(name, displayName, defaultValue), std::make_tuple(c...)};
}

template<typename...Constraints>
constexpr ParamSpec<LongT, Constraints...>LongParam(const char* name, const char* displayName, const LongT::type defaultValue,  const Constraints...c)
{
  return {LongT(name, displayName, defaultValue), std::make_tuple(c...)};
}

template<typename...Constraints>
constexpr ParamSpec<BufferT, Constraints...>BufferParam(const char* name, const char* displayName,  Constraints...c)
{
  return {BufferT(name, displayName), std::make_tuple(c...)};
}

template<typename...Constraints>
constexpr ParamSpec<EnumT, Constraints...>EnumParam(const char* name, const char* displayName,  Constraints...c)
{
  return {EnumT(name, displayName), std::make_tuple(c...)};
}

template<size_t N ,typename...Constraints>
constexpr ParamSpec<FloatArrayT, Constraints...>FloatArrayParam(const char* name, const char* displayName, FloatArrayT::type::value_type (&defaultValues) [N], Constraints...c)
{
  return {FloatArrayT(name, displayName, defaultValues), std::make_tuple(c...)};
}

template<size_t N ,typename...Constraints>
constexpr ParamSpec<LongArrayT, Constraints...>LongArrayParam(const char* name, const char* displayName, LongArrayT::type::value_type (&defaultValues) [N], Constraints...c)
{
  return {LongArrayT(name, displayName, defaultValues), std::make_tuple(c...)};
}

template<typename...Constraints>
constexpr ParamSpec<BufferArrayT, Constraints...>BufferArrayParam(const char* name, const char* displayName, Constraints...c)
{
  return {BufferArrayT(name, displayName), std::make_tuple(c...)};
}


template<typename T>
class ParameterValue
{
public:
  using ParameterType = T; 
  using type = typename T::type;
  
  ParameterValue(const T descriptor, type v): mDescriptor(descriptor), mValue(v) {}
  
  ParameterValue(const T descriptor):mDescriptor(descriptor), mValue(mDescriptor.defaultValue)
  {}

  auto& get() const noexcept { return mValue; }
  void set(type value) { mValue = value; mChanged = true; }
  bool enabled() const noexcept { return true; }
  bool changed() const noexcept { return mChanged; }
  const char* name() const noexcept {return mDescriptor.name;}
  const T descriptor() const noexcept { return mDescriptor; }
private:
  const T mDescriptor;
  type mValue;
  bool mChanged = false;
};
  
}
}

