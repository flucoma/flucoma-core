#pragma once

#include "FluidParams.hpp"
#include <vector>

namespace fluid {
namespace client {
  

enum class Type { kFloat, kLong, kBuffer, kEnum, kFloatArray, kLongArray, kBufferArray};

using float_t = double;
using long_t = long;
using enum_t = long;
using buffer_t = std::unique_ptr<BufferAdaptor>;
using floatarray_t = std::vector<float_t>;
using longarray_t = std::vector<long_t>;
using bufferarray_t = std::vector<buffer_t>;

struct ParamTypeBase {
  constexpr ParamTypeBase(const char* n): name(n){}
  const char* name;
};


struct FloatT: ParamTypeBase{
  static constexpr Type typeTag  = Type::kFloat;
  using type = float_t;
  constexpr FloatT(const char* name, const char* displayName, type defaultValue): ParamTypeBase(name) {}
  const std::size_t fixedSize = 1;
};

struct LongT:ParamTypeBase{
  static constexpr Type typeTag = Type::kLong;
  using type = long_t;
  constexpr LongT(const char* name, const char* displayName, type defaultValue): ParamTypeBase(name){}
  const std::size_t fixedSize = 1;
};

struct BufferT: ParamTypeBase{
  static constexpr Type typeTag = Type::kBuffer;
  using type = buffer_t;
  constexpr BufferT(const char* name, const char* displayName): ParamTypeBase(name){}
  const std::size_t fixedSize = 1;
}; // no non-relational conditions for buffer?

struct EnumT: ParamTypeBase{
  static constexpr Type typeTag = Type::kEnum;
  using type = enum_t;
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
  static constexpr Type typeTag = Type::kFloatArray;
  using type = floatarray_t;
  
  template<std::size_t N>
  FloatArrayT(const char* name, const char* displayName, type::value_type (&defaultValues) [N]):ParamTypeBase(name){}
  const std::size_t fixedSize;
};

struct LongArrayT: ParamTypeBase{
  static constexpr Type typeTag = Type::kLongArray;
  using type = longarray_t;
  template<std::size_t N>
  LongArrayT(const char* name, const char* displayName, type::value_type (&defaultValues) [N]):ParamTypeBase(name){}
  const std::size_t fixedSize;
};

struct BufferArrayT: ParamTypeBase{
  static constexpr Type typeTag = Type::kBufferArray;
  using type = bufferarray_t;
  BufferArrayT(const char* name, const char* displayName, const size_t size = 0):ParamTypeBase(name),fixedSize(size){}
  const std::size_t fixedSize;
};

//using AllowedTypes = std::tuple<FloatParam, LongParam,BufferParam, EnumParam,FloatArrayParam,LongArrayParam,BufferArrayParam>;

//template<std::size_t N>
//using GetType = typename std::tuple_element<N,AllowedTypes>::type;
//
//  ParameterTypes.hpp
//  fluid_decomposition
//
//  Created by Owen Green on 27/11/2018.
//

#ifndef ParameterTypes_h
#define ParameterTypes_h


#endif /* ParameterTypes_h */
