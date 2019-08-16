#pragma once

#include <clients/common/FluidBaseClient.hpp>
#include "../nrt/FluidSharedInstanceAdaptor.hpp"
#include <memory> 

namespace fluid {
namespace client {

template<typename T>
class SharedClientRef
{
  using WeakPointer = std::weak_ptr<const T>;
public:
  //Basic interface
  using SharedType = NRTSharedInstanceAdaptor<T>;
  using Client = T;
  SharedClientRef(){}
  SharedClientRef(const char* name):mName{name}{}
  WeakPointer get() { return {SharedType::lookup(mName)}; }
  void set(const char* name) {  mName = std::string(name); }
  const char* name() { return mName.c_str();  }
  
  //Supporting machinery for making new parameter types
  
  struct ParamType: ParamTypeBase
  {
    using type  = SharedClientRef;
    constexpr ParamType(const char *name, const char *displayName)
      : ParamTypeBase(name, displayName)
  {}
  const std::size_t fixedSize = 1;
  }; 
  
  template <typename IsFixed = Fixed<false>>
  static constexpr ParamSpec<ParamType, IsFixed>
  MakeParam(const char *name, const char *displayName)
  {
    return {ParamType(name, displayName), std::make_tuple(), IsFixed{}};
  }
  
private:
  std::string mName;
};

}
}
