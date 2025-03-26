/*
Part of the Fluid Corpus Manipulation Project (http://www.flucoma.org/)
Copyright University of Huddersfield.
Licensed under the BSD-3 License.
See license.md file in the project root for full license information.
This project has received funding from the European Research Council (ERC)
under the European Union’s Horizon 2020 research and innovation programme
(grant agreement No 725899).
*/

#pragma once

#include "../nrt/FluidSharedInstanceAdaptor.hpp"
#include "../../data/FluidMemory.hpp"
#include <clients/common/FluidBaseClient.hpp>
#include <memory>

namespace fluid {
namespace client {

template <typename T>
class SharedClientRef
{
  using WeakPointer = std::weak_ptr<T>;

public:
  using SharedType = NRTSharedInstanceAdaptor<std::decay_t<T>>;

  SharedClientRef() {}
  SharedClientRef(const char* name) : mName{name, FluidDefaultAllocator()} {}
  WeakPointer get() const { return {SharedType::lookup(mName)}; }
  void        set(const char* name) { mName = std::string(name); }
  const char* name() const { return mName.c_str(); }

  // Supporting machinery for making new parameter types

  struct ParamType : ParamTypeBase
  {
    using type = SharedClientRef;
    constexpr ParamType(const char* name, const char* displayName)
        : ParamTypeBase(name, displayName)
    {}
    const index fixedSize = 1;
  };

  template <typename IsFixed = Fixed<false>>
  static constexpr ParamSpec<ParamType, IsFixed>
  makeParam(const char* name, const char* displayName)
  {
    return {ParamType(name, displayName), std::make_tuple(), IsFixed{}};
  }

private:
  rt::string mName;
};

template <typename T>
using ConstSharedClientRef = SharedClientRef<const T>;

template <typename T>
using IsSharedClientRef = isSpecialization<std::decay_t<T>, SharedClientRef>;


} // namespace client
} // namespace fluid
