#pragma once

//#include "FluidParams.hpp"
#include "ParameterConstraints.hpp"
#include "ParameterDescriptor.hpp"
#include "ParameterInstanceList.hpp"
#include <cassert>
#include <map>
#include <vector>

namespace fluid {
namespace client {

struct Constraint_t {};

template <typename ParamEnum> class ParameterDescriptorList {
//  ParameterDescriptor p;
  using const_iterator = std::vector<ParameterDescriptor>::const_iterator;

public:
  const_iterator begin() const noexcept { return mContainer.cbegin(); }
  const_iterator end() const noexcept { return mContainer.cend(); }
  size_t size() const noexcept { return mContainer.size(); }

  const ParameterDescriptor &operator[](ParamEnum index) const {
    assert(static_cast<int>(index) < mContainer.size() &&
           "Parameter index out of range");
    return mContainer[static_cast<int>(index)];
  }

  ParameterDescriptorList() {
    static_assert(std::is_enum<ParamEnum>(),
                  "Parameter set indexer must be an enum type");
  }


  constexpr void addRelationalConstraint(ParamEnum first, ParamEnum second,
                                         Constraint_t f) {
    mConstraintSpecs.emplace(std::make_pair(first, second), f);
  }

  template <typename... Args> void add(Args &&... args) {
    mContainer.emplace_back(args...);
  }

  ParameterInstanceList<ParamEnum> makeInstances() { return {*this}; }

private:
  std::map<std::pair<ParamEnum, ParamEnum>, Constraint_t> mConstraintSpecs;
  std::vector<ParameterDescriptor> mContainer;
};

// template<typename ParamEnum>
// auto ParameterDescriptorList<ParamEnum>::mConstraints =  std::make_tuple();
//

} // namespace client
} // namespace fluid

