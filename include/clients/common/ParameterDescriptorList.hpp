#pragma once

//#include "FluidParams.hpp"
#include "ParameterDescriptor.hpp"
#include <vector>

namespace fluid {
namespace client {

class ParameterDescriptorList {
  ParameterDescriptor p; 
  using const_iterator = std::vector<ParameterDescriptor>::const_iterator;

public:
  const_iterator begin() const { return mContainer.cbegin(); }
  const_iterator end() const { return mContainer.cend(); }
  size_t size() const { return mContainer.size(); }
  const ParameterDescriptor &operator[](size_t index) const {
    return mContainer[index];
  }
  
protected:
  template <typename ...Args>
  ParameterDescriptor& add(Args...args) {mContainer.emplace_back(args...); return mContainer.back();}

  
  
private:
  std::vector<ParameterDescriptor> mContainer;
};

} // namespace client
} // namespace fluid
