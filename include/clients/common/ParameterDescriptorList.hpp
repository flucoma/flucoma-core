/**
 ParameterDescriptorList.hpp
 **/

#pragma once

#include "FluidParams.hpp"
#include <vector>

namespace fluid {
namespace client {
    
  using ParameterDescriptor = Descriptor;
    
  class ParameterDescriptorList
  {
    using const_iterator = std::vector<ParameterDescriptor>::const_iterator;
    
  public:
    const_iterator begin() const { return mContainer.cbegin(); }
    const_iterator end() const { return mContainer.cend(); }
    size_t size() const { return mContainer.size(); }
    
    const ParameterDescriptor& operator[](size_t index) const
    {
      return mContainer[index];
    }
    
  protected:
    template<typename T>
    void addParameterDescriptor(T x) { }
    
    std::vector<ParameterDescriptor> mContainer;
  };
    
}
}//namespace fluid


