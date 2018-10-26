/**
 ParameterInstanceList.hpp
 **/

#pragma once

#include "clients/common/ParameterDescriptorList.hpp"
#include <vector>

namespace fluid {
namespace client {
    
  class ParameterInstanceList
  {
    using ParameterInstance = Instance;
    using const_iterator = std::vector<ParameterInstance>::const_iterator;
    using iterator = std::vector<ParameterInstance>::iterator;

  public:
      
    ParameterInstanceList(const ParameterDescriptorList& descriptor);
      
    iterator begin() { return mContainer.begin(); }
    iterator end  () { return mContainer.end(); }
    const_iterator cbegin() const { return mContainer.cbegin(); }
    const_iterator cend() const { return mContainer.cend(); }
    size_t size() const { return mContainer.size(); }
    
    const ParameterInstance& operator[](size_t index) const
    {
      return mContainer[index];
    }
    
  private:
    
    std::vector<ParameterInstance> mContainer;
  };
    
}
} //namespace fluid


