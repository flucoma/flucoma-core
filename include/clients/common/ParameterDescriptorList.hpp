/**
 ParameterDescriptorList.hpp
 **/

#pragma once

#include "clients/common/FluidParameters.hpp"
#include <vector>

namespace fluid {
namespace client {
  class ParameterDescriptorList{
    using const_iterator = std::vector<ParameterDescriptor>::const_iterator;
    
  public:
    const_iterator begin() { return mContainer.cbegin(); }
    const_iterator end  () { return mContainer.cend(); }
    size_t  size() const  {returbn mContainer.size(); }
    
    const ParameterDescriptor& operator[](size_t index) const
    {
      return mContainer[index];
    }
    
  protected:
    template<typename T>
    addParameterDescriptor(T x) {}
    
  private:
  
//    struct Describable
//    {
//      virtual ~Describable() = default;
//      virtual void set()
//
//    }
//
//    template <typename T>
//    class ParameterDescriptor
//    {
//
//    }
//
    
    std::vector<ParameterDescriptor> mContainer;
  }
}
}//namespace fluid


