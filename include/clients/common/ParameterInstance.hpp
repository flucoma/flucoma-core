
#pragma once

#include "ParameterDescriptor.hpp"

namespace fluid{
namespace client{
    
  class ParameterInstance
  {
  public:
      
    ParameterInstance(const ParameterInstance& i) = delete;
    ParameterInstance& operator=(ParameterInstance i) = delete;
    
    ParameterInstance(const ParameterInstance&& i) {}
    
    ParameterInstance(ParameterDescriptor desc) {}
    
    ~ParameterInstance() {}
    
    void reset() {}
    
    void set(double v) {}
    void set(long v) {}
    void set(BufferAdaptor* p) {}
    
    double getFloat() const {}
    long getLong() const {}
    BufferAdaptor* getBuffer() const{}
    
    bool hasChanged() const{}
    const ParameterDescriptor& descriptor() const{}
  }; //ParameterInstance

} //namespace client
} //namespace fluid


