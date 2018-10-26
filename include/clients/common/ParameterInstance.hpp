
#pragma once

namespace fluid{
namespace client{
  
//  struct ParameterDescriptor;
  
  struct ParameterRangeCheck
  {
    enum class RangeErrorType { kNone, kMin, kMax };
    RangeErrorType condition;
    bool ok;
    
    operator bool() { return ok; }
  };
  
  struct ParameterInstance
  {
    
    ParameterInstance(const ParameterInstance& i) = delete;
    ParameterInstance& operator=(ParameterInstance i) = delete;
    
    ParameterInstance(const ParameterInstance&& i) {}
    
    ParameterInstance(ParameterDescriptor desc) {}
    
    ParameterInstance(ParameterDescriptor desc, double v) {}
    
    ~ParameterInstance()  {}
    
    void reset()  {}
    
    void setFloat(double v){}
    
    void setLong(long v){}
    
    void setBuffer(BufferAdaptor* p){}
    
    double getFloat() const {}
    
    long   getLong()  const {}
    
    BufferAdaptor* getBuffer() const{}
    
    ParameterRangeCheck checkRange(){}
    bool hasChanged() const{}
    const ParameterDescriptor& descriptor() const{}
    bool operator==(Instance& x) const{}
    bool operator!=(Instance& x) const{}
  }; //ParameterInstance

} //namespace client
} //namespace fluid


