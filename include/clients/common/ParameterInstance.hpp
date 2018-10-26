
#pragma once

namespace fluid{
namespace client{
  
//  struct ParameterDescriptor;
  
  class ParameterRangeCheck
  {
  public:
      
    enum class RangeErrorType { kNone, kMin, kMax };
      
    ParameterRangeCheck(RangeErrorType a) : mCondition(a) {}
    operator bool() { return mCondition == RangeErrorType::kNone; }
    operator RangeErrorType() { return mCondition; }

  private:
      
    RangeErrorType mCondition;
  };
  
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
    
    ParameterRangeCheck checkRange(){}
    bool hasChanged() const{}
    const ParameterDescriptor& descriptor() const{}
  }; //ParameterInstance

} //namespace client
} //namespace fluid


