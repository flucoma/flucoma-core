  /**
  @file FluidParams.hpp
 
 Templates for defining and querying parameters of fluid decomposition client objects,
 to enable argument checking from host wrappers
 
 These will generally be static members of their client classes
 
 **/
#pragma once


#include "data/FluidTensor.hpp"

#include <utility> //for swap
#include <iostream>//for ostream
#include <limits> //for numeric_limits
#include <string> //for std string
#include <cassert>//for assert
#include <vector> //for lookup

namespace fluid {
namespace client{
  /**
   Plain base class, just has a parameter name and whether its instantiation-only
   
   Exists so that we can have polymorphic containers of Params, irrespective of their mix-ins
   **/

enum class Type { kFloat, kLong, kBuffer, kEnum };

class BufferAdaptor //: public FluidTensorView<float,2>
{
public:
  class Access {
  public:
    Access(BufferAdaptor *adaptor) : mAdaptor(adaptor) {
      if (mAdaptor)
        mAdaptor->acquire();
    }

    ~Access() {
      if (mAdaptor)
        mAdaptor->release();
    }

    Access(const Access &) = delete;
    Access &operator=(const Access &) = delete;

    void destroy() {
      if (mAdaptor)
        mAdaptor->release();
      mAdaptor = nullptr;
    }

    bool valid() { return mAdaptor ? mAdaptor->valid() : false; }

    void resize(size_t frames, size_t channels, size_t rank) {
      if (mAdaptor)
        mAdaptor->resize(frames, channels, rank);
    }

    FluidTensorView<float, 1> samps(size_t channel, size_t rankIdx = 0) {
      assert(mAdaptor);
      return mAdaptor->samps(channel, rankIdx);
    }

    //      FluidTensorView<float,2> samps()
    //      {
    //        assert(mAdaptor);
    //        return mAdaptor->samps();
    //      }
    //
    FluidTensorView<float, 1> samps(size_t offset, size_t nframes,
                                    size_t chanoffset) {
      assert(mAdaptor);
      return mAdaptor->samps(offset, nframes, chanoffset);
    }

    size_t numFrames() const { return mAdaptor ? mAdaptor->numFrames() : 0; }

    size_t numChans() const { return mAdaptor ? mAdaptor->numChans() : 0; }

    size_t rank() const { return mAdaptor ? mAdaptor->rank() : 0; }

  private:
    BufferAdaptor *mAdaptor;
  };

  BufferAdaptor(BufferAdaptor &&rhs) = default;
  BufferAdaptor() = default;

  virtual ~BufferAdaptor() {
    //      destroy();
  }

  bool operator==(BufferAdaptor &rhs) const { return equal(&rhs); }

  bool operator!=(BufferAdaptor &rhs) const { return !(*this == rhs); }

private:
  virtual bool equal(BufferAdaptor *rhs) const = 0;
  //    virtual void assign() = 0;
  //    virtual void destroy() = 0;
  virtual void acquire() = 0;
  virtual void release() = 0;
  virtual bool valid() const = 0;
  virtual void resize(size_t frames, size_t channels, size_t rank) = 0;
  // Return a slice of the buffer
  virtual FluidTensorView<float, 1> samps(size_t channel,
                                          size_t rankIdx = 0) = 0;
  // Return a view of all the data
  //    virtual FluidTensorView<float,2> samps() = 0;
  virtual FluidTensorView<float, 1> samps(size_t offset, size_t nframes,
                                          size_t chanoffset) = 0;
  virtual size_t numFrames() const = 0;
  virtual size_t numChans() const = 0;
  virtual size_t rank() const = 0;
  };

  
  struct Descriptor{
    
    Descriptor(const Descriptor& d)
    {
      mName = d.mName;
      mDisplayName = d.mDisplayName;
      mInstantiation = d.mInstantiation;
      mType = d.mType;
      mMin = d.mMin;
      mMax = d.mMax;
      mDefault = d.mDefault;
      mHasDefault = d.mHasDefault;
    }
    
    Descriptor(const Descriptor&& d)
    {
      mName = d.mName;
      mDisplayName = d.mDisplayName;
      mInstantiation = d.mInstantiation;
      mType = d.mType;
      mMin = d.mMin;
      mMax = d.mMax;
      mDefault = d.mDefault;
      mHasDefault = d.mHasDefault;
    }
    
    Descriptor& operator=(Descriptor d)
    {
      std::swap(*this,d);
      return *this;
    }
    
    Descriptor(std::string name,std::string dispName, Type type):mName(name),mType(type),mDisplayName(dispName){

      if (mType == Type::kBuffer) {
        mMin = 0;
        mMax = 0;
      }
    }
    
    std::string  getDisplayName() const {return mDisplayName; }
    std::string  getName() const { return mName; }
    Type getType() const { return mType; }
    
    Descriptor& setMin(double  min)
    {
      assert(mType != Type::kBuffer);
      assert(min < mMax);
      mMin = min;
      return *this;
    }
    
    double getMin() const { return mMin; }
    
    Descriptor& setMax(double max)
    {
      assert(mType != Type::kBuffer);
      assert(max > mMin);
      mMax = max;
      return *this;
    }
    
    double getMax() const { return mMax; }
    
    Descriptor& setClip(double min, double max){
      assert(min < max);
      mMin = min;
      mMax = max;
      return *this;
    }
    
    Descriptor& setDefault(double def)
    {
      assert(mType != Type::kBuffer);
      assert(def <= mMax);
      assert(def >= mMin);
      mDefault = def;
      mHasDefault = true;
      return *this;
    }
    
    double getDefault()const { return mDefault;}
    
    Descriptor& setInstantiation(bool i)
    {
      mInstantiation = i;
      return *this;
    }

    bool instantiation() const {return mInstantiation;}
    bool hasDefault() const {
      return mType == Type::kBuffer ? false : mHasDefault;
    }
    
    bool hasMin() const {
      return mType == Type::kBuffer
                 ? false
                 : !(mMin == std::numeric_limits<double>::lowest());
    }
    
    bool hasMax() const {
      return mType == Type::kBuffer
                 ? false
                 : !(mMax == std::numeric_limits<double>::max());
    }
    
    bool operator==(const Descriptor& x) const
    {
      return mName == x.mName
            && (hasMin()     ? (mMin == x.mMin) : 1)
            && (hasMax()     ? (mMax == x.mMax) : 1)
            && (hasDefault() ? (mDefault == x.mDefault) : 1)
            && (mInstantiation == x.mInstantiation)
            && (mType == x.mType);
    }
    
    bool operator != (const Descriptor& x) const
    {
      return !(*this == x);
    }
    
    friend std::ostream& operator<< (std::ostream& out,const Descriptor& p)  {
      out << "Parameter " << p.mName << "{ \n";
      out << "\tType: " << static_cast<std::underlying_type<Type>::type>(p.mType) <<  '\n';
      out << "\tInstantiation " << p.mInstantiation << '\n';
      if (p.hasMin())
        out << "\tMin " << p.mMin << '\n';
      if (p.hasMax())
        out << "\tMax " << p.mMax << '\n';
      if (p.hasDefault())
        out << "\tDefault  " << p.mDefault << '\n';
      out << "}\n";
      
      return out;
    }
    
  private:
    
    std::string mName;
    bool mInstantiation = false;
    client::Type mType;
    std::string mDisplayName; 
    double mMin = std::numeric_limits<double>::lowest();
    double mMax = std::numeric_limits<double>::max();
    double mDefault = 0;
    bool mHasDefault = false;
  };
  
  class DescriptorList
  {
    
    
  private:
    
    
  };
  
  
  
  
  
  class Instance
  {
    union Value{
      BufferAdaptor* vBuffer;
      double vFloat;
      long vLong;
    };
  public:
    enum class RangeErrorType { kNone, kMin, kMax };

    Instance(const Instance& i) = delete;
    Instance& operator=(Instance i) = delete;
    
    Instance(const Instance&& i) : mDesc(i.mDesc)
    {
//      value = i.value;
      mHasChanged = i.mHasChanged;
      mValue = i.mValue;
    }

    Instance(Descriptor desc) : mDesc(desc), mValue{nullptr}
    {
      //TODO something that checks 0 is in actual range
//      value = mDesc.hasDefault() ? mDesc.getDefault() : 0;
      if(mDesc.hasDefault())
      {
        switch(mDesc.getType())
        {
        case Type::kFloat:
          mValue.vFloat = mDesc.getDefault();
          mHasChanged = false;
          break;
        case Type::kLong:
          mValue.vLong = mDesc.getDefault();
          mHasChanged = false;
          break;
        default:
          break;
        }
      }
    }

    Instance(Descriptor desc, double v) : mDesc(desc), mValue{0}
    {
      switch(mDesc.getType())
      {
      case Type::kFloat:
        mValue.vFloat = v;
        mHasChanged = true;
        break;
      case Type::kLong:
        mValue.vLong = v;
        mHasChanged = true;
        break;
      default:
        assert(false && "Why would you even?");
        break;
      }
    }
    
    ~Instance()
    {
      if (mDesc.getType() == Type::kBuffer && mValue.vBuffer)
        delete mValue.vBuffer;
    }
    
    void reset()
    {
      switch(mDesc.getType())
      {
      case Type::kFloat:
        mValue.vFloat = mDesc.hasDefault() ? mDesc.getDefault() : 0;
        mHasChanged = false;
        break;
      case Type::kLong:
        mValue.vLong = mDesc.hasDefault() ? mDesc.getDefault() : 0;
        break;
      case Type::kBuffer:
        if (mValue.vBuffer)
          delete mValue.vBuffer;
        mValue.vBuffer = nullptr;
        break;
      default:
        break;
      }
      mHasChanged = false;
      
    }
    
    
    
    void setFloat(double v)
    {
//
//      if(!v == mDesc.getDefault())
//      {
//        mHasChanged = true;
//      }

      
      switch (mDesc.getType())
      {
      case Type::kFloat:
        mValue.vFloat = v;
        break;
      case Type::kLong:
        mValue.vLong = v;
        break;
      case Type::kBuffer:
      default:
        assert(false && "Don't call this on this type of parameter");
      }
      mHasChanged = true;
    }
    
    void setLong(long v)
    {
//      if(!(static_cast<long>(v) == static_cast<long>(mDesc.getDefault())))
//      {
//        mHasChanged = true;
//      }
      
      switch (mDesc.getType())
      {
      case Type::kFloat:
        mValue.vFloat = v;
        break;
      case Type::kLong:
        mValue.vLong = v;
        break;
      case Type::kBuffer:
      default:
        assert(false && "Don't call this on this type of parameter");
      }
      mHasChanged = true;
    }
    
    void setBuffer(BufferAdaptor* p)
    {
      switch (mDesc.getType())
      {
      case Type::kBuffer: {
        if (mValue.vBuffer)
          delete mValue.vBuffer;
        mValue.vBuffer = p;
        break;
        }
        case Type::kFloat:
        case Type::kLong:
        default:
          assert(false && "Don't call this on a non-buffer parameter");
      }
      mHasChanged = true;
    }
    
    
    double getFloat() const {
      double value;
      switch (mDesc.getType())
      {
      case Type::kFloat:
        value = mValue.vFloat;
        break;
      case Type::kLong:
        value = mValue.vLong;
        break;
      case Type::kBuffer:
      default:
        value = 0; // shut the compiler up
        assert(false && "Don't call this on a non-buffer parameter");
      }
      return value;
      
    }
    
    long   getLong()  const {
      long value;
      switch (mDesc.getType())
      {
      case Type::kFloat:
        value = static_cast<long>(mValue.vFloat);
        break;
      case Type::kLong:
        value = mValue.vLong;
        break;
      case Type::kBuffer:
      default:
        value = 0; // shut the compiler up
        assert(false && "Don't call this on a buffer parameter");
      }
      return value;
    }
    
    BufferAdaptor* getBuffer() const
    {
      switch (mDesc.getType())
      {
      case Type::kBuffer:
        return mValue.vBuffer;
      case Type::kFloat:
      case Type::kLong:
      default:
        assert(false && "Don't call this on a non-buffer parameter");
        return nullptr; // shut the compiler up
      }
    }
    
    std::pair<bool, RangeErrorType> checkRange()
    {
      switch (mDesc.getType())
      {
      case Type::kBuffer:
        return std::make_pair(true, RangeErrorType::kNone);
        break;
      case Type::kFloat:
        if (mDesc.hasMin() && mValue.vFloat < mDesc.getMin()) {
          mValue.vFloat = mDesc.getMin();
          return std::make_pair(false, RangeErrorType::kMin);
        }
        if (mDesc.hasMax() && mValue.vFloat > mDesc.getMax()) {
          mValue.vFloat = mDesc.getMax();
          return std::make_pair(false, RangeErrorType::kMax);
        }
        break;
      case Type::kLong:
        if (mDesc.hasMin() && mValue.vLong < mDesc.getMin()) {
          mValue.vLong = static_cast<long>(mDesc.getMin());
          return std::make_pair(false, RangeErrorType::kMin);
        }
        if (mDesc.hasMax() && mValue.vLong > mDesc.getMax()) {
          mValue.vLong = mDesc.getMax();
          return std::make_pair(false, RangeErrorType::kMax);
        }
        break;
      default:
        break;
      }
      return std::make_pair(true, RangeErrorType::kNone);
    }
    
    bool hasChanged() const
    {
      return mHasChanged;
    }
    
    const Descriptor& getDescriptor() const
    {
      return mDesc; 
    }
    
    bool operator==(Instance& x) const
    {
      return mDesc == x.mDesc;
    }
    
    bool operator!=(Instance& x) const
    {
      return !(*this == x); 
    }
    
  private:
    const Descriptor mDesc;
//    double value;
    bool mHasChanged = false;
    Value mValue;
  };
  
  Instance& lookupParam(std::string key, std::vector<Instance>& params)
  {
    auto res = std::find_if(params.begin(), params.end(),
                         [&](const Instance& i)->bool{
                           return i.getDescriptor().getName() == key;
                         });
    
    assert(res != params.end()); //harsh, but fair    
    return *res;
  }
  
  
  // TODO: Come back to this when there's more time and spare neurons
  //Fancy schmancy mixins towards compile-time parameter description goodness
//
//
//  /**
//   Trait for attaching a minimum value to a parameter
//  **/
//  template<typename T>
//  struct Min
//  {
//    Min(T m):min(m){}
//    T min = std::numeric_limits<T>::lowest();
//  };
//
//  /**
//   Trait for attaching a maximum value to a parameter
//   **/
//  template<typename T>
//  struct Max
//  {
//    Max(T m):max(m){}
//    T max = std::numeric_limits<T>::max();
//  };
//
//  /**
//   Trait for associating a default with a parameter
//   **/
//  template<typename T>
//  struct DefaultValue
//  {
//    DefaultValue(T v): default_value(v){}
//    T default_value;
//  };
//
//  /**
//   Overloaded function for determining if a parameter has a minimum
//   **/
//  template<typename T>
//  bool hasMin(T o) { return false; }
//
//  template<typename T>
//  bool hasMin(Min<T> m) { return true; }
//
//  /**
//   Overloaded function for determining if a parameter has a maximum
//   **/
//  template<typename T>
//  bool hasMax(T o) { return false; }
//  template<typename T>
//  bool hasMax(Max<T> m) { return true; }
//
//  /**
//   Overloaded function for determining if a parameter has a default
//   **/
//  template<typename T>
//  bool hasDefault(T o) { return false; }
//
//  template<typename T>
//  bool hasDefault(DefaultValue<T> m) { return true; }
//
//
//
//  /**
//   Fancy param class that can be augmented with min, max, default
//   **/
//  template<typename T, template<typename> class ...Attr>
//  class MixParam: public Param, Attr<T>...
//  {
//  public:
//    using underlying_type = T;
//
//    template<typename ... U>
//    MixParam(std::string name, U...args): Param(name), Attr<T>(std::forward<U>(args))...{}
//
//    T& value() { return mValue; };
//
//  private:
//    T mValue;
//
//  };
  

}//namespace client
}//namespace fluid


