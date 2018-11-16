#pragma once

//#include "ParameterInstanceList.hpp"

#include <cassert>
#include <functional>
#include <iostream> //for ostream
#include <string>
#include <vector>

namespace fluid {
namespace client {

class ParameterInstance;
class ParameterInstanceList;
class BufferAdaptor;

class ParameterDescriptor {
  struct Type_t {};
public:
  struct ConstraintResult {

    ConstraintResult(bool ok, std::string errorStr)
        : mOk(ok), mErrorStr(errorStr) {}

    operator bool() { return mOk; }

  private:
    bool mOk;
    std::string mErrorStr;
  };

  using Constraints = std::function<ConstraintResult(
      const ParameterInstance &, const ParameterInstanceList &)>;

  enum class Type { kFloat, kLong, kBuffer, kEnum, kFloatArray, kLongArray, kBufferArray};

  
  struct Long_t       : Type_t{
    static constexpr Type typeTag = Type::kLong;

    template<typename...Conditions>
    constexpr Long_t(Conditions...coniditions)
    {
      //sanity check conditions? Make functors here?
    }
//    std::vector<std::function<ConstraintResult(long)>> conditions;
    const size_t fixedSize = 1;
  };
  
  struct Float_t      : Type_t{
    static constexpr Type typeTag  = Type::kFloat;

    template<typename...Conditions>
    constexpr Float_t(Conditions...coniditions)
    {
      
    }
    const size_t fixedSize = 1;
  };
  
  struct Buffer_t     : Type_t{
    static constexpr Type typeTag = Type::kBuffer;
    const size_t fixedSize = 1;
  }; // no non-relational conditions for buffer?
  
  struct Enum_t       : Type_t{
    static constexpr Type typeTag = Type::kEnum;
    
    template<size_t...N>
    Enum_t(const char* (&...string)[N]):fixedSize(sizeof...(N))
    {
      static_assert(sizeof...(N) > 0, "Fluid Param: No enum strings supplied!");
      static_assert(sizeof...(N) <= 16, "Fluid Param: : Maximum 16 things in an Enum param"  );
      
      strings[sizeof...(N)]  = {string...};
      
//      (void) std::initializer_list<int> {(strings.emplace_back(string),0)... };
    }
    
  private:


    const char* strings[16];//unilateral descision klaxon: if you have more than 16 things in an Enum, you need to rethink
    const size_t fixedSize;
  };
  
  struct FloatArray_t : Type_t{
    static constexpr Type typeTag = Type::kFloatArray;
    
    template<typename...Conditions>
    FloatArray_t(const size_t size = 0, Conditions...conditions):fixedSize(size){}
    const size_t fixedSize;
  };
  
  struct LongArray_t  : Type_t{
    static constexpr Type typeTag = Type::kLongArray;
    template<typename...Conditions>
    LongArray_t(const size_t size = 0, Conditions...conditions):fixedSize(size){}
    const size_t fixedSize;
  };
  
  struct BufferArray_t  : Type_t{
    static constexpr Type typeTag = Type::kBufferArray;
    BufferArray_t(const size_t size = 0):fixedSize(size){}
    const size_t fixedSize;
  };
  
  
//  ParameterDescriptor(std::string name, std::string dispName, Type type,
//                      double defaultVal, Constraints constraints,
//                      bool instantiation = false)
//      : mName(name), mDispName(dispName), mType(type), mDefaultVal(defaultVal),
//        mInstantiation(instantiation) {
//    assert(type != Type::kBuffer && "Cannot set default value for buffers");
//  }
//
//  ParameterDescriptor(std::string name, std::string dispName, Type type,
//                      Constraints constraints, bool instantiation = false)
//      : mName(name), mDispName(dispName), mType(type), mDefaultVal(0.0),
//        mInstantiation(instantiation) {}

  template<typename T> //TODO: SFINAE this constructor away if T != Type_t descendent
  ParameterDescriptor(const char* name, const char* displayName, T&& details)
  : mName(name), mDispName(displayName), mType(T::typeTag), mFixedSize(details.fixedSize)
  {
    
  }
  
  
  
  ParameterDescriptor(const ParameterDescriptor &d) = default;
  ParameterDescriptor(ParameterDescriptor &&d) = default;
  ParameterDescriptor &operator=(ParameterDescriptor &d) = default;
  ParameterDescriptor &operator=(ParameterDescriptor &&d) = default;

  std::string getDisplayName() const { return mName; }
  std::string getName() const { return mDispName; }
  const Type getType() const { return mType; }

  double getDefault() const { return mDefaultVal; }
  bool instantiation() const { return mInstantiation; }
  size_t getFixedSize() const { return mFixedSize; }

  bool operator==(const ParameterDescriptor &x) const {
    return mType == x.mType && mName == x.mName && mDispName == x.mDispName &&
           mInstantiation == x.mInstantiation && mDefaultVal == x.mDefaultVal;
  }
  bool operator!=(const ParameterDescriptor &x) const { return !(x == *this); }

  friend std::ostream &operator<<(std::ostream &out,
                                  const ParameterDescriptor &p) {
    out << "Parameter " << p.mName << "{ \n\tType: "
        << static_cast<std::underlying_type<Type>::type>(p.mType)
        << "\tInstantiation " << p.mInstantiation << "\tDefault "
        << p.mDefaultVal << "\n}\n";
    return out;
  }

private:
  Type_t mDetails;
  std::string mName;
  std::string mDispName;
  const Type mType;
  Type_t typeInfo;
  double mDefaultVal;
  bool mInstantiation;
  size_t mFixedSize = 0;
  std::vector<std::string> mEnumStrings;
  
 
  
};

} // namespace client
} // namespace fluid
