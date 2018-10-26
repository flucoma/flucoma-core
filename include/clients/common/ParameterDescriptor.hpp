#pragma once

#include <iostream>//for ostream
#include <string>
#include <functional>
#include <cassert>

namespace fluid {
namespace client {
    
  class ParameterDescriptor
  {
    using Constraints = std::function<void(void)>;
      
  public:
    
    enum class Type { kFloat, kLong, kBuffer, kEnum };
    
    ParameterDescriptor(std::string name, std::string dispName, Type type, double defaultVal, Constraints constraints, bool instantiation = false)
      : mName(name), mDispName(dispName), mType(type), mDefaultVal(defaultVal), mInstantiation(instantiation)
    {
      assert(type != Type::kBuffer && "Cannot set default value for buffers");
    }
    
    ParameterDescriptor(std::string name, std::string dispName, Type type, Constraints constraints, bool instantiation = false)
      : mName(name), mDispName(dispName), mType(type), mDefaultVal(0.0), mInstantiation(instantiation)
    {
    }
      
    ParameterDescriptor(const ParameterDescriptor& d) = default;
    ParameterDescriptor(ParameterDescriptor&& d) = default;
    ParameterDescriptor& operator=(ParameterDescriptor& d) = default;
    ParameterDescriptor& operator=(ParameterDescriptor&& d) = default;

    std::string getDisplayName() const { return mName; }
    std::string getName() const { return mDispName; }
    Type getType() const { return mType; }

    double getDefault() const { return mDefaultVal; }
    bool instantiation() const { return mInstantiation; }
    
    bool operator == (const ParameterDescriptor& x) const {
      return  mType          == x.mType
          &&  mName          == x.mName
          &&  mDispName      == x.mDispName
          &&  mInstantiation == x.mInstantiation
          &&  mDefaultVal    == x.mDefaultVal;
    }
    bool operator != (const ParameterDescriptor& x) const { return !(x == *this); }
    
    friend std::ostream& operator << (std::ostream& out,const ParameterDescriptor& p) {
      out << "Parameter "
          << p.mName
          << "{ \n\tType: "
          << static_cast<std::underlying_type<Type>::type>(p.mType)
          << "\tInstantiation " << p.mInstantiation
          << "\tDefault " << p.mDefaultVal
          << "\n}\n";
          return out;
    }
      
  private:
      
    std::string mName;
    std::string mDispName;
    Type mType;
    double mDefaultVal;
    bool mInstantiation;
  };

}
}
