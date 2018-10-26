#pragma once

#include <string>

namespace fluid {
namespace client {
    
  class ParameterDescriptor
  {
  public:
    
    enum class Type { kFloat, kLong, kBuffer, kEnum };
    
    ParameterDescriptor(const ParameterDescriptor& d) {}
    ParameterDescriptor(const ParameterDescriptor&& d) {}
    ParameterDescriptor& operator=(ParameterDescriptor d) {}
    ParameterDescriptor(std::string name,std::string dispName, Type type) {}
    
    std::string  getDisplayName() const {}
    std::string  getName() const {}
    Type getType() const {}
    
    ParameterDescriptor& setInstantiation(bool i) {}
    ParameterDescriptor& setMax(double max) {}
    ParameterDescriptor& setMin(double min) {}
    ParameterDescriptor& setDefault(double def) {}

    double getMin() const {}
    double getDefault() const {}
    double getMax() const {}
    bool instantiation() const {}

//    ParameterDescriptor& setClip(double min, double max){}
    
    bool hasDefault() const {}
    bool hasMin() const {}
    bool hasMax() const {}
    
    bool operator==(const ParameterDescriptor& x) const {}
    bool operator != (const ParameterDescriptor& x) const {}
    
    friend std::ostream& operator<< (std::ostream& out,const ParameterDescriptor& p) {}
  private:
  };

}
}
