#pragma once

#include <cassert>
#include <functional>
#include <iostream> //for ostream
#include <string>

namespace fluid {
namespace client {

class ParameterInstance;
class ParameterInstanceList;

class ParameterDescriptor {
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

  enum class Type { kFloat, kLong, kBuffer, kEnum };

  ParameterDescriptor(std::string name, std::string dispName, Type type,
                      double defaultVal, Constraints constraints,
                      bool instantiation = false)
      : mName(name), mDispName(dispName), mType(type), mDefaultVal(defaultVal),
        mInstantiation(instantiation) {
    assert(type != Type::kBuffer && "Cannot set default value for buffers");
  }

  ParameterDescriptor(std::string name, std::string dispName, Type type,
                      Constraints constraints, bool instantiation = false)
      : mName(name), mDispName(dispName), mType(type), mDefaultVal(0.0),
        mInstantiation(instantiation) {}

  ParameterDescriptor(const ParameterDescriptor &d) = default;
  ParameterDescriptor(ParameterDescriptor &&d) = default;
  ParameterDescriptor &operator=(ParameterDescriptor &d) = default;
  ParameterDescriptor &operator=(ParameterDescriptor &&d) = default;

  std::string getDisplayName() const { return mName; }
  std::string getName() const { return mDispName; }
  Type getType() const { return mType; }

  double getDefault() const { return mDefaultVal; }
  bool instantiation() const { return mInstantiation; }

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
  std::string mName;
  std::string mDispName;
  Type mType;
  double mDefaultVal;
  bool mInstantiation;
};

} // namespace client
} // namespace fluid
