#pragma once

#include "ParameterConstraints.hpp"
#include "ParameterTypes.hpp"

#include <cassert>
#include <functional>
#include <iostream> //for ostream
#include <string>
#include <vector>

namespace fluid {
namespace client {

class ParameterInstance;

template <typename> class ParameterInstanceList;

class BufferAdaptor;

class ParameterDescriptor {
  struct Type_t {};

public:
  //  using Constraints = std::function<ConstraintResult(
  //      const ParameterInstance &, const ParameterInstanceList &)>;

  //  ParameterDescriptor(std::string name, std::string dispName, Type type,
  //                      double defaultVal, Constraints constraints,
  //                      bool instantiation = false)
  //      : mName(name), mDispName(dispName), mType(type),
  //      mDefaultVal(defaultVal),
  //        mInstantiation(instantiation) {
  //    assert(type != Type::kBuffer && "Cannot set default value for buffers");
  //  }
  //
  //  ParameterDescriptor(std::string name, std::string dispName, Type type,
  //                      Constraints constraints, bool instantiation = false)
  //      : mName(name), mDispName(dispName), mType(type), mDefaultVal(0.0),
  //        mInstantiation(instantiation) {}

  template <typename T>
  constexpr ParameterDescriptor(const char *name, const char *displayName,
                                bool instantiation, T &&details) noexcept
      : mDetails(details), mName(name), mDispName(displayName),
        mType(T::typeTag), mFixedSize(details.fixedSize),
        mInstantiation(instantiation) {
    static_assert(
        std::is_base_of<Type_t, T>(),
        "Fluid Param: Parameter Descriptor details must descend from Type_t");
  }

  ParameterDescriptor(const ParameterDescriptor &d) = default;
  ParameterDescriptor(ParameterDescriptor &&d) = default;
  ParameterDescriptor &operator=(ParameterDescriptor &d) = default;
  ParameterDescriptor &operator=(ParameterDescriptor &&d) = default;

  std::string getDisplayName() const { return mName; }
  std::string getName() const { return mDispName; }
  constexpr Type getType() const { return mType; }

  double getDefault() const { return mDefaultVal; }
  bool instantiation() const { return mInstantiation; }
  size_t getFixedSize() const { return mFixedSize; }

//  template <Type T> constexpr auto detail() const {
//
//    using t_t = GetType<static_cast<int>(T)>;
//    return static_cast<t_t>(mDetails);
//  }

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
  const Type_t mDetails;
  const char *mName;
  const char *mDispName;
  const Type mType;
  //  Type_t typeInfo;
  double mDefaultVal;
  size_t mFixedSize = 0;
  bool mInstantiation;
};

} // namespace client
} // namespace fluid

