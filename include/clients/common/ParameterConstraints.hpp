#pragma once

#include "ParameterDescriptor.hpp"
#include "ParameterDescriptorList.hpp"
#include "ParameterInstance.hpp"
namespace fluid {
namespace client {

using Constraint = ParameterDescriptor::Constraints;
using Result = ParameterDescriptor::ConstraintResult;
class ConstraintResult {
public:
  ConstraintResult(const bool ok, const char *errorStr) noexcept
      : mOk(ok), mErrorStr(errorStr) {}
  operator bool() const noexcept { return mOk; }
  const char *message() const noexcept { return mErrorStr; }

private:
  bool mOk;
  const char *mErrorStr;
};


struct minConstraint {
  Result operator()(const ParameterInstance &x,
                    const ParameterInstanceList &params) {
    return Result{true, ""};
  }
};
} // namespace client
} // namespace fluid
