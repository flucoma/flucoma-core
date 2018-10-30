#pragma once

#include "ParameterDescriptor.hpp"
#include "ParameterDescriptorList.hpp"
#include "ParameterInstance.hpp"
namespace fluid {
namespace client {

using Constraint = ParameterDescriptor::Constraints;
using Result = ParameterDescriptor::ConstraintResult;

struct minConstraint {
  Result operator()(const ParameterInstance &x,
                    const ParameterInstanceList &params) {
    return Result{true, ""};
  }
};
} // namespace client
} // namespace fluid
