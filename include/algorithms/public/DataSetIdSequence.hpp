#pragma once

#include "data/FluidIndex.hpp"
#include "data/FluidTensor.hpp"
#include <iomanip>
#include <iostream>
#include <string>


namespace fluid {
namespace algorithm {

struct DataSetIdSequence {
  using string = std::string;

public:
  DataSetIdSequence(string prefix, index start = 0, index end = 0)
      : mPrefix(prefix), mStart(start), mEnd(end), mNextId(start) {}

  string next() {
    std::stringstream id;
    id << mPrefix << mNextId++;
    if (mEnd > 0 && mNextId > mEnd)
      mNextId = mStart;
    return id.str();
  }

  void generate(FluidTensorView<string,1> ids){
      for(index i = 0; i < ids.rows(); i++){
        ids(i) = next();
      }
  }

  void reset() { mNextId = mStart; }

private:
  string mPrefix;
  index mStart{0};
  index mEnd{0};
  index mNextId{0};
};
} // namespace algorithm
} // namespace fluid
