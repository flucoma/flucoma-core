#pragma once

#include "data/FluidIndex.hpp"
#include "data/FluidTensor.hpp"
#include "data/TensorTypes.hpp"
#include <iomanip>
#include <iostream>
#include <string>
#include <unordered_map>

namespace fluid {
namespace algorithm {

class LabelSetEncoder {
  using string = std::string;
  using StringVector = FluidTensor<string, 1>;
  using LabelSet = FluidDataSet<string, string, 1>;

public:
  void fit(LabelSet labels) {
    StringVector tmp(1);
    auto ids = labels.getIds();
    for (index i = 0; i < labels.size(); i++) {
      labels.get(ids(i), tmp);
      string label = tmp(0);
      auto pos = mLabelsMap.find(label);
      if (pos == mLabelsMap.end())
        mLabelsMap[label] = mNumLabels++;
    }
    mLabels = StringVector(mNumLabels);
    for (auto l : mLabelsMap)
      mLabels(l.second) = l.first;
  }

  index encodeIndex(string label) const{
    auto pos = mLabelsMap.find(label);
    if (pos != mLabelsMap.end()) return pos->second;
    else return -1;
  }

  std::string decodeIndex(index in) const{
    if(in < mLabels.size()) return mLabels(in);
    else return "";
  }

  void encodeOneHot(string label, RealVectorView out) const{
    assert(out.size() == mNumLabels);
    RealVector result(mNumLabels);
    result.fill(0.0);
    out.fill(0.0);
    auto pos = mLabelsMap.find(label);
    if (pos != mLabelsMap.end()) out(pos->second) = 1.0;
  }

  std::string decodeOneHot(RealVectorView in) const{
    double maxVal = 0;
    index maxIndex = 0;
    for (index i = 0; i < in.size(); i++) {
      if (in(i) > maxVal) {
        maxIndex = i;
        maxVal = in(i);
      }
    }
    return mLabels(maxIndex);
  }
  index numLabels() const{
    return mNumLabels;
  }

  void init(FluidTensor<string, 1> labels){
    mLabelsMap.clear();
    mLabels = labels;
    for(index i = 0; i < mLabels.size(); i++){
      mLabelsMap[mLabels(i)] = i;
    }
    mNumLabels = mLabels.size();
  }
  void getLabels(FluidTensorView<string, 1> out) const{
    out = mLabels;
  }

private:
  index mNumLabels{0};
  std::unordered_map<string, index> mLabelsMap;
  FluidTensor<string, 1> mLabels;
};
} // namespace algorithm
} // namespace fluid
