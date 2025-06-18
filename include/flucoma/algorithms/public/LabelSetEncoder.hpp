/*
Part of the Fluid Corpus Manipulation Project (http://www.flucoma.org/)
Copyright University of Huddersfield.
Licensed under the BSD-3 License.
See license.md file in the project root for full license information.
This project has received funding from the European Research Council (ERC)
under the European Union’s Horizon 2020 research and innovation programme
(grant agreement No 725899).
*/

#pragma once

#include "../../data/FluidIndex.hpp"
#include "../../data/FluidTensor.hpp"
#include "../../data/TensorTypes.hpp"
#include <iomanip>
#include <iostream>
#include <string>
#include <unordered_map>

namespace fluid {
namespace algorithm {

class LabelSetEncoder
{
  using string = std::string;
  using StringVector = FluidTensor<string, 1>;
  using LabelSet = FluidDataSet<string, string, 1>;

public:
  void fit(LabelSet labels)
  {
    StringVector tmp(1);
    auto         ids = labels.getIds();
    for (index i = 0; i < labels.size(); i++)
    {
      labels.get(ids(i), tmp);
      string label = tmp(0);
      auto   pos = mLabelsMap.find(label);
      if (pos == mLabelsMap.end()) mLabelsMap[label] = mNumLabels++;
    }
    mLabels = StringVector(mNumLabels);
    for (auto l : mLabelsMap) mLabels(l.second) = l.first;
    mInitialized = true;
  }

  index encodeIndex(string const& label) const
  {
    auto pos = mLabelsMap.find(label);
    if (pos != mLabelsMap.end())
      return pos->second;
    else
      return -1;
  }

  //todo: is this ever used? Will it be?
  std::string decodeIndex(index in) const
  {
    if (in < mLabels.size())
      return mLabels(in);
    else
      return "";
  }

  void encodeOneHot(string const& label, RealVectorView out) const
  {
    assert(out.size() == mNumLabels);
    RealVector result(mNumLabels);
    result.fill(0.0);
    out.fill(0.0);
    auto pos = mLabelsMap.find(label);
    if (pos != mLabelsMap.end()) out(pos->second) = 1.0;
  }

  std::string const& decodeOneHot(RealVectorView in) const
  {
    double maxVal = 0;
    index  maxIndex = 0;
    for (index i = 0; i < in.size(); i++)
    {
      if (in(i) > maxVal)
      {
        maxIndex = i;
        maxVal = in(i);
      }
    }
    assert(maxIndex < mLabels.size());
    return mLabels(maxIndex);
  }
  index numLabels() const { return mNumLabels; }

  // from JSON: expecting unique labels, as opposed to fit
  void init(FluidTensorView<string, 1> labels)
  {
    mLabelsMap.clear();
    mLabels = FluidTensor<string, 1>(labels);
    for (index i = 0; i < mLabels.size(); i++) { mLabelsMap[mLabels(i)] = i; }
    mNumLabels = mLabels.size();
    mInitialized = true;
  }

  void getLabels(FluidTensorView<string, 1> out) const { out <<= mLabels; }

  bool initialized() const { return mInitialized; }

  void clear()
  {
    mLabelsMap.clear();
    mLabels = FluidTensor<string, 1>();
    mNumLabels = 0;
    mInitialized = false;
  }

private:
  index                             mNumLabels{0};
  std::unordered_map<string, index> mLabelsMap;
  FluidTensor<string, 1>            mLabels;
  bool                              mInitialized{false};
};
} // namespace algorithm
} // namespace fluid
