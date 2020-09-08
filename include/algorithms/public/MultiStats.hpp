/*
Part of the Fluid Corpus Manipulation Project (http://www.flucoma.org/)
Copyright 2017-2019 University of Huddersfield.
Licensed under the BSD-3 License.
See license.md file in the project root for full license information.
This project has received funding from the European Research Council (ERC)
under the European Union’s Horizon 2020 research and innovation programme
(grant agreement No 725899).
*/
#pragma once

#include "../util/FluidEigenMappings.hpp"
#include "../util/OutlierDetection.hpp"
#include "../../data/FluidIndex.hpp"
#include "../../data/TensorTypes.hpp"
#include "Stats.hpp"
#include "WeightedStats.hpp"
#include <Eigen/Core>
#include <cassert>
#include <cmath>

namespace fluid {
namespace algorithm {

class MultiStats
{
public:
  using ArrayXd = Eigen::ArrayXd;
  using ArrayXi = Eigen::ArrayXi;

  void init(index numDerivatives, double low, double mid, double high)
  {
    assert(numDerivatives <= 2);
    mNumDerivatives = numDerivatives;
    mLow = low / 100.0;
    mMiddle = mid / 100.0;
    mHigh = high / 100.0;
  }

  index numStats() { return 7; }

  ArrayXd diff(Eigen::Ref<ArrayXd> in){
    return in.segment(1, in.size() - 1) - in.segment(0, in.size() - 1);
  }

  void process(const RealMatrixView in, RealMatrixView out, double cutoff = -1,
               RealVectorView w = RealVectorView(nullptr, 0, 0), double threshold = 0)
  {
    using namespace Eigen;
    using namespace _impl;
    using fluid::Slice;
    assert(out.size() == in.rows() * numStats() * (mNumDerivatives + 1));
    bool weighted = w.size() > 0;
    ArrayXXd input = asEigen<Array>(in);
    ArrayXd weights = asEigen<Array>(w);
    index numChannels = input.rows();
    index numFrames = input.cols();
    ArrayXi mask = ArrayXi::Ones(numFrames);
    if(cutoff >= 0){
      for(index i = 0; i < numChannels; i++){
        ArrayXd _row = input.row(i);
        OutlierDetection().process(_row, mask, cutoff);
      }
    }
    if(weighted){
      for(index i = 0; i < numFrames; i++){
        if(weights(i) < threshold) mask(i) = 0;
      }
    }
    index numCleanFrames = mask.sum();
    Eigen::ArrayXXd filtered = ArrayXXd::Zero(numChannels, numCleanFrames);
    Eigen::ArrayXd filteredWeights;
    if(weighted) filteredWeights = ArrayXd::Zero(numCleanFrames);
    index k = 0;
    for(index j = 0; j < mask.size(); j++){
      if(mask(j) > 0) filtered.col(k) = input.col(j);
      if(weighted) filteredWeights(k) = weights(j);
      k++;
    }
    if(weighted) filteredWeights = filteredWeights / filteredWeights.sum();
    ArrayXXd result(numChannels, numStats() * (mNumDerivatives + 1));
    for(index i = 0; i < numChannels; i++){
      ArrayXd d1, d2, d1Weights, d2Weights;
      ArrayXd channel = filtered.row(i);
      result.block(i, 0, 1, numStats()) = weighted?
        WeightedStats().process(channel, weights, mLow, mMiddle, mHigh).matrix().transpose():
        Stats().process(channel, mLow, mMiddle, mHigh).matrix().transpose();
      if (mNumDerivatives > 0){
        d1 = diff(channel);
        if(weighted)d1Weights = filteredWeights.segment(1, numCleanFrames - 1);
        result.block(i, numStats(), 1, numStats()) = weighted?
          WeightedStats().process(d1, d1Weights, mLow, mMiddle, mHigh).matrix().transpose():
          Stats().process(d1, mLow, mMiddle, mHigh).matrix().transpose();
      }
      if (mNumDerivatives > 1){
        d2 = diff(d1);
        if(weighted)d2Weights = filteredWeights.segment(2, numCleanFrames - 2);
        result.block(i, 2 * numStats(), 1, numStats()) = weighted?
          WeightedStats().process(d2, d2Weights, mLow, mMiddle, mHigh).matrix().transpose():
          Stats().process(d2, mLow, mMiddle, mHigh).matrix().transpose();
      }
    }
    out = asFluid(result);
  }

  index  mNumDerivatives{0};
  double mLow{0};
  double mMiddle{0.5};
  double mHigh{1};
};
} // namespace algorithm
} // namespace fluid
