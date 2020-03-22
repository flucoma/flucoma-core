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

#include "TransientExtraction.hpp"
#include "../../data/TensorTypes.hpp"
#include "../../data/FluidIndex.hpp"

namespace fluid {
namespace algorithm {

class TransientSegmentation : public algorithm::TransientExtraction
{

public:
  TransientSegmentation(index order, index iterations, double robustFactor)
      : TransientExtraction(order, iterations, robustFactor)
  {}

  void setDetectionParameters(double power, double threshHi, double threshLo,
                              index halfWindow = 7, index hold = 25,
                              index minSegment = 50)
  {
    TransientExtraction::setDetectionParameters(power, threshHi, threshLo,
                                                halfWindow, hold);
    mMinSegment = minSegment;
  }

  void init(index order, index iterations, double robustFactor, index blockSize,
            index padSize)
  {
    TransientExtraction::init(order, iterations, robustFactor, blockSize,
                              padSize);
    mLastDetection = false;
    mDebounce = 0;
  }

  index modelOrder() const { return TransientExtraction::modelOrder(); }
  index blockSize() const { return TransientExtraction::blockSize(); }
  index hopSize() const { return TransientExtraction::hopSize(); }
  index padSize() const { return TransientExtraction::padSize(); }
  index inputSize() const { return TransientExtraction::inputSize(); }
  index analysisSize() const { return TransientExtraction::analysisSize(); }

  void process(const RealVectorView input, RealVectorView output)
  {
    detect(input.data(), input.extent(0));
    const double* transientDetection = getDetect();
    for (index i = 0; i < std::min<index>(hopSize(), output.size()); i++)
    {
      output(i) = (transientDetection[i] != 0 && !mLastDetection && !mDebounce);
      mDebounce = output(i) == 1.0 ? mMinSegment : std::max<index>(0, mDebounce - 1);
      mLastDetection = transientDetection[i] == 1.0;
    }
  }

private:
  index mMinSegment{25};
  index mDebounce{0};
  bool  mLastDetection{false};
};

}; // namespace algorithm
}; // namespace fluid
