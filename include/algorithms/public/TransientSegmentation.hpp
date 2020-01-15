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

namespace fluid {
namespace algorithm {

class TransientSegmentation : public algorithm::TransientExtraction
{

public:
  TransientSegmentation(size_t order, size_t iterations, double robustFactor)
      : TransientExtraction(order, iterations, robustFactor, false)
  {}

  void setDetectionParameters(double power, double threshHi, double threshLo,
                              int halfWindow = 7, int hold = 25,
                              int minSegment = 50)
  {
    TransientExtraction::setDetectionParameters(power, threshHi, threshLo,
                                                halfWindow, hold);
    mMinSegment = minSegment;
  }

  void init(size_t order, size_t iterations, double robustFactor, int blockSize,
            int padSize)
  {
    TransientExtraction::init(order, iterations, robustFactor, false, blockSize,
                              padSize);
    mLastDetection = false;
    mDebounce = 0;
  }

  int modelOrder() const { return TransientExtraction::modelOrder(); }
  int blockSize() const { return TransientExtraction::blockSize(); }
  int hopSize() const { return TransientExtraction::hopSize(); }
  int padSize() const { return TransientExtraction::padSize(); }
  int inputSize() const { return TransientExtraction::inputSize(); }
  int analysisSize() const { return TransientExtraction::analysisSize(); }

  void process(const RealVectorView input, RealVectorView output)
  {
    detect(input.data(), input.extent(0));
    const double* transientDetection = getDetect();
    for (int i = 0; i < std::min<size_t>(hopSize(), output.size()); i++)
    {
      output(i) = (transientDetection[i] && !mLastDetection && !mDebounce);
      mDebounce = output(i) == 1.0 ? mMinSegment : std::max(0, --mDebounce);
      mLastDetection = transientDetection[i] == 1.0;
    }
  }

private:
  int  mMinSegment{25};
  int  mDebounce{0};
  bool mLastDetection{false};
};

}; // namespace algorithm
}; // namespace fluid
