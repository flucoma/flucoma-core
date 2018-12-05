#pragma once

#include "../../data/TensorTypes.hpp"
#include "TransientExtraction.hpp"

namespace fluid {
namespace algorithm {

class TransientSegmentation : private algorithm::TransientExtraction {

public:
  TransientSegmentation(size_t order, size_t iterations, double robustFactor)
      : TransientExtraction(order, iterations, robustFactor, false),
        mHoldTime(25), mDebounce(0) {}

  void setDetectionParameters(double power, double threshHi, double threshLo,
                              int halfWindow = 7, int hold = 25) {
    TransientExtraction::setDetectionParameters(power, threshHi, threshLo,
                                                halfWindow, hold);
    mHoldTime = hold;
  }

  void prepareStream(int blockSize, int padSize) {
    TransientExtraction::prepareStream(blockSize, padSize);
    mDebounce = 0;
  }

  int modelOrder() const { return TransientExtraction::modelOrder(); }
  int blockSize() const { return TransientExtraction::blockSize(); }
  int hopSize() const { return TransientExtraction::hopSize(); }
  int padSize() const { return TransientExtraction::padSize(); }
  int inputSize() const { return TransientExtraction::inputSize(); }
  int analysisSize() const { return TransientExtraction::analysisSize(); }

  void process(const RealVector input, RealVector output) {
    detect(input.data(), input.extent(0));
    const double *transientDetection = getDetect();
    for (int i = 0; i < hopSize(); i++) {
      output(i) = (transientDetection[i] && !mDebounce);
      mDebounce = transientDetection[i] ? mHoldTime : std::max(0, --mDebounce);
    }
  }

private:
  //void resizeStorage() { mDetect.resize(hopSize(), 0.0); }

private:
  int mHoldTime;
  int mDebounce;
  // std::vector<double> mDetect;
};

}; // namespace algorithm
}; // namespace fluid
