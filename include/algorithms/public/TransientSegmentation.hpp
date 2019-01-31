#pragma once

#include "../../data/TensorTypes.hpp"
#include "TransientExtraction.hpp"

namespace fluid {
namespace algorithm {

class TransientSegmentation : private algorithm::TransientExtraction {

public:
  TransientSegmentation(size_t order, size_t iterations, double robustFactor)
      : TransientExtraction(order, iterations, robustFactor, false),
        mMinSegment(25), mDebounce(0), mLastDetection(false) {}

  void setDetectionParameters(double power, double threshHi, double threshLo,
                              int halfWindow = 7, int hold = 25,
                              int minSegment = 50) {
    TransientExtraction::setDetectionParameters(power, threshHi, threshLo,
                                                halfWindow, hold);
    mMinSegment = minSegment;
  }

  void prepareStream(int blockSize, int padSize) {
    TransientExtraction::prepareStream(blockSize, padSize);
    mLastDetection = false;
    mDebounce = 0;
  }

  int modelOrder() const { return TransientExtraction::modelOrder(); }
  int blockSize() const { return TransientExtraction::blockSize(); }
  int hopSize() const { return TransientExtraction::hopSize(); }
  int padSize() const { return TransientExtraction::padSize(); }
  int inputSize() const { return TransientExtraction::inputSize(); }
  int analysisSize() const { return TransientExtraction::analysisSize(); }

  void process(const RealVectorView input, RealVectorView output) {
    detect(input.data(), input.extent(0));
    const double *transientDetection = getDetect();
    for (int i = 0; i < std::min<size_t>(hopSize(), output.size()); i++) {
      output(i) = (transientDetection[i] && !mLastDetection && !mDebounce);
      mDebounce = output(i) == 1.0 ? mMinSegment : std::max(0, --mDebounce);
      mLastDetection = transientDetection[i] == 1.0;
    }
  }

private:
  // void resizeStorage() { mDetect.resize(hopSize(), 0.0); }

private:
  int mMinSegment;
  int mDebounce;
  bool mLastDetection;
  // std::vector<double> mDetect;
};

}; // namespace algorithm
}; // namespace fluid
