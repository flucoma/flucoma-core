#pragma once

#include "algorithms/OptimalTransport.hpp"
#include "algorithms/RTPGHI.hpp"
#include "algorithms/DistanceFuncs.hpp"
#include "algorithms/public/STFT.hpp"
#include "algorithms/util/AlgorithmUtils.hpp"
#include "algorithms/util/FluidEigenMappings.hpp"
#include "algorithms/util/Munkres.hpp"
#include "data/TensorTypes.hpp"
#include <Eigen/Core>
#include <Eigen/Dense>
#include <vector>

namespace fluid {
namespace algorithm {

class NMFMorph {

public:

  void init(RealMatrixView W1, RealMatrixView W2, RealMatrixView H,
            index winSize, index fftSize, index hopSize, bool assign) {
    using namespace Eigen;
    using namespace _impl;
    mW1 = asEigen<Matrix>(W1).transpose();
    mH = asEigen<Matrix>(H);
    if (assign) {
      MatrixXd tmpW2 = asEigen<Matrix>(W2).transpose();
      ArrayXXd cost = ArrayXXd::Zero(mW1.cols(), tmpW2.cols());
      for (index i = 0; i < mW1.cols(); i++) {
        for (index j = 0; j < tmpW2.cols(); j++) {
          cost(i, j) = DistanceFuncs::map()[DistanceFuncs::Distance::kKL](mW1.col(i),tmpW2.col(j));
        }
      }
      Munkres munk;
      munk.init(mW1.cols(), tmpW2.cols());
      ArrayXi result = ArrayXi::Zero(mW1.cols());
      munk.process(cost, result);
      mW2 = MatrixXd::Zero(tmpW2.rows(), tmpW2.cols());
      for (index i = 0; i < result.size(); i++) {
        mW2.col(i) = tmpW2.col(result(i));
      }
    } else {
      mW2 = asEigen<Matrix>(W2).transpose();
    }
    mWindowSize = winSize;
    mFFTSize = fftSize;
    mHopSize = hopSize;
    mSTFT = STFT(winSize, fftSize, hopSize);
    mISTFT = ISTFT(winSize, fftSize, hopSize);
    mRTPGHI.init(fftSize);

    index rank = mW1.cols();
    mOT = std::vector<OptimalTransport>(rank);
    for (index i = 0; i < rank; i++) {
      mOT[i].init(mW1.col(i), mW2.col(i));
    }
  }

  void processFrame(ComplexVectorView v, double interpolation) {
    using namespace Eigen;
    using namespace _impl;
    MatrixXd W = MatrixXd::Zero(mW1.rows(), mW1.cols());
    for (int i = 0; i < W.cols(); i++) {
      ArrayXd out = ArrayXd::Zero(mW2.rows());
      mOT[i].interpolate(interpolation, out);
      W.col(i) = out;
    }

    VectorXd hFrame = mH.col(mPos);
    VectorXd frame = W * hFrame;
    RealVectorView mag1 = asFluid(frame);
    mRTPGHI.processFrame(mag1, v, mWindowSize, mFFTSize, mHopSize, 1e-6);
    mPos = (mPos + 1) % mH.cols();
  }

private:
  MatrixXd mW1;
  MatrixXd mW2;
  MatrixXd mH;
  index mWindowSize;
  index mHopSize;
  index mFFTSize;
  STFT mSTFT{1024, 1024, 512};
  ISTFT mISTFT{1024, 1024, 512};
  RTPGHI mRTPGHI;
  std::vector<OptimalTransport> mOT;
  int mPos{0};
  int mIterations;
};
} // namespace algorithm
} // namespace fluid
