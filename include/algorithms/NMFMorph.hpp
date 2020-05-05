#pragma once

#include "algorithms/OptimalTransport.hpp"
#include "algorithms/RTPGHI.hpp"
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
  using  MatrixXd = Eigen::MatrixXd;

  void init(RealMatrixView W1, RealMatrixView W2, RealMatrixView H,
            index winSize, index fftSize, index hopSize, bool assign) {
    using namespace Eigen;
    using namespace _impl;
    using namespace std;
    mW1 = asEigen<Matrix>(W1).transpose();
    mH = asEigen<Matrix>(H);
    MatrixXd tmpW2 = asEigen<Matrix>(W2).transpose();
    ArrayXXd cost = ArrayXXd::Zero(mW1.cols(), tmpW2.cols());
    if (assign) {
        for (index i = 0; i < mW1.cols(); i++) {
          for (index j = 0; j < tmpW2.cols(); j++) {
            OptimalTransport tmpOT;
            tmpOT.init(mW1.col(i), tmpW2.col(j));
            cost(i, j) = tmpOT.mDistance;
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
  RTPGHI mRTPGHI;
  std::vector<OptimalTransport> mOT;
  int mPos{0};
};
} // namespace algorithm
} // namespace fluid
