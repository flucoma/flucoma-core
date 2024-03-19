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

#include "STFT.hpp"
#include "../util/AlgorithmUtils.hpp"
#include "../util/FluidEigenMappings.hpp"
#include "../util/Munkres.hpp"
#include "../util/OptimalTransport.hpp"
#include "../util/RTPGHI.hpp"
#include "../../data/FluidMemory.hpp"
#include "../../data/TensorTypes.hpp"
#include <Eigen/Core>
#include <Eigen/Dense>
#include <vector>

namespace fluid {
namespace algorithm {

class NMFMorph
{

public:
  using MatrixXd = Eigen::MatrixXd;

  NMFMorph(index maxFFTSize, Allocator& alloc)
      : mW1(0, 0, alloc), mW2(0, 0, alloc), mH(0, 0, alloc),
        mRTPGHI(maxFFTSize, alloc), mOT(alloc)
  {}

  void init(RealMatrixView W1, RealMatrixView W2, RealMatrixView H,
            index winSize, index fftSize, index hopSize, bool assign,
            Allocator& alloc)
  {
    using namespace Eigen;
    using namespace _impl;
    using namespace std;
    mInitialized = false;
    index maxSize = std::max(W1.cols(), W2.cols());
    mW1 = ScopedEigenMap<MatrixXd>(W1.cols(), W1.rows(), alloc);
    mW1 = asEigen<Matrix>(W1).transpose();

    mH = ScopedEigenMap<MatrixXd>(H.rows(), H.cols(), alloc);
    mH = asEigen<Matrix>(H);

    if (assign)
    {
      ScopedEigenMap<MatrixXd> tmpW2(W2.cols(), W2.rows(), alloc);
      tmpW2 = asEigen<Matrix>(W2).transpose();
      ScopedEigenMap<ArrayXXd> cost(mW1.cols(), tmpW2.cols(), alloc);
      cost.setZero();
      OptimalTransport tmpOT(maxSize, alloc);
      for (index i = 0; i < mW1.cols(); i++)
      {
        for (index j = 0; j < tmpW2.cols(); j++)
        {
          tmpOT.init(mW1.col(i), tmpW2.col(j), alloc);
          if (!tmpOT.initialized()) return;
          cost(i, j) = tmpOT.mDistance;
        }
      }
      Munkres munk(mW1.cols(), tmpW2.cols(), alloc);
      ScopedEigenMap<ArrayXi> result(mW1.cols(), alloc);
      result.setZero();
      munk.process(cost, result, alloc);
      mW2 = ScopedEigenMap<MatrixXd>(tmpW2.rows(), tmpW2.cols(), alloc);
      mW2.setZero();
      for (index i = 0; i < result.size(); i++)
      {
        mW2.col(i) = tmpW2.col(result(i));
      }
    }
    else
    {
      mW2 = ScopedEigenMap<MatrixXd>(W2.cols(), W2.rows(), alloc);
      mW2 = asEigen<Matrix>(W2).transpose();
    }
    mWindowSize = winSize;
    mFFTSize = fftSize;
    mHopSize = hopSize;
    mRTPGHI.init(fftSize);

    index rank = mW1.cols();
    mOT = rt::vector<OptimalTransport>(alloc);
    mOT.reserve(asUnsigned(rank));
    for (index i = 0; i < rank; i++)
    {
      mOT.emplace_back(maxSize, alloc);
      mOT.back().init(mW1.col(i), mW2.col(i), alloc);
    }
    mPos = 0;
    mInitialized = true;
  }

  bool initialized() const { return mInitialized; }

  void processFrame(ComplexVectorView v, double interpolation, Allocator& alloc)
  {
    using namespace Eigen;
    using namespace _impl;
    ScopedEigenMap<MatrixXd> W(mW1.rows(), mW1.cols(), alloc);
    W.setZero();
    ScopedEigenMap<ArrayXd> out(mW2.rows(), alloc);
    for (int i = 0; i < W.cols(); i++)
    {
      out.setZero();
      mOT[asUnsigned(i)].interpolate(interpolation, out);
      W.col(i) = out;
    }
    ScopedEigenMap<VectorXd> hFrame(mH.rows(), alloc);
    hFrame = mH.col(mPos);
    ScopedEigenMap<VectorXd> frame(W.rows(), alloc);
    frame = W * hFrame;
    RealVectorView mag1 = asFluid(frame);
    mRTPGHI.processFrame(mag1, v, mWindowSize, mFFTSize, mHopSize, 1e-6, alloc);
    mPos = (mPos + 1) % mH.cols();
  }

private:
  ScopedEigenMap<MatrixXd>     mW1;
  ScopedEigenMap<MatrixXd>     mW2;
  ScopedEigenMap<MatrixXd>     mH;
  index                        mWindowSize;
  index                        mHopSize;
  index                        mFFTSize;
  RTPGHI                       mRTPGHI;
  rt::vector<OptimalTransport> mOT;
  int                          mPos{0};
  bool                         mInitialized;
};
} // namespace algorithm
} // namespace fluid
