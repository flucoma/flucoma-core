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

#include "../util/FluidEigenMappings.hpp"
#include "../../data/FluidDataSet.hpp"
#include "../../data/FluidIndex.hpp"
#include "../../data/FluidMemory.hpp"
#include "../../data/FluidTensor.hpp"
#include "../../data/TensorTypes.hpp"
#include <Eigen/Core>
#include <random>

namespace fluid {
namespace algorithm {

// many thanks to Nico from https://nicodjimenez.github.io/2014/08/08/lstm.html,
// you saved my sanity here

class LSTMParam
{
  using VectorXd = Eigen::VectorXd;
  using MatrixXd = Eigen::MatrixXd;
  using ArrayXd = Eigen::ArrayXd;

  using EigenMatrixMap = Eigen::Map<MatrixXd>;
  using EigenVectorMap = Eigen::Map<VectorXd>;
  using EigenArrayMap = Eigen::Map<ArrayXd>;

public:
  LSTMParam(index inputSize, index outputSize)
      : mInSize{inputSize}, mLayerSize{inputSize + outputSize},
        mOutSize{outputSize},

        // allocate the memory for the weights
        mWi(mOutSize, mLayerSize), mWg(mOutSize, mLayerSize),
        mWf(mOutSize, mLayerSize), mWo(mOutSize, mLayerSize),

        // allocate the memory for the weight derivatives
        mDWi(mOutSize, mLayerSize), mDWg(mOutSize, mLayerSize),
        mDWf(mOutSize, mLayerSize), mDWo(mOutSize, mLayerSize),

        // allocate the memory for the biases
        mBi(mOutSize), mBg(mOutSize), mBf(mOutSize), mBo(mOutSize),

        // allocate the memory for the bias derivatives
        mDBi(mOutSize), mDBg(mOutSize), mDBf(mOutSize), mDBo(mOutSize),

        // create eigen maps for the weights
        mEWi(mWi.data(), mWi.rows(), mWi.cols()),
        mEWg(mWg.data(), mWg.rows(), mWg.cols()),
        mEWf(mWf.data(), mWf.rows(), mWf.cols()),
        mEWo(mWo.data(), mWo.rows(), mWo.cols()),

        // create eigen maps for the wight derivatives
        mEDWi(mDWi.data(), mDWi.rows(), mDWi.cols()),
        mEDWg(mDWg.data(), mDWg.rows(), mDWg.cols()),
        mEDWf(mDWf.data(), mDWf.rows(), mDWf.cols()),
        mEDWo(mDWo.data(), mDWo.rows(), mDWo.cols()),

        // create eigen maps for the biases
        mEBi(mBi.data(), mBi.size()), mEBg(mBg.data(), mBg.size()),
        mEBf(mBf.data(), mBf.size()), mEBo(mBo.data(), mBo.size()),

        // create eigen maps for the bias derivatives
        mEDBi(mDBi.data(), mDBi.size()), mEDBg(mDBg.data(), mDBg.size()),
        mEDBf(mDBf.data(), mDBf.size()), mEDBo(mDBo.data(), mDBo.size())
  {
    std::random_device rnd_device;
    std::mt19937       mersenne_engine{rnd_device()};

    std::uniform_real_distribution<double> dist{-1.0, 1.0};
    auto gen = [&dist, &mersenne_engine]() { return dist(mersenne_engine); };

    std::generate(mWi.begin(), mWi.end(), gen);
    std::generate(mWi.begin(), mWi.end(), gen);
    std::generate(mWi.begin(), mWi.end(), gen);
    std::generate(mWi.begin(), mWi.end(), gen);

    std::generate(mWi.begin(), mWi.end(), gen);
    std::generate(mWi.begin(), mWi.end(), gen);
    std::generate(mWi.begin(), mWi.end(), gen);
    std::generate(mWi.begin(), mWi.end(), gen);
  };

  void apply(double lr)
  {
    mEWi -= lr * mEDWi;
    mEWg -= lr * mEDWg;
    mEWf -= lr * mEDWf;
    mEWo -= lr * mEDWo;

    mEBi -= lr * mEDBi;
    mEBg -= lr * mEDBg;
    mEBf -= lr * mEDBf;
    mEBo -= lr * mEDBo;

    // clear weight derivatives
    mDWi.fill(0.0);
    mDWg.fill(0.0);
    mDWf.fill(0.0);
    mDWo.fill(0.0);

    // clear bias derivatives
    mDBi.fill(0.0);
    mDBg.fill(0.0);
    mDBf.fill(0.0);
    mDBo.fill(0.0);
  }

  index mInSize, mLayerSize, mOutSize;

  // parameters
  RealMatrix mWi, mWg, mWf, mWo;
  RealMatrix mDWi, mDWg, mDWf, mDWo;
  RealVector mBi, mBg, mBf, mBo;
  RealVector mDBi, mDBg, mDBf, mDBo;

  // eigen maps to the parameters
  EigenMatrixMap mEWi, mEWg, mEWf, mEWo;
  EigenMatrixMap mEDWi, mEDWg, mEDWf, mEDWo;
  EigenVectorMap mEBi, mEBg, mEBf, mEBo;
  EigenVectorMap mEDBi, mEDBg, mEDBf, mEDBo;
};

class LSTMState
{
  using VectorXd = Eigen::VectorXd;
  using MatrixXd = Eigen::MatrixXd;
  using ArrayXd = Eigen::ArrayXd;

  using EigenMatrixMap = Eigen::Map<MatrixXd>;
  using EigenVectorMap = Eigen::Map<VectorXd>;
  using EigenArrayMap = Eigen::Map<ArrayXd>;

public:
  LSTMState(index inputSize, index outputSize)
      : mI(outputSize), mG(outputSize), mF(outputSize), mO(outputSize),
        mC(outputSize), mH(outputSize), mDC(outputSize), mDH(outputSize),

        mEI(mI.data(), mI.size()), mEG(mG.data(), mG.size()),
        mEF(mF.data(), mF.size()), mEO(mO.data(), mO.size()),
        mEC(mC.data(), mC.size()), mEH(mH.data(), mH.size()),
        mEDC(mDC.data(), mDC.size()), mEDH(mDH.data(), mDH.size())
  {}

  // state at time t
  RealVector mI, mG, mF, mO, mC, mH;
  RealVector mDC, mDH;

  // eigen maps to the states for schmancy maths
  EigenVectorMap mEI, mEG, mEF, mEO, mEC, mEH;
  EigenVectorMap mEDC, mEDH;
};

class LSTMCell
{
  using VectorXd = Eigen::VectorXd;
  using MatrixXd = Eigen::MatrixXd;
  using ArrayXd = Eigen::ArrayXd;

  using EigenMatrixMap = Eigen::Map<MatrixXd>;
  using EigenVectorMap = Eigen::Map<VectorXd>;
  using EigenArrayMap = Eigen::Map<ArrayXd>;

public:
  LSTMCell(LSTMParam& param)
      : mParam(param), mState(param.mInSize, param.mOutSize){};

  void forwardFrame(InputRealVectorView inData, InputRealVectorView prevState,
                    InputRealVectorView prevData, RealVectorView outState,
                    RealVectorView outData,
                    Allocator&     alloc = FluidDefaultAllocator())
  {}

  LSTMState  mState;
  LSTMParam& mParam;
};

} // namespace algorithm
} // namespace fluid