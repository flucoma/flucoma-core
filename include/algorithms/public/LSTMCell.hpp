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

// MATRIX WIGHT NAMING SYSTEM BECAUSE IT IS VERY SUCCINCT SINCE I DONT WANT
// REALLY LONG LINES
// regex: m(E[MA])?D?[WB][igfo]
//
// m prefix => code style, member of class
// E => not real matrix, is a view (Eigen::Map)
// [MA] => eigen matrix view and eigen array view, respectively
// D => derivative of relevant matrix
// [WB] => weight matrix or bias vector
// [igfo] => what the weight is for: input gate, input weighting, forget gate,
//           output gate, respectively
// p => previous state
//
// e.g. mWi -> input gate weights
//      mCp -> previous cell state vector
//      mDBo -> derivative of output gate biases
//      mEMWf -> eigen matrix map of the forget gate weights
//      mEADBf -> eigen array map of the derivative of the forget gate biases

namespace fluid {
namespace algorithm {

// many thanks to Nico from https://nicodjimenez.github.io/2014/08/08/lstm.html,
// you saved my sanity here

class LSTMParam
{
  using VectorXd = Eigen::VectorXd;
  using MatrixXd = Eigen::MatrixXd;
  using ArrayXd = Eigen::ArrayXd;
  using ArrayXXd = Eigen::ArrayXXd;

  using EigenMatrixMap = Eigen::Map<MatrixXd>;
  using EigenVectorMap = Eigen::Map<VectorXd>;
  using EigenArrayXMap = Eigen::Map<ArrayXd>;
  using EigenArrayXXMap = Eigen::Map<ArrayXXd>;

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

        // create matrix eigen maps for the weights
        mEMWi(mWi.data(), mWi.rows(), mWi.cols()),
        mEMWg(mWg.data(), mWg.rows(), mWg.cols()),
        mEMWf(mWf.data(), mWf.rows(), mWf.cols()),
        mEMWo(mWo.data(), mWo.rows(), mWo.cols()),

        // create matrix eigen maps for the biases
        mEMBi(mBi.data(), mBi.size()), mEMBg(mBg.data(), mBg.size()),
        mEMBf(mBf.data(), mBf.size()), mEMBo(mBo.data(), mBo.size()),

        // create matrix eigen maps for the weight derivatives
        mEMDWi(mDWi.data(), mDWi.rows(), mDWi.cols()),
        mEMDWg(mDWg.data(), mDWg.rows(), mDWg.cols()),
        mEMDWf(mDWf.data(), mDWf.rows(), mDWf.cols()),
        mEMDWo(mDWo.data(), mDWo.rows(), mDWo.cols()),

        // create array eigen maps for the bias derivatives
        mEMDBi(mDBi.data(), mDBi.size()), mEMDBg(mDBg.data(), mDBg.size()),
        mEMDBf(mDBf.data(), mDBf.size()), mEMDBo(mDBo.data(), mDBo.size()),

        // create array eigen maps for the weights
        mEAWi(mWi.data(), mWi.rows(), mWi.cols()),
        mEAWg(mWg.data(), mWg.rows(), mWg.cols()),
        mEAWf(mWf.data(), mWf.rows(), mWf.cols()),
        mEAWo(mWo.data(), mWo.rows(), mWo.cols()),

        // create array eigen maps for the biases
        mEABi(mBi.data(), mBi.size()), mEABg(mBg.data(), mBg.size()),
        mEABf(mBf.data(), mBf.size()), mEABo(mBo.data(), mBo.size()),

        // create array eigen maps for the weight derivatives
        mEADWi(mDWi.data(), mDWi.rows(), mDWi.cols()),
        mEADWg(mDWg.data(), mDWg.rows(), mDWg.cols()),
        mEADWf(mDWf.data(), mDWf.rows(), mDWf.cols()),
        mEADWo(mDWo.data(), mDWo.rows(), mDWo.cols()),

        // create array eigen maps for the bias derivatives
        mEADBi(mDBi.data(), mDBi.size()), mEADBg(mDBg.data(), mDBg.size()),
        mEADBf(mDBf.data(), mDBf.size()), mEADBo(mDBo.data(), mDBo.size())
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
    mEAWi -= lr * mEADWi;
    mEAWg -= lr * mEADWg;
    mEAWf -= lr * mEADWf;
    mEAWo -= lr * mEADWo;

    mEABi -= lr * mEADBi;
    mEABg -= lr * mEADBg;
    mEABf -= lr * mEADBf;
    mEABo -= lr * mEADBo;

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

  const index mInSize, mLayerSize, mOutSize;

  // parameters
  RealMatrix mWi, mWg, mWf, mWo;
  RealMatrix mDWi, mDWg, mDWf, mDWo;
  RealVector mBi, mBg, mBf, mBo;
  RealVector mDBi, mDBg, mDBf, mDBo;

  // eigen maps to the parameters for linear algebra
  EigenMatrixMap mEMWi, mEMWg, mEMWf, mEMWo;
  EigenMatrixMap mEMDWi, mEMDWg, mEMDWf, mEMDWo;
  EigenVectorMap mEMBi, mEMBg, mEMBf, mEMBo;
  EigenVectorMap mEMDBi, mEMDBg, mEMDBf, mEMDBo;

  // eigen maps to the parameters for element-wise algebra
  EigenArrayXXMap mEAWi, mEAWg, mEAWf, mEAWo;
  EigenArrayXXMap mEADWi, mEADWg, mEADWf, mEADWo;
  EigenArrayXMap  mEABi, mEABg, mEABf, mEABo;
  EigenArrayXMap  mEADBi, mEADBg, mEADBf, mEADBo;
};

class LSTMState
{
  using VectorXd = Eigen::VectorXd;
  using ArrayXd = Eigen::ArrayXd;

  using EigenVectorMap = Eigen::Map<VectorXd>;
  using EigenArrayXMap = Eigen::Map<ArrayXd>;

public:
  LSTMState(index inputSize, index outputSize)
      : mXH(inputSize + outputSize), mCp(outputSize), mHp(outputSize),
        mI(outputSize), mG(outputSize), mF(outputSize), mO(outputSize),
        mC(outputSize), mH(outputSize), mDC(outputSize), mDH(outputSize),

        mEMXH(mXH.data(), mXH.size()), mEMI(mI.data(), mI.size()),
        mEMCp(mCp.data(), mCp.size()), mEMHp(mHp.data(), mHp.size()),
        mEMG(mG.data(), mG.size()), mEMF(mF.data(), mF.size()),
        mEMO(mO.data(), mO.size()), mEMC(mC.data(), mC.size()),
        mEMH(mH.data(), mH.size()),

        mEMDXH(mDXH.data(), mDXH.size()), mEMDC(mDC.data(), mDC.size()),
        mEMDH(mDH.data(), mDH.size()),

        mEAXH(mXH.data(), mXH.size()), mEAI(mI.data(), mI.size()),
        mEACp(mCp.data(), mCp.size()), mEAHp(mHp.data(), mHp.size()),
        mEAG(mG.data(), mG.size()), mEAF(mF.data(), mF.size()),
        mEAO(mO.data(), mO.size()), mEAC(mC.data(), mC.size()),
        mEAH(mH.data(), mH.size()),

        mEADXH(mDXH.data(), mDXH.size()), mEADC(mDC.data(), mDC.size()),
        mEADH(mDH.data(), mDH.size())
  {}

  // state at time t
  RealVector mXH, mCp, mHp, mI, mG, mF, mO, mC, mH;
  RealVector mDXH, mDC, mDH;

  // eigen maps to the states for linear algebra
  EigenVectorMap mEMXH, mEMCp, mEMHp, mEMI, mEMG, mEMF, mEMO, mEMC, mEMH;
  EigenVectorMap mEMDXH, mEMDC, mEMDH;

  // eigen maps to the states for element-wise algebra
  EigenArrayXMap mEAXH, mEACp, mEAHp, mEAI, mEAG, mEAF, mEAO, mEAC, mEAH;
  EigenArrayXMap mEADXH, mEADC, mEADH;
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
  {
    assert(inData.size() == mParam.mInSize);
    assert(prevState.size() == mParam.mOutSize);
    assert(prevData.size() == mParam.mOutSize);

    ScopedEigenMap<ArrayXd> cp(mParam.mOutSize, alloc),
        Zi(mParam.mOutSize, alloc), Zg(mParam.mOutSize, alloc),
        Zf(mParam.mOutSize, alloc), Zo(mParam.mOutSize, alloc);

    // previous state as eigen array
    cp << _impl::asEigen<Eigen::Array>(prevState);

    // concatentate input and previous output
    mState.mEMXH << _impl::asEigen<Eigen::Matrix>(inData),
        _impl::asEigen<Eigen::Matrix>(prevData);

    // matrix mult
    Zi = mParam.mEMWi * mState.mEMXH + mParam.mEMBi;
    Zg = mParam.mEMWg * mState.mEMXH + mParam.mEMBg;
    Zf = mParam.mEMWf * mState.mEMXH + mParam.mEMBf;
    Zo = mParam.mEMWo * mState.mEMXH + mParam.mEMBo;

    mState.mEAI = logistic(Zi);
    mState.mEAG = tanh(Zg);
    mState.mEAF = logistic(Zf);
    mState.mEAO = logistic(Zo);

    // elem-wise mult and sum
    mState.mEAC = mState.mEAG * mState.mEAI + cp * mState.mEAF;
    mState.mEAH = mState.mEAC * mState.mEAC;
  }

  void backwardFrame(InputRealVectorView dLdh, InputRealVectorView dLdc,
                     Allocator& alloc = FluidDefaultAllocator())
  {
    ScopedEigenMap<ArrayXd> dC(mParam.mOutSize, alloc),
        dI(mParam.mOutSize, alloc), dG(mParam.mOutSize, alloc),
        dF(mParam.mOutSize, alloc), dO(mParam.mOutSize, alloc),
        dZi(mParam.mOutSize, alloc), dZg(mParam.mOutSize, alloc),
        dZf(mParam.mOutSize, alloc), dZo(mParam.mOutSize, alloc);
  }

  LSTMState  mState;
  LSTMParam& mParam;
};

} // namespace algorithm
} // namespace fluid