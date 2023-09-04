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
    std::generate(mWg.begin(), mWg.end(), gen);
    std::generate(mWf.begin(), mWf.end(), gen);
    std::generate(mWo.begin(), mWo.end(), gen);

    std::generate(mBi.begin(), mBi.end(), gen);
    std::generate(mBg.begin(), mBg.end(), gen);
    std::generate(mBf.begin(), mBf.end(), gen);
    std::generate(mBo.begin(), mBo.end(), gen);
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

  using ParamPtr = std::weak_ptr<LSTMParam>;
  using ParamLock = std::shared_ptr<LSTMParam>;

  LSTMState(ParamLock p)
      : mXH(p->mLayerSize), mCp(p->mOutSize), mHp(p->mOutSize), mI(p->mOutSize),
        mG(p->mOutSize), mF(p->mOutSize), mO(p->mOutSize), mC(p->mOutSize),
        mH(p->mOutSize), mDC(p->mOutSize), mDH(p->mOutSize),

        mEMXH(mXH.data(), mXH.size()), mEMI(mI.data(), mI.size()),
        mEMCp(mCp.data(), mCp.size()), mEMHp(mHp.data(), mHp.size()),
        mEMG(mG.data(), mG.size()), mEMF(mF.data(), mF.size()),
        mEMO(mO.data(), mO.size()), mEMC(mC.data(), mC.size()),
        mEMH(mH.data(), mH.size()),

        mEMDC(mDC.data(), mDC.size()), mEMDH(mDH.data(), mDH.size()),

        mEAXH(mXH.data(), mXH.size()), mEAI(mI.data(), mI.size()),
        mEACp(mCp.data(), mCp.size()), mEAHp(mHp.data(), mHp.size()),
        mEAG(mG.data(), mG.size()), mEAF(mF.data(), mF.size()),
        mEAO(mO.data(), mO.size()), mEAC(mC.data(), mC.size()),
        mEAH(mH.data(), mH.size()),

        mEADC(mDC.data(), mDC.size()), mEADH(mDH.data(), mDH.size())
  {}

public:
  LSTMState(ParamPtr p) : LSTMState(p.lock()){};

  // state at time t
  RealVector mXH, mCp, mHp, mI, mG, mF, mO, mC, mH;
  RealVector mDC, mDH;

  // eigen maps to the states for linear algebra
  EigenVectorMap mEMXH, mEMCp, mEMHp, mEMI, mEMG, mEMF, mEMO, mEMC, mEMH;
  EigenVectorMap mEMDC, mEMDH;

  // eigen maps to the states for element-wise algebra
  EigenArrayXMap mEAXH, mEACp, mEAHp, mEAI, mEAG, mEAF, mEAO, mEAC, mEAH;
  EigenArrayXMap mEADC, mEADH;
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
  using ParamType = LSTMParam;
  using ParamPtr = std::weak_ptr<ParamType>;
  using ParamLock = std::shared_ptr<ParamType>;

  LSTMCell(ParamPtr p) : mParams(p), mState(p) {}

  LSTMState& getState() { return mState; }

  void forwardFrame(InputRealVectorView inData, InputRealVectorView prevState,
                    InputRealVectorView prevData, RealVectorView outState,
                    RealVectorView outData,
                    Allocator&     alloc = FluidDefaultAllocator())
  {
    ParamLock params = mParams.lock();

    assert(inData.size() == params->mInSize);
    assert(prevState.size() == params->mOutSize);
    assert(prevData.size() == params->mOutSize);

    ScopedEigenMap<ArrayXd> cp(params->mOutSize, alloc),
        Zi(params->mOutSize, alloc), Zg(params->mOutSize, alloc),
        Zf(params->mOutSize, alloc), Zo(params->mOutSize, alloc);

    // previous state as eigen array
    cp << _impl::asEigen<Eigen::Array>(prevState);

    // concatentate input and previous output
    mState.mEMXH << _impl::asEigen<Eigen::Matrix>(inData),
        _impl::asEigen<Eigen::Matrix>(prevData);

    // matrix mult
    Zi = params->mEMWi * mState.mEMXH + params->mEMBi;
    Zg = params->mEMWg * mState.mEMXH + params->mEMBg;
    Zf = params->mEMWf * mState.mEMXH + params->mEMBf;
    Zo = params->mEMWo * mState.mEMXH + params->mEMBo;

    mState.mEAI = logistic(Zi);
    mState.mEAG = tanh(Zg);
    mState.mEAF = logistic(Zf);
    mState.mEAO = logistic(Zo);

    // elem-wise mult and sum
    mState.mEAC = mState.mEAG * mState.mEAI + cp * mState.mEAF;
    mState.mEAH = mState.mEAC * mState.mEAC;

    outState <<= mState.mC;
    outData <<= mState.mH;
  }

  void backwardFrame(InputRealVectorView dataDerivative,
                     InputRealVectorView stateDerivative,
                     RealVectorView      prevDataDerivative,
                     RealVectorView      prevStateDerivative,
                     Allocator&          alloc = FluidDefaultAllocator())
  {
    ParamLock params = mParams.lock();

    ScopedEigenMap<ArrayXd> dC(params->mOutSize, alloc),
        dLdh(params->mOutSize, alloc), dLdc(params->mOutSize, alloc),
        dI(params->mOutSize, alloc), dG(params->mOutSize, alloc),
        dF(params->mOutSize, alloc), dO(params->mOutSize, alloc);
    ScopedEigenMap<VectorXd> dXH(params->mLayerSize, alloc),
        dZi(params->mOutSize, alloc), dZg(params->mOutSize, alloc),
        dZf(params->mOutSize, alloc), dZo(params->mOutSize, alloc);

    dLdh = _impl::asEigen<Eigen::Array>(dataDerivative);
    dLdc = _impl::asEigen<Eigen::Array>(stateDerivative);

    dC = mState.mEAO * dLdh + dLdc;
    dI = mState.mEAG * dC;
    dG = mState.mEAI * dC;
    dF = mState.mEACp * dC;
    dO = mState.mEAC * dLdh;

    dZi = mState.mEAI * (1.0 - mState.mEAI) * dI;
    dZg = (1.0 - (mState.mEAG) * (mState.mEAG)) * dG;
    dZf = mState.mEAF * (1.0 - mState.mEAF) * dF;
    dZo = mState.mEAO * (1.0 - mState.mEAO) * dO;

    params->mEMDWi += dZi * mState.mEMXH.transpose();
    params->mEMDWg += dZg * mState.mEMXH.transpose();
    params->mEMDWf += dZf * mState.mEMXH.transpose();
    params->mEMDWo += dZo * mState.mEMXH.transpose();

    params->mEMDBi += dZi;
    params->mEMDBg += dZg;
    params->mEMDBf += dZf;
    params->mEMDBo += dZo;

    dXH = params->mEMWi.transpose() * dZi + params->mEMWg.transpose() * dZg +
          params->mEMWf.transpose() * dZf + params->mEMWo.transpose() * dZo;

    mState.mEADC = dC * mState.mEAF;
    mState.mEADH = dXH(Eigen::lastN(params->mOutSize));

    prevStateDerivative <<= mState.mDC;
    prevDataDerivative <<= mState.mDH;
  }

private:
  LSTMState mState;
  ParamPtr  mParams;
};

} // namespace algorithm
} // namespace fluid