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

#include "Recur.hpp"
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
        mDBi(mOutSize), mDBg(mOutSize), mDBf(mOutSize), mDBo(mOutSize)
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
    mWi.array() -= lr * mDWi.array();
    mWg.array() -= lr * mDWg.array();
    mWf.array() -= lr * mDWf.array();
    mWo.array() -= lr * mDWo.array();

    mBi.array() -= lr * mDBi.array();
    mBg.array() -= lr * mDBg.array();
    mBf.array() -= lr * mDBf.array();
    mBo.array() -= lr * mDBo.array();

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
  MatrixParam mWi, mWg, mWf, mWo;
  MatrixParam mDWi, mDWg, mDWf, mDWo;
  VectorParam mBi, mBg, mBf, mBo;
  VectorParam mDBi, mDBg, mDBf, mDBo;
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
      : mInSize(p->mInSize), mLayerSize(p->mLayerSize), mOutSize(p->mOutSize),
        mX(mInSize), mXH(mLayerSize), mCp(mOutSize), mHp(mOutSize),
        mI(mOutSize), mG(mOutSize), mF(mOutSize), mO(mOutSize), mC(mOutSize),
        mH(mOutSize), mDC(mOutSize), mDH(mOutSize)
  {}

public:
  LSTMState(ParamPtr p) : LSTMState{p.lock()} {};

  RealVector output() { return mH; }

  index mInSize, mLayerSize, mOutSize;

  // state at time t
  VectorParam mX, mXH, mCp, mHp, mI, mG, mF, mO, mC, mH;
  VectorParam mDC, mDH;
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
  // Recur typedefs
  using StateType = LSTMState;
  using ParamType = LSTMParam;
  using ParamPtr = std::weak_ptr<ParamType>;
  using ParamLock = std::shared_ptr<ParamType>;

  LSTMCell(ParamPtr p) : mParams(p), mState(p) {}

  StateType& getState() { return mState; }

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
    mState.mXH.matrix() << _impl::asEigen<Eigen::Matrix>(inData),
        _impl::asEigen<Eigen::Matrix>(prevData);

    // matrix mult
    Zi = params->mWi.matrix() * mState.mXH.matrix() + params->mBi.matrix();
    Zg = params->mWg.matrix() * mState.mXH.matrix() + params->mBg.matrix();
    Zf = params->mWf.matrix() * mState.mXH.matrix() + params->mBf.matrix();
    Zo = params->mWo.matrix() * mState.mXH.matrix() + params->mBo.matrix();

    mState.mI.array() = logistic(Zi);
    mState.mG.array() = tanh(Zg);
    mState.mF.array() = logistic(Zf);
    mState.mO.array() = logistic(Zo);

    // elem-wise mult and sum
    mState.mC.array() =
        mState.mG.array() * mState.mI.array() + cp * mState.mF.array();
    mState.mH.array() = mState.mC.array() * mState.mC.array();

    outState <<= mState.mC;
    outData <<= mState.mH;
  }

  // in many to one, all cells except the last one have no target output
  double backwardDatalessFrame(InputRealVectorView dataDerivative,
                               InputRealVectorView stateDerivative,
                               Allocator& alloc = FluidDefaultAllocator())
  {
    backwardFrame(mState.mH, dataDerivative, stateDerivative, alloc);
    return 0.0;
  }

  double backwardFrame(InputRealVectorView dataTarget,
                       InputRealVectorView dataDerivative,
                       InputRealVectorView stateDerivative,
                       Allocator&          alloc = FluidDefaultAllocator())
  {
    using _impl::asEigen;

    ParamLock params = mParams.lock();

    ScopedEigenMap<ArrayXd> dC(params->mOutSize, alloc),
        dLdh(params->mOutSize, alloc), dLdc(params->mOutSize, alloc),
        dI(params->mOutSize, alloc), dG(params->mOutSize, alloc),
        dF(params->mOutSize, alloc), dO(params->mOutSize, alloc);
    ScopedEigenMap<VectorXd> dXH(params->mLayerSize, alloc),
        dZi(params->mOutSize, alloc), dZg(params->mOutSize, alloc),
        dZf(params->mOutSize, alloc), dZo(params->mOutSize, alloc);

    dLdh = asEigen<Eigen::Array>(dataDerivative) +
           2 * (mState.mH.array() - asEigen<Eigen::Array>(dataTarget));
    dLdc = asEigen<Eigen::Array>(stateDerivative);

    dC = mState.mO.array() * dLdh + dLdc;
    dI = mState.mG.array() * dC;
    dG = mState.mI.array() * dC;
    dF = mState.mCp.array() * dC;
    dO = mState.mC.array() * dLdh;

    dZi = mState.mI.array() * (1.0 - mState.mI.array()) * dI;
    dZg = (1.0 - (mState.mG.array()) * (mState.mG.array())) * dG;
    dZf = mState.mF.array() * (1.0 - mState.mF.array()) * dF;
    dZo = mState.mO.array() * (1.0 - mState.mO.array()) * dO;

    params->mDWi.matrix() += dZi * mState.mXH.matrix().transpose();
    params->mDWg.matrix() += dZg * mState.mXH.matrix().transpose();
    params->mDWf.matrix() += dZf * mState.mXH.matrix().transpose();
    params->mDWo.matrix() += dZo * mState.mXH.matrix().transpose();

    params->mDBi.matrix() += dZi;
    params->mDBg.matrix() += dZg;
    params->mDBf.matrix() += dZf;
    params->mDBo.matrix() += dZo;

    dXH = params->mWi.matrix().transpose() * dZi +
          params->mWg.matrix().transpose() * dZg +
          params->mWf.matrix().transpose() * dZf +
          params->mWo.matrix().transpose() * dZo;

    mState.mDC.array() = dC * mState.mF.array();
    mState.mDH.array() = dXH(Eigen::lastN(params->mOutSize));

    return (mState.mH.matrix() - asEigen<Eigen::Matrix>(dataTarget)).norm();
  }

private:
  LSTMState mState;
  ParamPtr  mParams;
};

} // namespace algorithm
} // namespace fluid