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

  void update(double lr)
  {
    using _impl::asEigen, Eigen::Matrix, Eigen::Array;

    asEigen<Array>(mWi) -= lr * asEigen<Array>(mDWi);
    asEigen<Array>(mWg) -= lr * asEigen<Array>(mDWg);
    asEigen<Array>(mWf) -= lr * asEigen<Array>(mDWf);
    asEigen<Array>(mWo) -= lr * asEigen<Array>(mDWo);

    asEigen<Array>(mBi) -= lr * asEigen<Array>(mDBi);
    asEigen<Array>(mBg) -= lr * asEigen<Array>(mDBg);
    asEigen<Array>(mBf) -= lr * asEigen<Array>(mDBf);
    asEigen<Array>(mBo) -= lr * asEigen<Array>(mDBo);

    // clear weight derivatives
    std::fill(mDWi.begin(), mDWi.end(), 0.0);
    std::fill(mDWg.begin(), mDWg.end(), 0.0);
    std::fill(mDWf.begin(), mDWf.end(), 0.0);
    std::fill(mDWo.begin(), mDWo.end(), 0.0);

    // clear bias derivatives
    std::fill(mDBi.begin(), mDBi.end(), 0.0);
    std::fill(mDBg.begin(), mDBg.end(), 0.0);
    std::fill(mDBf.begin(), mDBf.end(), 0.0);
    std::fill(mDBo.begin(), mDBo.end(), 0.0);
  }

  const index mInSize, mLayerSize, mOutSize;

  // parameters
  RealMatrix mWi, mWg, mWf, mWo;
  RealMatrix mDWi, mDWg, mDWf, mDWo;
  RealVector mBi, mBg, mBf, mBo;
  RealVector mDBi, mDBg, mDBf, mDBo;
};

class LSTMState
{
  using ParamPtr = std::weak_ptr<LSTMParam>;
  using ParamLock = std::shared_ptr<LSTMParam>;

public:
  LSTMState(ParamLock p)
      : mInSize(p->mInSize), mLayerSize(p->mLayerSize), mOutSize(p->mOutSize),
        mX(mInSize), mXH(mLayerSize), mCp(mOutSize), mHp(mOutSize),
        mI(mOutSize), mG(mOutSize), mF(mOutSize), mO(mOutSize), mC(mOutSize),
        mH(mOutSize), mDX(mInSize), mDC(mOutSize), mDH(mOutSize)
  {}

  LSTMState(ParamPtr p) : LSTMState{p.lock()} {};

  RealVectorView output() { return mH; }
  RealVectorView inputDerivative() { return mDX; }

  index mInSize, mLayerSize, mOutSize;

  // state at time t
  RealVector mX, mXH, mCp, mHp, mI, mG, mF, mO, mC, mH;
  RealVector mDX, mDC, mDH;
};

class LSTMCell
{
  using VectorXd = Eigen::VectorXd;
  using MatrixXd = Eigen::MatrixXd;
  using ArrayXd = Eigen::ArrayXd;

public:
  // Recur typedefs
  using StateType = LSTMState;
  using ParamType = LSTMParam;
  using ParamPtr = std::weak_ptr<ParamType>;
  using ParamLock = std::shared_ptr<ParamType>;

  LSTMCell(ParamPtr p) : mParams(p), mState(p) {}

  StateType& getState() { return mState; }

  void forwardFrame(InputRealVectorView inData, StateType& prevState,
                    Allocator& alloc = FluidDefaultAllocator())
  {
    using _impl::asEigen, Eigen::Array, Eigen::Matrix, Eigen::ArrayXd;

    ParamLock params = mParams.lock();

    assert(inData.size() == mState.mInSize);
    assert(prevState.mInSize == mState.mInSize);
    assert(prevState.mOutSize == mState.mOutSize);

    ScopedEigenMap<ArrayXd> Zi(params->mOutSize, alloc),
        Zg(params->mOutSize, alloc), Zf(params->mOutSize, alloc),
        Zo(params->mOutSize, alloc);

    mState.mX <<= inData;
    mState.mCp <<= prevState.mC;
    mState.mHp <<= prevState.mH;

    // concatentate input and previous output
    asEigen<Matrix>(mState.mXH) << asEigen<Matrix>(inData),
        asEigen<Matrix>(prevState.mH);

    // matrix mult
    Zi = asEigen<Matrix>(params->mWi) * asEigen<Matrix>(mState.mXH) +
         asEigen<Matrix>(params->mBi);
    Zg = asEigen<Matrix>(params->mWg) * asEigen<Matrix>(mState.mXH) +
         asEigen<Matrix>(params->mBg);
    Zf = asEigen<Matrix>(params->mWf) * asEigen<Matrix>(mState.mXH) +
         asEigen<Matrix>(params->mBf);
    Zo = asEigen<Matrix>(params->mWo) * asEigen<Matrix>(mState.mXH) +
         asEigen<Matrix>(params->mBo);

    asEigen<Array>(mState.mI) = Zi.logistic();
    asEigen<Array>(mState.mG) = Zg.tanh();
    asEigen<Array>(mState.mF) = Zf.logistic();
    asEigen<Array>(mState.mO) = Zo.logistic();

    // elem-wise mult and sum
    asEigen<Array>(mState.mC) =
        asEigen<Array>(mState.mG) * asEigen<Array>(mState.mI) +
        asEigen<Array>(mState.mCp) * asEigen<Array>(mState.mF);
    asEigen<Array>(mState.mH) =
        asEigen<Array>(mState.mC) * asEigen<Array>(mState.mO);
  }

  void backwardFrame(StateType& upState, StateType& nextState,
                     Allocator& alloc = FluidDefaultAllocator())
  {
    calculateBackwardFrame(upState.inputDerivative(), nextState, alloc);
  }

  double backwardFrame(InputRealVectorView target, StateType& nextState,
                       Allocator& alloc = FluidDefaultAllocator())
  {
    using _impl::asEigen, _impl::asFluid, Eigen::Matrix, Eigen::Array;

    assert(target.size() == mState.output().size());
    ScopedEigenMap<Eigen::ArrayXd> inputDerivative(target.size(), alloc);

    inputDerivative = asEigen<Array>(mState.output()) - asEigen<Array>(target);
    calculateBackwardFrame(asFluid(inputDerivative), nextState, alloc);

    return (asEigen<Matrix>(mState.output()) - asEigen<Matrix>(target))
               .squaredNorm() /
           target.size();
  }

  void backwardFrame(StateType& nextState,
                     Allocator& alloc = FluidDefaultAllocator())
  {
    RealVector zeroes(mState.mOutSize);
    calculateBackwardFrame(zeroes, nextState, alloc);
  }

private:
  LSTMState mState;
  ParamPtr  mParams;

  void calculateBackwardFrame(RealVectorView inputDerivative,
                              StateType&     nextState,
                              Allocator&     alloc = FluidDefaultAllocator())
  {
    using _impl::asEigen, Eigen::Array, Eigen::Matrix;

    assert(inputDerivative.size() == mState.mOutSize);
    assert(nextState.mInSize == mState.mInSize);
    assert(nextState.mOutSize == mState.mOutSize);

    ParamLock params = mParams.lock();

    ScopedEigenMap<ArrayXd> dC(params->mOutSize, alloc),
        dLdh(params->mOutSize, alloc), dLdc(params->mOutSize, alloc),
        dI(params->mOutSize, alloc), dG(params->mOutSize, alloc),
        dF(params->mOutSize, alloc), dO(params->mOutSize, alloc);
    ScopedEigenMap<VectorXd> dXH(params->mLayerSize, alloc),
        dZi(params->mOutSize, alloc), dZg(params->mOutSize, alloc),
        dZf(params->mOutSize, alloc), dZo(params->mOutSize, alloc);

    dLdh = asEigen<Array>(nextState.mDH) + asEigen<Array>(inputDerivative);
    dLdc = asEigen<Array>(nextState.mDC);

    dC = asEigen<Array>(mState.mO) * dLdh + dLdc;
    dI = asEigen<Array>(mState.mG) * dC;
    dG = asEigen<Array>(mState.mI) * dC;
    dF = asEigen<Array>(mState.mCp) * dC;
    dO = asEigen<Array>(mState.mC) * dLdh;

    dZi = asEigen<Array>(mState.mI) * (1.0 - asEigen<Array>(mState.mI)) * dI;
    dZg =
        (1.0 - (asEigen<Array>(mState.mG)) * (asEigen<Array>(mState.mG))) * dG;
    dZf = asEigen<Array>(mState.mF) * (1.0 - asEigen<Array>(mState.mF)) * dF;
    dZo = asEigen<Array>(mState.mO) * (1.0 - asEigen<Array>(mState.mO)) * dO;

    asEigen<Matrix>(params->mDWi) +=
        dZi * asEigen<Matrix>(mState.mXH).transpose();
    asEigen<Matrix>(params->mDWg) +=
        dZg * asEigen<Matrix>(mState.mXH).transpose();
    asEigen<Matrix>(params->mDWf) +=
        dZf * asEigen<Matrix>(mState.mXH).transpose();
    asEigen<Matrix>(params->mDWo) +=
        dZo * asEigen<Matrix>(mState.mXH).transpose();

    asEigen<Matrix>(params->mDBi) += dZi;
    asEigen<Matrix>(params->mDBg) += dZg;
    asEigen<Matrix>(params->mDBf) += dZf;
    asEigen<Matrix>(params->mDBo) += dZo;

    dXH = asEigen<Matrix>(params->mWi).transpose() * dZi +
          asEigen<Matrix>(params->mWg).transpose() * dZg +
          asEigen<Matrix>(params->mWf).transpose() * dZf +
          asEigen<Matrix>(params->mWo).transpose() * dZo;

    asEigen<Array>(mState.mDC) = dC * asEigen<Array>(mState.mF);
    asEigen<Array>(mState.mDX) = dXH(Eigen::seqN(0, params->mInSize));
    asEigen<Array>(mState.mDH) = dXH(Eigen::lastN(params->mOutSize));
  }
};

} // namespace algorithm
} // namespace fluid