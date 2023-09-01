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

// xt is the input at time t, ht is the output at time t, htp at t-1, ct is the
// state at time t, ctp at t-1. wi is the input gate weights, wg is is the input
// filter weights, wf is the forget gate weights and wo is the output gate
// weights. The same naming applies to the bis vectors
class LSTMCell
{
  using VectorXd = Eigen::VectorXd;
  using MatrixXd = Eigen::MatrixXd;
  using ArrayXd = Eigen::ArrayXd;

  using EigenMatrixMap = Eigen::Map<MatrixXd>;
  using EigenVectorMap = Eigen::Map<VectorXd>;
  using EigenArrayMap = Eigen::Map<ArrayXd>;

public:
  using StateType = RealVector;

  explicit LSTMCell(index inputSize, index outputSize,
                    Allocator& alloc = FluidDefaultAllocator())
      : inSize{inputSize}, layerSize{inputSize + outputSize},
        outSize{outputSize},

        // allocate the memory for the weights
        mWi(outSize, layerSize), mWg(outSize, layerSize),
        mWf(outSize, layerSize), mWo(outSize, layerSize),

        // allocate the memory for the weight derivatives
        mDWi(outSize, layerSize), mDWg(outSize, layerSize),
        mDWf(outSize, layerSize), mDWo(outSize, layerSize),

        // allocate the memory for the biases
        mBi(outSize), mBg(outSize), mBf(outSize), mBo(outSize),

        // allocate the memory for the bias derivatives
        mDBi(outSize), mDBg(outSize), mDBf(outSize), mDBo(outSize),

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
        mEDBf(mDBf.data(), mDBf.size()), mEDBo(mDBo.data(), mDBo.size()){};

  ~LSTMCell() = default;

  void init()
  {
    resetParameters();
    resetDerivates();

    mInitialized = true;
  }

  void processFrame(InputRealVectorView inData, InputRealVectorView inState,
                    InputRealVectorView prevOutput, RealVectorView outState,
                    RealVectorView outData,
                    Allocator&     alloc = FluidDefaultAllocator())
  {
    using namespace _impl;

    ScopedEigenMap<VectorXd> xthtp(inData.size() + prevOutput.size(),
                                   alloc); // xt and htp concatenated
    ScopedEigenMap<ArrayXd>  ct(outState.size(), alloc),
        ctp(inState.size(), alloc), ht(outData.size(), alloc);

    xthtp << asEigen<Eigen::Matrix>(inData), asEigen<Eigen::Array>(inData);
    ctp = asEigen<Eigen::Array>(inData);

    forwardFrame(xthtp, ctp, ct, ht, alloc);

    asEigen<Eigen::Array>(outState) = ct;
    asEigen<Eigen::Matrix>(outData) = ht;
  };

  void forwardFrame(Eigen::Ref<VectorXd> xthtp, Eigen::Ref<ArrayXd> ctp,
                    Eigen::Ref<ArrayXd> ct, Eigen::Ref<ArrayXd> ht,
                    Allocator& alloc = FluidDefaultAllocator())
  {
    using namespace Eigen;
    using namespace _impl;

    assert(ctp.size() == ct.size());
    assert(ctp.size() == ht.size());

    index size = ctp.size();

    ScopedEigenMap<ArrayXd> inputGate(size, alloc), forgetGate(size, alloc),
        outputGate(size, alloc);

    ScopedEigenMap<ArrayXd> Zi(size, alloc), Zg(size, alloc), Zf(size, alloc),
        Zo(size, alloc);

    Zi = mEWi * xthtp + mEBi;
    Zg = mEWg * xthtp + mEBg;
    Zf = mEWf * xthtp + mEBf;
    Zo = mEWo * xthtp + mEBo;

    ct = ctp * logistic(Zf) + logistic(Zi) * tanh(Zg);
    ht = logistic(Zo) * ct;
  };

private:
  index inSize, layerSize, outSize;

  RealMatrix mWi, mWg, mWf, mWo;
  RealMatrix mDWi, mDWg, mDWf, mDWo;
  RealVector mBi, mBg, mBf, mBo;
  RealVector mDBi, mDBg, mDBf, mDBo;

  // eigen maps to the real parameters
  EigenMatrixMap mEWi, mEWg, mEWf, mEWo;
  EigenMatrixMap mEDWi, mEDWg, mEDWf, mEDWo;
  EigenVectorMap mEBi, mEBg, mEBf, mEBo;
  EigenVectorMap mEDBi, mEDBg, mEDBf, mEDBo;

  bool mInitialized{false};

  void resetParameters()
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
  }

  void resetDerivates()
  {
    // weight derivatives
    mDWi.fill(0.0);
    mDWg.fill(0.0);
    mDWf.fill(0.0);
    mDWo.fill(0.0);

    // bias derivatives
    mDBi.fill(0.0);
    mDBg.fill(0.0);
    mDBf.fill(0.0);
    mDBo.fill(0.0);
  }
};

} // namespace algorithm
} // namespace fluid