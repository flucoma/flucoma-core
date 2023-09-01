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

class LSTMCell
{
public:
  using StateType = RealVector;

  explicit LSTMCell() = default;
  ~LSTMCell() = default;

  void init(index inputSize, index outputSize)
  {
    mWi.resize(outputSize, inputSize);
    mWg.resize(outputSize, inputSize);
    mWf.resize(outputSize, inputSize);
    mWo.resize(outputSize, inputSize);

    mWi.fill(0);
    mWg.fill(0);
    mWf.fill(0);
    mWo.fill(0);

    mBi.resize(outputSize);
    mBg.resize(outputSize);
    mBf.resize(outputSize);
    mBo.resize(outputSize);

    mBi.fill(0);
    mBg.fill(0);
    mBf.fill(0);
    mBo.fill(0);

    mInitialized = true;
  }

  void processFrame(InputRealVectorView inData, InputRealVectorView inState,
                    InputRealVectorView prevOutput, RealVectorView outState,
                    RealVectorView outData, bool isFirstFrame,
                    Allocator& alloc = FluidDefaultAllocator())
  {
    ScopedEigenMap<Eigen::VectorXd> xthtp(inData.size() + prevOutput.size(),
                                          alloc);
    ScopedEigenMap<Eigen::ArrayXd>  ct(outState.size(), alloc),
        ctp(inState.size(), alloc), ht(outData.size(), alloc);


    xthtp << _impl::asEigen<Eigen::Matrix>(inData),
        _impl::asEigen<Eigen::Array>(inData);
    ctp = _impl::asEigen<Eigen::Array>(inData);

    forwardFrame(xthtp, ctp, ct, ht, isFirstFrame, alloc);

    _impl::asEigen<Eigen::Array>(outState) = ct;
    _impl::asEigen<Eigen::Matrix>(outData) = ht;
  };

  void forwardFrame(Eigen::Ref<Eigen::VectorXd> xthtp,
                    Eigen::Ref<Eigen::ArrayXd>  ctp,
                    Eigen::Ref<Eigen::ArrayXd>  ct,
                    Eigen::Ref<Eigen::ArrayXd> ht, bool isFirstFrame,
                    Allocator& alloc = FluidDefaultAllocator())
  {
    assert(ctp.size() == ct.size());
    assert(ctp.size() == ht.size());

    index size = ctp.size();

    ScopedEigenMap<Eigen::ArrayXd> inputGate(size, alloc),
        forgetGate(size, alloc), outputGate(size, alloc);

    ScopedEigenMap<Eigen::ArrayXd> Zi(size, alloc), Zg(size, alloc),
        Zf(size, alloc), Zo(size, alloc);

    ScopedEigenMap<Eigen::MatrixXd> Wi(mWi.rows(), mWi.cols(), alloc),
        Wg(mWg.rows(), mWg.cols(), alloc), Wf(mWf.rows(), mWf.cols(), alloc),
        Wo(mWo.rows(), mWo.cols(), alloc);

    ScopedEigenMap<Eigen::VectorXd> Bi(mBi.size(), alloc),
        Bg(mBg.size(), alloc), Bf(mBf.size(), alloc), Bo(mBo.size(), alloc);

    Wi = _impl::asEigen<Eigen::Matrix>(mWi);
    Wg = _impl::asEigen<Eigen::Matrix>(mWg);
    Wf = _impl::asEigen<Eigen::Matrix>(mWf);
    Wo = _impl::asEigen<Eigen::Matrix>(mWo);

    Bi = _impl::asEigen<Eigen::Matrix>(mBi);
    Bg = _impl::asEigen<Eigen::Matrix>(mBg);
    Bf = _impl::asEigen<Eigen::Matrix>(mBf);
    Bo = _impl::asEigen<Eigen::Matrix>(mBo);

    Zi = Wi * xthtp + Bi;
    Zg = Wg * xthtp + Bg;
    Zf = Wf * xthtp + Bf;
    Zo = Wo * xthtp + Bo;

    ct = ctp * logistic(Zf) + logistic(Zi) * tanh(Zg);
    ht = logistic(Zo) * ct;
  };

private:
  RealMatrix mWi, mWg, mWf, mWo;
  RealMatrix mDWi, mDWg, mDWf, mDWo;
  RealVector mBi, mBg, mBf, mBo;
  RealVector mDBi, mDBg, mDBf, mDBo;

  bool mInitialized{false};
};

} // namespace algorithm
} // namespace fluid