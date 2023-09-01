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

template <class Cell>
class Recur
{
public:
  explicit Recur() = default;
  ~Recur() = default;

  index size() const { return mInitialized ? asSigned(mNodes.size()) : 0; }
  index initialized() const { return mInitialized; }
  index dims() const { return mInitialized ? mOutSize : 0; }

  bool trained() const { return mInitialized ? mTrained : false; }
  void setTrained() const
  {
    if (mInitialized) mTrained = true;
  }

  Cell::ParamPtr getParams() const { return mParams; }

  void clear()
  {
    mParams.reset();
    mNodes.clear();
  };

  void init(index inSize, index outSize)
  {
    mInSize = inSize;
    mOutSize = outSize;
    mParams = std::make_shared<Cell::ParamType>(inSize, outSize);

    mInitialized = true;
  };

  void fit(InputRealMatrixView input, RealMatrixView output){};
  void process(){};

  void process(InputRealMatrixView input, RealMatrixView output)
  {
    assert(input.cols() == inSize);
    assert(output.cols() == outSize);
    assert(input.rows() == output.rows() || output.rows() == 1);

    Cell       lstm(mParams);
    RealVector nowState(mParams->mOutSize), nowHidden(mParams->mOutSize),
        nextState(mParams->mOutSize), nextHidden(mParams->mOutSize);

    nowState.fill(0.0);
    nowHidden.fill(0.0);

    for (index row = 0; row < input.rows(); ++row)
    {
      InputRealVectorView in = input.row(i);
      lstm.forwardFrame(in, nowState, nowHidden, nextState, nextHidden);

      if (output.rows() > 1 || row == input.rows() - 1)
      {
        output.row(i) <<= nextHidden;
      }

      nowState << nextState;
      nowHidden << nextHidden;
    }
  };

private:
  bool mInitialized{false};
  bool mTrained{false};

  index mInSize, mOutSize;

  // rt vector of cells (each have ptr to params)
  // pointer so Recur can be default constructible
  rt::vector<Cell> mNodes;
  Cell::ParamLock  mParams;
};

} // namespace algorithm
} // namespace fluid