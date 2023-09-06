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

template <class CellType>
class Recur
{
  using CellSeries = rt::vector<CellType>;
  using CellState = typename CellType::StateType;
  using ParamType = typename CellType::ParamType;
  using ParamPtr = typename CellType::ParamPtr;
  using ParamLock = typename CellType::ParamLock;

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

  ParamPtr getParams() const { return mParams; }

  void clear() { mParams.reset(); };

  void init(index inSize, index outSize)
  {
    mParams = std::make_shared<ParamType>(inSize, outSize);

    mInSize = inSize;
    mOutSize = outSize;

    mInitialized = true;
  };

  double fit(InputRealMatrixView input, InputRealMatrixView output)
  {
    // check the input sizes and check either N-N or N-1
    assert(input.cols() == mInSize);
    assert(output.cols() == mOutSize);
    assert(input.rows() == output.rows() || output.rows() == 1);

    CellSeries mNodes;
    mNodes.emplace_back(mParams);

    for (index i = 0; i < input.rows(); ++i)
    {
      mNodes.emplace_back(mParams);
      mNodes[i + 1].forwardFrame(input.row(i), mNodes[i].getState());
    }

    mNodes.emplace_back(mParams);

    double loss = 0.0;
    for (index i = input.rows(); i > 1; --i)
    {
      if (output.rows() > 1 || i == input.rows())
        loss += mNodes[i].backwardFrame(output.row(i - 1),
                                        mNodes[i + 1].getState());
      else
        mNodes[i].backwardDatalessFrame(mNodes[i + 1].getState());
    }

    return loss / input.rows();
  };

  void update(double lr = 0.5) { mParams->apply(lr); }

  void process(InputRealMatrixView input, RealMatrixView output)
  {
    assert(input.cols() == mInSize);
    assert(output.cols() == mOutSize);
    assert(input.rows() == output.rows() || output.rows() == 1);

    CellType cell(mParams);

    for (index i = 0; i < input.rows(); ++i)
    {
      cell.forwardFrame(input.row(i), *mState);
      *mState = cell.getState();

      if (output.rows() > 1 || i == input.rows() - 1)
        output.row(i) <<= mState->output();
    }
  };

private:
  bool mInitialized{false};
  bool mTrained{false};

  index mInSize, mOutSize;

  // rt vector of cells (each have ptr to params)
  // pointer rather than ref so Recur can be default constructible
  ParamPtr mParams;
  StatePtr mState;
};

} // namespace algorithm
} // namespace fluid