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
public:
  using CellSeries = rt::vector<CellType>;

  using StateType = typename CellType::StateType;
  using StatePtr = std::unique_ptr<StateType>;

  using ParamType = typename CellType::ParamType;
  using ParamWeakPtr = typename CellType::ParamPtr;
  using ParamPtr = typename CellType::ParamLock;

public:
  explicit Recur() = default;
  ~Recur() = default;

  index initialized() const { return mInitialized; }
  index size() const { return mInitialized ? mInSize : 0; }
  index dims() const { return mInitialized ? mOutSize : 0; }

  bool trained() const { return mInitialized ? mTrained : false; }
  void setTrained()
  {
    if (mInitialized) mTrained = true;
  }

  ParamWeakPtr getParams() const { return mParams; }

  void clear()
  {
    mParams.reset();
    mParams = std::make_shared<ParamType>(inSize, outSize);
  }
  void reset()
  {
    if (mInitialized)
    {
      mState.reset();
      mState = std::make_unique<StateType>(mParams);
    }
  }

  void update(double lr) { mParams->update(lr); }

  void init(index inSize, index outSize)
  {
    mParams = std::make_shared<ParamType>(inSize, outSize);
    mState = std::make_unique<StateType>(mParams);

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

  void process(InputRealMatrixView input, RealMatrixView output)
  {
    assert(input.rows() == output.rows() || output.rows() == 1);

    for (index i = 0; i < input.rows(); ++i)
    {
      processFrame(input.row(i));
      output.row(output.rows() == 1 ? 0 : i) <<= mState->output();
    }
  };

  void processFrame(InputRealVectorView input, RealVectorView output)
  {
    assert(output.size() == mOutSize);
    processFrame(input);
    output <<= mState->output();
  }

  void processFrame(InputRealVectorView input)
  {
    assert(input.size() == mInSize);

    // static so only allocate memory once
    static CellType cell(mParams);

    cell.forwardFrame(input, *mState);
    *mState = cell.getState();
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