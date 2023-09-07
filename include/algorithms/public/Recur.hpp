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
  using CellPtr = std::unique_ptr<CellType>;

  using StateType = typename CellType::StateType;
  using StatePtr = std::unique_ptr<StateType>;

  using ParamType = typename CellType::ParamType;
  using ParamWeakPtr = typename CellType::ParamPtr;
  using ParamPtr = typename CellType::ParamLock;

public:
  explicit Recur() = default;
  ~Recur() = default;

  Recur(Recur<CellType>& other)
  {
    mInSize = other.mInSize;
    mHiddenSize = other.mHiddenSize;
    mOutSize = other.mOutSize;

    mInitialized = true;

    mBottomParams = std::make_shared<ParamType>(*other.mBottomParams);
    mTopParams = std::make_shared<ParamType>(*other.mTopParams);

    mBottomState = std::make_unique<StateType>(*other.mBottomState);
    mTopState = std::make_unique<StateType>(*other.mTopState);

    mBottomCell = std::make_unique<CellType>(mBottomParams);
    mTopCell = std::make_unique<CellType>(mTopParams);
  }

  Recur<CellType>& operator=(Recur<CellType>& other)
  {
    mInSize = other.mInSize;
    mHiddenSize = other.mHiddenSize;
    mOutSize = other.mOutSize;

    mTrained = false;
    mInitialized = true;

    mBottomParams = std::make_shared<ParamType>(*other.mBottomParams);
    mTopParams = std::make_shared<ParamType>(*other.mTopParams);

    mBottomState = std::make_unique<StateType>(*other.mBottomState);
    mTopState = std::make_unique<StateType>(*other.mTopState);

    mBottomCell = std::make_unique<CellType>(mBottomParams);
    mTopCell = std::make_unique<CellType>(mTopParams);

    return *this;
  }

  index initialized() const { return mInitialized; }
  index dims() const { return mInitialized ? mInSize : 0; }
  index size() const { return mInitialized ? mOutSize : 0; }

  bool trained() const { return mInitialized ? mTrained : false; }
  void setTrained()
  {
    if (mInitialized) mTrained = true;
  }

  ParamWeakPtr getTopParams() const { return mTopParams; }
  ParamWeakPtr getBottomParams() const { return mBottomParams; }

  void clear()
  {
    mBottomParams = std::make_shared<ParamType>(mInSize, mHiddenSize);
    mTopParams = std::make_shared<ParamType>(mHiddenSize, mOutSize);

    mBottomState = std::make_unique<StateType>(mBottomParams);
    mTopState = std::make_unique<StateType>(mTopParams);

    mBottomCell = std::make_unique<CellType>(mBottomParams);
    mTopCell = std::make_unique<CellType>(mTopParams);
  }
  void reset()
  {
    mBottomState = std::make_unique<StateType>(mBottomParams);
    mTopState = std::make_unique<StateType>(mTopParams);

    mBottomCell = std::make_unique<CellType>(mBottomParams);
    mTopCell = std::make_unique<CellType>(mTopParams);
  }

  void update(double lr)
  {
    mTopParams->update(lr);
    mBottomParams->update(lr);
  }

  void init(index inSize, index hiddenSize, index outSize)
  {

    mInSize = inSize;
    mHiddenSize = hiddenSize;
    mOutSize = outSize;

    mInitialized = true;

    mBottomParams = std::make_shared<ParamType>(inSize, hiddenSize);
    mTopParams = std::make_shared<ParamType>(hiddenSize, outSize);

    mBottomState = std::make_unique<StateType>(mBottomParams);
    mTopState = std::make_unique<StateType>(mTopParams);

    mBottomCell = std::make_unique<CellType>(mBottomParams);
    mTopCell = std::make_unique<CellType>(mTopParams);
  };

  double fit(InputRealMatrixView input, InputRealMatrixView output)
  {
    // check the input sizes and check either N-N or N-1
    assert(input.cols() == mInSize);
    assert(output.cols() == mOutSize);
    assert(input.rows() == output.rows());

    CellSeries bottomNodes, topNodes;

    bottomNodes.emplace_back(mBottomParams);
    topNodes.emplace_back(mTopParams);

    for (index i = 0; i < input.rows(); ++i)
    {
      bottomNodes.emplace_back(mBottomParams);
      bottomNodes[i + 1].forwardFrame(input.row(i), bottomNodes[i].getState());

      topNodes.emplace_back(mTopParams);
      topNodes[i + 1].forwardFrame(bottomNodes[i + 1].getState().output(),
                                   topNodes[i].getState());
    }

    bottomNodes.emplace_back(mBottomParams);
    topNodes.emplace_back(mTopParams);

    double loss = 0.0;
    for (index i = input.rows(); i > 1; --i)
    {
      loss += topNodes[i].backwardFrame(output.row(i - 1),
                                        topNodes[i + 1].getState());
      bottomNodes[i].backwardFrame(topNodes[i].getState(),
                                   bottomNodes[i + 1].getState());
    }
    return loss / input.rows();
  };

  double fit(InputRealMatrixView input, InputRealVectorView output)
  {
    // check the input sizes and check either N-N or N-1
    assert(input.cols() == mInSize);
    assert(output.size() == mOutSize);

    CellSeries bottomNodes, topNodes;

    bottomNodes.emplace_back(mBottomParams);
    topNodes.emplace_back(mTopParams);

    for (index i = 0; i < input.rows(); ++i)
    {
      bottomNodes.emplace_back(mBottomParams);
      bottomNodes[i + 1].forwardFrame(input.row(i), bottomNodes[i].getState());

      topNodes.emplace_back(mTopParams);
      topNodes[i + 1].forwardFrame(bottomNodes[i + 1].getState().output(),
                                   topNodes[i].getState());
    }

    bottomNodes.emplace_back(mBottomParams);
    topNodes.emplace_back(mTopParams);

    double loss = topNodes[input.rows()].backwardFrame(
        output, topNodes.back().getState());
    bottomNodes[input.rows()].backwardFrame(topNodes[input.rows()].getState(),
                                            bottomNodes.back().getState());

    for (index i = input.rows() - 1; i > 1; --i)
    {
      topNodes[i].backwardFrame(topNodes[i + 1].getState());
      bottomNodes[i].backwardFrame(topNodes[i].getState(),
                                   bottomNodes[i + 1].getState());
    }

    return loss;
  };

  void process(InputRealMatrixView input, RealMatrixView output)
  {
    assert(input.rows() == output.rows());
    for (index i = 0; i < input.rows(); ++i)
      processFrame(input.row(i), output.row(i));
  };

  void process(InputRealMatrixView input, RealVectorView output)
  {
    assert(input.cols() == output.size());
    for (index i = 0; i < input.rows(); ++i) processFrame(input.row(i), output);
  };

  void processFrame(InputRealVectorView input, RealVectorView output)
  {
    assert(output.size() == mOutSize);
    processFrame(input);
    output <<= mTopState->output();
  }

  void processFrame(InputRealVectorView input)
  {
    assert(input.size() == mInSize);

    mBottomCell->forwardFrame(input, *mBottomState);
    mBottomState = std::make_unique<StateType>(mBottomCell->getState());

    mTopCell->forwardFrame(mBottomState->output(), *mTopState);
    mTopState = std::make_unique<StateType>(mTopCell->getState());
  };

private:
  bool mInitialized{false};
  bool mTrained{false};

  index mInSize, mHiddenSize, mOutSize;

  // pointers rather than ref so Recur can be default constructible
  ParamPtr mBottomParams, mTopParams; // parameters of two layers
  StatePtr mBottomState, mTopState;   // previous state
  CellPtr  mBottomCell, mTopCell;     // the recurrent cell themselves
};

} // namespace algorithm
} // namespace fluid