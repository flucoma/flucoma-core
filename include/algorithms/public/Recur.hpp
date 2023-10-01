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
  using CellVectorVector = rt::vector<rt::vector<CellType>>;
  using CellPtr = std::unique_ptr<CellType>;

  using StateType = typename CellType::StateType;
  using StatePtr = std::unique_ptr<StateType>;

  using ParamType = typename CellType::ParamType;
  using ParamWeakPtr = typename CellType::ParamPtr;
  using ParamPtr = typename CellType::ParamLock;

public:
  using IndexVector = FluidTensor<index, 1>;
  using IndexVectorView = FluidTensorView<index, 1>;
  using IndexInputVectorView = FluidTensorView<const index, 1>;

  using ParamVector = rt::vector<ParamPtr>;
  using StateVector = rt::vector<StatePtr>;
  using CellVector = rt::vector<CellPtr>;

  explicit Recur() = default;
  ~Recur() = default;

  Recur(Recur<CellType>& other)
  {
    mInSize = other.mInSize;
    mHiddenSize = other.mHiddenSize;
    mOutSize = other.mOutSize;

    mTrained = other.mTrained;
    mInitialized = true;

    mBottomParams = std::make_shared<ParamType>(*other.mBottomParams);
    mTopParams = std::make_shared<ParamType>(*other.mTopParams);

    mBottomState = std::make_unique<StateType>(*other.mBottomState);
    mTopState = std::make_unique<StateType>(*other.mTopState);

    // clear initialise the cells
    mBottomCell = std::make_unique<CellType>(mBottomParams);
    mTopCell = std::make_unique<CellType>(mTopParams);
  }

  Recur(Recur<CellType>&& other)
  {
    mInSize = other.mInSize;
    mHiddenSize = other.mHiddenSize;
    mOutSize = other.mOutSize;

    mTrained = other.mTrained;
    mInitialized = true;

    mBottomParams = other.mBottomParams;
    mTopParams = other.mTopParams;

    // unique pointers so must release other's ownership of the memory then
    // reclaim it
    mBottomState = std::move(other.mBottomState);
    mTopState = std::move(other.mTopState);

    // clear initialise the cells
    mBottomCell = std::make_unique<CellType>(mBottomParams);
    mTopCell = std::make_unique<CellType>(mTopParams);
  }

  Recur<CellType>& operator=(Recur<CellType>& other)
  {
    mInSize = other.mInSize;
    mHiddenSize = other.mHiddenSize;
    mOutSize = other.mOutSize;

    mTrained = other.mTrained;
    mInitialized = true;

    mBottomParams = std::make_shared<ParamType>(*other.mBottomParams);
    mTopParams = std::make_shared<ParamType>(*other.mTopParams);

    mBottomState = std::make_unique<StateType>(*other.mBottomState);
    mTopState = std::make_unique<StateType>(*other.mTopState);

    // clear initialise the cells
    mBottomCell = std::make_unique<CellType>(mBottomParams);
    mTopCell = std::make_unique<CellType>(mTopParams);

    return *this;
  }

  Recur<CellType>& operator=(Recur<CellType>&& other)
  {
    mInSize = other.mInSize;
    mHiddenSize = other.mHiddenSize;
    mOutSize = other.mOutSize;

    mTrained = other.mTrained;
    mInitialized = true;

    // shared pointers so takes co-ownership, extends lifetime
    mBottomParams = other.mBottomParams;
    mTopParams = other.mTopParams;

    // unique pointers so must release other's ownership of the memory then
    // reclaim it
    mBottomState = std::move(other.mBottomState);
    mTopState = std::move(other.mTopState);

    // clear initialise the cells
    mBottomCell = std::make_unique<CellType>(mBottomParams);
    mTopCell = std::make_unique<CellType>(mTopParams);

    return *this;
  }

  index initialized() const { return mInitialized; }
  index dims() const { return mInitialized ? mInSize : 0; }
  index size() const { return mInitialized ? mOutSize : 0; }

  index inputDims() const { return mInitialized ? mInSize : 0; }
  index hiddenDims() const { return mInitialized ? mHiddenSize : 0; }
  index outputDims() const { return mInitialized ? mOutSize : 0; }

  bool trained() const { return mTrained; }
  void setTrained(bool set = true) { mTrained = set; }

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

    mTrained = false;
  }

  void reset()
  {
    mBottomParams->reset();
    mTopParams->reset();

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

    mBottomParams = std::make_shared<ParamType>(mInSize, mHiddenSize);
    mTopParams = std::make_shared<ParamType>(mHiddenSize, mOutSize);

    mBottomState = std::make_unique<StateType>(mBottomParams);
    mTopState = std::make_unique<StateType>(mTopParams);

    mBottomCell = std::make_unique<CellType>(mBottomParams);
    mTopCell = std::make_unique<CellType>(mTopParams);

    mInitialized = true;
  };

  double fit(InputRealMatrixView input, InputRealMatrixView output)
  {
    assert(input.cols() == mInSize);
    assert(output.cols() == mOutSize);
    assert(input.rows() == output.rows());

    CellSeries bottomNodes, topNodes;

    bottomNodes.emplace_back(mBottomParams);
    bottomNodes[0].forwardFrame(input.row(0), StateType{mBottomParams});

    topNodes.emplace_back(mTopParams);
    topNodes[0].forwardFrame(bottomNodes[0].getState().output(),
                             StateType{mTopParams});

    for (index i = 1; i < input.rows(); ++i)
    {
      bottomNodes.emplace_back(mBottomParams);
      bottomNodes[i].forwardFrame(input.row(i), bottomNodes[i - 1].getState());

      topNodes.emplace_back(mTopParams);
      topNodes[i].forwardFrame(bottomNodes[i].getState().output(),
                               topNodes[i - 1].getState());
    }

    double loss = 0.0;
    loss += topNodes.back().backwardFrame(output.row(output.rows() - 1),
                                          StateType{mTopParams});
    bottomNodes.back().backwardFrame(topNodes.back().getState(),
                                     StateType{mBottomParams});

    for (index i = input.rows() - 2; i >= 0; --i)
    {
      loss +=
          topNodes[i].backwardFrame(output.row(i), topNodes[i + 1].getState());
      bottomNodes[i].backwardFrame(topNodes[i].getState(),
                                   bottomNodes[i + 1].getState());
    }

    return loss / input.rows();
  };

  double fit(InputRealMatrixView input, InputRealVectorView output)
  {
    assert(input.cols() == mInSize);
    assert(output.size() == mOutSize);

    CellSeries bottomNodes, topNodes;

    {
      StateType bottomState{mBottomParams}, topState{mTopParams};

      bottomNodes.emplace_back(mBottomParams);
      bottomNodes[0].forwardFrame(input.row(0), bottomState);

      topNodes.emplace_back(mTopParams);
      topNodes[0].forwardFrame(bottomNodes[0].getState().output(),
                              topState);
    }

    for (index i = 1; i < input.rows(); ++i)
    {
      bottomNodes.emplace_back(mBottomParams);
      bottomNodes[i].forwardFrame(input.row(i), bottomNodes[i - 1].getState());

      topNodes.emplace_back(mTopParams);
      topNodes[i].forwardFrame(bottomNodes[i].getState().output(),
                               topNodes[i - 1].getState());
    }
    
    double loss;

    {
      StateType bottomState{mBottomParams}, topState{mTopParams};

      loss = topNodes.back().backwardFrame(output, topState);
      bottomNodes.back().backwardFrame(topNodes.back().getState(),
                                      bottomState);

      for (index i = input.rows() - 2; i >= 0; --i)
      {
        topNodes[i].backwardFrame(topNodes[i + 1].getState());
        bottomNodes[i].backwardFrame(topNodes[i].getState(),
                                    bottomNodes[i + 1].getState());
      }
    }

    return loss;
  };

  double fit(InputRealMatrixView data)
  {
    assert(data.cols() == mInSize);
    assert(data.cols() == mOutSize);

    CellSeries bottomNodes, topNodes;

    {
      StateType bottomState{mBottomParams}, topState{mTopParams};

      bottomNodes.emplace_back(mBottomParams);
      bottomNodes[0].forwardFrame(data.row(0), bottomState);

      topNodes.emplace_back(mTopParams);
      topNodes[0].forwardFrame(bottomNodes[0].getState().output(),
                              topState);
    }

    for (index i = 1; i < data.rows() - 1; ++i)
    {
      bottomNodes.emplace_back(mBottomParams);
      bottomNodes[i].forwardFrame(data.row(i), bottomNodes[i - 1].getState());

      topNodes.emplace_back(mTopParams);
      topNodes[i].forwardFrame(bottomNodes[i].getState().output(),
                               topNodes[i - 1].getState());
    }

    double loss = 0.0;
    
    {
      StateType bottomState{mBottomParams}, topState{mTopParams};

      loss += topNodes.back().backwardFrame(data.row(data.rows() - 1),
                                            topState);
      bottomNodes.back().backwardFrame(topNodes.back().getState(),
                                      bottomState);
    }

    for (index i = data.rows() - 3; i >= 0; --i)
    {
      loss += topNodes[i].backwardFrame(data.row(i + 1),
                                        topNodes[i + 1].getState());
      bottomNodes[i].backwardFrame(topNodes[i].getState(),
                                   bottomNodes[i + 1].getState());
    }

    return loss / (data.rows() - 1);
  };

  void process(InputRealMatrixView input, RealMatrixView output)
  {
    assert(input.rows() == output.rows());
    for (index i = 0; i < input.rows(); ++i)
      processFrame(input.row(i), output.row(i));
  };

  void process(InputRealMatrixView input, RealVectorView output)
  {
    for (index i = 0; i < input.rows(); ++i) processFrame(input.row(i));
    output <<= mTopState->output();
  };

  void process(InputRealMatrixView input)
  {
    for (index i = 0; i < input.rows(); ++i) processFrame(input.row(i));
  };

  void processFrame(InputRealVectorView input, RealVectorView output)
  {
    assert(output.size() == mOutSize);
    processFrame(input);
    output <<= mTopState->output();
  }

  void processFrame(InputRealVectorView input)
  {
    assert(mTrained);
    assert(input.size() == mInSize);

    mBottomCell->forwardFrame(input, *mBottomState);
    mBottomState = std::make_unique<StateType>(mBottomCell->getState());

    mTopCell->forwardFrame(mBottomState->output(), *mTopState);
    mTopState = std::make_unique<StateType>(mTopCell->getState());
  };

private:
  bool mInitialized{false};
  bool mTrained{false};

  IndexVector mSizes;
  index mSize;

  // pointers rather than ref so Recur can be default constructible
  ParamVector mParams; // parameters of the layers
  StateVector mStates; // previous states
  CellVector  mCells;  // the recurrent cells themselves
};

} // namespace algorithm
} // namespace fluid