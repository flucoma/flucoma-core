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
    resize(other.mSizes);

    mTrained = other.mTrained;
    mInitialized = true;

    for (index i = 0; i < mSize; i++) {
      mParams[i] = std::make_shared<ParamType>(*other.mParams[i]);
      mStates[i] = std::make_unique<StateType>(*other.mStates[i]);
      mCells[i] = std::make_unique<CellType>(mParams[i]);
    }
  }

  Recur(Recur<CellType>&& other)
  {
    mSizes = std::move(other.mSizes);
    mSize = mSizes.size() - 1;

    mCells.resize(mSize);
    mParams.resize(mSize);
    mStates.resize(mSize);

    mTrained = other.mTrained;
    mInitialized = true;

    for (index i = 0; i < mSize; i++) {
      mParams[i] = other.mParams[i];
      mStates[i] = std::move(other.mStates[i]);
      mCells[i] = std::make_unique<CellType>(mParams[i]);
    }
  }

  Recur<CellType>& operator=(Recur<CellType>& other)
  {
    resize(other.mSizes);

    mTrained = other.mTrained;
    mInitialized = true;

    for (index i = 0; i < mSize; i++) {
      mParams[i] = std::make_shared<ParamType>(*other.mParams[i]);
      mStates[i] = std::make_unique<StateType>(*other.mStates[i]);
      mCells[i] = std::make_unique<CellType>(mParams[i]);
    }

    return *this;
  }

  Recur<CellType>& operator=(Recur<CellType>&& other)
  {
    mSizes = std::move(other.mSizes);
    mSize = mSizes.size() - 1;

    mCells.resize(mSize);
    mParams.resize(mSize);
    mStates.resize(mSize);

    mTrained = other.mTrained;
    mInitialized = true;

    for (index i = 0; i < mSize; i++) {
      mParams[i] = other.mParams[i];
      mStates[i] = std::move(other.mStates[i]);
      mCells[i] = std::make_unique<CellType>(mParams[i]);
    }

    return *this;
  }

  index initialized() const { return mInitialized; }
  index dims() const { return mInitialized ? mSizes[0] : 0; }
  index size() const { return mInitialized ? mSize : 0; }

  IndexInputVectorView getSizes() const { return mSizes; }

  index inputDims() const { return mInitialized ? mSizes[0] : 0; }
  index outputDims() const { return mInitialized ? mSizes[mSize] : 0; }

  bool trained() const { return mTrained; }
  void setTrained(bool set = true) { mTrained = set; }

  ParamWeakPtr getNthParams(index i) const 
  { 
    assert(i >= 0 && i < mSize); 
    return mParams[i]; 
  }

  void clear()
  {
    for (index i = 0; i < mSize; i++) {
      mParams[i] = std::make_shared<ParamType>(mSizes[i], mSizes[i + 1]);
      mStates[i] = std::make_unique<StateType>(mParams[i]);
      mCells[i] = std::make_unique<CellType>(mParams[i]);
    }

    mTrained = false;
  }

  void reset()
  {
    for (index i = 0; i < mSize; i++) {
      mParams[i]->reset();
      mStates[i] = std::make_unique<StateType>(mParams[i]);
      mCells[i] = std::make_unique<CellType>(mParams[i]);
    }
  }

  void update(double lr)
  {
    for (index i = 0; i < mSize; i++)
      mParams[i]->update(lr);
  }

  void resize(IndexVectorView sizes)
  {
    mSizes.resize(sizes.size());
    mSizes <<= sizes;

    mSize = mSizes.size() - 1;

    mCells.resize(mSize);
    mParams.resize(mSize);
    mStates.resize(mSize);

    mTrained = false;
  }

  void init(index inSize, IndexVectorView hiddenSizes, index outSize)
  {
    IndexVector sizes(hiddenSizes.size() + 2);

    sizes[0] = inSize, sizes[hiddenSizes.size() + 1] = outSize;
    std::copy(hiddenSizes.begin(), hiddenSizes.end(), sizes.begin() + 1);

    init(sizes);
  };

  void init(IndexVectorView sizes)
  {
    mSizes.resize(sizes.size());
    mSizes <<= sizes;
    mSize = mSizes.size() - 1;

    mCells.resize(mSize);
    mParams.resize(mSize);
    mStates.resize(mSize);

    for (index i = 0; i < mSize; i++)
    {
      mParams[i] = std::make_shared<ParamType>(mSizes[i], mSizes[i + 1]);
      mStates[i] = std::make_unique<StateType>(mParams[i]);
      mCells[i] = std::make_unique<CellType>(mParams[i]);
    }

    mInitialized = true;
  }

// TODO: parallelise the diagonals for propagating data.
  double fit(InputRealMatrixView input, InputRealMatrixView output)
  {
    assert(input.cols() == mSizes[0]);
    assert(output.cols() == mSizes[mSize]);
    assert(input.rows() == output.rows());

    // forward pass
    CellVectorVector nodes(mSize);

    {
      StateType emptyState{mParams[0]};

      nodes[0].emplace_back(mParams[0]);
      nodes[0][0].forwardFrame(input.row(0), emptyState);
    }

    for (index n = 1; n < mSize; ++n)
    {
      StateType emptyState{mParams[n]};

      nodes[n].emplace_back(mParams[n]);
      nodes[n][0].forwardFrame(nodes[n - 1][0].getState().output(), emptyState);
    }

    for (index i = 1; i < input.rows(); ++i)
    {
      nodes[0].emplace_back(mParams[0]);
      nodes[0][i].forwardFrame(input.row(i), nodes[0][i - 1].getState());

      for (index n = 1; n < mSize; ++n)
      {
        nodes[n].emplace_back(mParams[n]);
        nodes[n][i].forwardFrame(nodes[n - 1][i].getState().output(), nodes[n][i - 1].getState());
      }
    }

    // backpropagation
    double loss = 0.0;

    {
      StateType emptyState{mParams.back()};
      loss += nodes.back().back().backwardFrame(output.row(output.rows() - 1), emptyState);
    }
    
    for (index n = nodes.size() - 2; n >= 0; --n)
    {
      StateType emptyState{mParams[n]};
      nodes[n].back().backwardFrame(nodes[n + 1].back().getState(), emptyState);
    }

    for (index i = input.rows() - 2; i >= 0; --i)
    {
      loss += nodes.back()[i].backwardFrame(output.row(i), nodes.back()[i + 1].getState());
      for (index n = nodes.size() - 2; n >= 0; --n)
      {
        nodes[n][i].backwardFrame(nodes[n + 1][i].getState(), nodes[n][i + 1].getState());
      }
    }

    return loss / input.rows();
  };

  double fit(InputRealMatrixView input, InputRealVectorView output)
  {    
    assert(input.cols() == mSizes[0]);
    assert(output.size() == mSizes[mSize]);

    // forward pass
    CellVectorVector nodes(mSize);

    {
      StateType emptyState{mParams[0]};

      nodes[0].emplace_back(mParams[0]);
      nodes[0][0].forwardFrame(input.row(0), emptyState);
    }

    for (index n = 1; n < mSize; ++n)
    {
      StateType emptyState{mParams[n]};

      nodes[n].emplace_back(mParams[n]);
      nodes[n][0].forwardFrame(nodes[n - 1][0].getState().output(), emptyState);
    }

    for (index i = 1; i < input.rows(); ++i)
    {
      nodes[0].emplace_back(mParams[0]);
      nodes[0][i].forwardFrame(input.row(i), nodes[0][i - 1].getState());

      for (index n = 1; n < mSize; ++n)
      {
        nodes[n].emplace_back(mParams[n]);
        nodes[n][i].forwardFrame(nodes[n - 1][i].getState().output(), nodes[n][i - 1].getState());
      }
    }

    // backpropagation
    double loss;

    {
      StateType emptyState{mParams.back()};
      loss = nodes.back().back().backwardFrame(output, emptyState);
    }

    for (index n = nodes.size() - 2; n >= 0; --n)
    {
      StateType emptyState{mParams[n]};
      nodes[n].back().backwardFrame(nodes[n + 1].back().getState(), emptyState);
    }

    for (index i = input.rows() - 2; i >= 0; --i)
    {
      nodes.back()[i].backwardFrame(nodes.back()[i + 1].getState());
      for (index n = nodes.size() - 2; n >= 0; --n)
      {
        nodes[n][i].backwardFrame(nodes[n + 1][i].getState(), nodes[n][i + 1].getState());
      }
    }

    return loss;
  };

  double fit(InputRealMatrixView data)
  {
    assert(data.cols() == mSizes[0]);
    assert(data.cols() == mSizes[mSize]);

    // forward pass
    CellVectorVector nodes(mSize);

    {
      StateType emptyState{mParams[0]};

      nodes[0].emplace_back(mParams[0]);
      nodes[0][0].forwardFrame(data.row(0), emptyState);
    }

    for (index n = 1; n < mSize; ++n)
    {
      StateType emptyState{mParams[n]};

      nodes[n].emplace_back(mParams[n]);
      nodes[n][0].forwardFrame(nodes[n - 1][0].getState().output(), emptyState);
    }

    for (index i = 1; i < data.rows() - 1; ++i)
    {
      nodes[0].emplace_back(mParams[0]);
      nodes[0][i].forwardFrame(data.row(i), nodes[0][i - 1].getState());

      for (index n = 1; n < mSize; ++n)
      {
        nodes[n].emplace_back(mParams[n]);
        nodes[n][i].forwardFrame(nodes[n - 1][i].getState().output(), nodes[n][i - 1].getState());
      }
    }

    // backpropagation
    double loss = 0.0;

    {
      StateType emptyState{mParams.back()};
      loss += nodes.back().back().backwardFrame(data.row(data.rows() - 1), emptyState);
    }

    for (index n = nodes.size() - 2; n >= 0; --n)
    {
      StateType emptyState{mParams[n]};
      nodes[n].back().backwardFrame(nodes[n + 1].back().getState(), emptyState);
    }

    for (index i = data.rows() - 3; i >= 0; --i)
    {
      loss += nodes.back()[i].backwardFrame(data.row(i + 1), nodes.back()[i + 1].getState());
      for (index n = nodes.size() - 2; n >= 0; --n)
      {
        nodes[n][i].backwardFrame(nodes[n + 1][i].getState(), nodes[n][i + 1].getState());
      }
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
    output <<= mStates.back()->output();
  };

  void process(InputRealMatrixView input)
  {
    for (index i = 0; i < input.rows(); ++i) processFrame(input.row(i));
  };

  void processFrame(InputRealVectorView input, RealVectorView output)
  {
    assert(output.size() == mSizes[mSize]);
    processFrame(input);
    output <<= mStates.back()->output();
  }

  void processFrame(InputRealVectorView input)
  {
    assert(mTrained);
    assert(input.size() == mSizes[0]);

    assert(mParams.size() == mSize);
    assert(mStates.size() == mSize);
    assert(mCells.size() == mSize);

    mCells[0]->forwardFrame(input, *mStates[0]);
    mStates[0] = std::make_unique<StateType>(mCells[0]->getState());

    for (index i = 1; i < mSize; ++i)
    {
      mCells[i]->forwardFrame(mStates[i - 1]->output(), *mStates[i]);
      mStates[i] = std::make_unique<StateType>(mCells[i]->getState());
    }
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