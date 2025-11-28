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
#include "../FluidTensor.hpp"
#include <algorithm>
#include <iterator>
#include <optional>
#include <random>
#include <vector>

namespace fluid::detail {
template <class Derived>
class DataSampler
{
  struct BatchIterator
  {
    using iterator_category = std::input_iterator_tag;
    using value_type = std::optional<FluidTensorView<index, 2>>;
    using difference_type = std::ptrdiff_t;
    using pointer = value_type*;
    using reference = value_type&;

    BatchIterator(DataSampler& sampler, value_type&& val)
        : mRef(&sampler), mVal(std::move(val))
    {}

    BatchIterator& operator++()
    {
      mVal = mRef->nextBatch();
      return *this;
    }
    BatchIterator operator++(int)
    {
      BatchIterator res = *this;
      ++(*this);
      return res;
    }
    bool operator==(BatchIterator other) const
    {
      return mRef == other.mRef && (mVal == other.mVal);
    }
    bool operator!=(BatchIterator other) const { return !(*this == other); }

    reference         operator*() { return mVal; }
    value_type const& operator*() const { return mVal; }

  private:
    DataSampler* mRef;
    value_type   mVal;
  };

  bool                  mShuffle;
  index                 mSeed;
  index                 mTrainCount;    
  std::mt19937          mGen;
  std::vector<index>    mIdx;
  index                 mBatchSize;
  FluidTensor<index, 2> mBatch;
  FluidTensor<index, 2> mValidation;
  FluidTensor<index, 2> mTraining; // is there a way to avoid this?
  index                 mBatchCount{0};

  std::vector<index> makeIndex(index size, bool shuffle)
  {
    using std::begin, std::end;
    std::vector<index> result(asUnsigned(size));
    std::iota(begin(result), end(result), 0);
    if (shuffle) std::shuffle(begin(result), end(result), mGen);
    return result;
  }

protected:
  DataSampler(index size, index batchSize, double validationFraction,
              bool shuffle, index seed)
      : mShuffle{shuffle}, mSeed{seed},
        mTrainCount{
            std::lrint((1 - std::clamp(validationFraction, 0.0, 1.0)) * size)},
        mGen(static_cast<size_t>(seed > 0 ? seed : std::random_device()())),
        mIdx(makeIndex(size, mShuffle)),
        mBatchSize{std::min(mTrainCount, batchSize)},
        mBatch(batchSize + (mTrainCount % mBatchSize), 2),
        mValidation(size - mTrainCount, 2), mTraining(mTrainCount, 2)
  {}
public:
  void reset()
  {
    if (mSeed > 0) mGen.seed(asUnsigned(mSeed));
    mBatchCount = 0;
    mIdx = makeIndex(mIdx.size(), mShuffle);
  }

  // Returns in / out indices for this batch (not the data)
  std::optional<FluidTensorView<index, 2>> nextBatch()
  {
    using std::begin, std::end, std::transform;

    if (mBatchCount >= mTrainCount) return {};

    index thisBatchSize = mBatchCount + mBatchSize < mTrainCount
                              ? mBatchSize
                              : mTrainCount - mBatchCount;

    // if there's a remainder from n batches into the training data size,
    // stick the extra on the first batch
    if (index remainder = (mTrainCount % mBatchSize);
        remainder && mBatchCount == 0)
      thisBatchSize += remainder;

    auto batchStart = mIdx.begin() + mBatchCount;
    auto batchEnd = batchStart + thisBatchSize;
    mBatchCount += thisBatchSize;

    return derived().map(batchStart, batchEnd, mBatch)(Slice(0, thisBatchSize),
                                                       Slice(0));
  }

  index maxBatchSize() { return mBatchSize + (mTrainCount % mBatchSize); }

  std::optional<FluidTensorView<index, 2>> validationSet()
  {
    if (mTrainCount == asSigned(mIdx.size())) return {}; // no validation

    using std::begin, std::end;

    auto validationStart = begin(mIdx) + mTrainCount;
    auto validationEnd = end(mIdx);

    return derived().map(validationStart, validationEnd, mValidation);
  }

  std::optional<FluidTensorView<index, 2>> trainingSet()
  {
    if (mTrainCount == 0) return {}; // no training data, which would be weird

    using std::begin, std::end;

    auto trainingStart = begin(mIdx);
    auto trainingEnd = begin(mIdx) + mTrainCount;

    return derived().map(trainingStart, trainingEnd, mTraining);
  }


  index batchSize() const { return mBatchSize; }

  bool operator==(DataSampler const& other)
  {
    return &mIdx == &(other.mIdx) && mBatchCount == other.mBatchCount;
  }

  bool operator!=(DataSampler const& other) { return !(*this == other); }

  BatchIterator begin()
  {
    reset();

    auto b = nextBatch();

    return BatchIterator(*this, std::move(b));
  }

  BatchIterator end()
  {
    return BatchIterator(*this, std::optional<FluidTensorView<index, 2>>{});
  }

private:
  Derived& derived() { return *(static_cast<Derived*>(this)); }
};
} // namespace fluid::detail
