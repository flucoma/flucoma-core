#pragma once
#include "data/FluidTensor.hpp"
#include <Eigen/Dense>
#include <limits>

namespace fluid {
namespace medianfilter {

using Eigen::ArrayXd;
using Eigen::Ref;

const double maxDouble = std::numeric_limits<double>::max();

// This implements the algorithm described in
// Suomela, J., Median Filtering is Equivalent to Sorting
// https://arxiv.org/abs/1406.1717
// it is based on the author's own C++11 and python implementations
class MedianFilter {

  struct Block {
    struct Link {
      int prev;
      int next;
    };

    Block(size_t filterSize)
        : mFilterSize(filterSize), mLinks(filterSize), mSorted(filterSize) {}

    void sortData() {
      for (int i = 0; i < mFilterSize; i++) {
        mSorted[i] = std::make_pair(mData[i], i);
      }
      std::sort(mSorted.begin(), mSorted.end());
    }

    void construct(const double *data) {
      mData = data;
      sortData();
      int a = mFilterSize;
      for (int i = 0; i < mFilterSize; i++) {
        int b = mSorted[i].second;
        mLinks[a].next = b;
        mLinks[b].prev = a;
        a = b;
      }
      mLinks[a].next = mFilterSize;
      mLinks[mFilterSize].prev = a;
      mMedianIndex = mSorted[mFilterSize / 2].second;
      mSmallIndex = mFilterSize / 2;
    }

    void unwind() {
      for (int i = 0; i < mFilterSize; i++) {
        int j = mFilterSize - 1 - i;
        Link l = mLinks[j];
        mLinks[l.prev].next = l.next;
        mLinks[l.next].prev = l.prev;
      }
      mMedianIndex = mFilterSize;
      mSmallIndex = 0;
    }

    void del(int i) {
      Link l = mLinks[i];
      mLinks[l.prev].next = l.next;
      mLinks[l.next].prev = l.prev;
      if (belowMedian(i)) {
        mSmallIndex--;
      } else {
        if (i == mMedianIndex) {
          mMedianIndex = l.next;
        }
        if (mSmallIndex > 0) {
          mMedianIndex = mLinks[mMedianIndex].prev;
          mSmallIndex--;
        }
      }
    }

    void undelete(int i) {
      Link l = mLinks[i];
      mLinks[l.prev].next = i;
      mLinks[l.next].prev = i;
      if (belowMedian(i)) {
        mMedianIndex = mLinks[mMedianIndex].prev;
      }
    }

    void advance() {
      mMedianIndex = mLinks[mMedianIndex].next;
      mSmallIndex++;
    }

    double peek() const {
      return mMedianIndex == mFilterSize ? maxDouble : mData[mMedianIndex];
    }

    bool atEnd() const { return mMedianIndex == mFilterSize; }

    bool belowMedian(int i) const {
      return atEnd() || mData[i] < mData[mMedianIndex] ||
             (mData[i] == mData[mMedianIndex] && i < mMedianIndex);
    }
    size_t mFilterSize;
    std::vector<Link> mLinks;
    const double *mData;
    std::vector<std::pair<double, int>> mSorted;
    int mMedianIndex;
    int mSmallIndex;

  };

public:
  MedianFilter(size_t size)
      : mSize(size), mHalfSize((size - 1) / 2), a(size), b(size) {
    assert(mSize % 2);
  }

  void processBlock(double *out, int index) {
    for (int i = 0; i < mSize; i++) {
      a.del(i);
      b.undelete(i);
      assert(a.mSmallIndex + b.mSmallIndex <= mHalfSize);
      if (a.mSmallIndex + b.mSmallIndex < mHalfSize) {
        nextBlock()->advance();
      }
      assert(a.mSmallIndex + b.mSmallIndex <= mHalfSize);
      out[(index - 1) * mSize + i + 1] = std::min(a.peek(), b.peek());
    }
    assert(a.mSmallIndex == 0);
    assert(b.mSmallIndex == mHalfSize);
  }

  Block *nextBlock() {
    if (b.atEnd()) {
      return &a;
    } else if (a.atEnd()) {
      return &b;
    } else {
      return a.peek() <= b.peek() ? &a : &b;
    }
  }

  void process(const Ref<const ArrayXd> &in, Ref<ArrayXd> out) {
    // assuming size of in is multiple of block size ...
    // client code should pad
    int nBlocks = in.size() / mSize;
    b.construct(in.data() + mSize * 0);
    out[0] = b.peek();
    for (int i = 1; i < nBlocks; i++) {
      std::swap(a, b);
      b.construct(in.data() + mSize * i);
      b.unwind();
      processBlock(out.data(), i);
    }
  }

private:
  size_t mSize;
  size_t mHalfSize;
  Block a;
  Block b;
};
} // namespace medianfilter
} // namespace fluid
