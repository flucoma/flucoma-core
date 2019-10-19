/*
Copyright 2017-2019 University of Huddersfield.
Licensed under the BSD-3 License.
See LICENSE file in the project root for full license information.
This project has received funding from the European Research Council (ERC)
under the European Union’s Horizon 2020 research and innovation programme
(grant agreement No 725899).
*/
#pragma once

#include "../../data/FluidTensor.hpp"

#include <Eigen/Core>
#include <cassert>
#include <limits>

namespace fluid {
namespace algorithm {


const double maxDouble = std::numeric_limits<double>::max();

// This implements the algorithm described in
// Suomela, J., Median Filtering is Equivalent to Sorting
// https://arxiv.org/abs/1406.1717
// it is based on the author's own C++11 and python implementations
class MedianFilter
{

  using ArrayXd = Eigen::ArrayXd;


  struct Block
  {
    struct Link
    {
      int prev;
      int next;
    };

    Block(double* data, int size)
        : mData(data), mSize(size), mLinks(size + 1), mSorted(size)
    {}

    void sort()
    {
      for (int i = 0; i < mSize; i++)
      { mSorted[i] = std::make_pair(mData[i], i); }
      std::sort(mSorted.begin(), mSorted.end());
    }

    double insertRight(double newVal)
    {
      double oldVal = mData[0];
      int    insertP = 0, deleteP = 0;
      for (int i = 0; i < mSize; i++)
      {
        if (mSorted[i].second == 0)
          deleteP = i;
        else
          mSorted[i].second--; // decrease pos as we remove first
        if (mSorted[mSize - i - 1].first > newVal) insertP = mSize - i - 1;
      }
      if (newVal >= mSorted[mSize - 1].first)
        insertP = mSize; // will be decreased later
      auto newPair = std::make_pair(newVal, mSize - 1);
      if (insertP < deleteP)
      {
        std::memmove(mSorted.data() + insertP + 1, mSorted.data() + insertP,
                     (deleteP - insertP) * sizeof(std::pair<double, int>));
      } else if (insertP > deleteP)
      {
        insertP--;
        std::memmove(mSorted.data() + deleteP, mSorted.data() + deleteP + 1,
                     (insertP - deleteP) * sizeof(std::pair<double, int>));
      }
      mSorted[insertP] = newPair;
      std::memmove(mData, mData + 1, (mSize - 1) * sizeof(double));
      mData[mSize - 1] = newVal;
      init();
      return oldVal;
    }

    void init()
    {
      int a = mSize;
      for (int i = 0; i < mSize; i++)
      {
        int b = mSorted[i].second;
        mLinks[a].next = b;
        mLinks[b].prev = a;
        a = b;
      }
      mLinks[a].next = mSize;
      mLinks[mSize].prev = a;
      mMedianIndex = mSorted[mSize / 2].second;
      mSmallIndex = mSize / 2;
    }

    void unwind()
    {
      for (int i = 0; i < mSize; i++)
      {
        int  j = mSize - 1 - i;
        Link l = mLinks[j];
        mLinks[l.prev].next = l.next;
        mLinks[l.next].prev = l.prev;
      }
      mMedianIndex = mSize;
      mSmallIndex = 0;
    }

    void del(int i)
    {
      Link l = mLinks[i];
      mLinks[l.prev].next = l.next;
      mLinks[l.next].prev = l.prev;
      if (belowMedian(i))
      {
        mSmallIndex--;
      } else
      {
        if (i == mMedianIndex) { mMedianIndex = l.next; }
        if (mSmallIndex > 0)
        {
          mMedianIndex = mLinks[mMedianIndex].prev;
          mSmallIndex--;
        }
      }
    }

    void undelete(int i)
    {
      Link l = mLinks[i];
      mLinks[l.prev].next = i;
      mLinks[l.next].prev = i;
      if (belowMedian(i)) { mMedianIndex = mLinks[mMedianIndex].prev; }
    }

    void advance()
    {
      mMedianIndex = mLinks[mMedianIndex].next;
      mSmallIndex++;
    }

    double peek() const
    {
      return mMedianIndex == mSize ? maxDouble : mData[mMedianIndex];
    }

    bool atEnd() const { return mMedianIndex == mSize; }

    bool belowMedian(int i) const
    {
      return atEnd() || mData[i] < mData[mMedianIndex] ||
             (mData[i] == mData[mMedianIndex] && i < mMedianIndex);
    }

    friend std::ostream& operator<<(std::ostream& os, const Block& b)
    {
      os << "--" << std::endl;
      for (int i = 0; i < b.mSize; i++) os << b.mData[i] << " ";
      os << std::endl;
      for (int i = 0; i < b.mSize; i++)
        os << b.mSorted[i].first << " " << b.mSorted[i].second << " - ";
      os << std::endl;
      for (int i = 0; i < b.mLinks.size(); i++)
        os << b.mLinks[i].prev << " " << b.mLinks[i].next << " - ";
      os << std::endl;
      os << b.mSize << " " << b.mMedianIndex << " " << b.mSmallIndex
         << std::endl;
      return os;
    }

    double*                             mData;
    size_t                              mSize;
    std::vector<Link>                   mLinks;
    std::vector<std::pair<double, int>> mSorted;
    int                                 mMedianIndex;
    int                                 mSmallIndex;
  };

public:
  void processBlock(double* out, int index)
  {
    for (int i = 0; i < mSize; i++)
    {
      a.del(i);
      b.undelete(i);
      assert(a.mSmallIndex + b.mSmallIndex <= mHalfSize);
      if (a.mSmallIndex + b.mSmallIndex < mHalfSize) { nextBlock()->advance(); }
      assert(a.mSmallIndex + b.mSmallIndex <= mHalfSize);
      out[(index - 1) * mSize + i + 1] = std::min(a.peek(), b.peek());
    }
    assert(a.mSmallIndex == 0);
    assert(b.mSmallIndex == mHalfSize);
  }

  Block* nextBlock()
  {
    if (b.atEnd())
    {
      return &a;
    } else if (a.atEnd())
    {
      return &b;
    } else
    {
      return a.peek() <= b.peek() ? &a : &b;
    }
  }

  MedianFilter(const Eigen::Ref<ArrayXd> in, int size)
      : mSize(size), mHalfSize((size - 1) / 2), a(nullptr, 0), b(nullptr, 0)
  {
    assert(mSize % 2);
    int nBlocks = in.size() / mSize;
    mInput = ArrayXd(in);
    for (int i = 0; i < nBlocks; i++)
    {
      blocks.emplace_back(Block(mInput.data() + mSize * i, mSize));
      blocks.back().sort();
    }
  }

  void insertRight(double val)
  {
    for (auto x = blocks.rbegin(); x != blocks.rend(); x++)
    { val = x->insertRight(val); }
  }

  void process(Eigen::Ref<ArrayXd> out)
  {
    // assuming size of in is multiple of block size ...
    // client code should pad
    int nBlocks = blocks.size();
    b = blocks.front();
    b.init();
    out[0] = b.peek();
    int i = 0;

    for (Block& x : blocks)
    {
      if (i > 0)
      {
        a = b;
        b = x;
        b.init();
        b.unwind();
        processBlock(out.data(), i);
      }
      i++;
    }
  }
  ArrayXd mInput;

private:
  int   mSize{3};
  int   mHalfSize{1};
  Block a;
  Block b;

  std::vector<Block> blocks;
};
} // namespace algorithm
} // namespace fluid
