/*
Part of the Fluid Corpus Manipulation Project (http://www.flucoma.org/)
Copyright 2017-2019 University of Huddersfield.
Licensed under the BSD-3 License.
See license.md file in the project root for full license information.
This project has received funding from the European Research Council (ERC)
under the European Union’s Horizon 2020 research and innovation programme
(grant agreement No 725899).
*/
#pragma once

#include "../../data/FluidTensor.hpp"
#include <cassert>
#include <deque>

namespace fluid {
namespace algorithm {

class MedianFilter
{

public:
  void init(size_t size)
  {
    assert(size % 2);
    mFilterSize = size;
    mMiddle = (mFilterSize - 1) / 2;
    mUnsorted = std::deque<double>(mFilterSize, 0);
    mSorted = std::deque<double>(mFilterSize, 0);
  }

  double processSample(double val)
  {
    mUnsorted.push_back(val);
    double old = mUnsorted.front();
    mUnsorted.pop_front();
    for (auto it = mSorted.begin(); it != mSorted.end(); ++it)
    {
      if ((*it) == old)
      {
        it = mSorted.erase(it);
        break;
      }
    }
    if (val <= mSorted.front())
      mSorted.push_front(val);
    else if (val >= mSorted.back())
      mSorted.push_back(val);
    else
    {
      auto it = mSorted.begin();
      while (*it < val) it++;
      mSorted.insert(it, val);
    }
    auto it = mSorted.begin();
    std::advance(it, mMiddle);
    return *it;
  }
  
  size_t size() { return mFilterSize; }

private:
  size_t mFilterSize;
  size_t mMiddle;

  std::deque<double> mUnsorted{5, 0};
  std::deque<double> mSorted{5, 0};
};
} // namespace algorithm
} // namespace fluid
