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

#include "../../data/FluidIndex.hpp"
#include "../../data/FluidMemory.hpp"
#include "../../data/FluidTensor.hpp"
#include <cassert>
#include <deque>

namespace fluid {
namespace algorithm {

class MedianFilter
{

public:
  MedianFilter(index maxSize, Allocator& alloc)
      : mUnsorted(alloc), mSorted(alloc)
  {
    mUnsorted.reserve(asUnsigned(maxSize));
    mSorted.reserve(asUnsigned(maxSize));
    init(maxSize);
  }

  void init(index size)
  {
    assert(size >= 3);
    assert(size % 2);
    assert(asUnsigned(size) <= mUnsorted.capacity());
    mFilterSize = asUnsigned(size);
    mUnsorted.resize(mFilterSize, 0);
    mSorted.resize(mFilterSize, 0);
    std::fill(mUnsorted.begin(), mUnsorted.end(), 0);
    std::fill(mSorted.begin(), mSorted.end(), 0);
    mInitialized = true;
  }

  double processSample(double val)
  {
    assert(mInitialized);
    double old = mUnsorted.front();
    std::rotate(mUnsorted.begin(), mUnsorted.begin() + 1, mUnsorted.end());
    mUnsorted[mFilterSize - 1] = val;
    mSorted.erase(std::lower_bound(mSorted.begin(), mSorted.end(), old));
    mSorted.insert(std::upper_bound(mSorted.begin(), mSorted.end(), val), val);
    return mSorted[size_t(mSorted.size() / 2)];
  }

  index size() { return asSigned(mFilterSize); }

  bool initialized() const { return mInitialized; }

private:
  std::size_t mFilterSize{0};
  bool        mInitialized{false};

  rt::vector<double> mUnsorted;
  rt::vector<double> mSorted;
};

} // namespace algorithm
} // namespace fluid
