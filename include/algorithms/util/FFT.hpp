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

#include "HISSTools_FFT/HISSTools_FFT.h"
#include "../../data/FluidIndex.hpp"
#include <Eigen/Core>

namespace fluid {
namespace algorithm {

class FFT
{

public:
  using ArrayXcd = Eigen::ArrayXcd;
  using ArrayXcdRef = Eigen::Ref<ArrayXcd>;
  using ArrayXd = Eigen::ArrayXd;
  using ArrayXdRef = Eigen::Ref<const ArrayXd>;

  FFT() = delete;

  FFT(index size)
      : mMaxSize(size), mSize(size), mFrameSize(size / 2 + 1),
        mLog2Size(static_cast<index>(std::log2(size))),
        mOutputBuffer(mFrameSize), mRealBuffer(mFrameSize),
        mImagBuffer(mFrameSize)
  {
    hisstools_create_setup(&mSetup, asUnsigned(mLog2Size));
    mSplit.realp = mRealBuffer.data();
    mSplit.imagp = mImagBuffer.data();
  }

  ~FFT() { hisstools_destroy_setup(mSetup); }

  FFT(const FFT& other) = delete;

  FFT(FFT&& other) { *this = std::move(other); }

  FFT& operator=(const FFT&) = delete;

  FFT& operator=(FFT&& other)
  {
    mMaxSize = other.mMaxSize;
    mSize = other.mSize;
    mFrameSize = other.mFrameSize;
    mLog2Size = other.mLog2Size;
    std::swap(mOutputBuffer, other.mOutputBuffer);
    std::swap(mRealBuffer, other.mRealBuffer);
    std::swap(mImagBuffer, other.mImagBuffer);
    std::swap(mSplit, other.mSplit);
    std::swap(mSetup, other.mSetup);
    return *this;
  }

  void resize(index newSize)
  {
    assert(newSize <= mMaxSize);
    mFrameSize = newSize / 2 + 1;
    mLog2Size = static_cast<index>(std::log2(newSize));
    mSize = newSize;
  }

  Eigen::Ref<ArrayXcd> process(const ArrayXdRef& input)
  {
    hisstools_rfft(mSetup, input.data(), &mSplit, asUnsigned(input.size()),
                   asUnsigned(mLog2Size));
    mSplit.realp[mFrameSize - 1] = mSplit.imagp[0];
    mSplit.imagp[mFrameSize - 1] = 0;
    mSplit.imagp[0] = 0;
    for (index i = 0; i < mFrameSize; i++)
    {
      mOutputBuffer(i) =
          0.5 * std::complex<double>(mSplit.realp[i], mSplit.imagp[i]);
    }
    return mOutputBuffer.segment(0, mFrameSize);
  }

protected:
  index mMaxSize{16384};
  index mSize{1024};
  index mFrameSize{513};
  index mLog2Size{10};

  FFT_SETUP_D         mSetup;
  FFT_SPLIT_COMPLEX_D mSplit;

private:
  ArrayXcd mOutputBuffer;
  ArrayXd  mRealBuffer;
  ArrayXd  mImagBuffer;
};

class IFFT : public FFT
{

public:
  IFFT(index size) : FFT(size), mOutputBuffer(size) {}

  using ArrayXcdRef = Eigen::Ref<const ArrayXcd>;
  using ArrayXdRef = Eigen::Ref<ArrayXd>;

  Eigen::Ref<ArrayXd> process(const Eigen::Ref<const ArrayXcd>& input)
  {
    for (index i = 0; i < input.size(); i++)
    {
      mSplit.realp[i] = input[i].real();
      mSplit.imagp[i] = input[i].imag();
    }
    mSplit.imagp[0] = mSplit.realp[mFrameSize - 1];
    hisstools_rifft(mSetup, &mSplit, mOutputBuffer.data(),
                    asUnsigned(mLog2Size));
    return mOutputBuffer.segment(0, mSize);
  }

private:
  ArrayXd mOutputBuffer;
};
} // namespace algorithm
} // namespace fluid
