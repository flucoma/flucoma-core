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
#include <Eigen/Core>
#include <fft/fft.hpp>

namespace fluid {
namespace algorithm {

namespace impl {
class FFTSetup
{
public:
  FFTSetup(index maxSize) : mMaxSize{maxSize}
  {
    assert(maxSize > 0 && "FFT Max Size must be > 0!");
    htl::create_fft_setup(&mSetup,
                          asUnsigned(static_cast<index>(std::log2(maxSize))));
  }

  ~FFTSetup()
  {
    htl::destroy_fft_setup(mSetup);
    mSetup = 0;
  }

  FFTSetup(FFTSetup const&) = delete;
  FFTSetup& operator=(FFTSetup const&) = delete;

  FFTSetup(FFTSetup&& other) { *this = std::move(other); };
  FFTSetup& operator=(FFTSetup&& other)
  {
    using std::swap;
    swap(mMaxSize, other.mMaxSize);
    swap(mSetup, other.mSetup);
    return *this;
  }

  htl::setup_type<double> operator()() const noexcept { return mSetup; }
  index                   maxSize() const noexcept { return mMaxSize; }

private:
  htl::setup_type<double> mSetup{nullptr};
  index                   mMaxSize;
};
} // namespace impl

class FFT
{

public:
  using MapXcd = Eigen::Map<Eigen::ArrayXcd>;

  static void setup() { getFFTSetup(); }

  FFT() = delete;

  FFT(index size, Allocator& alloc = FluidDefaultAllocator()) noexcept
      : mMaxSize(size), mSize(size), mFrameSize(size / 2 + 1),
        mLog2Size(static_cast<index>(std::log2(size))), mSetup(getFFTSetup()),
        mRealBuffer(asUnsigned(mFrameSize), alloc),
        mImagBuffer(asUnsigned(mFrameSize), alloc),
        mOutputBuffer(asUnsigned(mFrameSize), alloc)
  {}

  FFT(const FFT& other) = delete;
  FFT(FFT&& other) noexcept = default;

  FFT& operator=(const FFT&) = delete;
  FFT& operator=(FFT&& other) noexcept = default;

  void resize(index newSize) noexcept
  {
    assert(newSize <= mMaxSize);
    mFrameSize = newSize / 2 + 1;
    mLog2Size = static_cast<index>(std::log2(newSize));
    mSize = newSize;
  }

  MapXcd process(const Eigen::Ref<const Eigen::ArrayXd>& input)
  {

    mSplit.realp = mRealBuffer.data();
    mSplit.imagp = mImagBuffer.data();
    htl::rfft(mSetup, input.derived().data(), &mSplit, asUnsigned(input.size()),
              asUnsigned(mLog2Size));
    mSplit.realp[mFrameSize - 1] = mSplit.imagp[0];
    mSplit.imagp[mFrameSize - 1] = 0;
    mSplit.imagp[0] = 0;
    for (index i = 0; i < mFrameSize; i++)
    {
      mOutputBuffer[asUnsigned(i)] =
          0.5 * std::complex<double>(mSplit.realp[i], mSplit.imagp[i]);
    }
    return {mOutputBuffer.data(), mFrameSize};
  }

protected:
  static htl::setup_type<double> getFFTSetup()
  {
    static const impl::FFTSetup static_setup(65536);
    return static_setup();
  }

  index mMaxSize{16384};
  index mSize{1024};
  index mFrameSize{513};
  index mLog2Size{10};

  htl::setup_type<double> mSetup;
  htl::split_type<double> mSplit;
  rt::vector<double>      mRealBuffer;
  rt::vector<double>      mImagBuffer;

private:
  rt::vector<std::complex<double>> mOutputBuffer;
};

class IFFT : public FFT
{

public:
  IFFT(index size, Allocator& alloc = FluidDefaultAllocator())
      : FFT(size, alloc), mOutputBuffer(asUnsigned(size), alloc)
  {}

  using MapXd = Eigen::Map<Eigen::ArrayXd>;

  MapXd process(const Eigen::Ref<const Eigen::ArrayXcd>& input)
  {
    assert(input.size() == mFrameSize);

    mSplit.realp = mRealBuffer.data();
    mSplit.imagp = mImagBuffer.data();
    for (index i = 0; i < input.size(); i++)
    {
      mSplit.realp[i] = input(i).real();
      mSplit.imagp[i] = input(i).imag();
    }
    mSplit.imagp[0] = mSplit.realp[mFrameSize - 1];
    htl::rifft(mSetup, &mSplit, mOutputBuffer.data(), asUnsigned(mLog2Size));
    return {mOutputBuffer.data(), mSize};
  }

private:
  rt::vector<double> mOutputBuffer;
};
} // namespace algorithm
} // namespace fluid
