#pragma once

#include <clients/common/ParameterTypes.hpp>
#include <clients/common/ParameterConstraints.hpp>
#include <clients/common/FluidBaseClient.hpp>
#include <clients/common/MakeParams.hpp>
#include <clients/common/DeriveSTFTParams.hpp>
#include <clients/rt/BufferedProcess.hpp>
#include <algorithms/public/NMF.hpp>
#include <clients/common/ParameterTrackChanges.hpp>
namespace fluid {
namespace client {

enum NMFMatchParamIndex{kFilterbuf,kRank,kIterations,kWinSize,kHopSize,kFFTSize,kMaxWinSize};

auto constexpr NMFMatchParams =
AddSTFTParams<1024,256,-1>(
  std::make_tuple(
    BufferParam("filterBuf", "Filters Buffer"),
    LongParam("rank", "Rank", 1, Min(1)),
    LongParam<Fixed<true>>("maxRank","Maximum Rank",-1,Min(1)),
    LongParam("iterations", "Iterations", 10, Min(1))));

using ParamsT = decltype(NMFMatchParams);

template <typename T, typename U = T>
class NMFMatch : public FluidBaseClient<ParamsT,NMFMatchParams>, public AudioIn, public ControlOut {
  using HostVector = HostVector<U>;
public:

  NMFMatch(size_t maxRank, size_t maxWin):FluidBaseClient<ParamsT,NMFMatchParams>(NMFMatchParams)
  {
    audioChannelsIn(1);
    controlChannelsOut(1);
  }

  size_t latency() { return get<kWinSize>(); }

  Result process(std::vector<HostVector> &input, std::vector<HostVector> &output)
  {
    if(!input[0].data()) return;
    
    if (get<kFilterbuf>().get()) {

      auto filterBuffer = BufferAdaptor::Access(get<kFilterbuf>().get());

      if (!filterBuffer.valid()) {
        return {Result::Status::kError,"Filter buffer invalid"};
      }

      size_t winSize, hopSize, fftSize;
      std::tie(winSize,hopSize,fftSize) = impl::deriveSTFTParams<kWinSize,kHopSize,kFFTSize>(*this);

      size_t nBins = fftSize / 2 + 1;
      size_t rank  = get<kRank>();

      if (filterBuffer.numChans() != rank || filterBuffer.numFrames() != nBins)
      {
        return {Result::Status::kError, "Filters buffer needs to be (fftsize / 2 + 1) frames by "
                     "rank channels"};
      }

      if(mTrackValues.changed(rank, nBins))
      {
        tmpFilt.resize(nBins,rank);
        tmpMagnitude.resize(1,nBins);
        tmpOut.resize(rank);
        mNMF.reset(new algorithm::NMF(rank, get<kIterations>()));
      }

      for (size_t i = 0; i < tmpFilt.cols(); ++i)
        tmpFilt.col(i) = filterBuffer.samps(0, i);


      mSTFTProcessor.process(*this, input, output,
        [&](ComplexMatrixView in, ComplexMatrixView out)
        {
          algorithm::STFT::magnitude(in, tmpMagnitude);
          mNMF->processFrame(tmpMagnitude.row(0), tmpFilt, tmpOut);
        });
        output[0] = tmpOut;
    }
  }

private:

  ParameterTrackChanges<size_t,size_t> mTrackValues;
  STFTBufferedProcess<T, U, NMFMatch, kMaxWinSize, kWinSize, kHopSize, kFFTSize, false> mSTFTProcessor;
  std::unique_ptr<algorithm::NMF> mNMF;

  FluidTensor<double, 2> tmpFilt;
  FluidTensor<double, 2> tmpMagnitude;
  FluidTensor<double, 1> tmpOut;
  
  size_t mNBins{0};
  size_t mRank{0};
};
} // namespace client
} // namespace fluid
