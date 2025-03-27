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

#include "../common/BufferedProcess.hpp"
#include "../common/FluidBaseClient.hpp"
#include "../common/FluidNRTClientWrapper.hpp"
#include "../common/ParameterConstraints.hpp"
#include "../common/ParameterTypes.hpp"
#include "../../algorithms/public/STFT.hpp"

namespace fluid {
namespace client {
namespace bufstft {
enum BufferSTFTParamIndex {
  kSource,
  kOffset,
  kNumFrames,
  kStartChan,
  kMag,
  kPhase,
  kResynth,
  kInvert,
  kPadding,
  kFFT
};

constexpr auto BufSTFTParams = defineParameters(
    InputBufferParam("source", "Source Buffer"),
    LongParam("startFrame", "Source Offset", 0, Min(0)),
    LongParam("numFrames", "Number of Frames", -1),
    LongParam("startChan", "Start Channel", 0, Min(0)),
    BufferParam("magnitude", "Magnitude Buffer"),
    BufferParam("phase", "Phase Buffer"),
    BufferParam("resynth", "Resynthesis Buffer"),
    LongParam("inverse", "Inverse Transform", 0, Min(0), Max(1)),
    EnumParam("padding", "Added Padding", 1, "None", "Default", "Full"),
    FFTParam("fftSettings", "FFT Settings", 1024, -1, -1));

class BufferSTFTClient : public FluidBaseClient,
                         public OfflineIn,
                         public OfflineOut
{
public:
  using ParamDescType = decltype(BufSTFTParams);

  using ParamSetViewType = ParameterSetView<ParamDescType>;
  std::reference_wrapper<ParamSetViewType> mParams;

  void setParams(ParamSetViewType& p) { mParams = p; }

  template <size_t N>
  auto& get() const
  {
    return mParams.get().template get<N>();
  }

  static constexpr auto& getParameterDescriptors() { return BufSTFTParams; }


  BufferSTFTClient(ParamSetViewType& p, FluidContext&) : mParams(p) {}

  template <typename T>
  Result process(FluidContext& c)
  {
    if (get<kInvert>() == 0)
      return processFwd<T>(c);
    else
      return processInverse<T>(c);
  }

private:
  template <typename T>
  Result processFwd(FluidContext&)
  {
    auto s = get<kSource>().get();
    if (!s) return {Result::Status::kError, "No input buffer supplied"};

    auto m = get<kMag>().get();
    auto p = get<kPhase>().get();

    bool haveMag = m != nullptr;
    bool havePhase = p != nullptr;

    if (!haveMag && !havePhase)
      return {Result::Status::kError,
              "Neither magnitude nor phase buffer supplied"};

    auto mags = BufferAdaptor::Access(m);
    auto phases = BufferAdaptor::Access(p);

    index  offset = get<kOffset>();
    index  numFrames = get<kNumFrames>();
    index  numChans = 1;
    Result rangeOK = bufferRangeCheck(get<kSource>().get(), offset, numFrames,
                                      get<kStartChan>(), numChans);

    if (!rangeOK.ok()) return rangeOK;

    auto source = BufferAdaptor::ReadAccess(s);

    if (haveMag && !mags.exists())
      return {Result::Status::kError, "Magnitude buffer not found"};

    if (havePhase && !phases.exists())
      return {Result::Status::kError, "Phase buffer not found"};

    index fftSize = get<kFFT>().fftSize();
    index winSize = get<kFFT>().winSize();
    index hopSize = get<kFFT>().hopSize();

    index  padding = FFTParams::padding(get<kFFT>(), get<kPadding>());
    double totalPadding = padding << 1;

    index paddedLength = static_cast<index>(numFrames + totalPadding);
    if (get<kPadding>() == 2)
      paddedLength = static_cast<index>(
          std::ceil(double(paddedLength) / hopSize) * hopSize);

    index numHops =
        static_cast<index>(1 + std::floor((paddedLength - winSize) / hopSize));

    index numBins = (fftSize >> 1) + 1;

    // I'm thinking that most things can only really deal with 65536 channels
    // (if that) => limitation on multichannel inputs vs FFT size
    if (numChans * numBins >= 65536)
      return {
          Result::Status::kError,
          "Can produce up to 65536 channels. Split your data up and try again"};

    if (m)
    {
      auto r = mags.resize(numHops, numBins * numChans,
                           source.sampleRate() / hopSize);
      if (!r.ok()) return r;
    }

    if (p)
    {
      auto r = phases.resize(numHops, numBins * numChans,
                             source.sampleRate() / hopSize);
      if (!r.ok()) return r;
    }

    auto input = source.samps(0)(Slice(offset, numFrames));

    FluidTensor<double, 1> paddedInput(paddedLength);

    auto paddingSlice = Slice(padding, input.size());
    paddedInput(paddingSlice) <<= input;

    FluidTensor<double, 2> tmpMags(numHops, numBins);
    FluidTensor<double, 2> tmpPhase(numHops, numBins);

    FluidTensor<std::complex<double>, 2> tmpComplex(numHops, numBins);

    auto stft = algorithm::STFT(winSize, fftSize, hopSize);

    for (index i = 0; i < numHops; ++i)
      stft.processFrame(paddedInput(Slice(i * hopSize, winSize)),
                        tmpComplex.row(i));

    if (haveMag)
    {
      algorithm::STFT::magnitude(tmpComplex, tmpMags);
      mags.allFrames().transpose() <<= tmpMags(Slice(0, numHops), Slice(0));
    }

    if (havePhase)
    {
      algorithm::STFT::phase(tmpComplex, tmpPhase);
      phases.allFrames().transpose() <<= tmpPhase(Slice(0, numHops), Slice(0));
    }
    return {};
  }

  template <typename T>
  Result processInverse(FluidContext&)
  {
    auto m = get<kMag>().get();
    auto p = get<kPhase>().get();

    bool haveMag = m != nullptr;
    bool havePhase = p != nullptr;

    if (!haveMag || !havePhase)
      return {Result::Status::kError,
              "Need both magnutude and phase buffers for inverse transform"};

    auto r = get<kResynth>().get();

    if (!r) return {Result::Status::kError, "No resynthesis buffer supplied"};

    auto mags = BufferAdaptor::ReadAccess(m);
    auto phases = BufferAdaptor::ReadAccess(p);

    if (mags.numFrames() != phases.numFrames() ||
        mags.numChans() != phases.numChans())
      return {Result::Status::kError,
              "Magnitude and Phase buffer sizes don't match"};

    index fftSize = get<kFFT>().fftSize();
    index winSize = get<kFFT>().winSize();
    index hopSize = get<kFFT>().hopSize();

    if (mags.numChans() != (fftSize >> 1) + 1)
      return {Result::Status::kError,
              "Wrong number of channels for FFT sizee of ",
              fftSize,
              " got ",
              mags.numChans(),
              " expected ",
              (fftSize >> 1) + 1};

    auto resynth = BufferAdaptor::Access(r);

    index numFrames = mags.numFrames();
    index padding = FFTParams::padding(get<kFFT>(), get<kPadding>());


    index paddedOutputSize = (mags.numFrames() - 1) * hopSize + winSize;
    index finalOutputSize = paddedOutputSize - padding;
    auto  resizeResult =
        resynth.resize(finalOutputSize, 1, mags.sampleRate() * hopSize);
    if (!resizeResult.ok()) return resizeResult;

    FluidTensor<double, 1> tmpOut(paddedOutputSize);
    FluidTensor<double, 1> normalizer(paddedOutputSize);

    FluidTensor<std::complex<double>, 2> tmpComplex(tmpOut.size() / hopSize,
                                                    mags.numChans());

    FluidTensor<double, 1> frame(winSize);

    auto magsView = mags.allFrames().transpose();
    auto phaseView = phases.allFrames().transpose();

    std::transform(magsView.begin(), magsView.end(), phaseView.begin(),
                   tmpComplex.begin(),
                   [](auto& m, auto& p) { return std::polar(m, p); });

    auto istft = algorithm::ISTFT(winSize, fftSize, hopSize);

    FluidTensor<double, 1> windowSquared(istft.window());
    windowSquared.apply([](double& x) { x *= x; });

    auto addIn = [](double& x, double& y) { x += y; };

    for (index i = 0; i < numFrames; ++i)
    {
      istft.processFrame(tmpComplex.row(i), frame);
      auto thisSlice = Slice(i * hopSize, winSize);
      tmpOut(thisSlice).apply(frame, addIn);
      normalizer(thisSlice).apply(windowSquared, addIn);
    }

    std::transform(tmpOut.begin(), tmpOut.end(), normalizer.begin(),
                   tmpOut.begin(), [](double x, double y) {
                     constexpr double epsilon =
                         std::numeric_limits<double>::epsilon();
                     return x / std::max(y, epsilon);
                   });

    resynth.samps(0) <<= tmpOut(Slice(padding, finalOutputSize));

    return {};
  }
};
} // namespace bufstft

using NRTThreadedBufferSTFTClient =
    NRTThreadingAdaptor<ClientWrapper<bufstft::BufferSTFTClient>>;

} // namespace client
} // namespace fluid
