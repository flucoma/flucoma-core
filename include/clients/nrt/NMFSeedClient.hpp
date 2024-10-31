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

#include "CommonResults.hpp"
#include "../common/FluidBaseClient.hpp"
#include "../common/FluidNRTClientWrapper.hpp"
#include "../common/OfflineClient.hpp"
#include "../common/ParameterConstraints.hpp"
#include "../common/ParameterSet.hpp"
#include "../common/ParameterTypes.hpp"
#include "../../algorithms/public/NNDSVD.hpp"
#include "../../algorithms/public/STFT.hpp"
#include "../../data/FluidTensor.hpp"

namespace fluid {
namespace client {
namespace nndsvd {

enum NMFSeedParamIndex {
  kSource,
  kFilters,
  kEnvelopes,
  kMinRank,
  kMaxRank,
  kCoverage,
  kMethod,
  kFFT
};

constexpr auto NMFSeedParams =
    defineParameters(InputBufferParam("source", "Source Buffer"),
                     BufferParam("bases", "Bases Buffer"),
                     BufferParam("activations", "Activations Buffer"),
                     LongParam("minComponents", "Minimum Number of Components",
                               1, Min(1), UpperLimit<kMaxRank>()),
                     LongParam("maxComponents", "Maximum Number of Components",
                               200, Min(1), LowerLimit<kMinRank>()),
                     FloatParam("coverage", "Coverage", 0.5, Min(0), Max(1)),
                     EnumParam("method", "Initialization Method", 0, "NMF-SVD",
                               "NNDSVDar", "NNDSVDa", "NNDSVD"),
                     FFTParam("fftSettings", "FFT Settings", 1024, -1, -1));

class NMFSeedClient : public FluidBaseClient, public OfflineIn, public OfflineOut
{
public:
  using ParamDescType = decltype(NMFSeedParams);

  using ParamSetViewType = ParameterSetView<ParamDescType>;
  std::reference_wrapper<ParamSetViewType> mParams;

  void setParams(ParamSetViewType& p) { mParams = p; }

  template <size_t N>
  auto& get() const
  {
    return mParams.get().template get<N>();
  }

  static constexpr auto getParameterDescriptors() { return NMFSeedParams; }

  NMFSeedClient(ParamSetViewType& p, FluidContext&) : mParams{p} {}

  template <typename T>
  Result process(FluidContext&)
  {
    auto source = BufferAdaptor::ReadAccess(get<kSource>().get());

    if (!source.exists())
      return {Result::Status::kError, "Source Buffer Supplied But Invalid"};

    double sampleRate = source.sampleRate();
    index  nFrames = source.numFrames();
    auto   fftParams = get<kFFT>();
    index  nWindows = static_cast<index>(
        std::floor((nFrames + fftParams.hopSize()) / fftParams.hopSize()));
    index nBins = fftParams.frameSize();

    if (source.numChans() > 1)
    { return {Result::Status::kError, "Only one channel supported"}; }

    auto stft = algorithm::STFT(fftParams.winSize(), fftParams.fftSize(),
                                fftParams.hopSize());
    auto tmp = RealVector(nFrames);
    tmp <<= source.samps(0, nFrames, 0);
    auto spectrum = ComplexMatrix(nWindows, nBins);
    auto magnitude = RealMatrix(nWindows, nBins);
    auto outputFilters = RealMatrix(get<kMaxRank>(), nBins);
    auto outputEnvelopes = RealMatrix(nWindows, get<kMaxRank>());

    stft.process(tmp, spectrum);
    algorithm::STFT::magnitude(spectrum, magnitude);

    auto nndsvd = algorithm::NNDSVD();

    index rank = nndsvd.process(magnitude, outputFilters, outputEnvelopes,
                                get<kMinRank>(), get<kMaxRank>(),
                                get<kCoverage>(), get<kMethod>());

    auto   filters = BufferAdaptor::Access{get<kFilters>().get()};
    Result resizeResult =
        filters.resize(nBins, rank, sampleRate / fftParams.fftSize());
    if (!resizeResult.ok()) return resizeResult;

    for (index j = 0; j < rank; ++j)
    { filters.samps(j) <<= outputFilters.row(j); }

    auto envelopes = BufferAdaptor::Access{get<kEnvelopes>().get()};
    resizeResult = envelopes.resize((nFrames / fftParams.hopSize()) + 1, rank,
                                    sampleRate / fftParams.hopSize());
    if (!resizeResult.ok()) return resizeResult;
    auto maxH =
        *std::max_element(outputEnvelopes.begin(), outputEnvelopes.end());
    auto scale = 1. / (maxH);
    for (index j = 0; j < rank; j++)
    {
      auto env = envelopes.samps(j);
      env <<= outputEnvelopes.col(j);
      env.apply([scale](float& x) { x *= static_cast<float>(scale); });
    }
    return OK();
  }
};
} // namespace nndsvd

using NRTThreadedNMFSeedClient =
    NRTThreadingAdaptor<ClientWrapper<nndsvd::NMFSeedClient>>;

} // namespace client
} // namespace fluid
