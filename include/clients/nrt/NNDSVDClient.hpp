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

#include "algorithms/NNDSVD.hpp"
#include "algorithms/public/STFT.hpp"
#include "data/FluidTensor.hpp"
#include "clients/common/FluidBaseClient.hpp"
#include "clients/common/FluidNRTClientWrapper.hpp"
#include "clients/common/OfflineClient.hpp"
#include "clients/common/ParameterConstraints.hpp"
#include "clients/common/ParameterSet.hpp"
#include "clients/common/ParameterTypes.hpp"
#include <algorithm> //for max_element
#include <cassert>
#include <sstream> //for ostringstream
#include <string>
#include <unordered_set>
#include <utility> //for std make_pair
#include <vector>  //for containers of params, and for checking things

namespace fluid {
namespace client {

class NNDSVDClient : public FluidBaseClient,
                    public OfflineIn,
                    public OfflineOut {

  enum NNDSVDParamIndex {
    kSource,
    kFilters,
    kEnvelopes,
    kMinRank,
    kMaxRank,
    kCoverage,
    kMethod,
    kFFT
  };

public:
  FLUID_DECLARE_PARAMS(InputBufferParam("source", "Source Buffer"),
                       BufferParam("bases", "Bases Buffer"),
                       BufferParam("activations", "Activations Buffer"),
                       LongParam("minComponents",
                                 "Minimum Number of Components", 1, Min(1),
                                 UpperLimit<kMaxRank>()),
                       LongParam("maxComponents",
                                 "Maximum Number of Components", 200, Min(1),
                                 LowerLimit<kMinRank>()),
                       FloatParam("coverage", "Coverage", 0.5, Min(0), Max(1)),
                       EnumParam("method", "Initialization method", 0,
                                 "NMF-SVD", "NNDSVDar", "NNDSVDa", "NNDSVD"),
                       FFTParam("fftSettings", "FFT Settings", 1024, -1, -1));

  NNDSVDClient(ParamSetViewType &p) : mParams{p} {}

  template <typename T> Result process(FluidContext&) {
    auto source = BufferAdaptor::ReadAccess(get<kSource>().get());

    if (!source.exists())
      return {Result::Status::kError, "Source Buffer Supplied But Invalid"};

    double sampleRate = source.sampleRate();
    index nFrames = source.numFrames();
    auto fftParams = get<kFFT>();
    index nWindows = static_cast<index>(
        std::floor((nFrames + fftParams.hopSize()) / fftParams.hopSize()));
    index nBins = fftParams.frameSize();

    if (source.numChans() > 1) {
      return {Result::Status::kError, "Only one channel supported"};
    }

    auto stft = algorithm::STFT(fftParams.winSize(), fftParams.fftSize(),
                                fftParams.hopSize());
    auto tmp = FluidTensor<double, 1>(nFrames);
    tmp = source.samps(0, nFrames, 0);
    auto spectrum = FluidTensor<std::complex<double>, 2>(nWindows, nBins);
    auto magnitude = FluidTensor<double, 2>(nWindows, nBins);
    auto outputFilters = FluidTensor<double, 2>(get<kMaxRank>(), nBins);
    auto outputEnvelopes = FluidTensor<double, 2>(nWindows, get<kMaxRank>());

    stft.process(tmp, spectrum);
    algorithm::STFT::magnitude(spectrum, magnitude);

    auto nndsvd = algorithm::NNDSVD();

    index rank = nndsvd.process(magnitude, outputFilters, outputEnvelopes,
                                get<kMinRank>(), get<kMaxRank>(),
                                get<kCoverage>(), get<kMethod>());

    auto filters = BufferAdaptor::Access{get<kFilters>().get()};
    Result resizeResult =
        filters.resize(nBins, rank, sampleRate / fftParams.fftSize());
    if (!resizeResult.ok())
      return resizeResult;

    for (index j = 0; j < rank; ++j) {
      filters.samps(j) = outputFilters.row(j);
    }

    auto envelopes = BufferAdaptor::Access{get<kEnvelopes>().get()};
    resizeResult = envelopes.resize((nFrames / fftParams.hopSize()) + 1, rank,
                                    sampleRate / fftParams.hopSize());
    if (!resizeResult.ok())
      return resizeResult;
    auto maxH =
        *std::max_element(outputEnvelopes.begin(), outputEnvelopes.end());
    auto scale = 1. / (maxH);
    for (index j = 0; j < rank; j++) {
      auto env = envelopes.samps(j);
      env = outputEnvelopes.col(j);
      env.apply([scale](float &x) { x *= static_cast<float>(scale); });
    }
    return {Result::Status::kOk, ""};
  }
};

using NRTThreadedNNDSVDClient =
    NRTThreadingAdaptor<ClientWrapper<NNDSVDClient>>;

} // namespace client
} // namespace fluid
