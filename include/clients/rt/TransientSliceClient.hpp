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

#include "../common/AudioClient.hpp"
#include "../common/BufferedProcess.hpp"
#include "../common/FluidBaseClient.hpp"
#include "../common/FluidNRTClientWrapper.hpp"
#include "../common/ParameterConstraints.hpp"
#include "../common/ParameterSet.hpp"
#include "../common/ParameterTrackChanges.hpp"
#include "../common/ParameterTypes.hpp"
#include "../../algorithms/public/TransientSegmentation.hpp"
#include <tuple>

namespace fluid {
namespace client {

class TransientSliceClient : public FluidBaseClient,
                             public AudioIn,
                             public AudioOut
{

public:
  enum TransientParamIndex {
    kOrder,
    kBlockSize,
    kPadding,
    kSkew,
    kThreshFwd,
    kThreshBack,
    kWinSize,
    kDebounce,
    kMinSeg
  };

  FLUID_DECLARE_PARAMS(
      LongParam("order", "Order", 20, Min(10), LowerLimit<kWinSize>(),
                UpperLimit<kBlockSize>()),
      LongParam("blockSize", "Block Size", 256, Min(100), LowerLimit<kOrder>()),
      LongParam("padSize", "Padding", 128, Min(0)),
      FloatParam("skew", "Skew", 0, Min(-10), Max(10)),
      FloatParam("threshFwd", "Forward Threshold", 2, Min(0)),
      FloatParam("threshBack", "Backward Threshold", 1.1, Min(0)),
      LongParam("windowSize", "Window Size", 14, Min(0), UpperLimit<kOrder>()),
      LongParam("clumpLength", "Clumping Window Length", 25, Min(0)),
      LongParam("minSliceLength", "Minimum Length of Slice", 1000));

  TransientSliceClient(ParamSetViewType& p)
      : mParams{p}
  {
    audioChannelsIn(1);
    audioChannelsOut(1);
  }

  template <typename T>
  void process(std::vector<HostVector<T>>& input,
               std::vector<HostVector<T>>& output, FluidContext& c)
  {
    using namespace std;

    if (!input[0].data() || !output[0].data()) return;

    index order = get<kOrder>();
    index blockSize = get<kBlockSize>();
    index padding = get<kPadding>();
    index hostVecSize = input[0].size();
    index maxWinIn = 2 * blockSize + padding;
    index maxWinOut = maxWinIn; // blockSize - padding;

    if (mTrackValues.changed(order, blockSize, padding, hostVecSize) ||
        !mExtractor.initialized())
    {
      mExtractor.init(order, blockSize, padding);
      mBufferedProcess.hostSize(hostVecSize);
      mBufferedProcess.maxSize(maxWinIn, maxWinOut,
                               FluidBaseClient::audioChannelsIn(),
                               FluidBaseClient::audioChannelsOut());
    }

    double skew = pow(2, get<kSkew>());
    double threshFwd = get<kThreshFwd>();
    double thresBack = get<kThreshBack>();
    index  halfWindow = lrint(get<kWinSize>() / 2);
    index  debounce = get<kDebounce>();
    index  minSeg = get<kMinSeg>();

    mExtractor.setDetectionParameters(skew, threshFwd, thresBack, halfWindow,
                                      debounce, minSeg);

    RealMatrix in(1, hostVecSize);

    in.row(0) = input[0]; // need to convert float->double in some hosts
    mBufferedProcess.push(RealMatrixView(in));

    mBufferedProcess.process(mExtractor.inputSize(), mExtractor.hopSize(),
                             mExtractor.hopSize(), c,
                             [this](RealMatrixView in, RealMatrixView out) {
                               mExtractor.process(in.row(0), out.row(0));
                             });

    RealMatrix out(1, hostVecSize);
    mBufferedProcess.pull(RealMatrixView(out));

    if (output[0].data()) output[0] = out.row(0);
  }

  index latency()
  {
    return get<kPadding>() + get<kBlockSize>() - get<kOrder>();
  }

  void reset()
  {
    mBufferedProcess.reset();
    mExtractor.init(get<kOrder>(), get<kBlockSize>(), get<kPadding>());
  }

private:
  ParameterTrackChanges<index, index, index, index> mTrackValues;
  algorithm::TransientSegmentation                  mExtractor;

  BufferedProcess        mBufferedProcess;
  FluidTensor<double, 1> mTransients;
};

using RTTransientSliceClient = ClientWrapper<TransientSliceClient>;

auto constexpr NRTTransientSliceParams = makeNRTParams<TransientSliceClient>(
    InputBufferParam("source", "Source Buffer"),
    BufferParam("indices", "Indices Buffer"));

using NRTTransientSliceClient =
    NRTSliceAdaptor<TransientSliceClient, decltype(NRTTransientSliceParams),
                    NRTTransientSliceParams, 1, 1>;

using NRTThreadedTransientSliceClient =
    NRTThreadingAdaptor<NRTTransientSliceClient>;

} // namespace client
} // namespace fluid
