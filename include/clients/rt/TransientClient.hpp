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

#include "../common/BufferedProcess.hpp"
#include "../common/FluidBaseClient.hpp"
#include "../common/FluidNRTClientWrapper.hpp"
#include "../common/ParameterConstraints.hpp"
#include "../common/ParameterSet.hpp"
#include "../common/ParameterTrackChanges.hpp"
#include "../common/ParameterTypes.hpp"
#include "../../algorithms/public/TransientExtraction.hpp"
#include "../../data/TensorTypes.hpp"
#include <complex>
#include <string>
#include <tuple>

namespace fluid {
namespace client {
namespace transient {

enum TransientParamIndex {
  kOrder,
  kBlockSize,
  kPadding,
  kSkew,
  kThreshFwd,
  kThreshBack,
  kWinSize,
  kDebounce
};

constexpr auto TransientParams = defineParameters(
    LongParam("order", "Order", 20, Min(10), LowerLimit<kWinSize>(),
              UpperLimit<kBlockSize>()),
    LongParam("blockSize", "Block Size", 256, Min(100), LowerLimit<kOrder>()),
    LongParam("padSize", "Padding", 128, Min(0)),
    FloatParam("skew", "Skew", 0, Min(-10), Max(10)),
    FloatParam("threshFwd", "Forward Threshold", 2, Min(0)),
    FloatParam("threshBack", "Backward Threshold", 1.1, Min(0)),
    LongParam("windowSize", "Window Size", 14, Min(0), UpperLimit<kOrder>()),
    LongParam("clumpLength", "Clumping Window Length", 25, Min(0)));

class TransientClient : public FluidBaseClient, public AudioIn, public AudioOut
{
public:
  using ParamDescType = decltype(TransientParams);

  using ParamSetViewType = ParameterSetView<ParamDescType>;
  std::reference_wrapper<ParamSetViewType> mParams;

  void setParams(ParamSetViewType& p) { mParams = p; }

  template <size_t N>
  auto& get() const
  {
    return mParams.get().template get<N>();
  }

  static constexpr auto& getParameterDescriptors() { return TransientParams; }

  TransientClient(ParamSetViewType& p) : mParams(p)
  {
    audioChannelsIn(1);
    audioChannelsOut(2);
    setInputLabels({"audio input"});
    setOutputLabels({"transient component","residual"});
  }

  template <typename T>
  void process(std::vector<HostVector<T>>& input,
               std::vector<HostVector<T>>& output, FluidContext& c)
  {
    if (!input[0].data() || (!output[0].data() && !output[1].data())) return;

    index order = get<kOrder>();
    index blockSize = get<kBlockSize>();
    index padding = get<kPadding>();
    index hostVecSize = input[0].size();
    index maxWinIn = 2 * blockSize + padding;
    index maxWinOut = blockSize - order;

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
    index  halfWindow = static_cast<index>(round(get<kWinSize>() / 2));
    index  debounce = get<kDebounce>();

    mExtractor.setDetectionParameters(skew, threshFwd, thresBack, halfWindow,
                                      debounce);

    RealMatrix in(1, hostVecSize);

    in.row(0) <<= input[0]; // need to convert float->double in some hosts
    mBufferedProcess.push(RealMatrixView(in));

    mBufferedProcess.process(
        mExtractor.inputSize(), mExtractor.hopSize(), mExtractor.hopSize(), c,
        [this](RealMatrixView in, RealMatrixView out) {
          mExtractor.process(in.row(0), out.row(0), out.row(1));
        });

    RealMatrix out(2, hostVecSize);
    mBufferedProcess.pull(RealMatrixView(out));

    if (output[0].data()) output[0] <<= out.row(0);
    if (output[1].data()) output[1] <<= out.row(1);
  }

  index latency()
  {
    return get<kPadding>() + get<kBlockSize>() - get<kOrder>();
  }

  void reset() { mBufferedProcess.reset(); }

private:
  ParameterTrackChanges<index, index, index, index> mTrackValues;
  algorithm::TransientExtraction                    mExtractor;
  BufferedProcess                                   mBufferedProcess;
};
} // namespace transient

using RTTransientClient = ClientWrapper<transient::TransientClient>;

auto constexpr NRTTransientParams = makeNRTParams<transient::TransientClient>(
    InputBufferParam("source", "Source Buffer"),
    BufferParam("transients", "Transients Buffer"),
    BufferParam("residual", "Residual Buffer"));

using NRTTransientsClient =
    NRTStreamAdaptor<transient::TransientClient, decltype(NRTTransientParams),
                     NRTTransientParams, 1, 2>;

using NRTThreadedTransientsClient = NRTThreadingAdaptor<NRTTransientsClient>;

} // namespace client
} // namespace fluid
