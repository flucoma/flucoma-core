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

#include "DataSetClient.hpp"
#include "NRTClient.hpp"
#include "../../algorithms/public/Standardization.hpp"

namespace fluid {
namespace client {
namespace standardize {

enum { kInvert, kInputBuffer, kOutputBuffer };

constexpr auto StandardizeParams = defineParameters(
    EnumParam("invert", "Inverse Transform", 0, "False", "True"),
    BufferParam("inputPointBuffer", "Input Point Buffer"),
    BufferParam("predictionBuffer", "Prediction Buffer"));

class StandardizeClient : public FluidBaseClient,
                          AudioIn,
                          ControlOut,
                          ModelObject,
                          public DataClient<algorithm::Standardization>
{
public:
  using string = std::string;
  using BufferPtr = std::shared_ptr<BufferAdaptor>;
  using StringVector = FluidTensor<string, 1>;

  template <typename T>
  Result process(FluidContext&)
  {
    return {};
  }

  using ParamDescType = decltype(StandardizeParams);

  using ParamSetViewType = ParameterSetView<ParamDescType>;
  std::reference_wrapper<ParamSetViewType> mParams;

  void setParams(ParamSetViewType& p) { mParams = p; }

  template <size_t N>
  auto& get() const
  {
    return mParams.get().template get<N>();
  }

  static constexpr auto getParameterDescriptors() { return StandardizeParams; }

  StandardizeClient(ParamSetViewType& p) : mParams(p)
  {
    audioChannelsIn(1);
    controlChannelsOut({1,1});
  }

  template <typename T>
  void process(std::vector<FluidTensorView<T, 1>>& input,
               std::vector<FluidTensorView<T, 1>>& output, FluidContext&)
  {
    if (!mAlgorithm.initialized()) return;
    InOutBuffersCheck bufCheck(mAlgorithm.dims());
    if (!bufCheck.checkInputs(get<kInputBuffer>().get(),
                              get<kOutputBuffer>().get()))
      return;
    auto outBuf = BufferAdaptor::Access(get<kOutputBuffer>().get());
    if (outBuf.samps(0).size() < mAlgorithm.dims()) return;
    RealVector src(mAlgorithm.dims());
    RealVector dest(mAlgorithm.dims());
    src = BufferAdaptor::ReadAccess(get<kInputBuffer>().get())
              .samps(0, mAlgorithm.dims(), 0);
    mTrigger.process(input, output, [&]() {
      mAlgorithm.processFrame(src, dest, get<kInvert>() == 1);
      outBuf.samps(0, mAlgorithm.dims(), 0) = dest;
    });
  }

  index latency() { return 0; }

  MessageResult<void> fit(DataSetClientRef datasetClient)
  {
    auto weakPtr = datasetClient.get();
    if (auto datasetClientPtr = weakPtr.lock())
    {
      auto dataset = datasetClientPtr->getDataSet();
      if (dataset.size() == 0) return Error(EmptyDataSet);
      mAlgorithm.init(dataset.getData());
    }
    else
    {
      return Error(NoDataSet);
    }
    return {};
  }

  MessageResult<void> transform(DataSetClientRef sourceClient,
                                DataSetClientRef destClient) const
  {
    return _transform(sourceClient, destClient, get<kInvert>() == 1);
  }

  MessageResult<void> transformPoint(BufferPtr in, BufferPtr out) const
  {
    if (!mAlgorithm.initialized()) return Error(NoDataFitted);
    InOutBuffersCheck bufCheck(mAlgorithm.dims());
    if (!bufCheck.checkInputs(in.get(), out.get()))
      return Error(bufCheck.error());
    BufferAdaptor::Access outBuf(out.get());
    Result                resizeResult =
        outBuf.resize(mAlgorithm.dims(), 1, outBuf.sampleRate());
    if (!resizeResult.ok()) return Error(BufferAlloc);
    RealVector src(mAlgorithm.dims());
    RealVector dest(mAlgorithm.dims());
    src = BufferAdaptor::ReadAccess(in.get()).samps(0, mAlgorithm.dims(), 0);
    mAlgorithm.processFrame(src, dest, get<kInvert>() == 1);
    outBuf.samps(0, mAlgorithm.dims(), 0) = dest;
    return OK();
  }

  MessageResult<void> fitTransform(DataSetClientRef sourceClient,
                                   DataSetClientRef destClient)
  {
    auto result = fit(sourceClient);
    if (!result.ok()) return result;
    result = _transform(sourceClient, destClient, false);
    return result;
  }

  static auto getMessageDescriptors()
  {
    return defineMessages(
        makeMessage("fit", &StandardizeClient::fit),
        makeMessage("fitTransform", &StandardizeClient::fitTransform),
        makeMessage("transform", &StandardizeClient::transform),
        makeMessage("transformPoint", &StandardizeClient::transformPoint),
        makeMessage("cols", &StandardizeClient::dims),
        makeMessage("clear", &StandardizeClient::clear),
        makeMessage("size", &StandardizeClient::size),
        makeMessage("load", &StandardizeClient::load),
        makeMessage("dump", &StandardizeClient::dump),
        makeMessage("read", &StandardizeClient::read),
        makeMessage("write", &StandardizeClient::write));
  }

private:
  MessageResult<void> _transform(DataSetClientRef sourceClient,
                                 DataSetClientRef destClient, bool invert) const
  {
    using namespace std;
    auto srcPtr = sourceClient.get().lock();
    auto destPtr = destClient.get().lock();
    if (srcPtr && destPtr)
    {
      auto srcDataSet = srcPtr->getDataSet();
      if (srcDataSet.size() == 0) return Error(EmptyDataSet);
      StringVector ids{srcDataSet.getIds()};
      RealMatrix   data(srcDataSet.size(), srcDataSet.pointSize());
      if (!mAlgorithm.initialized()) return Error(NoDataFitted);
      mAlgorithm.process(srcDataSet.getData(), data, invert);
      FluidDataSet<string, double, 1> result(ids, data);
      destPtr->setDataSet(result);
    }
    else
    {
      return Error(NoDataSet);
    }
    return OK();
  }

  FluidInputTrigger mTrigger;
};
} // namespace standardize

using RTStandardizeClient = ClientWrapper<standardize::StandardizeClient>;

} // namespace client
} // namespace fluid
