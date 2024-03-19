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

#include "DataSetClient.hpp"
#include "NRTClient.hpp"
#include "../../algorithms/public/Standardization.hpp"

namespace fluid {
namespace client {
namespace standardize {

constexpr auto StandardizeParams =
    defineParameters(StringParam<Fixed<true>>("name", "Name"));

class StandardizeClient : public FluidBaseClient,
                          OfflineIn,
                          OfflineOut,
                          ModelObject,
                          public DataClient<algorithm::Standardization>
{
  enum { kName };

public:
  using string = std::string;
  using BufferPtr = std::shared_ptr<BufferAdaptor>;
  using InputBufferPtr = std::shared_ptr<const BufferAdaptor>;
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

  StandardizeClient(ParamSetViewType& p, FluidContext&) : mParams(p) {}

  MessageResult<void> fit(InputDataSetClientRef datasetClient)
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

  MessageResult<void> transform(InputDataSetClientRef sourceClient,
                                DataSetClientRef destClient) const
  {
    return _transform(sourceClient, destClient, false);
  }

  MessageResult<void> transformPoint(InputBufferPtr in, BufferPtr out) const
  {
     return _transformPoint(in,out,false);
  }

  MessageResult<void> inverseTransform(InputDataSetClientRef sourceClient,
                                DataSetClientRef destClient) const
  {
    return _transform(sourceClient, destClient, true);
  }

  MessageResult<void> inverseTransformPoint(InputBufferPtr in, BufferPtr out) const
  {
    return _transformPoint(in,out,true);
  }

  MessageResult<void> fitTransform(InputDataSetClientRef sourceClient,
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
        makeMessage("inverseTransform", &StandardizeClient::inverseTransform),
        makeMessage("inverseTransformPoint",
                    &StandardizeClient::inverseTransformPoint),
        makeMessage("cols", &StandardizeClient::dims),
        makeMessage("clear", &StandardizeClient::clear),
        makeMessage("size", &StandardizeClient::size),
        makeMessage("load", &StandardizeClient::load),
        makeMessage("dump", &StandardizeClient::dump),
        makeMessage("read", &StandardizeClient::read),
        makeMessage("write", &StandardizeClient::write));
  }

private:
  MessageResult<void> _transform(InputDataSetClientRef sourceClient,
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
  
  MessageResult<void> _transformPoint(InputBufferPtr in, BufferPtr out, bool invert) const
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
    src <<= BufferAdaptor::ReadAccess(in.get()).samps(0, mAlgorithm.dims(), 0);
    mAlgorithm.processFrame(src, dest, invert);
    outBuf.samps(0, mAlgorithm.dims(), 0) <<= dest;
    return OK();
  }
  
};

using StandardizeRef = SharedClientRef<const StandardizeClient>;

constexpr auto StandardizeQueryParams = defineParameters(
    StandardizeRef::makeParam("model", "Source Model"),
    EnumParam("invert", "Inverse Transform", 0, "False", "True"),
    InputBufferParam("inputPointBuffer", "Input Point Buffer"),
    BufferParam("predictionBuffer", "Prediction Buffer"));

class StandardizeQuery : public FluidBaseClient, ControlIn, ControlOut
{
  enum { kModel, kInvert, kInputBuffer, kOutputBuffer };

public:
  using ParamDescType = decltype(StandardizeQueryParams);

  using ParamSetViewType = ParameterSetView<ParamDescType>;
  std::reference_wrapper<ParamSetViewType> mParams;

  void setParams(ParamSetViewType& p) { mParams = p; }

  template <size_t N>
  auto& get() const
  {
    return mParams.get().template get<N>();
  }

  static constexpr auto getParameterDescriptors()
  {
    return StandardizeQueryParams;
  }

  StandardizeQuery(ParamSetViewType& p, FluidContext&) : mParams(p)
  {
    controlChannelsIn(1);
    controlChannelsOut({1, 1});
  }

  template <typename T>
  void process(std::vector<FluidTensorView<T, 1>>& input,
               std::vector<FluidTensorView<T, 1>>& output, FluidContext&)
  {
    output[0] <<= input[0];
    if (input[0](0) > 0)
    {
      auto stdPtr = get<kModel>().get().lock();
      if (!stdPtr)
      {
        // report error ?
        return;
      }

      algorithm::Standardization const& algorithm = stdPtr->algorithm();

      if (!algorithm.initialized()) return;
      InOutBuffersCheck bufCheck(algorithm.dims());
      if (!bufCheck.checkInputs(get<kInputBuffer>().get(),
                                get<kOutputBuffer>().get()))
        return;
      auto outBuf = BufferAdaptor::Access(get<kOutputBuffer>().get());
      if (outBuf.samps(0).size() < algorithm.dims()) return;
      RealVector src(algorithm.dims());
      RealVector dest(algorithm.dims());
      src <<= BufferAdaptor::ReadAccess(get<kInputBuffer>().get())
                .samps(0, algorithm.dims(), 0);
      algorithm.processFrame(src, dest, get<kInvert>() == 1);
      outBuf.samps(0, algorithm.dims(), 0) <<= dest;
    }
  }

  index latency() const { return 0; }
};


} // namespace standardize

using NRTThreadedStandardizeClient =
    NRTThreadingAdaptor<typename standardize::StandardizeRef::SharedType>;

using RTStandardizeQueryClient = ClientWrapper<standardize::StandardizeQuery>;

} // namespace client
} // namespace fluid
