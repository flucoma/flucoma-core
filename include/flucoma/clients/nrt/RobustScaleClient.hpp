/*
Part of the Fluid Corpus Manipulation Project (http://www.flucoma.org/)
Copyright University of Huddersfield.
Licensed under the BSD-3 License.
See license.md file in the project root for full license information.
This project has received funding from the European Research Council (ERC)
under the European Union’s Horizon 2020 research and innovation programme
(grant agreement No 725899).
*/

// modified version of NormalizeClient.hpp code
#pragma once

#include "DataSetClient.hpp"
#include "NRTClient.hpp"
#include "../../algorithms/public/RobustScaling.hpp"

namespace fluid {
namespace client {
namespace robustscale {

constexpr auto RobustScaleParams = defineParameters(
    StringParam<Fixed<true>>("name", "Name"),
    FloatParam("low", "Low Percentile", 25, Min(0), Max(100)),
    FloatParam("high", "High Percentile", 75, Min(0), Max(100)));

class RobustScaleClient : public FluidBaseClient,
                          OfflineIn,
                          OfflineOut,
                          ModelObject,
                          public DataClient<algorithm::RobustScaling>
{
  enum { kName, kLow, kHigh };

public:
  using string = std::string;
  using BufferPtr = std::shared_ptr<BufferAdaptor>;
  using InputBufferPtr = std::shared_ptr<const BufferAdaptor>;
  using StringVector = FluidTensor<string, 1>;

  using ParamDescType = decltype(RobustScaleParams);

  using ParamSetViewType = ParameterSetView<ParamDescType>;
  std::reference_wrapper<ParamSetViewType> mParams;

  void setParams(ParamSetViewType& p) { mParams = p; }

  template <size_t N>
  auto& get() const
  {
    return mParams.get().template get<N>();
  }

  static constexpr auto& getParameterDescriptors() { return RobustScaleParams; }

  RobustScaleClient(ParamSetViewType& p, FluidContext&) : mParams(p) {}

  template <typename T>
  Result process(FluidContext&)
  {
      return{};
  }

  MessageResult<void> fit(InputDataSetClientRef datasetClient)
  {
    auto weakPtr = datasetClient.get();
    if (auto datasetClientPtr = weakPtr.lock())
    {
      auto dataset = datasetClientPtr->getDataSet();
      if (dataset.size() == 0) return Error(EmptyDataSet);
      mAlgorithm.init(get<kLow>(), get<kHigh>(), dataset.getData());
    }
    else
    {
      return Error(NoDataSet);
    }
    return {};
  }
  MessageResult<void> transform(InputDataSetClientRef sourceClient,
                                DataSetClientRef destClient)
  {
    return _transform(sourceClient, destClient, false);
  }

  MessageResult<void> fitTransform(InputDataSetClientRef sourceClient,
                                   DataSetClientRef destClient)
  {
    auto result = fit(sourceClient);
    if (!result.ok()) return result;
    result = _transform(sourceClient, destClient, false);
    return result;
  }

  MessageResult<void> transformPoint(InputBufferPtr in, BufferPtr out)
  {
    return _transformPoint(in, out, false);
  }
  
  MessageResult<void> inverseTransform(InputDataSetClientRef sourceClient,
                                DataSetClientRef destClient)
  {
    return _transform(sourceClient, destClient, true);
  }
  
  MessageResult<void> inverseTransformPoint(InputBufferPtr in, BufferPtr out)
  {
    return _transformPoint(in, out, true);
  }

  static auto getMessageDescriptors()
  {
    return defineMessages(
        makeMessage("fit", &RobustScaleClient::fit),
        makeMessage("fitTransform", &RobustScaleClient::fitTransform),
        makeMessage("transform", &RobustScaleClient::transform),
        makeMessage("transformPoint", &RobustScaleClient::transformPoint),
        makeMessage("inverseTransform", &RobustScaleClient::inverseTransform),
        makeMessage("inverseTransformPoint",
                    &RobustScaleClient::inverseTransformPoint),
        makeMessage("cols", &RobustScaleClient::dims),
        makeMessage("clear", &RobustScaleClient::clear),
        makeMessage("size", &RobustScaleClient::size),
        makeMessage("load", &RobustScaleClient::load),
        makeMessage("dump", &RobustScaleClient::dump),
        makeMessage("read", &RobustScaleClient::read),
        makeMessage("write", &RobustScaleClient::write));
  }

private:
  MessageResult<void> _transform(InputDataSetClientRef sourceClient,
                                 DataSetClientRef destClient, bool invert)
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
  
  MessageResult<void> _transformPoint(InputBufferPtr in, BufferPtr out, bool invert)
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

using RobustScaleRef = SharedClientRef<const RobustScaleClient>;

constexpr auto RobustScaleQueryParams = defineParameters(
    RobustScaleRef::makeParam("model", "Source Model"),
    EnumParam("invert", "Inverse Transform", 0, "False", "True"),
    InputBufferParam("inputPointBuffer", "Input Point Buffer"),
    BufferParam("predictionBuffer", "Prediction Buffer"));

class RobustScaleQuery : public FluidBaseClient, ControlIn, ControlOut
{
  enum { kModel, kInvert, kInputBuffer, kOutputBuffer };

public:
  using string = std::string;
  using BufferPtr = std::shared_ptr<BufferAdaptor>;
  using StringVector = FluidTensor<string, 1>;

  using ParamDescType = decltype(RobustScaleQueryParams);

  using ParamSetViewType = ParameterSetView<ParamDescType>;
  std::reference_wrapper<ParamSetViewType> mParams;

  void setParams(ParamSetViewType& p) { mParams = p; }

  template <size_t N>
  auto& get() const
  {
    return mParams.get().template get<N>();
  }

  static constexpr auto& getParameterDescriptors()
  {
    return RobustScaleQueryParams;
  }

  RobustScaleQuery(ParamSetViewType& p, FluidContext&) : mParams(p)
  {
    controlChannelsIn(1);
    controlChannelsOut({1, 1});
  }

  template <typename T>
  void process(std::vector<FluidTensorView<T, 1>>& input,
               std::vector<FluidTensorView<T, 1>>& output, FluidContext&)
  {
    output[0](0) = 0;
    if (input[0](0) > 0)
    {
      auto robustPtr = get<kModel>().get().lock();
      if (!robustPtr)
      {
        // report error?
        return;
      }
      algorithm::RobustScaling const& algorithm = robustPtr->algorithm();
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
      output[0](0) = 1;
    }
  }

  index latency() const { return 0; }
};

} // namespace robustscale

using NRTThreadedRobustScaleClient =
    NRTThreadingAdaptor<typename robustscale::RobustScaleRef::SharedType>;

using RTRobustScaleQueryClient = ClientWrapper<robustscale::RobustScaleQuery>;

} // namespace client
} // namespace fluid
