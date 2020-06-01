#pragma once

#include "NRTClient.hpp"
#include "algorithms/Normalization.hpp"

namespace fluid {
namespace client {

class NormalizeClient : public FluidBaseClient, OfflineIn, OfflineOut, ModelObject  {
  enum {kMin, kMax};

public:
  using string = std::string;
  using BufferPtr = std::shared_ptr<BufferAdaptor>;
  using StringVector = FluidTensor<string, 1>;

  template <typename T> Result process(FluidContext &) { return {}; }

  FLUID_DECLARE_PARAMS(FloatParam("min", "Minimum value", 0.0),
                       FloatParam("max", "Maximum value", 1.0, LowerLimit<kMin>()));

  NormalizeClient(ParamSetViewType &p)
      : mParams(p), mDataClient(mAlgorithm) {
  }

  MessageResult<void> fit(DataSetClientRef datasetClient) {
    auto weakPtr = datasetClient.get();
    if (auto datasetClientPtr = weakPtr.lock()) {
      auto dataset = datasetClientPtr->getDataSet();
      if (dataset.size() == 0)
        return Error(EmptyDataSet);
      mDims = dataset.pointSize();
      mAlgorithm.init(get<kMin>(), get<kMax>(), dataset.getData());
    } else {
      return Error(NoDataSet);
    }
    return {};
  }
  MessageResult<void> transform(DataSetClientRef sourceClient,
                            DataSetClientRef destClient) {
    using namespace std;
    auto srcPtr = sourceClient.get().lock();
    auto destPtr = destClient.get().lock();
    if (srcPtr && destPtr) {
      auto srcDataSet = srcPtr->getDataSet();
      if (srcDataSet.size() == 0)
        return Error(EmptyDataSet);
      StringVector ids{srcDataSet.getIds()};
      RealMatrix data(srcDataSet.size(), srcDataSet.pointSize());
      if(!mAlgorithm.initialized()) return Error(NoDataFitted);
      mAlgorithm.setMin(get<kMin>());
      mAlgorithm.setMax(get<kMax>());
      mAlgorithm.process(srcDataSet.getData(), data);
      FluidDataSet<string, double, 1> result(ids, data);
      destPtr->setDataSet(result);
    } else {
      return Error(NoDataSet);
    }
    return {};
  }

  MessageResult<void> fitTransform(DataSetClientRef sourceClient,
                                   DataSetClientRef destClient) {
            auto result = fit(sourceClient);
            if (!result.ok()) return result;
            result = transform(sourceClient, destClient);
            return result;
  }

  MessageResult<void> transformPoint(BufferPtr in, BufferPtr out) {
    if (!in || !out) return Error(NoBuffer);
    BufferAdaptor::Access inBuf(in.get());
    BufferAdaptor::Access outBuf(out.get());
    if(!inBuf.exists()) return Error(InvalidBuffer);
    if(!outBuf.exists()) return Error(InvalidBuffer);
    if (inBuf.numFrames() != mDims) return Error(WrongPointSize);
    if(!mAlgorithm.initialized()) return Error(NoDataFitted);
    Result resizeResult = outBuf.resize(mDims, 1, inBuf.sampleRate());
    if(!resizeResult.ok()) return Error(BufferAlloc);
    RealVector src(mDims);
    RealVector dest(mDims);
    src = inBuf.samps(0, mDims, 0);
    mAlgorithm.setMin(get<kMin>());
    mAlgorithm.setMax(get<kMax>());
    mAlgorithm.processFrame(src, dest);
    outBuf.samps(0) = dest;
    return {};
  }

  MessageResult<index> size() { return mDataClient.size(); }
  MessageResult<index> cols() { return mDataClient.dims(); }
  MessageResult<void> write(string fn) {return mDataClient.write(fn);}
  MessageResult<void> read(string fn) {return mDataClient.read(fn);}
  MessageResult<string> dump() { return mDataClient.dump();}
  MessageResult<void> load(string  s) { return mDataClient.load(s);}

  FLUID_DECLARE_MESSAGES(makeMessage("fit", &NormalizeClient::fit),
                         makeMessage("fitTransform", &NormalizeClient::fitTransform),
                         makeMessage("transform", &NormalizeClient::transform),
                         makeMessage("transformPoint", &NormalizeClient::transformPoint),
                         makeMessage("cols", &NormalizeClient::cols),
                         makeMessage("size", &NormalizeClient::size),
                         makeMessage("load", &NormalizeClient::load),
                         makeMessage("dump", &NormalizeClient::dump),
                         makeMessage("read", &NormalizeClient::read),
                         makeMessage("write", &NormalizeClient::write));

private:
  algorithm::Normalization mAlgorithm;
  DataClient<algorithm::Normalization> mDataClient;
  index mDims{0};
};

using NRTThreadedNormalizeClient =
    NRTThreadingAdaptor<ClientWrapper<NormalizeClient>>;

} // namespace client
} // namespace fluid
