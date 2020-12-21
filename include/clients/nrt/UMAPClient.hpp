#pragma once

#include "DataSetClient.hpp"
#include "NRTClient.hpp"
#include "algorithms/UMAP.hpp"

namespace fluid {
namespace client {

class UMAPClient :
  public FluidBaseClient, OfflineIn, OfflineOut, ModelObject,
  public DataClient<algorithm::UMAP>  {

public:
  using string = std::string;
  using BufferPtr = std::shared_ptr<BufferAdaptor>;
  using StringVector = FluidTensor<string, 1>;

  template <typename T> Result process(FluidContext &) { return {}; }

  enum { kNumDimensions, kNumNeighbors, kMinDistance, kNumIter, kLearningRate };

  FLUID_DECLARE_PARAMS(
      LongParam("numDimensions", "Target Number of Dimensions", 2, Min(1)),
      LongParam("numNeighbours", "Number of Nearest Neighbours", 15, Min(1)),
      FloatParam("minDist", "Minimum Distance", 0.1, Min(0)),
      LongParam("iterations", "Number of Iterations", 200, Min(1)),
      FloatParam("learnRate", "Learning Rate", 0.1, Min(0.0), Max(1.0))
    );

  UMAPClient(ParamSetViewType &p) : mParams(p) {}

  MessageResult<void> fitTransform(DataSetClientRef sourceClient,
                                   DataSetClientRef destClient) {
    index k = get<kNumDimensions>();
    auto srcPtr = sourceClient.get().lock();
    auto destPtr = destClient.get().lock();
    if (!srcPtr || !destPtr)
      return Error(NoDataSet);
    auto src = srcPtr->getDataSet();
    auto dest = destPtr->getDataSet();
    if (src.size() == 0)
      return Error(EmptyDataSet);
    if(get<kNumNeighbors>() > src.size())
      return Error("Number of Neighbours is larger than dataset");
    FluidDataSet<string, double, 1> result;
    result =
        mAlgorithm.train(src, get<kNumNeighbors>(), k, get<kMinDistance>(),
                           get<kNumIter>(), get<kLearningRate>());
    destPtr->setDataSet(result);
    return OK();
  }

  MessageResult<void> fit(DataSetClientRef sourceClient) {
    index k = get<kNumDimensions>();
    auto srcPtr = sourceClient.get().lock();
    if (!srcPtr) return Error(NoDataSet);
    auto src = srcPtr->getDataSet();
    if (src.size() == 0)
      return Error(EmptyDataSet);
    if(get<kNumNeighbors>() > src.size())
      return Error("Number of Neighbours is larger than dataset");
    StringVector ids{src.getIds()};
    FluidDataSet<string, double, 1> result;
    result =
        mAlgorithm.train(src, get<kNumNeighbors>(), k, get<kMinDistance>(),
                           get<kNumIter>(), get<kLearningRate>());
    return OK();
  }

  MessageResult<void> transform(DataSetClientRef sourceClient,
                                   DataSetClientRef destClient) {
    auto srcPtr = sourceClient.get().lock();
    auto destPtr = destClient.get().lock();
    if (!srcPtr || !destPtr) return Error(NoDataSet);
    auto src = srcPtr->getDataSet();
    auto dest = destPtr->getDataSet();
    if (src.size() == 0) return Error(EmptyDataSet);
    if(get<kNumNeighbors>() > src.size())
      return Error("Number of Neighbours is larger than dataset");
    if(!mAlgorithm.initialized()) return Error(NoDataFitted);
    StringVector ids{src.getIds()};
    FluidDataSet<string, double, 1> result;
    result = mAlgorithm.transform(src, get<kNumIter>(), get<kLearningRate>());
    destPtr->setDataSet(result);
    return OK();
  }


  FLUID_DECLARE_MESSAGES(
    makeMessage("fitTransform",&UMAPClient::fitTransform),
    makeMessage("fit",&UMAPClient::fit),
    makeMessage("transform",&UMAPClient::transform),
    makeMessage("cols", &UMAPClient::dims),
    makeMessage("clear", &UMAPClient::clear),
    makeMessage("size", &UMAPClient::size),
    makeMessage("load", &UMAPClient::load),
    makeMessage("dump", &UMAPClient::dump),
    makeMessage("write", &UMAPClient::write),
    makeMessage("read", &UMAPClient::read)
  );

};

using NRTThreadedUMAPClient = NRTThreadingAdaptor<ClientWrapper<UMAPClient>>;

} // namespace client
} // namespace fluid
