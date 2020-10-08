#pragma once

#include "DataSetClient.hpp"
#include "NRTClient.hpp"
#include "algorithms/UMAP.hpp"

namespace fluid {
namespace client {

class UMAPClient : public FluidBaseClient, OfflineIn, OfflineOut, ModelObject {

public:
  using string = std::string;
  using BufferPtr = std::shared_ptr<BufferAdaptor>;
  using StringVector = FluidTensor<string, 1>;

  template <typename T> Result process(FluidContext &) { return {}; }

  enum { kNumDimensions, kNumNeighbors, kMinDistance, kNumIter, kLearningRate, kBatchSize };

  FLUID_DECLARE_PARAMS(
      LongParam("numDimensions", "Target Number of Dimensions", 2, Min(1)),
      LongParam("numNeighbours", "Number of Nearest Neighbours", 15, Min(1)),
      FloatParam("minDist", "Minimum Distance", 0.1, Min(0)),
      LongParam("iterations", "Number of Iterations", 200, Min(1)),
      FloatParam("learnRate", "Learning Rate", 0.1, Min(0.0), Max(1.0)),
      LongParam("batchSize", "Batch Size", 50, Min(1))
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
    if(get<kBatchSize>() > src.size())
      return Error("Batch size is larger than dataset");
      if(get<kNumNeighbors>() > src.size())
        return Error("Number of Neighbours is larger than dataset");

    StringVector ids{src.getIds()};
    RealMatrix output(src.size(), k);
    FluidDataSet<string, double, 1> result;

    result =
        mAlgorithm.process(src, get<kNumNeighbors>(), k, get<kMinDistance>(),
                           get<kNumIter>(), get<kLearningRate>(), get<kBatchSize>());
    destPtr->setDataSet(result);
    return OK();
  }

  FLUID_DECLARE_MESSAGES(makeMessage("fitTransform",
                                     &UMAPClient::fitTransform));

private:
  algorithm::UMAP mAlgorithm;
};

using NRTThreadedUMAPClient = NRTThreadingAdaptor<ClientWrapper<UMAPClient>>;

} // namespace client
} // namespace fluid
