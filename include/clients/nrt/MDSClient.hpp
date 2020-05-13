#pragma once

#include "NRTClient.hpp"
#include "algorithms/MDS.hpp"

namespace fluid {
namespace client {

class MDSClient : public FluidBaseClient, OfflineIn, OfflineOut {

public:
  using string = std::string;
  using BufferPtr = std::shared_ptr<BufferAdaptor>;

  template <typename T> Result process(FluidContext &) { return {}; }

  FLUID_DECLARE_PARAMS();

  MDSClient(ParamSetViewType &p) : mParams(p) {}

  MessageResult<void> fitTransform(DataSetClientRef sourceClient, index k,
                                   index dist, DataSetClientRef destClient) {
    auto srcPtr = sourceClient.get().lock();
    auto destPtr = destClient.get().lock();
    if (srcPtr && destPtr) {
      auto src = srcPtr->getDataSet();
      auto dest = destPtr->getDataSet();
      if (k <= 0)
        return SmallKError;
      if (dist < 0 || dist > 6)
        return {Result::Status::kError, "dist should be  between 0 and 6"};
      if (src.size() == 0)
        return EmptyDataSetError;
      FluidTensor<string, 1> ids{src.getIds()};
      FluidTensor<double, 2> output(src.size(), k);
      mAlgorithm.process(src.getData(), output, dist, k);
      FluidDataSet<string, double, 1> result(ids, output);
      destPtr->setDataSet(result);
    } else {
      return NoDataSetError;
    }
    return {};
  }

  FLUID_DECLARE_MESSAGES(makeMessage("fitTransform", &MDSClient::fitTransform));

private:
  algorithm::MDS mAlgorithm;
};

using NRTThreadedMDSClient = NRTThreadingAdaptor<ClientWrapper<MDSClient>>;

} // namespace client
} // namespace fluid
