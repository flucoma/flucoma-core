#pragma once

#include "DataSetClient.hpp"
#include "DataSetErrorStrings.hpp"
#include "algorithms/MDS.hpp"
#include "data/FluidDataSet.hpp"

#include <clients/common/FluidBaseClient.hpp>
#include <clients/common/FluidNRTClientWrapper.hpp>
#include <clients/common/MessageSet.hpp>
#include <clients/common/OfflineClient.hpp>
#include <clients/common/ParameterSet.hpp>
#include <clients/common/ParameterTypes.hpp>
#include <clients/common/Result.hpp>
#include <data/FluidFile.hpp>
#include <data/FluidIndex.hpp>
#include <data/FluidTensor.hpp>
#include <data/TensorTypes.hpp>
#include <nlohmann/json.hpp>
#include <string>

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
        return {Result::Status::kError, "k should be at least 1"};
      if (dist < 0 || dist > 6)
        return {Result::Status::kError, "dist should be  between 0 and 6"};
      if (src.size() == 0)
        return {Result::Status::kError, EmptyDataSetError};
      FluidTensor<string, 1> ids{src.getIds()};
      FluidTensor<double, 2> output(src.size(), k);
      mAlgorithm.process(src.getData(), output, dist, k);
      FluidDataSet<string, double, 1> result(ids, output);
      destPtr->setDataSet(result);
    } else {
      return {Result::Status::kError, "DataSet doesn't exist"};
    }
    return {};
  }

  FLUID_DECLARE_MESSAGES(makeMessage("fitTransform", &MDSClient::fitTransform));

private:
  MessageResult<void> mOKResult{Result::Status::kOk};
  algorithm::MDS mAlgorithm;
};

using NRTThreadedMDSClient = NRTThreadingAdaptor<ClientWrapper<MDSClient>>;

} // namespace client
} // namespace fluid
