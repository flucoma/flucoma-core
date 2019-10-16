#pragma once

#include "DataSetClient.hpp"
#include "DataSetErrorStrings.hpp"
#include "algorithms/KDTree.hpp"
#include "data/FluidDataSet.hpp"

#include <clients/common/FluidBaseClient.hpp>
#include <clients/common/MessageSet.hpp>
#include <clients/common/OfflineClient.hpp>
#include <clients/common/ParameterSet.hpp>
#include <clients/common/ParameterTypes.hpp>
#include <clients/common/Result.hpp>
#include <clients/nrt/FluidNRTClientWrapper.hpp>
#include <data/FluidTensor.hpp>
#include <data/TensorTypes.hpp>
#include <string>

namespace fluid {
namespace client {

class KDTreeClient : public FluidBaseClient, OfflineIn, OfflineOut {
  enum { kNDims };

public:
  using string = std::string;
  using BufferPtr = std::shared_ptr<BufferAdaptor>;


  template <typename T> Result process(FluidContext &) { return {}; }

  FLUID_DECLARE_PARAMS(LongParam<Fixed<true>>("nDims", "Dimension size", 1,
                                              Min(1)));

  KDTreeClient(ParamSetViewType &p) : mParams(p), mDataSet(get<kNDims>()) {
    mDims = get<kNDims>();
  }

  MessageResult<void> index(DataSetClientRef datasetClient) {
    auto weakPtr = datasetClient.get();
    if (auto datasetClientPtr = weakPtr.lock()) {
      auto dataset = datasetClientPtr->getDataSet();
      if (dataset.size() == 0) return {Result::Status::kError, EmptyDataSetError};
      mTree = algorithm::KDTree(dataset);
    } else {
      return {Result::Status::kError, "DataSet doesn't exist"};
    }
    return {Result::Status::kOk};
  }

  MessageResult<FluidTensor<std::string, 1>> knn(BufferPtr data, int k) const {
    if (!data)
      return {Result::Status::kError, NoBufferError};
    BufferAdaptor::Access buf(data.get());
    if (buf.numFrames() != mDims)
      return {Result::Status::kError, WrongPointSizeError};
    if (k > mTree.nPoints()){
      return {Result::Status::kError, SmallDataSetError};
    }
    if(k <= 0 ){
      return {Result::Status::kError, "k should be at least 1"};
    }
    FluidTensor<double, 1> point(mDims);
    point = buf.samps(0, mDims, 0);
    FluidDataSet<std::string, double,1> nearest =
        mTree.kNearest(point, k);
    FluidTensor<std::string, 1> result{nearest.getIds()};
    return result;
  }

  MessageResult<void> read(string fileName) {
    auto file = FluidFile(fileName, "r");
    if(!file.valid()){return {Result::Status::kError, file.error()};}
    if(!file.read()){return {Result::Status::kError, ReadError};}
    if(!file.checkKeys({"tree","data","rows","cols"})){
      return {Result::Status::kError, file.error()};
    }
    size_t rows, cols;
    file.get("cols", cols);
    file.get("rows", rows);
    algorithm::KDTree::FlatData treeData(rows, cols);
    file.get("tree", treeData.tree, rows, 2);
    file.get("data", treeData.data, rows, cols);
    mTree.fromFlat(treeData);
    mTree.print();
    return mOKResult;
  }

  MessageResult<void> write(string fileName){
    auto file = FluidFile(fileName, "w");
    if(!file.valid()){return {Result::Status::kError, file.error()};}
    mTree.print();
    algorithm::KDTree::FlatData treeData = mTree.toFlat();
    file.add("tree", treeData.tree);
    return file.write()? mOKResult:mWriteError;
  }

  FLUID_DECLARE_MESSAGES(makeMessage("index", &KDTreeClient::index),
                         makeMessage("knn", &KDTreeClient::knn),
                         makeMessage("write", &KDTreeClient::write),
                         makeMessage("read", &KDTreeClient::read)
  );

private:
  MessageResult<void> mOKResult{Result::Status::kOk};
  MessageResult<void> mWriteError{Result::Status::kError, WriteError};
  mutable FluidDataSet<string, double, 1> mDataSet;
  mutable algorithm::KDTree mTree{1};
  size_t mDims;
};

using NRTThreadedKDTreeClient =
    NRTThreadingAdaptor<ClientWrapper<KDTreeClient>>;

} // namespace client
} // namespace fluid
