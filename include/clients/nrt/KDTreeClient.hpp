#pragma once

#include "NRTClient.hpp"
#include "algorithms/KDTree.hpp"
#include <string>

namespace fluid {
namespace client {

class KDTreeClient : public FluidBaseClient, OfflineIn, OfflineOut {


public:
  using string = std::string;
  using BufferPtr = std::shared_ptr<BufferAdaptor>;


  template <typename T> Result process(FluidContext &) { return {}; }

  FLUID_DECLARE_PARAMS();

  KDTreeClient(ParamSetViewType &p) : mParams(p) {}

  MessageResult<void> fit(DataSetClientRef datasetClient) {

    auto datasetClientPtr = datasetClient.get().lock();
    if(!datasetClientPtr) return NoDataSetError;
    auto dataset = datasetClientPtr->getDataSet();
    if (dataset.size() == 0) return EmptyDataSetError;

    mTree = algorithm::KDTree(dataset);
    return OKResult;
  }

  MessageResult<fluid::index> cols(){return mTree.nDims();}

  MessageResult<FluidTensor<std::string, 1>> kNearest(BufferPtr data, int k) const {

    if (!data) return NoBufferError;
    BufferAdaptor::Access buf(data.get());
    if (buf.numFrames() != mTree.nDims()) return WrongPointSizeError;
    if (k > mTree.nPoints()) return SmallDataSetError;
    if(k <= 0 ) return SmallKError;

    FluidTensor<double, 1> point(mTree.nDims());
    point = buf.samps(0, mTree.nDims(), 0);
    FluidDataSet<std::string, double,1> nearest = mTree.kNearest(point, k);
    FluidTensor<std::string, 1> result{nearest.getIds()};
    return result;
  }

  MessageResult<FluidTensor<double, 1>> kNearestDist(BufferPtr data, int k) const {
    // TODO: refactor with kNearest
    if (!data)
      return NoBufferError;
    BufferAdaptor::Access buf(data.get());
    if (buf.numFrames() != mTree.nDims())
      return WrongPointSizeError;
    if (k > mTree.nPoints()){
      return SmallDataSetError;
    }
    if(k <= 0 ){
      return SmallKError;
    }
    FluidTensor<double, 1> point(mTree.nDims());
    point = buf.samps(0, mTree.nDims(), 0);
    FluidDataSet<std::string, double,1> nearest = mTree.kNearest(point, k);
    FluidTensor<double, 1> result{nearest.getData().col(0)};
    return result;
  }

  MessageResult<void> read(string fileName) {
    auto file = FluidFile(fileName, "r");
    if(!file.valid()){return {Result::Status::kError, file.error()};}
    if(!file.read()){return ReadError;}
    if(!file.checkKeys({"tree", "data", "rows", "cols", "ids"})){
      return {Result::Status::kError, file.error()};
    }
    fluid::index rows, cols;
    file.get("cols", cols);
    file.get("rows", rows);
    algorithm::KDTree::FlatData treeData(rows, cols);
    file.get("tree", treeData.tree, rows, 2);
    file.get("data", treeData.data, rows, cols);
    file.get("ids", treeData.ids, rows);
    mTree.fromFlat(treeData);
    return OKResult;
  }

  MessageResult<void> write(string fileName){
    auto file = FluidFile(fileName, "w");
    if(!file.valid()){return {Result::Status::kError, file.error()};}
    algorithm::KDTree::FlatData treeData = mTree.toFlat();
    file.add("tree", treeData.tree);
    file.add("cols", treeData.data.cols());
    file.add("rows", treeData.data.rows());
    file.add("data", treeData.data);
    file.add("ids", treeData.ids);
    return file.write()? OKResult:WriteError;
  }

  FLUID_DECLARE_MESSAGES(makeMessage("fit", &KDTreeClient::fit),
                         makeMessage("kNearest", &KDTreeClient::kNearest),
                         makeMessage("kNearestDist", &KDTreeClient::kNearestDist),
                         makeMessage("cols", &KDTreeClient::cols),
                         makeMessage("write", &KDTreeClient::write),
                         makeMessage("read", &KDTreeClient::read)
  );

private:
  algorithm::KDTree mTree{1};
};

using NRTThreadedKDTreeClient =
    NRTThreadingAdaptor<ClientWrapper<KDTreeClient>>;

} // namespace client
} // namespace fluid
