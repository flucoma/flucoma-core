#pragma once

#include "NRTClient.hpp"
#include "algorithms/KDTree.hpp"
#include <string>

namespace fluid {
namespace client {

class KDTreeClient : public FluidBaseClient, OfflineIn, OfflineOut, ModelObject
{

public:
  using string = std::string;
  using BufferPtr = std::shared_ptr<BufferAdaptor>;
  using StringVector = FluidTensor<string, 1>;

  template <typename T> Result process(FluidContext &) { return {}; }

  FLUID_DECLARE_PARAMS();

  KDTreeClient(ParamSetViewType &p) : mParams(p) {}

  MessageResult<void> fit(DataSetClientRef datasetClient) {
    auto datasetClientPtr = datasetClient.get().lock();
    if(!datasetClientPtr) return Error(NoDataSet);
    auto dataset = datasetClientPtr->getDataSet();
    if (dataset.size() == 0) return Error(EmptyDataSet);
    mTree = algorithm::KDTree(dataset);
    return OK();
  }

  MessageResult<fluid::index> cols(){return mTree.nDims();}

  MessageResult<StringVector> kNearest(BufferPtr data, int k) const {
    if (!data) return Error<StringVector>(NoBuffer);
    BufferAdaptor::Access buf(data.get());
    if(!buf.exists()) return Error<StringVector>(InvalidBuffer);
    if (buf.numFrames() != mTree.nDims()) return Error<StringVector>(WrongPointSize);
    if (k > mTree.nPoints()) return Error<StringVector>(SmallDataSet);
    if(k <= 0 ) return Error<StringVector>(SmallK);

    RealVector point(mTree.nDims());
    point = buf.samps(0, mTree.nDims(), 0);
    FluidDataSet<std::string, double,1> nearest = mTree.kNearest(point, k);
    StringVector result{nearest.getIds()};
    return result;
  }

  MessageResult<RealVector> kNearestDist(BufferPtr data, int k) const {
    // TODO: refactor with kNearest
    if (!data)
      return Error<RealVector>(NoBuffer);
    BufferAdaptor::Access buf(data.get());
    if(!buf.exists()) return Error<RealVector>(InvalidBuffer);
    if (buf.numFrames() != mTree.nDims())
      return Error<RealVector>(WrongPointSize);
    if (k > mTree.nPoints()){
      return Error<RealVector>(SmallDataSet);
    }
    if(k <= 0 ){
      return Error<RealVector>(SmallK);
    }
    RealVector point(mTree.nDims());
    point = buf.samps(0, mTree.nDims(), 0);
    FluidDataSet<std::string, double,1> nearest = mTree.kNearest(point, k);
    RealVector result{nearest.getData().col(0)};
    return result;
  }

  MessageResult<void> read(string fileName) {
    auto file = FluidFile(fileName, "r");
    if(!file.valid()){return Error(file.error());}
    if(!file.read()){return Error(FileRead);}
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
    return OK();
  }

  MessageResult<void> write(string fileName){
    auto file = FluidFile(fileName, "w");
    if(!file.valid()){return Error(file.error());}
    algorithm::KDTree::FlatData treeData = mTree.toFlat();
    file.add("tree", treeData.tree);
    file.add("cols", treeData.data.cols());
    file.add("rows", treeData.data.rows());
    file.add("data", treeData.data);
    file.add("ids", treeData.ids);
    return file.write()? OK():Error(FileWrite);
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
