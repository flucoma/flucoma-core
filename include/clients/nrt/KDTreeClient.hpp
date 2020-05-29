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

  KDTreeClient(ParamSetViewType &p) : mParams(p), mDataClient(mTree) {}

  MessageResult<void> fit(DataSetClientRef datasetClient) {
    auto datasetClientPtr = datasetClient.get().lock();
    if(!datasetClientPtr) return Error(NoDataSet);
    auto dataset = datasetClientPtr->getDataSet();
    if (dataset.size() == 0) return Error(EmptyDataSet);
    mTree = algorithm::KDTree(dataset);
    return OK();
  }

  MessageResult<StringVector> kNearest(BufferPtr data, int k) const {
    if (!data) return Error<StringVector>(NoBuffer);
    BufferAdaptor::Access buf(data.get());
    if(!buf.exists()) return Error<StringVector>(InvalidBuffer);
    if (buf.numFrames() != mTree.dims()) return Error<StringVector>(WrongPointSize);
    if (k > mTree.size()) return Error<StringVector>(SmallDataSet);
    if(k <= 0 ) return Error<StringVector>(SmallK);

    RealVector point(mTree.dims());
    point = buf.samps(0, mTree.dims(), 0);
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
    if (buf.numFrames() != mTree.dims())
      return Error<RealVector>(WrongPointSize);
    if (k > mTree.size()){
      return Error<RealVector>(SmallDataSet);
    }
    if(k <= 0 ){
      return Error<RealVector>(SmallK);
    }
    RealVector point(mTree.dims());
    point = buf.samps(0, mTree.dims(), 0);
    FluidDataSet<std::string, double,1> nearest = mTree.kNearest(point, k);
    RealVector result{nearest.getData().col(0)};
    return result;
  }

  MessageResult<index> size() { return mDataClient.size(); }
  MessageResult<index> cols() { return mDataClient.dims(); }
  MessageResult<void> write(string fn) {return mDataClient.write(fn);}
  MessageResult<void> read(string fn) {return mDataClient.read(fn);}
  MessageResult<string> dump() { return mDataClient.dump();}
  MessageResult<void> load(string  s) { return mDataClient.load(s);}

  FLUID_DECLARE_MESSAGES(makeMessage("fit", &KDTreeClient::fit),
                         makeMessage("kNearest", &KDTreeClient::kNearest),
                         makeMessage("kNearestDist", &KDTreeClient::kNearestDist),
                         makeMessage("cols", &KDTreeClient::cols),
                         makeMessage("size", &KDTreeClient::size),
                         makeMessage("load", &KDTreeClient::load),
                         makeMessage("dump", &KDTreeClient::dump),
                         makeMessage("write", &KDTreeClient::write),
                         makeMessage("read", &KDTreeClient::read)
  );

private:
  algorithm::KDTree mTree{1};
  DataClient<algorithm::KDTree> mDataClient;
};

using NRTThreadedKDTreeClient =
    NRTThreadingAdaptor<ClientWrapper<KDTreeClient>>;

} // namespace client
} // namespace fluid
