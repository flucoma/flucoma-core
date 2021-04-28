#pragma once

#include "DataSetClient.hpp"
#include "LabelSetClient.hpp"
#include "NRTClient.hpp"
#include "algorithms/public/KMeans.hpp"
#include <string>

namespace fluid {
namespace client {
namespace kmeans{

  enum {kNumClusters, kMaxIter, kInputBuffer, kOutputBuffer};

  constexpr auto KMeansParams = defineParameters(
      LongParam("numClusters","Number of Clusters", 4, Min(1)),
      LongParam("maxIter","Max number of Iterations", 100, Min(1)),
      BufferParam("inputPointBuffer","Input Point Buffer"),
      BufferParam("predictionBuffer","Prediction Buffer")
  );

class KMeansClient : public FluidBaseClient,
                     AudioIn,
                     ControlOut,
                     ModelObject,
                     public DataClient<algorithm::KMeans> {

public:
  using string = std::string;
  using BufferPtr = std::shared_ptr<BufferAdaptor>;
  using IndexVector = FluidTensor<index, 1>;
  using StringVector = FluidTensor<string, 1>;
  using StringVectorView = FluidTensorView<string, 1>;
  using LabelSet = FluidDataSet<string, string, 1>;

  using ParamDescType = decltype(KMeansParams);

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
    return KMeansParams;
  }

  KMeansClient(ParamSetViewType &p) : mParams(p)
  {
    audioChannelsIn(1);
    controlChannelsOut(1);
  }

  template <typename T>
  void process(std::vector<FluidTensorView<T, 1>> &input,
               std::vector<FluidTensorView<T, 1>> &output, FluidContext &)
  {
    if (!mAlgorithm.initialized()) return;
    InOutBuffersCheck bufCheck(mAlgorithm.dims());
    if(!bufCheck.checkInputs(
      get<kInputBuffer>().get(),
      get<kOutputBuffer>().get()))
      return;
    auto outBuf = BufferAdaptor::Access(get<kOutputBuffer>().get());
    if(outBuf.samps(0).size() != 1) return;
    RealVector point(mAlgorithm.dims());
    point = BufferAdaptor::ReadAccess(get<kInputBuffer>().get()).samps(0, mAlgorithm.dims(), 0);
    mTrigger.process(input, output, [&](){
       outBuf.samps(0)[0] = mAlgorithm.vq(point);
    });
  }

  MessageResult<IndexVector> fit(DataSetClientRef datasetClient) {
    index k = get<kNumClusters>();
    index maxIter = get<kMaxIter>();
    auto datasetClientPtr = datasetClient.get().lock();
    if (!datasetClientPtr) return Error<IndexVector>(NoDataSet);
    auto dataSet = datasetClientPtr->getDataSet();
    if (dataSet.size() == 0) return Error<IndexVector>(EmptyDataSet);
    if (k <= 1) return Error<IndexVector>(SmallK);
    mAlgorithm.train(dataSet, k, maxIter);
    IndexVector assignments(dataSet.size());
    mAlgorithm.getAssignments(assignments);
    return getCounts(assignments, k);
  }

  MessageResult<IndexVector> fitPredict(DataSetClientRef datasetClient,
                                        LabelSetClientRef labelsetClient) {
    index k = get<kNumClusters>();
    index maxIter = get<kMaxIter>();
    auto datasetClientPtr = datasetClient.get().lock();
    if (!datasetClientPtr)
      return Error<IndexVector>(NoDataSet);
    auto dataSet = datasetClientPtr->getDataSet();
    if (dataSet.size() == 0)
      return Error<IndexVector>(EmptyDataSet);
    auto labelsetClientPtr = labelsetClient.get().lock();
    if (!labelsetClientPtr)
      return Error<IndexVector>(NoLabelSet);
    if (k <= 1) return Error<IndexVector>(SmallK);
    if (maxIter <= 0) maxIter = 100;
    mAlgorithm.train(dataSet, k, maxIter);
    IndexVector assignments(dataSet.size());
    mAlgorithm.getAssignments(assignments);
    StringVectorView ids = dataSet.getIds();
    labelsetClientPtr->setLabelSet(getLabels(ids, assignments));
    return getCounts(assignments, k);
  }

  MessageResult<IndexVector> predict(DataSetClientRef datasetClient,
                                     LabelSetClientRef labelClient) const {
    auto dataPtr = datasetClient.get().lock();
    if (!dataPtr)
      return Error<IndexVector>(NoDataSet);
    auto labelsetClientPtr = labelClient.get().lock();
    if (!labelsetClientPtr)
      return Error<IndexVector>(NoLabelSet);
    auto dataSet = dataPtr->getDataSet();
    if (dataSet.size() == 0)
      return Error<IndexVector>(EmptyDataSet);
    if (!mAlgorithm.initialized())
      return Error<IndexVector>(NoDataFitted);
    if (dataSet.dims() != mAlgorithm.dims())
      return Error<IndexVector>(WrongPointSize);
    StringVectorView ids = dataSet.getIds();
    IndexVector assignments(dataSet.size());
    RealVector query(mAlgorithm.dims());
    for (index i = 0; i < dataSet.size(); i++) {
      dataSet.get(ids(i), query);
      assignments(i) = mAlgorithm.vq(query);
    }
    labelsetClientPtr->setLabelSet(getLabels(ids, assignments));
    return getCounts(assignments, mAlgorithm.getK());
  }


  MessageResult<void> transform(DataSetClientRef srcClient,
                                DataSetClientRef dstClient) const {
    auto srcPtr = srcClient.get().lock();
    if (!srcPtr) return Error<void>(NoDataSet);
    auto destPtr = dstClient.get().lock();
    if (!destPtr) return Error<void>(NoDataSet);

    auto srcDataSet = srcPtr->getDataSet();
    if (srcDataSet.size() == 0)
      return Error<void>(EmptyDataSet);
    if (!mAlgorithm.initialized())
      return Error<void>(NoDataFitted);
    if (srcDataSet.dims() != mAlgorithm.dims())
      return Error<void>(WrongPointSize);

    StringVectorView ids = srcDataSet.getIds();
    RealMatrix output(srcDataSet.size(), mAlgorithm.size());
    mAlgorithm.getDistances(srcDataSet.getData(), output);
    FluidDataSet<string, double, 1> result(ids, output);
    destPtr->setDataSet(result);
    return OK();
  }

  MessageResult<IndexVector> fitTransform(DataSetClientRef srcClient,
                                     DataSetClientRef dstClient) {
    index k = get<kNumClusters>();
    index maxIter = get<kMaxIter>();
    auto srcPtr = srcClient.get().lock();
    if (!srcPtr) return Error<IndexVector>(NoDataSet);
    auto destPtr = dstClient.get().lock();
    if (!destPtr) return Error<IndexVector>(NoDataSet);
    auto dataSet = srcPtr->getDataSet();
    if (dataSet.size() == 0)
      return Error<IndexVector>(EmptyDataSet);
    if (k <= 1) return Error<IndexVector>(SmallK);
    if (maxIter <= 0) maxIter = 100;
    mAlgorithm.train(dataSet, k, maxIter);
    IndexVector assignments(dataSet.size());
    mAlgorithm.getAssignments(assignments);
    transform(srcClient, dstClient);
    return getCounts(assignments, k);
  }

  MessageResult<index> predictPoint(BufferPtr data) const {
    if (!mAlgorithm.initialized()) return Error<index>(NoDataFitted);
    InBufferCheck bufCheck(mAlgorithm.dims());
    if(!bufCheck.checkInputs(data.get())) return Error<index>(bufCheck.error());
    RealVector point(mAlgorithm.dims());
    point = BufferAdaptor::ReadAccess(data.get()).samps(0, mAlgorithm.dims(), 0);
    return mAlgorithm.vq(point);
  }

  MessageResult<void> getMeans(DataSetClientRef dstClient) const {
    auto destPtr = dstClient.get().lock();
    if (!destPtr) return Error<void>(NoDataSet);
    if (!mAlgorithm.initialized())
      return Error<void>(NoDataFitted);
    RealMatrix output(mAlgorithm.size(), mAlgorithm.dims());
    mAlgorithm.getMeans(output);
    StringVector ids(mAlgorithm.size());
    std::generate(ids.begin(), ids.end(),[n = 0]() mutable {return std::to_string(n++);});
    FluidDataSet<string, double, 1> result(ids, output);
    destPtr->setDataSet(result);
    return OK();
  }

  MessageResult<void> setMeans(DataSetClientRef srcClient) {
    auto srcPtr = srcClient.get().lock();
    if (!srcPtr) return Error(NoDataSet);
    auto dataSet = srcPtr->getDataSet();
    if (dataSet.size() == 0)return Error(EmptyDataSet);
    if (dataSet.size() != get<kNumClusters>())return Error(WrongNumInitial);
    mAlgorithm.setMeans(dataSet.getData());
    return OK();
  }


MessageResult<void> transformPoint(BufferPtr in, BufferPtr out) const {
    if (!mAlgorithm.initialized()) return Error(NoDataFitted);
    InBufferCheck bufCheck(mAlgorithm.dims());
    if(!bufCheck.checkInputs(in.get())) return Error(bufCheck.error());
    BufferAdaptor::Access outBuf(out.get());
    Result resizeResult = outBuf.resize(mAlgorithm.size(), 1, outBuf.sampleRate());
    if (!resizeResult.ok())
      return Error(BufferAlloc);
    RealMatrix src(1, mAlgorithm.dims());
    RealMatrix dest(1, mAlgorithm.size());
    RealVector dest1(mAlgorithm.size());
    src.row(0) = BufferAdaptor::ReadAccess(in.get()).samps(0, mAlgorithm.dims(), 0);
    mAlgorithm.getDistances(src, dest);
    outBuf.samps(0) = dest.row(0);
    return OK();
  }


  index latency() { return 0; }

  static auto getMessageDescriptors()
  {
    return defineMessages(makeMessage("fit", &KMeansClient::fit),
                         makeMessage("predict", &KMeansClient::predict),
                         makeMessage("transform", &KMeansClient::transform),
                         makeMessage("predictPoint",
                                     &KMeansClient::predictPoint),
                         makeMessage("transformPoint",
                                      &KMeansClient::transformPoint),
                         makeMessage("fitTransform",
                                      &KMeansClient::fitTransform),
                         makeMessage("getMeans",&KMeansClient::getMeans),
                         makeMessage("setMeans",&KMeansClient::setMeans),
                         makeMessage("fitPredict", &KMeansClient::fitPredict),
                         makeMessage("cols", &KMeansClient::dims),
                         makeMessage("clear", &KMeansClient::clear),
                         makeMessage("size", &KMeansClient::size),
                         makeMessage("load", &KMeansClient::load),
                         makeMessage("dump", &KMeansClient::dump),
                         makeMessage("write", &KMeansClient::write),
                         makeMessage("read", &KMeansClient::read));
  }


private:
  IndexVector getCounts(IndexVector assignments, index k) const {
    IndexVector counts(k);
    counts.fill(0);
    for (auto a : assignments)
      counts[a]++;
    return counts;
  }

  LabelSet getLabels(StringVectorView &ids, IndexVector assignments) const {
    LabelSet result(1);
    for (index i = 0; i < ids.size(); i++) {
      StringVector point = {std::to_string(assignments(i))};
      result.add(ids(i), point);
    }
    return result;
  }

  FluidInputTrigger mTrigger;
};
}

using RTKMeansClient = ClientWrapper<kmeans::KMeansClient>;

} // namespace client
} // namespace fluid
