/*
Part of the Fluid Corpus Manipulation Project (http://www.flucoma.org/)
Copyright 2017-2019 University of Huddersfield.
Licensed under the BSD-3 License.
See license.md file in the project root for full license information.
This project has received funding from the European Research Council (ERC)
under the European Union’s Horizon 2020 research and innovation programme
(grant agreement No 725899).
*/

#pragma once

#include "DataSetClient.hpp"
#include "NRTClient.hpp"
#include "../../algorithms/public/KDTree.hpp"
#include <string>

namespace fluid {
namespace client {
namespace kdtree {

constexpr auto KDTreeParams = defineParameters(
    StringParam<Fixed<true>>("name", "Name"),
    LongParam("numNeighbours", "Number of Nearest Neighbours", 1),
    FloatParam("radius", "Maximum distance", 0, Min(0)));

class KDTreeClient : public FluidBaseClient,
                     OfflineIn,
                     OfflineOut,
                     ModelObject,
                     public DataClient<algorithm::KDTree>
{
  enum { kName, kNumNeighbors, kRadius };

public:
  using string = std::string;
  using BufferPtr = std::shared_ptr<BufferAdaptor>;
  using InputBufferPtr = std::shared_ptr<const BufferAdaptor>;
  using StringVector = FluidTensor<rt::string, 1>;
  using ParamDescType = decltype(KDTreeParams);

  using ParamSetViewType = ParameterSetView<ParamDescType>;
  std::reference_wrapper<ParamSetViewType> mParams;

  void setParams(ParamSetViewType& p) { mParams = p; }

  template <size_t N>
  auto& get() const
  {
    return mParams.get().template get<N>();
  }

  static constexpr auto& getParameterDescriptors() { return KDTreeParams; }

  KDTreeClient(ParamSetViewType& p, FluidContext&) : mParams(p)
  {
    audioChannelsIn(1);
    controlChannelsOut({1, 1});
  }

  template <typename T>
  Result process(FluidContext&)
  {
    return {};
  }

  MessageResult<void> fit(InputDataSetClientRef datasetClient)
  {
    mDataSetClient = datasetClient;
    auto datasetClientPtr = mDataSetClient.get().lock();
    if (!datasetClientPtr) return Error(NoDataSet);
    auto dataset = datasetClientPtr->getDataSet();
    if (dataset.size() == 0) return Error(EmptyDataSet);
    mAlgorithm = algorithm::KDTree(dataset);
    return OK();
  }

  MessageResult<StringVector> kNearest(InputBufferPtr data, Optional<index> nNeighbours) const
  {
    //we can deprecate ancillary parameters in favour of optional args by falling back to using parameters when arg not present
    index k = nNeighbours ? nNeighbours.value() : get<kNumNeighbors>();
    //alternatively we could just be hardcore and ignore parameters and have message handlers fallback to a default when arg missing (which would be eventual behaviour, I guess)
    //index k =  nNeighbours.value_or(1); 
    if (k > mAlgorithm.size()) return Error<StringVector>(SmallDataSet);
    // if (k <= 0 && get<kRadius>() <= 0) return Error<StringVector>(SmallK);
    if (!mAlgorithm.initialized()) return Error<StringVector>(NoDataFitted);
    InBufferCheck bufCheck(mAlgorithm.dims());
    if (!bufCheck.checkInputs(data.get()))
      return Error<StringVector>(bufCheck.error());
    RealVector point(mAlgorithm.dims());
    point <<=
        BufferAdaptor::ReadAccess(data.get()).samps(0, mAlgorithm.dims(), 0);
    auto [dists, ids] =  mAlgorithm.kNearest(point, k, get<kRadius>());
    StringVector result(asSigned(ids.size()));
    std::transform(ids.cbegin(), ids.cend(), result.begin(),
                   [](const std::string* x) {
                     return rt::string{*x, FluidDefaultAllocator()};
                   });
    return result;
  }

  MessageResult<RealVector> kNearestDist(InputBufferPtr data, Optional<index> nNeighbours) const
  {
    // TODO: refactor with kNearest
    index k = nNeighbours ? nNeighbours.value() : get<kNumNeighbors>();
    if (k > mAlgorithm.size()) return Error<RealVector>(SmallDataSet);
    // if (k <= 0 && get<kRadius>() <= 0) return Error<RealVector>(SmallK);
    if (!mAlgorithm.initialized()) return Error<RealVector>(NoDataFitted);
    InBufferCheck bufCheck(mAlgorithm.dims());
    if (!bufCheck.checkInputs(data.get()))
      return Error<RealVector>(bufCheck.error());
    RealVector point(mAlgorithm.dims());
    point <<=
        BufferAdaptor::ReadAccess(data.get()).samps(0, mAlgorithm.dims(), 0);
    auto [dist, ids] = mAlgorithm.kNearest(point, k, get<kRadius>());
    return {dist};
  }

  static auto getMessageDescriptors()
  {
    return defineMessages(
        makeMessage("fit", &KDTreeClient::fit),
        makeMessage("kNearest", &KDTreeClient::kNearest),
        makeMessage("kNearestDist", &KDTreeClient::kNearestDist),
        makeMessage("cols", &KDTreeClient::dims),
        makeMessage("clear", &KDTreeClient::clear),
        makeMessage("size", &KDTreeClient::size),
        makeMessage("load", &KDTreeClient::load),
        makeMessage("dump", &KDTreeClient::dump),
        makeMessage("write", &KDTreeClient::write),
        makeMessage("read", &KDTreeClient::read));
  }

  InputDataSetClientRef getDataSet() const { return mDataSetClient; }

  const algorithm::KDTree& algorithm() const { return mAlgorithm; }

private:
  InputDataSetClientRef mDataSetClient;
};

using KDTreeRef = SharedClientRef<const KDTreeClient>;

constexpr auto KDTreeQueryParams = defineParameters(
    KDTreeRef::makeParam("tree", "KDTree"),
    LongParam("numNeighbours", "Number of Nearest Neighbours", 1),
    FloatParam("radius", "Maximum distance", 0, Min(0)),
    InputDataSetClientRef::makeParam("dataSet", "DataSet Name"),
    InputBufferParam("inputPointBuffer", "Input Point Buffer"),
    BufferParam("predictionBuffer", "Prediction Buffer"));

class KDTreeQuery : public FluidBaseClient, ControlIn, ControlOut
{
  enum { kTree, kNumNeighbors, kRadius, kDataSet, kInputBuffer, kOutputBuffer };

public:
  using ParamDescType = decltype(KDTreeQueryParams);
  using ParamSetViewType = ParameterSetView<ParamDescType>;

  std::reference_wrapper<ParamSetViewType> mParams;

  void setParams(ParamSetViewType& p) { mParams = p; }

  template <size_t N>
  auto& get() const
  {
    return mParams.get().template get<N>();
  }

  static constexpr auto& getParameterDescriptors() { return KDTreeQueryParams; }

  KDTreeQuery(ParamSetViewType& p, FluidContext& c) : mParams(p), mRTBuffer(c.allocator()) 
  {
    controlChannelsIn(1);
    controlChannelsOut({1, 1});
  }

  index latency() { return 0; }

  template <typename T>
  void process(std::vector<FluidTensorView<T, 1>>& input,
               std::vector<FluidTensorView<T, 1>>& output, FluidContext& c)
  {
    output[0] <<= input[0];

    if (input[0](0) > 0)
    {
      auto kdtreeptr = get<kTree>().get().lock();
      if (!kdtreeptr)
      {
        // c.reportError("No FluidKDTree found"); //why are both this and line 197+214 commented?
        return;
      }

      if (!kdtreeptr->initialized())
      {
        // c.reportError("FluidKDTree not fitted");
        return;
      }

      index k = get<kNumNeighbors>();
      if (k > kdtreeptr->size() || k <= 0) return;
      index             dims = kdtreeptr->dims();
      InOutBuffersCheck bufCheck(dims);
      if (!bufCheck.checkInputs(get<kInputBuffer>().get(),
                                get<kOutputBuffer>().get()))
        return;
      auto datasetClientPtr = get<kDataSet>().get().lock();
      if (!datasetClientPtr)
        datasetClientPtr = kdtreeptr->getDataSet().get().lock();

      if (!datasetClientPtr)
      {
        // c.reportError("Could not obtain reference FluidDataSet");
        return;
      }

      auto  dataset = datasetClientPtr->getDataSet();
      index pointSize = dataset.pointSize();
      auto  outBuf = BufferAdaptor::Access(get<kOutputBuffer>().get());
      index realK = std::min(k, (outBuf.samps(0).size() / pointSize));
      if (realK <= 0) return;
      index outputSize =  realK * pointSize;

      RealVector point(dims, c.allocator());
      point <<= BufferAdaptor::ReadAccess(get<kInputBuffer>().get())
                    .samps(0, dims, 0);
      if (mRTBuffer.size() != outputSize)
      {
        mRTBuffer = RealVector(outputSize, c.allocator());
        mRTBuffer.fill(0);
      }

      auto [dists, ids] =
          kdtreeptr->algorithm().kNearest(point, realK, 0, c.allocator()); // i'd like to pass get<kRadius>() and output min(nbpoints,realK) somehow

      for (index i = 0; i < realK; i++)
      {
        dataset.get(*ids[asUnsigned(i)],
                    mRTBuffer(Slice(i * pointSize, pointSize)));
      }
      outBuf.samps(0, outputSize, 0) <<= mRTBuffer;
    }
  }


private:
  RealVector       mRTBuffer;
  InputDataSetClientRef mDataSetClient;
};

} // namespace kdtree

using NRTThreadedKDTreeClient =
    NRTThreadingAdaptor<typename kdtree::KDTreeRef::SharedType>;
using RTKDTreeQueryClient = ClientWrapper<kdtree::KDTreeQuery>;

} // namespace client
} // namespace fluid
