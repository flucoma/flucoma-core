/*
Part of the Fluid Corpus Manipulation Project (http://www.flucoma.org/)
Copyright University of Huddersfield.
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
    LongParam("numNeighbours", "Number of Nearest Neighbours", 1, Min(0)),
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

  MessageResult<StringVector> kNearest(InputBufferPtr  data,
                                       Optional<index> nNeighbours) const
  {
    index k = nNeighbours ? nNeighbours.value() : get<kNumNeighbors>();
      
    if (k < 0) return Error(SmallK);

    auto reply = computeKnearest(data, k);
    if (!reply.ok()) return reply;

    auto dists = reply.value().first;
    auto ids = reply.value().second;

    StringVector result(asSigned(ids.size()));
    std::transform(ids.cbegin(), ids.cend(), result.begin(),
                   [](const std::string* x) {
                     return rt::string{*x, FluidDefaultAllocator()};
                   });
    return result;
  }

  MessageResult<RealVector> kNearestDist(InputBufferPtr  data,
                                         Optional<index> nNeighbours) const
  {
    index k = nNeighbours ? nNeighbours.value() : get<kNumNeighbors>();
      
    if (k < 0) return Error(SmallK);

    auto  reply = computeKnearest(data, k);
    if (!reply.ok()) return reply;

    return {reply.value().first};
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

  MessageResult<algorithm::KDTree::KNNResult>
  computeKnearest(InputBufferPtr data, index k) const
  {
    if (k > mAlgorithm.size())
      return Error<algorithm::KDTree::KNNResult>(SmallDataSet);
    if (k < 0 && get<kRadius>() < 0)
      return Error<algorithm::KDTree::KNNResult>(SmallK);
    if (!mAlgorithm.initialized())
      return Error<algorithm::KDTree::KNNResult>(NoDataFitted);
    InBufferCheck bufCheck(mAlgorithm.dims());
    if (!bufCheck.checkInputs(data.get()))
      return Error<algorithm::KDTree::KNNResult>(bufCheck.error());
    RealVector point(mAlgorithm.dims());
    point <<=
        BufferAdaptor::ReadAccess(data.get()).samps(0, mAlgorithm.dims(), 0);
    return {mAlgorithm.kNearest(point, k, get<kRadius>())};
  }
};

using KDTreeRef = SharedClientRef<const KDTreeClient>;

constexpr auto KDTreeQueryParams = defineParameters(
    KDTreeRef::makeParam("tree", "KDTree"),
    LongParam("numNeighbours", "Number of Nearest Neighbours", 1, Min(0)),
    FloatParam("radius", "Maximum distance", 0, Min(0)),
    InputDataSetClientRef::makeParam("lookupDataSet", "Lookup DataSet Name"),
    InputBufferParam("inputPointBuffer", "Input Point Buffer"),
    BufferParam("predictionBuffer", "Prediction Buffer"));

class KDTreeQuery : public FluidBaseClient, ControlIn, ControlOut
{
  enum {
    kTree,
    kNumNeighbors,
    kRadius,
    kLookupDataSet,
    kInputBuffer,
    kOutputBuffer
  };

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

  KDTreeQuery(ParamSetViewType& p, FluidContext& c) : mParams(p)
  {
    controlChannelsIn(1);
    controlChannelsOut({1, 1});
  }

  index latency() const { return 0; }

  template <typename T>
  void process(std::vector<FluidTensorView<T, 1>>& input,
               std::vector<FluidTensorView<T, 1>>& output, FluidContext& c)
  {
    if (input[0](0) > 0)
    {
      output[0](0) = mLastNumPoints = 0;

      auto kdtreeptr = get<kTree>().get().lock();
      if (!kdtreeptr)
        return; // c.reportError("FluidKDTree RT Query: No FluidKDTree found");

      if (!kdtreeptr->initialized())
        return; // c.reportError("FluidKDTree RT Query: tree not fitted");

      index k = get<kNumNeighbors>();
      if (k > kdtreeptr->size() || k < 0)
        return; // c.reportError("FluidKDTree RT Query has wrong k size");

      index dims = kdtreeptr->dims();

      InOutBuffersCheck bufCheck(dims);
      if (!bufCheck.checkInputs(get<kInputBuffer>().get(),
                                get<kOutputBuffer>().get()))
        return; // c.reportError("FluidKDTree RT Query i/o buffers are
                // unavailable");

      auto lookupDSpointer = get<kLookupDataSet>().get().lock();

      index pointSize = lookupDSpointer ? lookupDSpointer->dims().value() : 1;

      auto outBuf = BufferAdaptor::Access(get<kOutputBuffer>().get());
      auto outSamps = outBuf.samps(0);

      index numPoints = outSamps.size() / pointSize;
      if (numPoints <= 0)
        return; // c.reportError("FluidKDTree RT Query output buffer is too
                // small for one point")

      RealVector point(dims, c.allocator());
      point <<= BufferAdaptor::ReadAccess(get<kInputBuffer>().get())
                    .samps(0, dims, 0);

      auto [dists, ids] = kdtreeptr->algorithm().kNearest(
          point, k, get<kRadius>(), c.allocator());

      index numValid = 0;

      if (lookupDSpointer)
      {
        auto lookupDS = lookupDSpointer->getDataSet();

        auto lookupFn = [&lookupDS, outSamps, pointSize, n = 0,
                         &numValid](auto id) mutable {
          if (auto point = lookupDS.get(*id); point.data() != nullptr)
          {
            outSamps(Slice(n, pointSize)) <<= point;
            n += pointSize;
            numValid += 1;
          }
        };

        std::for_each_n(ids.begin(), std::min(asSigned(ids.size()), numPoints),
                        lookupFn);
      }
      else
      {
        numValid = std::min(asSigned(ids.size()), numPoints);
        std::copy_n(dists.begin(), numValid, outSamps.begin());
      }

      mLastNumPoints = numValid;
    }

    output[0](0) =
        mLastNumPoints; // updates the output if successful or if not triggered
  }


private:
  index                 mLastNumPoints{0};
  InputDataSetClientRef mDataSetClient;
};

} // namespace kdtree

using NRTThreadedKDTreeClient =
    NRTThreadingAdaptor<typename kdtree::KDTreeRef::SharedType>;
using RTKDTreeQueryClient = ClientWrapper<kdtree::KDTreeQuery>;

} // namespace client
} // namespace fluid
