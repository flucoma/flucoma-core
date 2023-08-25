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
#include "DataSeriesClient.hpp"
#include "NRTClient.hpp"
#include "../../algorithms/public/DTW.hpp"
#include <string>

namespace fluid {
namespace client {
namespace dtw {

constexpr auto DTWParams = defineParameters(
    StringParam<Fixed<true>>("name", "Name"),
    LongParam("p", "LpNorm power (distance weighting)", 2, Min(1)));

class DTWClient : public FluidBaseClient,
                  OfflineIn,
                  OfflineOut,
                  ModelObject
{
  enum { kName, kQ };

public:
  using string = std::string;
  using BufferPtr = std::shared_ptr<BufferAdaptor>;
  using InputBufferPtr = std::shared_ptr<const BufferAdaptor>;
  using StringVector = FluidTensor<rt::string, 1>;

  using ParamDescType = decltype(DTWParams);
  using ParamSetViewType = ParameterSetView<ParamDescType>;
  using ParamValues = typename ParamSetViewType::ValueTuple;

  std::reference_wrapper<ParamSetViewType> mParams;

  void setParams(ParamSetViewType& p) { mParams = p; }

  template <size_t N>
  auto& get() const
  {
    return mParams.get().template get<N>();
  }

  static constexpr auto& getParameterDescriptors() { return DTWParams; }

  DTWClient(ParamSetViewType& p, FluidContext&) : mParams(p)
  {
    controlChannelsIn(1);
    controlChannelsOut({1, 1});
  }

  template <typename T>
  Result process(FluidContext&)
  {
    return {};
  }

  MessageResult<void> cost(InputDataSetClientRef datasetClient)
  {
    return OK();
  }

  MessageResult<double> bufCost(InputBufferPtr data1, InputBufferPtr data2)
  {
    if (!data1 || !data2) return Error<double>(NoBuffer);

    BufferAdaptor::ReadAccess buf1(data1.get()), buf2(data2.get());

    if (!buf1.exists() || !buf2.exists()) return Error<double>(InvalidBuffer);
    if (buf1.numChans() != buf2.numChans()) return Error<double>(WrongPointSize);

    double cost = algorithm::DTW<double>::process(buf1.allFrames().transpose(),
                                                  buf2.allFrames().transpose());

    return cost;
  }

  static auto getMessageDescriptors()
  {
    return defineMessages(
        makeMessage("cost",    &DTWClient::cost),
        makeMessage("bufCost", &DTWClient::bufCost)
    );
  }
};

using DTWRef = SharedClientRef<const DTWClient>;

constexpr auto DTWQueryParams = defineParameters(
    DTWRef::makeParam("tree", "DTW"),
    LongParam("p", "LpNorm power (distance weighting)", 2, Min(0)),
    InputBufferParam("inputPointBuffer", "Input Point Buffer"),
    BufferParam("predictionBuffer", "Prediction Buffer"));

class DTWQuery : public FluidBaseClient, ControlIn, ControlOut
{
  enum { kTree, kNumNeighbors, kRadius, kDataSet, kInputBuffer, kOutputBuffer };

public:
  using ParamDescType = decltype(DTWQueryParams);
  using ParamSetViewType = ParameterSetView<ParamDescType>;

  std::reference_wrapper<ParamSetViewType> mParams;

  void setParams(ParamSetViewType& p) { mParams = p; }

  template <size_t N>
  auto& get() const
  {
    return mParams.get().template get<N>();
  }

  static constexpr auto& getParameterDescriptors() { return DTWQueryParams; }

  DTWQuery(ParamSetViewType& p, FluidContext& c)
      : mParams(p), mRTBuffer(c.allocator())
  {
    controlChannelsIn(1);
    controlChannelsOut({1, 1});
  }

  index latency() { return 0; }

  template <typename T>
  void process(std::vector<FluidTensorView<T, 1>>& input,
               std::vector<FluidTensorView<T, 1>>& output, FluidContext& c)
  {
    if (input[0](0) > 0)
    {
      auto kdtreeptr = get<kTree>().get().lock();
      if (!kdtreeptr)
      {
        // c.reportError("FluidKDTree RT Query: No FluidKDTree found");
        return;
      }

      if (!kdtreeptr->initialized())
      {
        // c.reportError("FluidKDTree RT Query: tree not fitted");
        return;
      }

      index k = get<kNumNeighbors>();
      if (k > kdtreeptr->size() || k < 0)
        return; // c.reportError("FluidKDTree RT Query has wrong k size");
      index             dims = kdtreeptr->dims();
      InOutBuffersCheck bufCheck(dims);
      if (!bufCheck.checkInputs(get<kInputBuffer>().get(),
                                get<kOutputBuffer>().get()))
        return; // c.reportError("FluidKDTree RT Query i/o buffers are
                // unavailable");
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
      index maxK = outBuf.samps(0).size() / pointSize;
      if (maxK <= 0) return;
      index outputSize = maxK * pointSize;

      RealVector point(dims, c.allocator());
      point <<= BufferAdaptor::ReadAccess(get<kInputBuffer>().get())
                    .samps(0, dims, 0);
      if (mRTBuffer.size() != outputSize)
      {
        mRTBuffer = RealVector(outputSize, c.allocator());
        mRTBuffer.fill(0);
      }

      auto [dists, ids] = kdtreeptr->algorithm().kNearest(
          point, k, get<kRadius>(), c.allocator());

      mNumValidKs = std::min(asSigned(ids.size()), maxK);

      for (index i = 0; i < mNumValidKs; i++)
      {
        dataset.get(*ids[asUnsigned(i)],
                    mRTBuffer(Slice(i * pointSize, pointSize)));
      }
      outBuf.samps(0, outputSize, 0) <<= mRTBuffer;
    }

    output[0](0) = mNumValidKs;
  }


private:
  RealVector            mRTBuffer;
  index                 mNumValidKs = 0;
  InputDataSetClientRef mDataSetClient;
};

} // namespace DTW

using NRTThreadedDTWClient =
    NRTThreadingAdaptor<typename dtw::DTWRef::SharedType>;
using RTDTWQueryClient = ClientWrapper<dtw::DTWQuery>;

} // namespace client
} // namespace fluid
