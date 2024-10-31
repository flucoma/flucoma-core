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
#include "../../algorithms/public/MDS.hpp"

namespace fluid {
namespace client {
namespace mds {

enum { kNumDimensions, kDistance };

constexpr auto MDSParams = defineParameters(
    LongParam("numDimensions", "Target Number of Dimensions", 2, Min(1)),
    EnumParam("distanceMetric", "Distance Metric", 1, "Manhattan", "Euclidean",
              "Squared Euclidean", "Max Distance", "Min Distance",
              "KL Divergence"));

class MDSClient : public FluidBaseClient, OfflineIn, OfflineOut, ModelObject
{

public:
  using string = std::string;
  using BufferPtr = std::shared_ptr<BufferAdaptor>;
  using InputBufferPtr = std::shared_ptr<const BufferAdaptor>;
  using StringVector = FluidTensor<string, 1>;

  template <typename T>
  Result process(FluidContext&)
  {
    return {};
  }

  using ParamDescType = decltype(MDSParams);

  using ParamSetViewType = ParameterSetView<ParamDescType>;
  std::reference_wrapper<ParamSetViewType> mParams;

  void setParams(ParamSetViewType& p) { mParams = p; }

  template <size_t N>
  auto& get() const
  {
    return mParams.get().template get<N>();
  }

  static constexpr auto getParameterDescriptors() { return MDSParams; }

  MDSClient(ParamSetViewType& p, FluidContext&) : mParams(p) {}

  MessageResult<void> fitTransform(InputDataSetClientRef sourceClient,
                                   DataSetClientRef destClient)
  {
    index k = get<kNumDimensions>();
    index dist = get<kDistance>();
    auto  srcPtr = sourceClient.get().lock();
    auto  destPtr = destClient.get().lock();
    if (!srcPtr || !destPtr) return Error(NoDataSet);
    auto src = srcPtr->getDataSet();
    auto dest = destPtr->getDataSet();
    if (src.size() == 0) return Error(EmptyDataSet);
    if (k <= 0) return Error(SmallK);
    if (dist < 0 || dist > 6) return Error("dist should be  between 0 and 6");

    StringVector ids{src.getIds()};
    RealMatrix   output(src.size(), k);
    mAlgorithm.process(src.getData(), output, dist, k);
    FluidDataSet<string, double, 1> result(ids, output);
    destPtr->setDataSet(result);
    return OK();
  }

  static auto getMessageDescriptors()
  {
    return defineMessages(
        makeMessage("fitTransform", &MDSClient::fitTransform));
  }

private:
  algorithm::MDS mAlgorithm;
};
} // namespace mds

using NRTThreadedMDSClient = NRTThreadingAdaptor<ClientWrapper<mds::MDSClient>>;

} // namespace client
} // namespace fluid
