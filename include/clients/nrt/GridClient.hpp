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
#include "../../algorithms/public/Grid.hpp"

namespace fluid {
namespace client {
namespace grid {

enum {kResample, kExtent, kAxis};

constexpr auto GridParams = defineParameters(
    LongParam("oversample", "Oversampling factor", 1, Min(1)),
    LongParam("extent", "Extent", 0, Min(0)),
    EnumParam("axis", "Extent Axis", 0, "Horizontal", "Vertical"));

class GridClient : public FluidBaseClient, OfflineIn, OfflineOut, ModelObject
{

public:
  using string = std::string;
  using BufferPtr = std::shared_ptr<BufferAdaptor>;
  using StringVector = FluidTensor<string, 1>;

  template <typename T>
  Result process(FluidContext&)
  {
    return {};
  }

  using ParamDescType = decltype(GridParams);

  using ParamSetViewType = ParameterSetView<ParamDescType>;
  std::reference_wrapper<ParamSetViewType> mParams;

  void setParams(ParamSetViewType& p) { mParams = p; }

  template <size_t N>
  auto& get() const
  {
    return mParams.get().template get<N>();
  }

  static constexpr auto getParameterDescriptors() { return GridParams; }

  GridClient(ParamSetViewType& p, FluidContext&) : mParams(p) {}

  MessageResult<void> fitTransform(InputDataSetClientRef sourceClient,
                                   DataSetClientRef destClient)
  {
    auto srcPtr = sourceClient.get().lock();
    auto destPtr = destClient.get().lock();
    if (!srcPtr || !destPtr) return Error(NoDataSet);
    auto src = srcPtr->getDataSet();
    auto dest = destPtr->getDataSet();
    if (src.dims() != 2) return Error("Dataset should be 2D");
    if (src.size() == 0) return Error(EmptyDataSet);
    FluidDataSet<string, double, 1> result;
    result = mAlgorithm.process(src,
        get<kResample>(), get<kExtent>(), get<kAxis>());
    destPtr->setDataSet(result);
    return OK();
  }

  static auto getMessageDescriptors()
  {
    return defineMessages(
        makeMessage("fitTransform", &GridClient::fitTransform));
  }

private:
  algorithm::Grid mAlgorithm;
};
} // namespace Grid

using NRTThreadedGridClient = NRTThreadingAdaptor<ClientWrapper<grid::GridClient>>;

} // namespace client
} // namespace fluid
