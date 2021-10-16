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

#include "../common/FluidBaseClient.hpp"
#include "../common/FluidNRTClientWrapper.hpp"
#include "../common/OfflineClient.hpp"
#include "../common/ParameterSet.hpp"
#include "../common/ParameterTypes.hpp"
#include "../common/Result.hpp"
#include "../../data/FluidTensor.hpp"
#include "../../data/TensorTypes.hpp"
#include <numeric> //std::iota
#include <vector>

namespace fluid {
namespace client {
namespace bufselect {

static constexpr std::initializer_list<index> SelectionDefaults = {-1};

enum { kSource, kDest, kIndices, kChannels };

constexpr auto BufSelectParams =
    defineParameters(InputBufferParam("source", "Source Buffer"),
                     BufferParam("destination", "Destination Buffer"));

class BufTransposeClient : public FluidBaseClient, OfflineIn, OfflineOut
{

public:
  using ParamDescType = decltype(BufSelectParams);

  using ParamSetViewType = ParameterSetView<ParamDescType>;
  std::reference_wrapper<ParamSetViewType> mParams;

  void setParams(ParamSetViewType& p) { mParams = p; }

  template <size_t N>
  auto& get() const
  {
    return mParams.get().template get<N>();
  }

  static constexpr auto& getParameterDescriptors() { return BufSelectParams; }


  BufTransposeClient(ParamSetViewType& p) : mParams{p} {}

  template <typename T>
  Result process(FluidContext&)
  {

    if (!get<kSource>().get())
    { return {Result::Status::kError, "No source buffer "}; }
    if (!get<kDest>().get())
    { return {Result::Status::kError, "No destination buffer"}; }

    BufferAdaptor::ReadAccess source(get<kSource>().get());
    BufferAdaptor::Access     destination(get<kDest>().get());

    if (!(source.exists() && source.valid()))
      return {Result::Status::kError, "Source Buffer Not Found or Invalid"};

    if (!destination.exists())
      return {Result::Status::kError,
              "Destination Buffer Not Found or Invalid"};

    index numFrames = source.numFrames();
    index numChans = source.numChans();

    auto resizeResult =
        destination.resize(numChans, numFrames, source.sampleRate());

    if (!resizeResult.ok()) return resizeResult;

    auto frames = source.allFrames();

    std::copy(
      frames.transpose().begin(), 
      frames.transpose().end(),
      destination.allFrames().begin()
    );

    return {Result::Status::kOk};
  }
};
} // namespace bufselect

using NRTThreadingTransposeClient =
    NRTThreadingAdaptor<ClientWrapper<bufselect::BufTransposeClient>>;

} // namespace client
} // namespace fluid
