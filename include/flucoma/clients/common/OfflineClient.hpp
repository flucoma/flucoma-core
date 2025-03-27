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

#include "BufferAdaptor.hpp"

namespace fluid {
namespace client {
struct Offline
{};

struct OfflineIn : Offline
{};
struct OfflineOut : Offline
{};

struct BufferProcessSpec
{
  BufferProcessSpec() = default;

  BufferProcessSpec(const BufferAdaptor* b, intptr_t o, intptr_t nf,
                    intptr_t co, intptr_t nc)
      : buffer{b}, startFrame{o}, nFrames{nf}, startChan{co}, nChans{nc}
  {}

  const BufferAdaptor* buffer;
  intptr_t             startFrame = 0;
  intptr_t             nFrames = -1;
  intptr_t             startChan = 0;
  intptr_t             nChans = -1;
};


} // namespace client
} // namespace fluid
