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

#include "CommonResults.hpp"
#include "../common/BufferAdaptor.hpp"
#include "../common/Result.hpp"

namespace fluid {
namespace client {

class ClientInputCheck
{

public:
  std::string error() { return mErrorMessage; }

protected:
  std::string mErrorMessage;
};

class InBufferCheck : public ClientInputCheck
{
public:
  InBufferCheck(index size) : mInputSize(size){};
  bool checkInputs(const BufferAdaptor* inputPtr)
  {
    if (!inputPtr)
    {
      mErrorMessage = NoBuffer;
      return false;
    }
    BufferAdaptor::ReadAccess buf(inputPtr);
    if (!buf.exists() || !buf.valid())
    {
      mErrorMessage = InvalidBuffer;
      return false;
    }
    if (buf.numFrames() != mInputSize)
    {
      mErrorMessage = WrongPointSize;
      return false;
    }
    return true;
  }

protected:
  index mInputSize;
};

class InOutBuffersCheck : public InBufferCheck
{

public:
  using InBufferCheck::InBufferCheck;
  bool checkInputs(const BufferAdaptor* inputPtr, BufferAdaptor* outputPtr)
  {
    if (!InBufferCheck::checkInputs(inputPtr)) { return false; }
    if (!outputPtr)
    {
      mErrorMessage = NoBuffer;
      return false;
    }
    BufferAdaptor::Access buf(outputPtr);
    if (!buf.exists())
    {
      mErrorMessage = InvalidBuffer;
      return false;
    }
    return true;
  }
};

} // namespace client
} // namespace fluid
