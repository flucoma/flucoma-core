#pragma once

#include "CommonResults.hpp"
#include <clients/common/BufferAdaptor.hpp>
#include <clients/common/Result.hpp>

namespace fluid {
namespace client {

class ClientInputCheck {

public:
  std::string error() { return mErrorMessage; }
protected:
  std::string mErrorMessage;
};

class InBufferCheck : public ClientInputCheck {
public:
  InBufferCheck(index size) : mInputSize(size){};
  bool checkInputs(BufferAdaptor* inputPtr) {
    if (!inputPtr) {
      mErrorMessage = NoBuffer;
      return false;
    }
    BufferAdaptor::ReadAccess buf(inputPtr);
    if (!buf.exists() || !buf.valid()) {
      mErrorMessage = InvalidBuffer;
      return false;
    }
    if (buf.numFrames() != mInputSize) {
      mErrorMessage = WrongPointSize;
      return false;
    }
    return true;
  }

protected:
  index mInputSize;
};

class InOutBuffersCheck : public InBufferCheck {

public:
  using InBufferCheck::InBufferCheck;
  bool checkInputs(BufferAdaptor *inputPtr, BufferAdaptor *outputPtr) {
    if(!InBufferCheck::checkInputs(inputPtr)){return false;}
    if (!outputPtr) { mErrorMessage = NoBuffer;return false;}
    BufferAdaptor::Access buf(outputPtr);
    if (!buf.exists()) { mErrorMessage = InvalidBuffer; return false;}
    return true;
  }
};

} // namespace client
} // namespace fluid
