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
  bool checkInputs(BufferAdaptor *inputPtr) {
    if (!inputPtr) {
      mErrorMessage = NoBuffer;
      return false;
    }
    mIn = BufferAdaptor::ReadAccess(inputPtr);
    if (!mIn.exists() || !mIn.valid()) {
      mErrorMessage = InvalidBuffer;
      return false;
    }
    if (mIn.numFrames() != mInputSize) {
      mErrorMessage = WrongPointSize;
      return false;
    }
    return true;
  }
  BufferAdaptor::ReadAccess &in() { return mIn; }

protected:
  index mInputSize;
  BufferAdaptor::ReadAccess mIn{nullptr};
};

class InOutBuffersCheck : public InBufferCheck {

public:
  using InBufferCheck::InBufferCheck;
  bool checkInputs(BufferAdaptor *inputPtr, BufferAdaptor *outputPtr) {
    if(!InBufferCheck::checkInputs(inputPtr)){return false;}
    if (!outputPtr) { mErrorMessage = NoBuffer;return false;}
    mOut = BufferAdaptor::Access(outputPtr);
    if (!mOut.exists()) { mErrorMessage = InvalidBuffer; return false;}
    return true;
  }
  BufferAdaptor::Access &out() { return mOut; }

private:
  BufferAdaptor::Access mOut{nullptr};
};

} // namespace client
} // namespace fluid
