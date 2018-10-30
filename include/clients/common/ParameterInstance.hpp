
#pragma once

#include "ParameterDescriptor.hpp"
#include <memory>

namespace fluid {
namespace client {

class ParameterInstance {
  using Type = ParameterDescriptor::Type;

public:
  ParameterInstance(ParameterDescriptor desc)
      : mDesc(desc), mHasChanged(false) {
    reset();
  }

  ParameterInstance(const ParameterInstance &i) = delete;
  ParameterInstance &operator=(ParameterInstance &i) = delete;
  ParameterInstance(ParameterInstance &&i) = default;
  ParameterInstance &operator=(ParameterInstance &&i) = default;

  void reset() {
    switch (mDesc.getType()) {
    case Type::kFloat:
      mFloat = mDesc.getDefault();
      break;
    case Type::kLong:
      mLong = mDesc.getDefault();
      break;
    case Type::kBuffer:
      mBuffer = nullptr;
      break;
    default:
      break;
    }
    mHasChanged = false;
  }

  void set(double v) {
    switch (mDesc.getType()) {
    case Type::kFloat:
      mFloat = v;
      break;
    case Type::kLong:
      mLong = static_cast<long>(v);
      break;
    default:
      assert(false && "Don't call this on this type of parameter");
    }
    mHasChanged = true;
  }

  void set(long v) {
    switch (mDesc.getType()) {
    case Type::kFloat:
      mFloat = static_cast<double>(v);
      break;
    case Type::kLong:
      mLong = v;
      break;
    default:
      assert(false && "Don't call this on this type of parameter");
    }
    mHasChanged = true;
  }

  void set(BufferAdaptor *p) {
    switch (mDesc.getType()) {
    case Type::kBuffer:
      mBuffer = std::unique_ptr<BufferAdaptor>(p);
      break;
    default:
      assert(false && "Don't call this on a non-buffer parameter");
    }
    mHasChanged = true;
  }

  double getFloat() const {
    switch (mDesc.getType()) {
    case Type::kFloat:
      return mFloat;
    case Type::kLong:
      return static_cast<double>(mLong);
    default:
      assert(false && "Don't call this on a non-buffer parameter");
    }
    return 0.0;
  }

  long getLong() const {
    switch (mDesc.getType()) {
    case Type::kFloat:
      return static_cast<long>(mFloat);
    case Type::kLong:
      return mLong;
    default:
      assert(false && "Don't call this on a buffer parameter");
    }
    return 0;
  }

  BufferAdaptor *getBuffer() const {
    switch (mDesc.getType()) {
    case Type::kBuffer:
      return mBuffer.get();
    default:
      assert(false && "Don't call this on a non-buffer parameter");
    }
    return nullptr;
  }

  bool hasChanged() const { return mHasChanged; }

  const ParameterDescriptor &descriptor() const { return mDesc; }

private:
  const ParameterDescriptor mDesc;
  bool mHasChanged = false;
  std::unique_ptr<BufferAdaptor> mBuffer;
  double mFloat;
  long mLong;
}; // ParameterInstance

} // namespace client
} // namespace fluid
