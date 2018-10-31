
#pragma once

#include "ParameterDescriptorList.hpp"
#include "ParameterInstance.hpp"
#include <vector>

namespace fluid {
namespace client {

class ParameterInstanceList {
  using const_iterator = std::vector<ParameterInstance>::const_iterator;
  using iterator = std::vector<ParameterInstance>::iterator;

  class ParameterInstance {
    using Type = ParameterDescriptor::Type;
    friend ParameterInstanceList; 
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
  
  
public:
  ParameterInstanceList(const ParameterDescriptorList &descriptor) {
    for (auto &&d : descriptor)
      mContainer.emplace_back(d);
  }

  iterator begin() { return mContainer.begin(); }
  iterator end() { return mContainer.end(); }
  const_iterator cbegin() const { return mContainer.cbegin(); }
  const_iterator cend() const { return mContainer.cend(); }
  size_t size() const { return mContainer.size(); }

  ParameterInstance &operator[](size_t index) { return mContainer[index]; }
  const ParameterInstance &operator[](size_t index) const {
    return mContainer[index];
  }

  iterator lookup(std::string name) {
    for (auto it = begin(); it != end(); it++)
      if (it->descriptor().getName() == name)
        return it;

    return end();
  }

  const_iterator lookup(std::string name) const {
    for (auto it = cbegin(); it != cend(); it++)
      if (it->descriptor().getName() == name)
        return it;

    return cend();
  }

private:
  std::vector<ParameterInstance> mContainer;
};

} // namespace client
} // namespace fluid
