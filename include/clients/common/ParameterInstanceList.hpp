
#pragma once

#include "FluidParams.hpp"
#include "ParameterConstraints.hpp"
#include "ParameterDescriptor.hpp"
#include <functional>
#include <vector>

namespace fluid {
namespace client {

template <typename> class ParameterDescriptorList;

template <typename ParamEnum> class ParameterInstanceList {

  class ParameterInstance {

//    friend ParameterInstanceList;

  public:
    template<typename T>
    constexpr ParameterInstance(const ParameterDescriptor desc) noexcept
        : mDesc(desc), mHasChanged(false) {
      
//      using t = decltype(desc.template detail<desc.getType()>())::type;
      
      reset();
    }

    ParameterInstance(const ParameterInstance &i) = delete;
    ParameterInstance &operator=(ParameterInstance &i) = delete;
    ParameterInstance(ParameterInstance &&i) = default;
    ParameterInstance &operator=(ParameterInstance &&i) = default;

    template <typename F>
    void addConstraint(ParamEnum other, ParameterInstanceList &list,
                       F comparison, std::string message) {
      switch (mDesc.getType()) {
      case Type::kFloat:
        addFn(mFloat, other, list, comparison, message);
        return;
      case Type::kEnum:
      case Type::kLong:
        addFn(mLong, other, list, comparison, message);
        return;
      case Type::kBuffer:
        addFn(mBuffer.get(), other, list, comparison, message);
        return;
      case Type::kFloatArray:
        addFn(mFloatArray, other, list, comparison, message);
        return;
      case Type::kLongArray:
        addFn(mLongArray, other, list, comparison, message);
        return;
      case Type::kBufferArray:
        addFn(mBufferArray, other, list, comparison, message);
        return;
      }
    }

    template <typename F>
    void addConstraint(F comparison, std::string message) {
      switch (mDesc.getType()) {
      case Type::kFloat:
        addFn(mFloat, comparison, message);
        return;
      case Type::kEnum:
      case Type::kLong:
        addFn(mLong, comparison, message);
        return;
      case Type::kBuffer:
        addFn(mBuffer.get(), comparison, message);
        return;
      case Type::kFloatArray:
        addFn(mFloatArray, comparison, message);
        return;
      case Type::kLongArray:
        addFn(mLongArray, comparison, message);
        return;
      case Type::kBufferArray:
        addFn(mBufferArray, comparison, message);
        return;
      }
    }

    ConstraintResult check() const noexcept {
      for (auto &c : mContraints) {
        ConstraintResult res = c();
        if (!res)
          return res;
      }
      return {true, ""};
    }

    void reset() noexcept {
      switch (mDesc.getType()) {
      case Type::kFloat:
        mFloat = mDesc.getDefault();
        break;
      case Type::kEnum:
      case Type::kLong:
        mLong = mDesc.getDefault();
        break;
      case Type::kBuffer:
        mBuffer = nullptr;
        break;
      case Type::kEnum:
        mLong = mDesc.getDefault();
        break;
      case Type::kFloatArray:
        if (mDesc.getFixedSize())
          std::fill(mFloatArray.begin(), mFloatArray.end(), mDesc.getDefault());
        else
          mFloatArray.clear();
        break;
      case Type::kLongArray:
        if (mDesc.getFixedSize())
          std::fill(mLongArray.begin(), mLongArray.end(), mDesc.getDefault());
        else
          mFloatArray.clear();
        break;
      case Type::kBufferArray:
        if (mDesc.getFixedSize())
          std::for_each(mBufferArray.begin(), mBufferArray.end(),
                        [](std::unique_ptr<BufferAdaptor> &p) { p.reset(); });
        else
          mFloatArray.clear();
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
      case Type::kEnum:
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

    template<typename Iter>
    void set(Iter begin, Iter end) {
      switch (mDesc.getType()){
      case Type::kFloatArray:
        setArray(begin, end, mFloatArray);
      case Type::kLongArray:
        setArray(begin, end, mLongArray);
      case Type::kBufferArray:
        setArray(begin, end, mBufferArray);
      default:
        assert(false && "Don't call this on non-array parameters");
      }
      mHasChanged = true;
    }
//    template <typename F> void set(F(std::vector<long>&) f)    { f(mLongArray);  }
//    template <typename F> void set(F(std::vector<std::unique_ptr<BufferAdaptor>>) f) {}

    double getFloat() const {
      switch (mDesc.getType()) {
      case Type::kFloat:
        return mFloat;
      case Type::kLong:
        return static_cast<double>(mLong);
      default:
        assert(false && "Don't call this on a non-numeric parameter");
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

    long getEnum() const {
      switch (mDesc.getType()) {
      case Type::kEnum:
        return mLong;
      default:
        assert(false && "Don't call this on a non-enum parameter");
      }
      return 0;
    }

    std::vector<double> &getFloatArray() const {
      switch (mDesc.getType()) {
      case Type::kFloatArray:
        return mFloatArray;
      default:
        assert(false && "Don't call this on a non-float-array parameter");
      }
      return emptyArray<double>();
    }

    std::vector<long> &getLongArray() const {
      switch (mDesc.getType()) {
      case Type::kLongArray:
        return mLongArray;
      default:
        assert(false && "Don't call this on a non-long-array parameter");
      }
      return emptyArray<long>();
    }

    std::vector<std::unique_ptr<BufferAdaptor>> &getBufferArray() const {
      switch (mDesc.getType()) {
      case Type::kBufferArray:
        return mBufferArray;
      default:
        assert(false && "Don't call this on a non-buffer-array parameter");
      }
      return emptyArray<std::unique_ptr<BufferAdaptor>>();
    }

    bool hasChanged() const { return mHasChanged; }
    const ParameterDescriptor &descriptor() const { return mDesc; }

  private:
//    constexpr auto descriptorDetail() {
//      //      constexpr std::size_t N  =
//      //      static_cast<std::size_t>(mDesc.getType()); using t_t = typename
//      //      ParameterDescriptor::GetType<N>;
//      return mDesc.detail<mDesc.getType()>();
//    }

    const ParameterDescriptor mDesc;
    
    
    
    bool mHasChanged = false;

    // Menagerie of types
    std::unique_ptr<BufferAdaptor> mBuffer;
    double mFloat;
    long mLong;
    std::vector<std::string> mEnumStrings;
    std::vector<long> mLongArray;
    std::vector<double> mFloatArray;
    std::vector<std::unique_ptr<BufferAdaptor>> mBufferArray;

    std::vector<std::function<ConstraintResult()>>  mContraints;

    template <typename T, typename F>
    void addFn(T v, const ParamEnum other, const ParameterInstanceList &list,
               const F f, const std::string m) {
      mContraints.emplace_back(
          [this, &v, &list, other, f, m]() -> ConstraintResult {
            return {f(v, list[other]), m};
          });
    }

    template <typename T, typename F>
    void addFn(std::vector<T> &v, const ParamEnum other,
               const ParameterInstanceList &list, const F f,
               const std::string m) {
      mContraints.emplace_back(
          [this, &v, &list, other, f, m]() -> ConstraintResult {
            return {f(v, list[other]), m};
          });
    }

    template <typename T, typename F>
    void addFn(T v, const F f, const std::string m) {
      mContraints.emplace_back([this, v, f, m]() -> ConstraintResult {
        return {f(v), m};
      });
    }

    template <typename T, typename F>
    void addFn(std::vector<T> &v, const F f, const std::string m) {
      mContraints.emplace_back([this, &v, f, m]() -> ConstraintResult {
        return {f(v), m};
      });
    }

    template<typename IterIn, typename Array>
    void setArray(IterIn begin, IterIn end, Array& dst)
    {
      if(mDesc.getFixedSize())
        std::copy_n(begin,mDesc.getFixedSize(),dst.begin());
      else
        std::copy(begin,end,dst.begin());
    }
    

    template <typename T> static std::vector<T> &emptyArray() {
      static std::vector<T> empty;
      return empty;
    }
  }; // ParameterInstance

  friend ParameterDescriptor;
  using const_iterator =
      typename std::vector<ParameterInstance>::const_iterator;
  using iterator = typename std::vector<ParameterInstance>::iterator;

public:
  ParameterInstanceList(const ParameterDescriptorList<ParamEnum> &descriptor) {
    for (auto &&d : descriptor)
      mContainer.emplace_back(d);
  }
  
  template<typename ConstraintTagType>
  void addConstraint(ParamEnum first, ParamEnum second, ConstraintTagType c)
  {
//    this[first].addConstraint(makeConstraint(c),
  }
  

  iterator begin() noexcept { return mContainer.begin(); }
  iterator end() noexcept { return mContainer.end(); }
  const_iterator cbegin() const noexcept { return mContainer.cbegin(); }
  const_iterator cend() const noexcept { return mContainer.cend(); }
  size_t size() const noexcept { return mContainer.size(); }

  ParameterInstance &operator[](ParamEnum index) noexcept {
    return mContainer[static_cast<int>(index)];
  }
  const ParameterInstance &operator[](ParamEnum index) const noexcept {
    return mContainer[static_cast<int>(index)];
  }

  iterator lookup(const std::string &name) noexcept {
    return std::find_if(begin(), end(), [&name](ParameterInstance &p) {
      return p.descriptor().getName() == name;
    });
  }

  const_iterator lookup(const std::string &name) const noexcept {
    return std::find_if(begin(), end(), [&name](ParameterInstance &p) {
      return p.descriptor().getName() == name;
    });
  }

  iterator lookup(const ParameterDescriptor &d) noexcept {
    return std::find_if(begin(), end(), [&d](ParameterInstance &p) {
      return p.descriptor() == d;
    });
  }

  const_iterator lookup(const ParameterDescriptor &d) const {
    return std::find_if(begin(), end(), [&d](ParameterInstance &p) {
      return p.descriptor() == d;
    });
  }

  template <typename T> void set(std::string name, T value) {
    auto i = lookup(name);

    if (i != end()) {
      i->set(value);
    }
  }

  template <typename T> void set(ParamEnum index, T value) {
    auto i = *this[index];

    i.set(value);
  }

  void reset(std::string name) {
    auto i = lookup(name);

    if (i != end()) {
      i->reset();
    }
  }

  double getFloat(std::string name) const {
    return get(name, &ParameterInstance::getFloat, 0.0);
  }

  long getLong(std::string name) const {
    return get(name, &ParameterInstance::getLong, 0L);
  }

  BufferAdaptor *getBuffer(std::string name) const {
    return get<BufferAdaptor *>(name, &ParameterInstance::getBuffer, nullptr);
  }

private:
  template <typename T> using GetMethod = T (ParameterInstance::*)() const;

  template <typename T>
  T get(std::string name, GetMethod<T> Method, T falseReturn) const {
    const_iterator i = lookup(name);
    return i != cend() ? (*i.*Method)() : falseReturn;
  }

  std::vector<ParameterInstance> mContainer;
};

} // namespace client
} // namespace fluid
