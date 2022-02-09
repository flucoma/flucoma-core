#pragma once

namespace fluid {
template <typename T, typename Tag>
struct StrongType
{
  StrongType(T val) : mValue{val} {}
           operator const T&() const { return mValue; }
  const T& operator()() const { return mValue; }

private:
  T mValue;
};
} // namespace fluid
