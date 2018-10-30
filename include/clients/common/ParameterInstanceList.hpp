
#pragma once

#include "ParameterDescriptorList.hpp"
#include "ParameterInstance.hpp"
#include <vector>

namespace fluid {
namespace client {

class ParameterInstanceList {
  using const_iterator = std::vector<ParameterInstance>::const_iterator;
  using iterator = std::vector<ParameterInstance>::iterator;

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
