#include <clients/nrt/NMFClient.hpp>

#include <cstdio>

int main(int argc, char *argv[]) {
  using fluid::nmf::NMFClient;

  for (auto &&p : NMFClient::getParamDescriptors()) {
    std::cout << p << '\n';
  }

  std::cout << size_t(-1 - std::numeric_limits<size_t>::max());

  return 0;
}
