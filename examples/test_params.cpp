//
//  testparams.cpp
//  fdNMF
//
//  Created by Owen Green on 30/08/2018.
//

#include "clients/nrt/NMFClient.hpp"

#include <stdio.h>

int main(int argc, char* argv[])
{
  using fluid::nmf::NMFClient;
  
  
  for(auto&& p: NMFClient::getParamDescriptors())
  {
    std::cout << p << '\n';
  }
  
  
  std::cout << size_t(-1  - std::numeric_limits<size_t>::max()); 
  
  
  return 0;
}

