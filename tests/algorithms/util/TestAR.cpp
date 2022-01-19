// #include <catch2/catch_test_macros.hpp>
#include <catch2/catch_all.hpp>
#include "algorithms/util/ARModel.hpp"
#include <string> 
#include <cstdlib> 
#include <fstream> 
#include <array> 
#include <vector>
#include <iostream>
#include <sstream>
#include "burg.h"

TEST_CASE( "AR works", "[artest]" ) {
    // [0.9, 0.6 ,0.5]
    std::vector<double> testdata;         
    std::ifstream datain("/Users/owen/dev/flucoma-core/tests/algorithms/util/ar_test.txt"); 
      if (!datain.is_open()) {
        std::cout << "Couldn't open\n";
        return;   
      }
    
    for(std::string line; std::getline(datain,line);){      
      std::stringstream ss(line);
      double v; 
      ss >> v; 
      testdata.push_back(v);
    }
    
    fluid::algorithm::ARModel ar(3);     
    ar.estimate(testdata.data(), testdata.size(), 0,0); 
    // const double* params = ar.getParameters(); 
    
    std::vector<double> params(4); 
    std::copy(ar.getParameters(),ar.getParameters() + 3,params.begin() + 1); 
    // auto bp = BurgAlgorithm(testdata,3); 
    // auto bp = burg(testdata,3); 
    // const double* params = bp.data(); 
    // double var = ar.variance(); 
    // for(auto&& d: testdata) std::cout << d << '\n'; 
    // std::cout << testdata.size() << '\n'; 
    using Catch::Approx; 
    std::cout << params[0] << '\t'<< params[1] << '\t'<< params[2] << '\t' << params[3] << '\n';
    std::cout << "var\t" << ar.variance() << '\n'; 
    REQUIRE( params[1] == Approx(-0.9).epsilon(0.05) ); 
    REQUIRE( params[2] == Approx(-0.6).epsilon(0.05) ); 
    REQUIRE( params[3] == Approx(-0.5).epsilon(0.05) ); 
}
