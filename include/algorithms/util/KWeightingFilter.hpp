#pragma once
#include <cmath>
#include <iostream>

namespace fluid {
namespace algorithm {

class KWeightingFilter {
public:
  void init(double sampleRate) {
    // from https://github.com/jiixyj/libebur128/blob/master/ebur128/ebur128.c

    // Shelving filter
    double f0 = 1681.974450955533;
    double G = 3.999843853973347;
    double Q = 0.7071752369554196;

    double K = std::tan(M_PI * f0 / sampleRate);
    double Vh = std::pow(10.0, G / 20.0);
    double Vb = std::pow(Vh, 0.4996667741545416);

    double shelvB[3] = {0.0, 0.0, 0.0};
    double shelvA[3] = {1.0, 0.0, 0.0};

    double a0 = 1.0 + K / Q + K * K;
    shelvB[0] = (Vh + Vb * K / Q + K * K) / a0;
    shelvB[1] = 2.0 * (K * K - Vh) / a0;
    shelvB[2] = (Vh - Vb * K / Q + K * K) / a0;
    shelvA[1] = 2.0 * (K * K - 1.0) / a0;
    shelvA[2] = (1.0 - K / Q + K * K) / a0;

    // Hi-pass filter
    f0 = 38.13547087602444;
    Q = 0.5003270373238773;
    K = tan(M_PI * f0 / sampleRate);

    double hiB[3] = {1.0, -2.0, 1.0};
    double hiA[3] = {1.0, 0.0, 0.0};

    hiA[1] = 2.0 * (K * K - 1.0) / (1.0 + K / Q + K * K);
    hiA[2] = (1.0 - K / Q + K * K) / (1.0 + K / Q + K * K);

    // Combined
    mB[0] = shelvB[0] * hiB[0];
    mB[1] = shelvB[0] * hiB[1] + shelvB[1] * hiB[0];
    mB[2] = shelvB[0] * hiB[2] + shelvB[1] * hiB[1] + shelvB[2] * hiB[0];
    mB[3] = shelvB[1] * hiB[2] + shelvB[2] * hiB[1];
    mB[4] = shelvB[2] * hiB[2];

    mA[0] = shelvA[0] * hiA[0];
    mA[1] = shelvA[0] * hiA[1] + shelvA[1] * hiA[0];
    mA[2] = shelvA[0] * hiA[2] + shelvA[1] * hiA[1] + shelvA[2] * hiA[0];
    mA[3] = shelvA[1] * hiA[2] + shelvA[2] * hiA[1];
    mA[4] = shelvA[2] * hiA[2];
    for (int i = 0; i < 5; i++) {
      mX[i] = 0;
      mY[i] = 0;
    }
    //std::cout << sampleRate << std::endl;
    //std::cout << K << std::endl;
    for (int i = 0; i < 5; i++) {
      //std::cout << mA[i] << std::endl;
      //std::cout << mB[i] << std::endl;
    }
    for (int i = 0; i < 5; i++) {
      //std::cout << mX[i] << std::endl;
      //std::cout << mY[i] << std::endl;
    }
    //std::cout << "------"<< std::endl;
  //  std::cout << "------"<< std::endl;
  }

  double processSample(double x) {
    double y = 0;
    mX[0] = x;
    for (int i = 1; i < 5; i++) y-=mA[i] * mY[i - 1];
    for (int i = 0; i < 5; i++) y+=mB[i] * mX[i];

    /*for (int i = 0; i < 5; i++)
      y = y + mB[i] * mX[i] - mA[i] * mY[i];*/
    for (int i = 4; i > 0; i--) {
      mX[i] = mX[i - 1];
      mY[i] = mY[i - 1];
    }
    mY[0] = y;
    //std::cout << x << std::endl;
    //std::cout << y << std::endl;
    //std::cout << "------"<< std::endl;
    for (int i = 0; i < 5; i++) {
      //std::cout << mX[i] << std::endl;
      //std::cout << mY[i] << std::endl;
    }
    //std::cout << "------"<< std::endl;
    //std::cout << "------"<< std::endl;
    return y;
  }

private:
  double mA[5], mB[5], mX[5], mY[5];
};

}; // namespace algorithm
}; // namespace fluid