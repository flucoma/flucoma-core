#pragma once

#include "data/TensorTypes.hpp"
#include "algorithms/util/FluidEigenMappings.hpp"
#include "algorithms/public/STFT.hpp"
#include <Eigen/Core>
#include <Eigen/Dense>
#include <iostream>
#include <vector>

namespace fluid {
namespace algorithm {

using _impl::asEigen;
using _impl::asFluid;
using Eigen::Matrix;
using Eigen::Array;
using Eigen::ArrayXd;
using Eigen::ArrayXXd;
using Eigen::MatrixXd;
using Eigen::VectorXd;

class NMFCross {


public:
  //pass iteration number; returns true if able to continue (i.e. not cancelled)
  using ProgressCallback = std::function<bool(index)>;

  NMFCross(index nIterations)
      : mIterations(nIterations){}

  static void synthesize(const RealMatrixView h, const ComplexMatrixView w,
                       ComplexMatrixView out){
    using namespace Eigen;
    using namespace _impl;
    //double const epsilon = std::numeric_limits<double>::epsilon();
    MatrixXd H = asEigen<Matrix>(h);
    MatrixXcd W = asEigen<Matrix>(w);

    //double norm = epsilon + W.array().abs().real().sum();
    //W = W /norm;

    MatrixXcd V = H * W;
    out = asFluid(V);
  }

  void process(const RealMatrixView X, RealMatrixView H1,
               RealMatrixView W0, size_t r, size_t p) const{
    //double const epsilon = std::numeric_limits<double>::epsilon();
    index nFrames = X.extent(0);
    index nBins = X.extent(1);
    index rank = W0.extent(0);
    nBins = W0.extent(1);
    MatrixXd W = asEigen<Matrix>(W0).transpose();
    MatrixXd H;
    H = MatrixXd::Random(rank, nFrames) * 0.5 +
          MatrixXd::Constant(rank, nFrames, 0.5);
    MatrixXd V = asEigen<Matrix>(X).transpose();
    multiplicativeUpdates(V, W, H, r, p);
    MatrixXd HT = H.transpose();
    H1 = asFluid(HT);
  }

  void addProgressCallback(ProgressCallback&& callback)
  {
    mCallbacks.emplace_back(std::move(callback));
  }

private:
  index mIterations;
  std::vector<ProgressCallback> mCallbacks;

  std::vector<index> topC(Eigen::VectorXd vec, index c) const{
    using namespace std;
    vector<double> stdVec(vec.data(), vec.data()+vec.size());
    sort(stdVec.begin(), stdVec.end());
    vector<index> idx(vec.size());
    iota(idx.begin(), idx.end(), 0);
    sort(idx.begin(), idx.end(),
       [&vec](size_t i1, size_t i2) {return vec[i1] > vec[i2];});
    auto result = std::vector<index>(idx.begin(), idx.begin() + c);
    return result;
  }



  void multiplicativeUpdates(MatrixXd &V, MatrixXd &W, MatrixXd &H, size_t r, size_t p) const{
    using namespace std;
    using namespace Eigen;
    double const epsilon = std::numeric_limits<double>::epsilon();
    MatrixXd ones = MatrixXd::Ones(V.rows(), V.cols());
    H = H.array().max(epsilon).matrix();
    W = W.array().max(epsilon).matrix();
    //double norm = epsilon + W.sum();
    //W = W /norm;
    //W.colwise().normalize();
    //H.rowwise().normalize();

    for (index i = 0; i < mIterations; ++i)
    {
      if(i % 1 == 0){
        MatrixXd H1 = MatrixXd::Zero(H.rows(), H.cols());
        for (size_t j = 0; j < H.rows(); ++j){
          VectorXd row = H.row(j);
          for (size_t k = 0; k < H.cols(); ++k){
            MatrixXd::Index maxIndex;
            size_t start = k <= r? 0 : k - r;
            size_t length = k + r >= row.size()? row.size() - start - 1: 2 * r;
            VectorXd neighborhood = row.segment(start, length);
            neighborhood.maxCoeff(&maxIndex);
            if(start + maxIndex == k){
              H1(j,k) = H(j,k);
            }
            else{
              H1(j,k) = H(j,k) * (1 - (i + 1) / mIterations);
            }
          }
        }
        MatrixXd H2 = MatrixXd::Zero(H.rows(), H.cols());
        for (size_t k = 0; k < H.cols(); ++k){
          auto col = H1.col(k);
          auto top = topC(col, p);
          H2.col(k) = col * (1 - (i + 1) / mIterations);
          for (auto t: top) H2(t, k) = H1(t, k);
        }
      H = H2;
    }
    ArrayXXd V2 = (W * H).array().max(epsilon);
    ArrayXXd hnum = (W.transpose() * (V.array() / V2).matrix()).array();
    ArrayXXd hden = (W.transpose() * ones).array();
    H = (H.array() * hnum / hden.max(epsilon)).matrix();
    assert(H.allFinite());
    MatrixXd R = W * H;
    R = R.cwiseMax(epsilon);
    //double divergence = (V.cwiseProduct(V.cwiseQuotient(R)) - V + R).sum();
    for(auto& cb:mCallbacks)
      if(!cb(i + 1)) return;
    }
    //H.colwise().normalize();

    V = W * H;
  }
};
} // namespace algorithm
} // namespace fluid
