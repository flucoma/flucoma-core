#pragma once

#include "KDTree.hpp"
#include "algorithms/DistanceFuncs.hpp"
#include "algorithms/util/FluidEigenMappings.hpp"
#include "data/TensorTypes.hpp"
#include <unsupported/Eigen/NonLinearOptimization>
#include <unsupported/Eigen/NumericalDiff>

#include <Eigen/Core>
//#include <Eigen/Sparse> // TODO: use sparse matrix for mGraph
#include <Eigen/SVD>
#include <cassert>
#include <cmath>
#include <fstream>

namespace fluid {
namespace algorithm {

struct UMAPEmbeddingParamsFunctor {
  typedef double Scalar;
  enum {
    InputsAtCompileTime = 2,
    ValuesAtCompileTime = 300 // from UMAP python implementation
  };
  typedef Eigen::VectorXd InputType;
  typedef Eigen::VectorXd ValueType;
  typedef Eigen::MatrixXd JacobianType;

  UMAPEmbeddingParamsFunctor(double minDist, double spread = 1.0) {
    mX = Eigen::ArrayXd::LinSpaced(values(), 0, 3 * spread);// TODO: spread parameter?
    mY = (mX <= minDist).select(1, ((-mX + minDist) / spread).exp());
  }

  int operator()(const Eigen::VectorXd &x, Eigen::VectorXd &fvec) const {
    double a = x(0);
    double b = x(1);
    fvec = mY - (1 / (1 + a * mX.pow(2 * b)));
    return 0;
  }

  int values() const { return ValuesAtCompileTime; }
  int inputs() const { return InputsAtCompileTime; }

  Eigen::ArrayXd mX;
  Eigen::ArrayXd mY;
};

class UMAP {
public:
  using MatrixXd = Eigen::MatrixXd;
  using ArrayXXd = Eigen::ArrayXXd;
  using ArrayXd = Eigen::ArrayXd;
  using ArrayXXi = Eigen::ArrayXXi;
  using VectorXd = Eigen::VectorXd;
  // using SMatrixXd = Eigen::SparseMatrix<double>;
  using DataSet = FluidDataSet<std::string, double, 1>;

  /*
  double loss(Eigen::Ref<ArrayXXd> P, Eigen::Ref<ArrayXXd> Y, double a, double
  b){ using namespace Eigen; ArrayXXd D = DistanceMatrix(Y, 2); ArrayXXd Q = 1 /
  (1 + a * D.pow(b)); Q = Q + epsilon; ArrayXXd CE = - P * Q.log() - (1 - P) *
  (1e-6 + (1-Q)).log(); return CE.sum();
  }*/

  // https://towardsdatascience.com/how-to-program-umap-from-scratch-e6eff67f55fe
  ArrayXXd gradient(Eigen::Ref<ArrayXXd> P, Eigen::Ref<ArrayXXd> Y, double a,
                    double b) {
    using namespace Eigen;
    ArrayXXd D = DistanceMatrix(Y, 2);
    D.matrix().diagonal().setConstant(epsilon);
    ArrayXXd invDist = 1 / (1 + a * D.pow(b));
    MatrixXd Q = (1 - P).matrix() * (1 / D).matrix();
    Q.diagonal().setZero();
    Q = Q.array().rowwise() / Q.array().rowwise().sum().transpose();
    ArrayXXd fact = a * P * D.pow(b - 1) - Q.array();
    ArrayXXd result(Y.rows(), Y.cols());
    for (index i = 0; i < Y.rows(); i++) {
      ArrayXXd rowFact = fact.row(i).replicate(Y.cols(), 1).transpose();
      ArrayXXd rowInvDist = invDist.row(i).replicate(Y.cols(), 1).transpose();
      ArrayXXd yDiff = Y.row(i).replicate(Y.rows(), 1) - Y;
      result.row(i) = 2 * b * (rowFact * rowInvDist * yDiff).colwise().sum();
    }
    return result;
  }

  ArrayXd findSigma(index k, index maxIter = 64, double tolerance = 1e-5) {
    using namespace std;
    double target = log2(k);
    ArrayXd result = ArrayXd::Zero(mDists.rows());
    for (index i = 0; i < mDists.rows(); i++) {
      index iter = maxIter;
      double lo = 0;
      double hi = infinity;
      double mid = 1.0;
      double rho = mDists(i, 0);
      while (iter-- > 0) {
        double pSum = 0;
        for (index j = 1; j < mDists.cols(); j++) {
          double d = mDists(i, j) - rho;
          pSum += (d <= 0 ? 1.0 : exp(-(d / mid)));
        }
        if (abs(pSum - target) < tolerance)
          break;

        if (pSum > target) {
          hi = mid;
          mid = (lo + hi) / 2.0;
        } else {
          lo = mid;
          mid = (hi == infinity ? mid * 2 : (lo + hi) / 2.0);
        }
      }
      result(i) = mid;
    }
    return result;
  }

  ArrayXXd highDimProb(ArrayXd sigma) {
    ArrayXXd D = mGraph.array().colwise() - mDists.col(0);
    D = (D > 0).select(D, 0);
    D = D.rowwise() / sigma.transpose();
    ArrayXXd P = (-D).exp();
    P = (mGraph.array() > 0).select(P, 0);
    return P;
  }

  VectorXd findAB(double minDist) {
    using namespace Eigen;
    VectorXd ab(2);
    ab << 1.0, 1.0;
    UMAPEmbeddingParamsFunctor functor(minDist);
    NumericalDiff<UMAPEmbeddingParamsFunctor> numDiff(functor);
    Eigen::LevenbergMarquardt<NumericalDiff<UMAPEmbeddingParamsFunctor>> lm(
        numDiff);
    lm.minimize(ab);
    return ab;
  }

  ArrayXXd computeSpectralEmbedding(Eigen::Ref<ArrayXXd> P, index dims) {
    using namespace Eigen;
    MatrixXd D = P.matrix().colwise().sum().asDiagonal();
    MatrixXd Di = (D.array() > 0).select(1 / D.array().sqrt(), 0);
    Di = (D.array() == 0).select(0, Di);
    MatrixXd L = (Di * (D - P.matrix())) * Di;
    BDCSVD<MatrixXd> svd(L, ComputeThinV | ComputeThinU);
    MatrixXd U = svd.matrixU().rowwise().reverse();
    ArrayXXd Y = U.block(0, 1, U.rows(), dims).array();
    return Y;
  }

  void makeGraph(DataSet &in, index k) {
    algorithm::KDTree tree(in);
    mGraph = MatrixXd::Zero(in.size(), in.size());
    mDists = ArrayXXd::Zero(in.size(), k);
    auto data = in.getData();
    for (index i = 0; i < in.size(); i++) {
      auto nearest = tree.kNearest(data.row(i), k + 1); // discard self
      auto nearestIds = nearest.getIds();
      auto distances = nearest.getData().col(0);
      for (index j = 1; j < k; j++) {
        index neighborIndex = in.getIndex(nearestIds(j));
        mDists(i, j - 1) = distances(j);
        mGraph(i, neighborIndex) = distances(j);
      }
    }
  }

  DataSet process(DataSet &in, index k = 15, index dims = 2,
                  double minDist = 0.1, index maxIter = 200,
                  double learningRate = 0.1) {
    using namespace Eigen;
    using namespace _impl;
    makeGraph(in, k);
    ArrayXd sigma = findSigma(k);
    ArrayXXd P = highDimProb(sigma);
    P = (P + P.transpose()) - (P * P.transpose());
    auto ab = findAB(minDist);
    double a = ab(0), b = ab(1);
    ArrayXXd Y = computeSpectralEmbedding(P, dims);
    while (maxIter--) {
      MatrixXd G = gradient(P, Y, a, b);
      Y = Y - learningRate * G.array();
    }
    FluidTensor<double, 2> result(in.size(), dims);
    auto ids = in.getIds();
    result = _impl::asFluid(Y);
    FluidDataSet<std::string, double, 1> ds(2);
    for (index i = 0; i < in.size(); i++) {
      ds.add(ids(i), result.row(i));
    }
    return ds;
  }

private:
  MatrixXd mGraph;
  ArrayXXd mDists;
};
}; // namespace algorithm
}; // namespace fluid
