#pragma once
#include "KDTree.hpp"
#include "algorithms/DistanceFuncs.hpp"
#include "algorithms/SpectralEmbedding.hpp"
#include "algorithms/util/FluidEigenMappings.hpp"
#include "data/TensorTypes.hpp"
#include <cassert>
#include <cmath>
#include <random>
#include <unsupported/Eigen/NonLinearOptimization>
#include <unsupported/Eigen/NumericalDiff>
#include <Eigen/Core>
#include <Eigen/Sparse>
#include <random>

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
    mX = Eigen::ArrayXd::LinSpaced(values(), 0,
                                   3 * spread); // TODO: spread parameter?
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
  using VectorXd = Eigen::VectorXd;
  using Permutation = Eigen::PermutationMatrix<Eigen::Dynamic, Eigen::Dynamic>;
  using SparseMatrixXd = Eigen::SparseMatrix<double>;
  using DataSet = FluidDataSet<std::string, double, 1>;

  double loss(Eigen::Ref<ArrayXXd> P, Eigen::Ref<ArrayXXd> Y, double a,
              double b) {
    using namespace Eigen;
    ArrayXXd D = DistanceMatrix(Y, 2);
    ArrayXXd Q = 1 / (1 + a * D.pow(b));
    Q = Q + epsilon;
    ArrayXXd CE = -P * (Q+0.01).log() - (1 - P) * (1e-6 + (1 - Q+ 0.01)).log();
    return CE.sum();
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

  void computeHighDimProb(ArrayXd sigma) {
    for (int i = 0; i < mKNNGraph.outerSize(); i++){
      for (SparseMatrixXd::InnerIterator it(mKNNGraph, i); it; ++it){
        it.valueRef() = std::exp(-(it.value() - mDists(it.row(), 0)) / sigma(it.row()));
      }
    }
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

  void makeGraph(DataSet &in, index k) {
    algorithm::KDTree tree(in);
    mKNNGraph = SparseMatrixXd(in.size(), in.size());
    mKNNGraph.reserve(in.size() * k);
    mDists = ArrayXXd::Zero(in.size(), k);
    auto data = in.getData();
    for (index i = 0; i < in.size(); i++) {
      auto nearest = tree.kNearest(data.row(i), k + 1); // discard self
      auto nearestIds = nearest.getIds();
      auto distances = nearest.getData().col(0);
      for (index j = 1; j < k; j++) {
        index neighborIndex = in.getIndex(nearestIds(j));
        mDists(i, j - 1) = distances(j);
        mKNNGraph.insert(i, neighborIndex) = distances(j);
      }
    }
  }

  ArrayXXd normalizeEmbedding(Eigen::Ref<ArrayXXd> embedding){
   // based on umap python implementation
    double  expansion = 10.0 / embedding.abs().maxCoeff();
    ArrayXXd noise = 1e-4 * ArrayXXd::Random(embedding.rows(), embedding.cols()); // uniform
    ArrayXXd result = (embedding * expansion) + noise;
    ArrayXd min = result.colwise().minCoeff();
    ArrayXd max = result.colwise().maxCoeff();
    ArrayXd range =(max - min).max(epsilon);
    result = (result.rowwise() - min.transpose());
    result =  result.rowwise() / range.transpose();
    return 10.0 * result;
  }

  DataSet process(DataSet &in, index k = 15, index dims = 2,
                  double minDist = 0.1, index maxIter = 200,
                  double learningRate = 1.0) {
    using namespace Eigen;
    using namespace _impl;
    using namespace std;
    index n = in.size();
    random_device rd;
    mt19937 mt(rd());
    uniform_int_distribution<index> randomInt(0, n - 1);
    double negativeSampleRate = 5.0;
    makeGraph(in, k);
    ArrayXd sigma = findSigma(k);
    computeHighDimProb(sigma);
    SparseMatrixXd mKNNGraphT = mKNNGraph.transpose();
    mKNNGraph = (mKNNGraph + mKNNGraphT) - mKNNGraph.cwiseProduct(mKNNGraphT);
    auto ab = findAB(minDist);
    double a = ab(0), b = ab(1);
    ArrayXXd Y = mSpectralEmbedding.process(mKNNGraph, dims);
    Y = normalizeEmbedding(Y);
    //MatrixXd P = MatrixXd(mKNNGraph);
    mKNNGraph.makeCompressed();
    double maxVal = mKNNGraph.coeffs().maxCoeff();
    ArrayXi rowIndices(mKNNGraph.nonZeros());
    ArrayXi colIndices(mKNNGraph.nonZeros());
    ArrayXd epochsPerSample(mKNNGraph.nonZeros());

    index p = 0;
    for (int i = 0; i < mKNNGraph.outerSize(); i++){
      for (SparseMatrixXd::InnerIterator it(mKNNGraph, i); it; ++it){
        rowIndices(p) = it.row();
        colIndices(p) = it.col();
        epochsPerSample(p) = 1.0 / (it.value() / maxVal);
        p++;
      }
    }
    epochsPerSample = (epochsPerSample == 0).select(-1, epochsPerSample);
    ArrayXd epochsPerNegativeSample = epochsPerSample / negativeSampleRate;
    ArrayXd nextEpoch = epochsPerSample;
    ArrayXd nextNegEpoch = epochsPerNegativeSample;

    double alpha = learningRate;
    double gamma = 1.0;
    ArrayXd bound = VectorXd::Constant(dims, 4);//based on umap python implementation
    for (index i = 0; i < maxIter; i++) {
      for(index j = 0; j < mKNNGraph.nonZeros(); j++){
        if(nextEpoch(j) > i) continue;
        ArrayXd current = Y.row(rowIndices(j));
        ArrayXd other = Y.row(colIndices(j));
        double dist = DistanceFuncs::map()[DistanceFuncs::Distance::kSqEuclidean](current, other);
        double gradCoef = 0;
        ArrayXd grad;
        if (dist > 0){
            gradCoef =  -2.0 * a * b * pow(dist, b - 1.0);
            gradCoef /= a * pow(dist, b) + 1.0;
        }
        grad = (gradCoef * (current - other)).cwiseMin(bound).cwiseMax(-bound);
        current += grad * alpha;
        other += -grad * alpha; //TODO: disable when processing new points
        nextEpoch(j) += epochsPerSample(j);
        size_t numNegative = static_cast<size_t>((i - nextNegEpoch(j)) / epochsPerNegativeSample(j));
        for(index k = 0; k < numNegative; k++){
          index negativeIndex = randomInt(mt);
          if(negativeIndex == rowIndices(j)) continue;
          ArrayXd negative = Y.row(negativeIndex);
          dist = DistanceFuncs::map()[DistanceFuncs::Distance::kSqEuclidean](current, negative);
          gradCoef = 0;
          grad = VectorXd::Constant(dims, 4.0);
          if (dist > 0){
            gradCoef =  2.0 * gamma * b;
            gradCoef /= (0.001 + dist) * (a * pow(dist, b) + 1);
            grad = (gradCoef * (current - negative)).cwiseMin(bound).cwiseMax(-bound);
          }
          current += grad * alpha;
        }
        nextNegEpoch(j)+= numNegative * epochsPerNegativeSample(j);
        Y.row(rowIndices(j)) = current;
        Y.row(colIndices(j)) = other;
      }
      alpha = learningRate * (1.0 - (i / double(maxIter)));
    }
    DataSet out(in.getIds(), _impl::asFluid(Y));
    return out;
  }

private:
  SpectralEmbedding mSpectralEmbedding;
  SparseMatrixXd mKNNGraph;
  ArrayXXd mDists;
};
}; // namespace algorithm
}; // namespace fluid
