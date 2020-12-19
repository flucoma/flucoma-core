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
    mX = Eigen::ArrayXd::LinSpaced(values(), 0, 3 * spread);
    mY = (mX <= minDist).select(1, ((-mX + minDist) / spread).exp());
  }

  int operator()(const Eigen::VectorXd &x, Eigen::VectorXd &fvec) const {
    fvec = mY - (1 / (1 + x(0) * mX.pow(2 * x(1))));
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
  using ArrayXi = Eigen::ArrayXi;
  using VectorXd = Eigen::VectorXd;
  using Permutation = Eigen::PermutationMatrix<Eigen::Dynamic, Eigen::Dynamic>;
  using SparseMatrixXd = Eigen::SparseMatrix<double>;
  using DataSet = FluidDataSet<std::string, double, 1>;
  template <typename T> using Ref = Eigen::Ref<T>;

  double loss(Ref<ArrayXXd> P, Ref<ArrayXXd> Y, double a, double b) {
    ArrayXXd D = DistanceMatrix(Y, 2);
    ArrayXXd Q = 1 / (1 + a * D.pow(b));
    Q = Q + epsilon;
    ArrayXXd CE = -P * (Q+0.01).log() - (1 - P) * (1e-6 + (1 - Q+ 0.01)).log();
    return CE.sum();
  }

  ArrayXd findSigma(index k, Ref<ArrayXXd> dists,
                    index maxIter = 64, double tolerance = 1e-5) {
    using namespace std;
    double target = log2(k);
    ArrayXd result = ArrayXd::Zero(dists.rows());
    for (index i = 0; i < dists.rows(); i++) {
      index iter = maxIter;
      double lo = 0;
      double hi = infinity;
      double mid = 1.0;
      double rho = dists(i, 0);
      while (iter-- > 0) {
        double pSum = 0;
        for (index j = 1; j < dists.cols(); j++) {
          double d = dists(i, j) - rho;
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

  void computeHighDimProb(
    const Ref<ArrayXXd>& dists,
    const Ref<ArrayXd>& sigma,
    SparseMatrixXd& graph) {
      for (int i = 0; i < graph.outerSize(); i++){
        for (SparseMatrixXd::InnerIterator it(graph, i); it; ++it){
          it.valueRef() = std::exp(
            -(it.value() - dists(it.row(), 0)) / sigma(it.row())
          );
        }
      }
  }

  VectorXd findAB(double minDist) {
    using namespace Eigen;
    VectorXd ab(2);
    ab << 1.0, 1.0;
    UMAPEmbeddingParamsFunctor functor(minDist);
    NumericalDiff<UMAPEmbeddingParamsFunctor> numDiff(functor);
    LevenbergMarquardt<NumericalDiff<UMAPEmbeddingParamsFunctor>> lm(numDiff);
    lm.minimize(ab);
    return ab;
  }

  void makeGraph(
    const DataSet &in, const KDTree& tree, index k,
    SparseMatrixXd& graph, Ref<ArrayXXd> dists,
    bool discardFirst)
    {
    graph.reserve(in.size() * k);
    auto data = in.getData();
    for (index i = 0; i < in.size(); i++) {
      auto nearest = mTree.kNearest(data.row(i), discardFirst? k + 1:k);
      auto nearestIds = nearest.getIds();
      auto distances = nearest.getData().col(0);
      for (index j = 0; j < k; j++) {
        index pos = discardFirst?j+1:j;
        index neighborIndex = in.getIndex(nearestIds(pos));
        dists(i, j) = distances(pos);
        graph.insert(i, neighborIndex) = distances(pos);
      }
    }
  }

  ArrayXXd normalizeEmbedding(const Ref<ArrayXXd>& embedding){
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

void getGraphIndices(
    const SparseMatrixXd& graph,
    Ref<ArrayXi> rowIndices,
    Ref<ArrayXi> colIndices)
{
      index p = 0;
      for (int i = 0; i < mKNNGraph.outerSize(); i++){
        for (SparseMatrixXd::InnerIterator it(graph, i); it; ++it){
          rowIndices(p) = it.row();
          colIndices(p) = it.col();
          p++;
        }
      }
}

  void computeEpochsPerSample(
    const SparseMatrixXd& graph,
    Ref<ArrayXd> epochsPerSample)
  {
    index p = 0;
    double maxVal = graph.coeffs().maxCoeff();
    for (int i = 0; i < graph.outerSize(); i++){
      for (SparseMatrixXd::InnerIterator it(graph, i); it; ++it){
        epochsPerSample(p++) = 1.0 / (it.value() / maxVal);
      }
    }
  }

  void optimizeLayout(Ref<ArrayXXd> embedding, Ref<ArrayXXd> reference,
    Ref<ArrayXi> embIndices, Ref<ArrayXi> refIndices, Ref<ArrayXd> epochsPerSample,
    bool updateReference, double learningRate, index maxIter, double gamma = 1.0){
    using namespace std;
    double alpha = learningRate;
    double negativeSampleRate = 5.0;
    auto distance = DistanceFuncs::map()[DistanceFuncs::Distance::kSqEuclidean];
    double a = mAB(0);
    double b = mAB(1);
    random_device rd;
    mt19937 mt(rd());
    uniform_int_distribution<index> randomInt(0, reference.rows() - 1);
    ArrayXd epochsPerNegativeSample = epochsPerSample / negativeSampleRate;
    ArrayXd nextEpoch = epochsPerSample;
    ArrayXd nextNegEpoch = epochsPerNegativeSample;
    ArrayXd bound = VectorXd::Constant(embedding.cols(), 4);//based on umap python implementation
    for (index i = 0; i < maxIter; i++) {
      for(index j = 0; j < epochsPerSample.size(); j++){
        if(nextEpoch(j) > i) continue;
        ArrayXd current = embedding.row(embIndices(j));
        ArrayXd other = reference.row(refIndices(j));
        double dist = distance(current, other);//todo: try to have dist member
        double gradCoef = 0;
        ArrayXd grad;
        if (dist > 0){
            gradCoef =  -2.0 * a * b * pow(dist, b - 1.0);
            gradCoef /= a * pow(dist, b) + 1.0;
        }
        grad = (gradCoef * (current - other)).cwiseMin(bound).cwiseMax(-bound);
        current += grad * alpha;
        if(updateReference) other += -grad * alpha;
        nextEpoch(j) += epochsPerSample(j);
        size_t numNegative = static_cast<size_t>(
          (i - nextNegEpoch(j)) / epochsPerNegativeSample(j)
        );
        for(index k = 0; k < numNegative; k++){
          index negativeIndex = randomInt(mt);
          if(negativeIndex == embIndices(j)) continue;
          ArrayXd negative = reference.row(negativeIndex);
          dist = distance(current, negative);
          gradCoef = 0;
          grad = VectorXd::Constant(reference.cols(), 4.0);
          if (dist > 0){
            gradCoef =  2.0 * gamma * b;
            gradCoef /= (0.001 + dist) * (a * pow(dist, b) + 1);
            grad = (gradCoef * (current - negative)).cwiseMin(bound).cwiseMax(-bound);
          }
          current += grad * alpha;
        }
        nextNegEpoch(j) += numNegative * epochsPerNegativeSample(j);
        embedding.row(embIndices(j)) = current;
        if(updateReference) reference.row(refIndices(j)) = other;
      }
      alpha = learningRate * (1.0 - (i / double(maxIter)));
    }
  }

  ArrayXXd initTransformEmbedding(
    const SparseMatrixXd& graph,
    Ref<ArrayXXd> reference, index N) {
    ArrayXXd embedding = ArrayXXd::Zero(N, reference.cols());
    ArrayXd sums = ArrayXd::Zero(N);
    for (index i = 0; i < graph.outerSize(); i++){
        for (SparseMatrixXd::InnerIterator it(graph, i); it; ++it){
          embedding.row(it.row()) += (reference.row(it.col()) * it.value());
        }
      }
      return embedding;
  }

  void normalizeRows(
    const SparseMatrixXd& graph)
  {
    index p = 0;
    ArrayXd sums = ArrayXd::Zero(graph.innerSize());
    for (int i = 0; i < graph.outerSize(); i++){
      for (SparseMatrixXd::InnerIterator it(graph, i); it; ++it){
        sums(it.row())+= it.value();
      }
    }
    for (int i = 0; i < graph.outerSize(); i++){
      for (SparseMatrixXd::InnerIterator it(graph, i); it; ++it){
        it.valueRef() = it.value() / sums(it.row());
      }
    }
  }

  DataSet transform(DataSet &in, index maxIter = 200, double learningRate = 1.0) {
          SparseMatrixXd tmpGraph(in.size(), mKNNGraph.cols());
          ArrayXXd dists = ArrayXXd::Zero(in.size(), mK);
          makeGraph(in, mTree, mK, tmpGraph, dists, false);
          tmpGraph.makeCompressed();
          ArrayXd sigma = findSigma(mK, dists);
          computeHighDimProb(dists, sigma, tmpGraph);
          normalizeRows(tmpGraph);
          ArrayXXd embedding = initTransformEmbedding(tmpGraph, mEmbedding, in.size());
          ArrayXi rowIndices(tmpGraph.nonZeros());
          ArrayXi colIndices(tmpGraph.nonZeros());
          ArrayXd epochsPerSample(tmpGraph.nonZeros());
          getGraphIndices(tmpGraph, rowIndices, colIndices);
          computeEpochsPerSample(tmpGraph, epochsPerSample);
          epochsPerSample = (epochsPerSample == 0).select(-1, epochsPerSample);
          optimizeLayout(embedding, mEmbedding, rowIndices, colIndices, epochsPerSample,
            false, learningRate, maxIter);

          DataSet out(in.getIds(), _impl::asFluid(embedding));
          return out;
   }

  DataSet train(DataSet &in, index k = 15, index dims = 2,
                  double minDist = 0.1, index maxIter = 200,
                  double learningRate = 1.0) {
    using namespace Eigen;
    using namespace _impl;
    using namespace std;
    index n = in.size();
    mTree = KDTree(in);
    mKNNGraph = SparseMatrixXd(in.size(), in.size());
    mDists = ArrayXXd::Zero(in.size(), k);
    mK = k;
    makeGraph(in, mTree, mK, mKNNGraph, mDists, true);
    ArrayXd sigma = findSigma(k, mDists);
    computeHighDimProb(mDists, sigma, mKNNGraph);
    SparseMatrixXd mKNNGraphT = mKNNGraph.transpose();
    mKNNGraph = (mKNNGraph + mKNNGraphT) - mKNNGraph.cwiseProduct(mKNNGraphT);
    mAB = findAB(minDist);
    mEmbedding = mSpectralEmbedding.process(mKNNGraph, dims);
    mEmbedding = normalizeEmbedding(mEmbedding);
    mKNNGraph.makeCompressed();
    ArrayXi rowIndices(mKNNGraph.nonZeros());
    ArrayXi colIndices(mKNNGraph.nonZeros());
    ArrayXd epochsPerSample(mKNNGraph.nonZeros());
    getGraphIndices(mKNNGraph, rowIndices, colIndices);
    computeEpochsPerSample(mKNNGraph, epochsPerSample);
    epochsPerSample = (epochsPerSample == 0).select(-1, epochsPerSample);
    optimizeLayout(mEmbedding, mEmbedding, rowIndices, colIndices, epochsPerSample,
      true, learningRate, maxIter);
    DataSet out(in.getIds(), _impl::asFluid(mEmbedding));
    return out;
  }

private:
  SpectralEmbedding mSpectralEmbedding;
  SparseMatrixXd mKNNGraph;
  ArrayXXd mDists;
  KDTree mTree;
  index mK;
  VectorXd mAB;
  ArrayXXd mEmbedding;
};
}; // namespace algorithm
}; // namespace fluid
