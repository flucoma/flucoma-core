#pragma once
#include "data/FluidIndex.hpp"
#include <Eigen/Core>
#include <Eigen/Sparse>
#include <Spectra/MatOp/SparseSymMatProd.h>
#include <Spectra/SymEigsSolver.h>

namespace fluid {
namespace algorithm {


class SpectralEmbedding
{
public:
  using MatrixXd = Eigen::MatrixXd;
  using ArrayXXd = Eigen::ArrayXXd;
  using SparseMatrixXd = Eigen::SparseMatrix<double>;

  ArrayXXd train(SparseMatrixXd graph, index dims)
  {
    using namespace Eigen;
    using namespace Spectra;
    using namespace std;
    VectorXd diagData = graph * VectorXd::Ones(graph.cols());
    diagData = (1 / diagData.array().sqrt());
    SparseMatrixXd D = SparseMatrixXd(graph.rows(), graph.cols());
    D.reserve(graph.rows());
    for (index i = 0; i < D.rows(); i++) { D.insert(i, i) = diagData(i); }
    SparseMatrixXd I = SparseMatrixXd(D.rows(), D.cols());
    I.setIdentity();
    SparseMatrixXd L = I - (D * (graph * D));
    int            k = static_cast<int>(dims + 1);
    index          ncv = max(2 * k + 1, int(round(sqrt(L.rows()))));
    VectorXd       initV = VectorXd::Ones(L.rows());

    SparseSymMatProd<double> op(L);
    SymEigsSolver<SparseSymMatProd<double>> eig(op, k, ncv);    
    eig.init(initV.data());
    eig.compute(Spectra::SortRule::SmallestMagn, 1000, 1e-4); 
    // TODO: failback if not converging
    mEigenVectors = eig.eigenvectors();
    mEigenValues = eig.eigenvalues();
    ArrayXXd Y = mEigenVectors.block(0, 1, mEigenVectors.rows(), dims).array();
    return Y;
  }

  Eigen::MatrixXd eigenVectors() { return mEigenVectors; }

  Eigen::MatrixXd eigenValues() { return mEigenValues; }

private:
  Eigen::MatrixXd mEigenVectors;
  Eigen::MatrixXd mEigenValues;
};
}// namespace algorithm
}// namespace fluid
