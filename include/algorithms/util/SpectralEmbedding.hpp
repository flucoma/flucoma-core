/*
Part of the Fluid Corpus Manipulation Project (http://www.flucoma.org/)
Copyright 2017-2019 University of Huddersfield.
Licensed under the BSD-3 License.
See license.md file in the project root for full license information.
This project has received funding from the European Research Council (ERC)
under the European Union’s Horizon 2020 research and innovation programme
(grant agreement No 725899).
*/

#pragma once
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

  ArrayXXd process(SparseMatrixXd graph, index dims)
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
    SparseMatrixXd           L = I - (D * (graph * D));
    int                      k = static_cast<int>(dims + 1);
    index                    ncv = max(2 * k + 1, int(round(sqrt(L.rows()))));
    VectorXd                 initV = VectorXd::Ones(L.rows());
    SparseSymMatProd<double> op(L);
    SymEigsSolver<double, SMALLEST_MAGN, SparseSymMatProd<double>> eigs(&op, k,
                                                                        ncv);
    eigs.init(initV.data());
    auto nConverged = eigs.compute(
        D.cols(), 1e-4, SMALLEST_MAGN); // TODO: failback if not converging
    MatrixXd U = eigs.eigenvectors();
    ArrayXXd Y = U.block(0, 1, U.rows(), dims).array();
    return Y;
  }
};
}; // namespace algorithm
}; // namespace fluid
