/*
Part of the Fluid Corpus Manipulation Project (http://www.flucoma.org/)
Copyright University of Huddersfield.
Licensed under the BSD-3 License.
See license.md file in the project root for full license information.
This project has received funding from the European Research Council (ERC)
under the European Union’s Horizon 2020 research and innovation programme
(grant agreement No 725899).
*/

#pragma once

#include "../util/AlgorithmUtils.hpp"
#include "../util/FluidEigenMappings.hpp"
#include "../../data/FluidIndex.hpp"
#include "../../data/FluidMemory.hpp"
#include "../../data/TensorTypes.hpp"
#include <Eigen/Core>
#include <Eigen/Dense>
#include <cassert>
#include <cmath>

namespace fluid {
namespace algorithm {

class PolynomialRegressor
{
public:
  explicit PolynomialRegressor() = default;
  ~PolynomialRegressor() = default;

  void init(index degree, index dims, double tikhonov = 0.0)
  {
    mInitialized = true;
    setDegree(degree);
    setDims(dims);
    setTikhonov(tikhonov);
  };

  index  degree() const { return mInitialized ? asSigned(mDegree) : 0; };
  double tihkonov() const { return mInitialized ? mTikhonovFactor : 0.0; };
  index  dims() const { return mInitialized ? asSigned(mDims) : 0; };
  index  size() const { return mInitialized ? asSigned(mDegree) : 0; };

  void clear() { mRegressed = false; }

  bool regressed() const { return mRegressed; };
  bool initialized() const { return mInitialized; };

  void setDegree(index degree)
  {
    if (mDegree == degree) return;

    mDegree = degree;
    mRegressed = false;
  }

  void setDims(index dims)
  {
    if (mDims == dims) return;

    mDims = dims;
    mRegressed = false;
  }

  void setTikhonov(double tikhonov)
  {
    if (mTikhonovFactor == tikhonov) return;

    mTikhonovFactor = tikhonov;
    mRegressed = false;
  }


  void regress(InputRealMatrixView in, InputRealMatrixView out,
               Allocator& alloc = FluidDefaultAllocator())
  {
    using namespace _impl;
    using namespace Eigen;

    ScopedEigenMap<MatrixXd> input(in.rows(), in.cols(), alloc),
        output(out.rows(), out.cols(), alloc),
        transposeProduct(mDegree + 1, mDegree + 1, alloc);

    input = asEigen<Matrix>(in);
    output = asEigen<Matrix>(out);

    mCoefficients.resize(mDegree + 1, mDims);
    mTikhonovMatrix.resize(mDegree + 1, mDegree + 1);

    asEigen<Matrix>(mTikhonovMatrix) =
        mTikhonovFactor * MatrixXd::Identity(mDegree + 1, mDegree + 1);

    for (index i = 0; i < mDims; ++i)
    {
      generateDesignMatrix(input.col(i));

      // tikhonov/ridge regularisation, given Ax = y where x could be noisy
      // optimise the value _x = (A^T . A + R^T . R)^-1 . A^T . y
      // where R is a tikhonov filter matrix, in case of ridge regression of the
      // form a.I
      transposeProduct = asEigen<Matrix>(mDesignMatrix).transpose() *
                             asEigen<Matrix>(mDesignMatrix) +
                         asEigen<Matrix>(mTikhonovMatrix).transpose() *
                             asEigen<Matrix>(mTikhonovMatrix);
      asEigen<Matrix>(mCoefficients.col(i)) =
          transposeProduct.inverse() *
          asEigen<Matrix>(mDesignMatrix).transpose() * output.col(i);
    }

    mRegressed = true;
  };

  void getCoefficients(RealMatrixView coefficients) const
  {
    if (mInitialized) coefficients <<= mCoefficients;
  };

  void setCoefficients(InputRealMatrixView coefficients)
  {
    if (!mInitialized) mInitialized = true;

    setDegree(coefficients.rows() - 1);
    setDims(coefficients.cols());

    mCoefficients <<= coefficients;
    mRegressed = true;
  }

  void process(InputRealMatrixView in, RealMatrixView out,
               Allocator& alloc = FluidDefaultAllocator()) const
  {
    using namespace _impl;
    using namespace Eigen;

    ScopedEigenMap<VectorXd> coefficientsColumn(mCoefficients.rows(), alloc),
        inputColumn(in.rows(), alloc);

    for (index i = 0; i < mDims; ++i)
    {
      inputColumn = asEigen<Matrix>(in.col(i));
      coefficientsColumn = asEigen<Matrix>(mCoefficients.col(i));

      generateDesignMatrix(inputColumn);

      asEigen<Matrix>(out.col(i)) =
          asEigen<Matrix>(mDesignMatrix) * coefficientsColumn;
    }
  }

private:
  void generateDesignMatrix(Eigen::Ref<Eigen::VectorXd> in,
                            Allocator& alloc = FluidDefaultAllocator()) const
  {
    using namespace _impl;
    using namespace Eigen;

    ScopedEigenMap<ArrayXd> designColumn(in.size(), alloc),
        inArray(in.size(), alloc);

    designColumn = VectorXd::Ones(in.size());
    inArray = in.array();

    mDesignMatrix.resize(in.size(), mDegree + 1);

    for (index i = 0; i < mDegree + 1;
         ++i, designColumn = designColumn * inArray)
      asEigen<Matrix>(mDesignMatrix.col(i)) = designColumn;
  }

  index mDegree{2};
  index mDims{1};
  bool  mRegressed{false};
  bool  mInitialized{false};

  double mTikhonovFactor{0};

  RealMatrix mCoefficients;

  mutable RealMatrix mDesignMatrix;
  mutable RealMatrix mTikhonovMatrix;
};

} // namespace algorithm
} // namespace fluid