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

    void init(index degree, index dims)
    {
        mInitialized = true;
        setDegree(degree);
        setDims(dims);
    };

    index degree()  const { return mInitialized ? asSigned(mDegree) : 0; };
    index dims()    const { return mInitialized ? asSigned(mDims) : 0; };
    index size()    const { return mInitialized ? asSigned(mDegree) : 0; };

    void clear() { mRegressed = false; }

    bool    regressed()     const { return mRegressed; };
    bool    initialized()   const { return mInitialized; };

    void setDegree(index degree) {
        if (mDegree == degree) return;

        mDegree = degree;
        mCoefficients.conservativeResize(mDegree + 1, mDims);
        mRegressed = false;
    }

    void setDims(index dims) {
        if (mDims == dims) return;

        mDims = dims;
        mCoefficients.conservativeResize(mDegree + 1, mDims);
        mRegressed = false;
    }

    void regress(InputRealMatrixView in, 
                 InputRealMatrixView out,
                 Allocator& alloc = FluidDefaultAllocator())
    {
        using namespace _impl;

        ScopedEigenMap<Eigen::MatrixXd> input(in.rows(), in.cols(), alloc), 
          output(out.rows(), out.cols(), alloc);
        input = asEigen<Eigen::Array>(in);
        output = asEigen<Eigen::Array>(out);

        for(index i = 0; i < mDims; ++i)
        {
            generateDesignMatrix(input.col(i));

            Eigen::MatrixXd transposeProduct = mDesignMatrix.transpose() * mDesignMatrix;
            mCoefficients.col(i) = transposeProduct.inverse() * mDesignMatrix.transpose() * output.col(i);
        }
        

        mRegressed = true;
    };

    void getCoefficients(RealMatrixView coefficients) const
    {
       if (mInitialized) _impl::asEigen<Eigen::Array>(coefficients) = mCoefficients;   
    };

    void setCoefficients(InputRealMatrixView coefficients)
    {
        setDegree(coefficients.rows() - 1);
        setDims(coefficients.cols());

        mCoefficients = _impl::asEigen<Eigen::Array>(coefficients);
        mRegressed = true;
    }

    void process(InputRealMatrixView in, 
                 RealMatrixView out, 
                 Allocator& alloc = FluidDefaultAllocator()) const
    {
        using namespace _impl;

        ScopedEigenMap<Eigen::MatrixXd> input(in.rows(), in.cols(), alloc),
          output(out.rows(), out.cols(), alloc);
        input = asEigen<Eigen::Array>(in);
        output = asEigen<Eigen::Array>(out);

        calculateMappings(input, output);

        asEigen<Eigen::Array>(out) = output;
    }

private:
    void calculateMappings(Eigen::Ref<Eigen::MatrixXd> in, Eigen::Ref<Eigen::MatrixXd> out) const
    {
        for(index i = 0; i < mDims; ++i)
        {
            generateDesignMatrix(in.col(i));
            out.col(i) = mDesignMatrix * mCoefficients.col(i);
        }
    }

    void generateDesignMatrix(Eigen::Ref<Eigen::VectorXd> in) const
    {
        Eigen::VectorXd designColumn = Eigen::VectorXd::Ones(in.size());
        Eigen::ArrayXd inArray = in.array();

        mDesignMatrix.conservativeResize(in.size(), mDegree + 1);

        for(index i = 0; i < mDegree + 1; ++i, designColumn = designColumn.array() * inArray) 
            mDesignMatrix.col(i) = designColumn;
    }

    index mDegree       {2};
    index mDims         {1};
    bool  mRegressed    {false};
    bool  mInitialized  {false};

    mutable Eigen::MatrixXd mDesignMatrix;
    Eigen::MatrixXd mCoefficients;

};

} // namespace algorithm
} // namespace fluid