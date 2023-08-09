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
#include <cassert>
#include <cmath>

namespace fluid {
namespace algorithm {

template <typename T>
class PolynomialRegressor
{
    using ArrayXd =  Eigen::ArrayXd;
    using ArrayXXd = Eigen::ArrayXXd;

    using MatrixXd = Eigen::MatrixXd;
    using VectorXd = Eigen::VectorXd;

public:
    PolynomialRegressor() = default;
    ~PolynomialRegressor() = default;

    void init(index degree = 2)
    {
        mDegree = degree;
        mInitialised = true;
    };

    index   getDegree()     const { return mDegree;         };
    bool    initialised()   const { return mInitialised;    };
    bool    regressed()     const { return mRegressed;      };

    void setDegree(index degree) {
        if(mDegree == degree) return;

        mDegree = degree;
        resetMappingSpace(); 
    }

    void process(RealVectorView in, RealVectorView out)
    {
        setMappingSpace(in, out);
        process();
    };

    void process() 
    {
        assert(mInSet && mOutSet);
        calculateRegressionCoefficients();
    };

    void getCoefficients(RealVectorView coefficients) const
    {
        assert(mRegressed);
        asEigen<Vector>(coefficients) = mCoefficients;   
    };

    void setMappingSpace(RealVectorView in, RealVectorView out) const
    {
        VectorXd input = asEigen<Eigen::Vector>(in);
        VectorXd output = asEigen<Eigen::Vector>(out);

        setInputSpace(input);
        setOutputSpace(output);
    }

    void setMappingSpace(RealVectorView in, 
                         RealVectorView out, 
                         Allocator& alloc = FluidDefaultAllocator()) const
    {
        ScopedEigenMap<ArrayXd> input(in.size(), alloc), output(out.size(), alloc);
        input = asEigen<Eigen::Array>(in);
        output = asEigen<Eigen::Array>(in);

        setInputSpace(input);
        setInputSpace(output);
    };

    void resetMappingSpace() const { mInSet = mOutSet = mRegressed = false; };

private:

    void setInputSpace(Eigen::Ref<VectorXd> in) const 
    {
        VectorXd designColumn = VectorXd::Ones(in.size());
        ArrayXd inArray = in.array();
        mDesignMatrix.conservativeResize(in.size(), mDegree + 1);

        for(index i = 0; i < mDegree + 1; ++i) {
            mDesignMatrix.col(i) = designColumn;
            designColumn = designColumn.array() * inArray;
        }

        mIn = in;
        mInSet = true;
        mRegressed = false;
    };

    void setOutputSpace(Eigen::Ref<VectorXd> out) const 
    {
        mOut = out;
        mOutSet = true;
        mRegressed = false;
    };

    void calculateRegressionCoefficients()
    {
        MatrixXd transposeProduct = mDesignMatrix.transpose() * mDesignMatrix;
        mCoefficients = transposeProduct.inverse() * mDesignMatrix.transpose() * mOut;

        mRegressed = true;
    };

    index mDegree       {2};
    bool  mInitialised  {false};
    bool  mRegressed    {false};

    bool mInSet         {false};
    bool mOutSet        {false};

    mutable VectorXd mIn;
    mutable VectorXd mOut;

    MatrixXd mDesignMatrix;
    VectorXd mCoefficients;

}   

} // namespace algorithm
} // namespace fluid