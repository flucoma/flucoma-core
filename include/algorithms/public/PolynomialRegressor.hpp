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

class PolynomialRegressor
{
public:
    explicit PolynomialRegressor() = default;
    ~PolynomialRegressor() = default;

    void init(index degree = 2)
    {
        mInitialized = true;
        mDegree = degree;
    };

    index size() const { return asSigned(mInSet ? mIn.size() : 1); };
    index dims() const { return asSigned(mInSet ? mIn.size() : 0); };

    void clear() { mInSet = mOutSet = mRegressed = false; }

    index   getDegree()     const { return asSigned(mDegree); };
    bool    regressed()     const { return mRegressed; };
    bool    initialized()   const { return mInitialized; };

    void setDegree(index degree) {
        if (mDegree == degree)
        return;

        mDegree = degree;
        clear();
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
       _impl::asEigen<Eigen::Array>(coefficients) = mCoefficients;   
    };

    void setCoefficients(InputRealVectorView coefficients)
    {
        mCoefficients = _impl::asEigen<Eigen::Array>(coefficients);
        setDegree(coefficients.size() - 1);
        mRegressed = true;
    }

    void getMappedSpace(InputRealVectorView in, 
                        RealVectorView out, 
                        Allocator& alloc = FluidDefaultAllocator()) const
    {
        using namespace _impl;

        assert(mRegressed);

        ScopedEigenMap<Eigen::VectorXd> input(in.size(), alloc),
          output(out.size(), alloc);
        input = asEigen<Eigen::Array>(in);
        output = asEigen<Eigen::Array>(in);

        calculateMappings(input, output);

        asEigen<Eigen::Array>(out) = output;
    }

    void setMappingSpace(InputRealVectorView in, 
                         InputRealVectorView out, 
                         Allocator& alloc = FluidDefaultAllocator())
    {
        using namespace _impl;

        ScopedEigenMap<Eigen::VectorXd> input(in.size(), alloc), output(out.size(), alloc);
        input = asEigen<Eigen::Array>(in);
        output = asEigen<Eigen::Array>(in);

        setInputSpace(input);
        setOutputSpace(output);
    };

private:
    void setInputSpace(Eigen::Ref<Eigen::VectorXd> in) 
    {
        mIn = in;
        mInSet = true;
        mRegressed = false;
    };

    void setOutputSpace(Eigen::Ref<Eigen::VectorXd> out) 
    {
        mOut = out;
        mOutSet = true;
        mRegressed = false;
    };

    void calculateRegressionCoefficients()
    {
        generateDesignMatrix(mIn);

        Eigen::MatrixXd transposeProduct = mDesignMatrix.transpose() * mDesignMatrix;
        mCoefficients = transposeProduct.inverse() * mDesignMatrix.transpose() * mOut;

        mRegressed = true;
    };

    void calculateMappings(Eigen::Ref<Eigen::VectorXd> in, Eigen::Ref<Eigen::VectorXd> out) const
    {
        generateDesignMatrix(in);
        out = mDesignMatrix * mCoefficients;
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
    bool  mRegressed    {false};
    bool  mInitialized  {false};

    bool mInSet {false};
    bool mOutSet{false};

    Eigen::VectorXd mIn;
    Eigen::VectorXd mOut;
    mutable Eigen::MatrixXd mDesignMatrix;

    Eigen::VectorXd mCoefficients;

};

} // namespace algorithm
} // namespace fluid