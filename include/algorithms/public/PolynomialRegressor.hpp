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

    void init(index degree) 
    {
        mDegree = degree;
        mInitialised = true;
    };

    index   getDegree()     const { return mDegree;         };
    bool    initialised()   const { return mInitialised;    };
    bool    regressed()     const { return mRegressed;      };

    RealVectorView getCoefficients() const 
    {

    };

private:
    index mDegree       {2};
    bool  mInitialised  {false};
    bool  mRegressed    {false};

    VectorXd mCoefficients;

    mutable VectorXd mIn;
    mutable VectorXd mOut;

}   

}
}