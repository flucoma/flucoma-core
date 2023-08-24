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

#include "../util/FluidEigenMappings.hpp"
#include "../../data/FluidDataSet.hpp"
#include "../../data/FluidIndex.hpp"
#include "../../data/FluidTensor.hpp"
#include "../../data/TensorTypes.hpp"
#include "../../data/FluidMemory.hpp"
#include <Eigen/Core>
#include <random>

namespace fluid {
namespace algorithm {

class DTW
{
    using MatrixXd = Eigen::MatrixXd;
    using VectorXd = Eigen::VectorXd;
    using ArrayXd =  Eigen::ArrayXd;

public:
    explicit DTW() = default;
    ~DTW() = default;

    void init()
    {

    }

    double process(InputRealMatrixView x1, InputRealMatrixView x2, index q)
    {
        distanceMetrics.conservativeResize(x1.rows(), x2.rows());
        // simple brute force DTW is very inefficient, see FastDTW
        for (index i = 0; i < x1.rows(); i++)
        {
            for (index j = 0; j < x2.rows(); j++)
            {
                ArrayXd x1i = _impl::asEigen<Eigen::Array>(x1.row(i));
                ArrayXd x2j = _impl::asEigen<Eigen::Array>(x2.row(j));

                distanceMetrics(i, j) = euclidianDistToTheQ(x1i, x2j, q);

                if (i > 0 || j > 0)
                {
                    double minimum = std::numeric_limits<double>::max();

                    if (i > 0 && j > 0) 
                        minimum = std::min(minimum, distanceMetrics(i - 1, j - 1));
                    if (i > 0)
                        minimum = std::min(minimum, distanceMetrics(i - 1, j));
                    if (j > 0)
                        minimum = std::min(minimum, distanceMetrics(i, j - 1));

                    distanceMetrics(i, j) += minimum;
                }
            }
        }

        return std::pow(distanceMetrics.bottomLeftCorner<1, 1>().value(), 1.0 / q);
    }

private:
    mutable MatrixXd distanceMetrics;

    inline static double euclidianDistToTheQ(const Eigen::Ref<const VectorXd>& in, const Eigen::Ref<const VectorXd>& out, index q)
    {
        double euclidianSquared = (in * out).value();
        if(q == 2) 
            return euclidianSquared;
        return std::pow(euclidianSquared, 0.5 * q); // already squared, so (x^2)^(q/2) = x^q and _really_ optimises even values of q
    }
};

} // namespace algorithm
} // namespace fluid