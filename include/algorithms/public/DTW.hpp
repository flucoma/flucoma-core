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