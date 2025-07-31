#pragma once

#include <Eigen/Core>

namespace fluid {
namespace algorithm {
namespace _impl {

index incrementalMeanVariance(const Eigen::Ref<Eigen::ArrayXXd> data,
                              index                             lastSampleCount,
                              Eigen::Ref<Eigen::ArrayXd>        mean,
                              Eigen::Ref<Eigen::ArrayXd>        var)
{

  
  Eigen::VectorXd rowSums = data.isNaN().select(0, data).colwise().sum();
  index          newSampleCount = data.rows();
  Eigen::ArrayXd lastSum = mean * lastSampleCount;
  
  if (mean.cols() > 0)
  {
    index updatedSampleCount = lastSampleCount + newSampleCount;
    mean = ((mean * lastSampleCount) + rowSums.array()) / updatedSampleCount;

    if (var.rows() > 0)
    {
    
      Eigen::ArrayXXd tmp  = (data.transpose().colwise() - (rowSums.array() / newSampleCount)).transpose();
      Eigen::ArrayXd correction = tmp.colwise().sum();
      tmp = tmp.square();
    
      Eigen::ArrayXd newUnnormalisedVar = tmp.colwise().sum();
      newUnnormalisedVar -= correction.square() / newSampleCount;
    
      Eigen::ArrayXd lastUnormalisedVar = var * lastSampleCount;
  
      
      if(lastSampleCount > 0)
      {
          double lastCountOverNewCount = static_cast<double>(lastSampleCount) / newSampleCount;
          var =(
              lastUnormalisedVar
              + newUnnormalisedVar
              + lastCountOverNewCount
                / updatedSampleCount
                * (lastSum / lastCountOverNewCount - rowSums.array()).square()
                );
      }
      var /= updatedSampleCount;
    }
    return updatedSampleCount;
  }
  return lastSampleCount;
}

} // namespace _impl
} // namespace algorithm
} // namespace fluid
