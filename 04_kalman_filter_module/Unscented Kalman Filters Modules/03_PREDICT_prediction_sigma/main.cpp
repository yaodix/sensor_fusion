#include "Dense"
#include "ukf.h"

using Eigen::MatrixXd;

int main() {

  // Create a UKF instance
  UKF ukf;

  /**
   * Programming assignment calls
   */
  MatrixXd Xsig_pred = MatrixXd(5, 15);
  ukf.SigmaPointPrediction(&Xsig_pred);

  return 0;
}