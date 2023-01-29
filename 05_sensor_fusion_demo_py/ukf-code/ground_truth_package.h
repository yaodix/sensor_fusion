//
// Created by abang on 17-10-26.
//

#ifndef UKF_GROUND_TRUTH_PACKAGE_H
#define UKF_GROUND_TRUTH_PACKAGE_H

#include "Eigen/Dense"

class GroundTruthPackage {
public:
    long long timestamp_;

    enum SensorType{
        LASER,
        RADAR
    } sensor_type_;

    Eigen::VectorXd gt_values_;

};

#endif //UKF_GROUND_TRUTH_PACKAGE_H
