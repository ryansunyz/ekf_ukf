#include <iostream>
#include "tools.h"

using Eigen::VectorXd;
using Eigen::MatrixXd;
using std::vector;

Tools::Tools() {}

Tools::~Tools() {}

VectorXd Tools::CalculateRMSE(const vector<VectorXd> &estimations,
                              const vector<VectorXd> &ground_truth) {
  /**
  TODO:
    * Calculate the RMSE here.
  */
    VectorXd rmse(4);
    rmse << 0,0,0,0;
    
    // Check estimations dimension
    if (estimations.size() == 0 || estimations.size() != ground_truth.size()) {
        cout << "Invalid dimension of estimation vector" << endl;
        return rmse;
    }
    
    // Claculate RMSE
    int n = estimations.size();
    for(int i = 0; i < n; ++i) {
        VectorXd res = estimations[i] - ground_truth[i];
        VectorXd resSquare = res.array() * res.array();
        rmse += resSquare;
    }
    rmse = rmse / n;
    rmse = rmse.array().sqrt();
    return rmse;
}
