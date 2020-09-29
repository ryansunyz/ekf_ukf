#include "ukf.h"
#include "Eigen/Dense"
#include <iostream>

using namespace std;
using Eigen::MatrixXd;
using Eigen::VectorXd;
using std::vector;

/**
 * Initializes Unscented Kalman filter
 */
UKF::UKF() {
    // if this is false, laser measurements will be ignored (except during init)
    use_laser_ = true;

    // if this is false, radar measurements will be ignored (except during init)
    use_radar_ = true;

    // State dimension
    n_x_ = 5;
    
    // Augmented state dimension
    n_aug_ = 7;
    
    // initial state vector
    x_ = VectorXd(n_x_);
    
    // initial covariance matrix
    P_ = MatrixXd(n_x_, n_x_);
    
    // Process noise standard deviation longitudinal acceleration in m/s^2
    std_a_ = 0.5;

    // Process noise standard deviation yaw acceleration in rad/s^2
    std_yawdd_ = 0.5;

    // Laser measurement noise standard deviation position1 in m
    std_laspx_ = 0.15;

    // Laser measurement noise standard deviation position2 in m
    std_laspy_ = 0.15;

    // Radar measurement noise standard deviation radius in m
    std_radr_ = 0.3;

    // Radar measurement noise standard deviation angle in rad
    std_radphi_ = 0.03;

    // Radar measurement noise standard deviation radius change in m/s
    std_radrd_ = 0.3;

    // Complete the initialization.
    
    // Time stamp
    is_initialized_ = false;
    previous_timestamp_ = 0;
    time_us_ = 0.0;
    
    // Sigma point spreading parameter
    lambda_ = 3 - n_aug_;
    
    // Sigma point for predicted states
    Xsig_pred_ = MatrixXd(n_x_, 2 * n_aug_ + 1);
    Xsig_pred_.fill(0.0);
    
    // Weights
    weights_ = VectorXd(2 * n_aug_ + 1);
    weights_(0) = lambda_ / (lambda_ + n_aug_);
    for (int i = 1; i < 2 * n_aug_ + 1; i++) {
        weights_(i) = 0.5 / (n_aug_ + lambda_);
    }
    
    // Variance of radar measurement
    R_radar_ = MatrixXd(3, 3);
    R_radar_.fill(0.0);
    R_radar_(0, 0) = std_radr_ * std_radr_;
    R_radar_(1, 1) = std_radphi_ * std_radphi_;
    R_radar_(2, 2) = std_radrd_ * std_radrd_;
    
    // Variance of laser measurement
    R_laser_ = MatrixXd(2, 2);
    R_laser_.fill(0.0);
    R_laser_(0, 0) = std_laspx_ * std_laspx_;
    R_laser_(1, 1) = std_laspy_ * std_laspy_;
    
    cout << "Variable initialized" <<endl;
    cout << endl;
}

UKF::~UKF() {}

/**
 * @param {MeasurementPackage} meas_package The latest measurement data of
 * either radar or laser.
 */
void UKF::ProcessMeasurement(MeasurementPackage meas_package) {
    /**
     Overall procedures of UKF algorithm
     */
    cout << "----- UKF, new cycle starts ---- " << endl;
    if (!is_initialized_) {
        if (meas_package.sensor_type_ == MeasurementPackage::RADAR) {
            /**
             Initialize states with radar measurements
             */
            // Parse measurement
            double rho = meas_package.raw_measurements_[0];
            double phi = meas_package.raw_measurements_[1];
            double rhoDot = meas_package.raw_measurements_[2];
            // Convert into state space
            double px = rho*cos(phi);
            double py = rho*sin(phi);
            double vx = rhoDot * cos(phi);
            double vy = rhoDot * sin(phi);
            double v = sqrt(vx * vx + vy * vy);
            x_ << px, py, 0, 0, 0;
        } else if (meas_package.sensor_type_ == MeasurementPackage::LASER) {
            /**
             Initialize states with laser measurements
             */
            double px = meas_package.raw_measurements_[0];
            double py = meas_package.raw_measurements_[1];
            x_ << px, py, 0, 0, 0;
        }
        
        if (fabs(x_(0)) < 0.001 && fabs(x_(1)) < 0.001) {
            x_(0) = 0.001;
            x_(1) = 0.001;
        }
        
        P_ << 1, 0, 0, 0, 0,
              0, 1, 0, 0, 0,
              0, 0, 1, 0, 0,
              0, 0, 0, 0.5, 0,
              0, 0, 0, 0, 0.5;
        
        previous_timestamp_ = meas_package.timestamp_;
        is_initialized_ = true;
        
        // Output result
        cout << "State at first step is initiazated" <<endl;
        cout << "x = " << x_ << endl;
        cout << "P = " << P_ << endl;
        cout << "----- UKF, the cycle ends ------"  << endl;
        cout << endl;
        return;
    }
    
    // Predict states
    double delta_t = (meas_package.timestamp_ - previous_timestamp_) / 1000000.0;
    Prediction(delta_t);
    
    // Update measurement
    if (meas_package.sensor_type_ == MeasurementPackage::RADAR) {
        // Radar updates
        UpdateRadar(meas_package);
        
    } else {
        // Laser updates
        UpdateLidar(meas_package);
    }
    
    previous_timestamp_ = meas_package.timestamp_;
    // print the output
    cout << "----- UKF, the cycle ends ------" << endl;
    cout << endl;
}

/**
 * Predicts sigma points, the state, and the state covariance matrix.
 * @param {double} delta_t the change in time (in seconds) between the last
 * measurement and this one.
 */
void UKF::Prediction(double delta_t) {
    /**
     Estimate the object's location. Modify the state
     vector, x_. Predict sigma points, the state, and the state covariance matrix.
     */
    cout << "Start prediction" << endl;
    // Step 1: Generate Sigma Points for current step
    //create augmented mean state
    VectorXd x_aug_ = VectorXd(n_aug_);
    x_aug_.fill(0.0);
    x_aug_.head(n_x_) = x_;
    
    //create augmented covariance matrix
    MatrixXd P_aug_ = MatrixXd(n_aug_, n_aug_);
    P_aug_.fill(0.0);
    P_aug_.topLeftCorner(n_x_, n_x_) = P_;
    P_aug_(n_x_, n_x_) = std_a_ * std_a_;
    P_aug_(n_x_ + 1, n_x_ + 1) = std_yawdd_ * std_yawdd_;
    
    //create square root matrix
    MatrixXd L = P_aug_.llt().matrixL();
    
    //create augmented sigma points
    MatrixXd Xsig_aug_ = MatrixXd(n_aug_, 2 * n_aug_ + 1);
    Xsig_aug_.col(0) = x_aug_;
    for (int i = 0; i < n_aug_; i++) {
        Xsig_aug_.col(i + 1) = x_aug_ + sqrt(lambda_ + n_aug_)*L.col(i);
        Xsig_aug_.col(i + 1 + n_aug_) = x_aug_ - sqrt(lambda_ + n_aug_)*L.col(i);
    }
    
    // Step 2: Predict Sigma Points for next step, a-priori
    for (int i = 0; i < 2 * n_aug_ + 1; i++) {
        VectorXd xCur = Xsig_aug_.col(i);
        double px = xCur(0);
        double py = xCur(1);
        double v = xCur(2);
        double phi = xCur(3);
        double phiDot = xCur(4);
        double nu_a = xCur(5);
        double nu_phiDD = xCur(6);
        
        if (phiDot > 0.001) {
            //predict sigma points
            Xsig_pred_(0,i) = xCur(0) + v / phiDot * (sin(phi + delta_t * phiDot) - sin(phi))
            + 0.5 * delta_t * delta_t * cos(phi) * nu_a;
            Xsig_pred_(1,i) = xCur(1) + v / phiDot * (-cos(phi + delta_t * phiDot) + cos(phi))
            + 0.5 * delta_t * delta_t * sin(phi) * nu_a;
            Xsig_pred_(2,i) = xCur(2) + delta_t * nu_a;
            Xsig_pred_(3,i) = xCur(3) + phiDot * delta_t + 0.5 * delta_t * delta_t * nu_phiDD;
            Xsig_pred_(4,i) = xCur(4) + delta_t * nu_phiDD;
        } else {
            //avoid division by zero
            Xsig_pred_(0,i) = xCur(0) + v * cos(phi) * delta_t
            + 0.5 * delta_t * delta_t * cos(phi) * nu_a;
            Xsig_pred_(1,i) = xCur(1) + v * sin(phi) * delta_t
            + 0.5 * delta_t * delta_t * sin(phi) * nu_a;
            Xsig_pred_(2,i) = xCur(2) + 0 + delta_t * nu_a;
            Xsig_pred_(3,i) = xCur(3) + phiDot * delta_t + 0.5 * delta_t * delta_t * nu_phiDD;
            Xsig_pred_(4,i) = xCur(4) + 0 + delta_t * nu_phiDD;
        }
    }
    
    // Step 3: Predict Mean and Covariance for next step, a-priori
    //predict state mean
    x_.fill(0.0);
    for (int i = 0; i < 2 * n_aug_ + 1; i++) {
        x_ = x_ + weights_(i) * Xsig_pred_.col(i);
    }
    
    //predict state covariance matrix
    P_.fill(0.0);
    for (int i = 0; i < 2 * n_aug_ + 1; i++) {
        // state difference
        VectorXd x_diff = Xsig_pred_.col(i) - x_;
        //angle normalization
        while (x_diff(3)> M_PI) x_diff(3)-=2.*M_PI;
        while (x_diff(3)<-M_PI) x_diff(3)+=2.*M_PI;
        P_ = P_ + weights_(i) * x_diff * x_diff.transpose();
    }
    cout << "x = " << x_ << endl;
    cout << "P = " << P_ << endl;
    cout << "Prediction ends" << endl;
    cout << endl;
}

/**
 * Updates the state and the state covariance matrix using a laser measurement.
 * @param {MeasurementPackage} meas_package
 */
void UKF::UpdateLidar(MeasurementPackage meas_package) {
  /**
  Use lidar data to update the belief about the object's
  position. Modify the state vector, x_, and covariance, P_.
  You'll also need to calculate the lidar NIS.
  */
    cout << "Start Lidar update" << endl;

    // Step 4: Predict Measurement
    int n_z = 2;

    //create matrix for sigma points in measurement space
    MatrixXd Zsig = MatrixXd(n_z, 2 * n_aug_ + 1);
    Zsig.fill(0.0);
    
    //mean predicted measurement
    VectorXd z_pred = VectorXd(n_z);
    z_pred.fill(0.0);

    //measurement covariance matrix S
    MatrixXd S = MatrixXd(n_z,n_z);
    S.fill(0.0);
    //Incoming Lidar measurement
    VectorXd z = meas_package.raw_measurements_;
  
    //transform sigma points into measurement space
    for (int i = 0; i < 2 * n_aug_ + 1; i++) {
        // Parse variables
        double px = Xsig_pred_(0, i);
        double py = Xsig_pred_(1, i);
        Zsig(0, i) = px;
        Zsig(1, i) = py;
    }
    
    //calculate mean predicted measurement
    z_pred = Zsig * weights_;
    
    //calculate measurement covariance matrix S
    for (int i = 0; i < 2 * n_aug_ + 1; i++) {
        VectorXd dz = Zsig.col(i) - z_pred;
        S += weights_(i) * dz * dz.transpose();
    }
    S = S + R_laser_;
    
    // Step 5: Update State and convariance matrix, a-posteriori
    //calculate cross correlation matrix
    MatrixXd Tc = MatrixXd(n_x_, n_z);
    Tc.fill(0.0);
    
    for (int i = 0; i < 2 * n_aug_ + 1; i++) {
        VectorXd z_diff = Zsig.col(i) - z_pred;
        Tc += weights_(i) * (Xsig_pred_.col(i) - x_) * z_diff.transpose();
    }
    
    //calculate Kalman gain K;
    MatrixXd K = Tc * S.inverse();
    
    //update state mean and covariance matrix
    VectorXd z_diff = z - z_pred;
    x_ += K * z_diff;
    P_ -= K * S * K.transpose();
    
    // NIS Lidar Update
    NIS_laser_ = z_diff.transpose() * S.inverse() * z_diff;
    cout << "x = " << x_ << endl;
    cout << "P = " << P_ << endl;
    cout << "Lidar update ends" << endl;
    cout << endl;
}

/**
 * Updates the state and the state covariance matrix using a radar measurement.
 * @param {MeasurementPackage} meas_package
 */
void UKF::UpdateRadar(MeasurementPackage meas_package) {
  /**
   Use radar data to update the belief about the object's
  position. Modify the state vector, x_, and covariance, P_.
  You'll also need to calculate the radar NIS.
  */
    cout << "Start Radar update" << endl;
    
    // Step 4: Predict Measurement
    int n_z = 3;
    //Incoming radar measurement
    VectorXd z = meas_package.raw_measurements_;
 
    //create matrix for sigma points in measurement space
    MatrixXd Zsig = MatrixXd(n_z, 2 * n_aug_ + 1);
    Zsig.fill(0.0);
    
    //measurement covariance matrix S
    MatrixXd S = MatrixXd(n_z,n_z);
    S.fill(0.0);
    
    //mean predicted measurement
    VectorXd z_pred = VectorXd(n_z);
    z_pred.fill(0.0);

    //transform sigma points into measurement space
    for (int i = 0; i < 2 * n_aug_ + 1; i++) {
        // Parse variables
        double px = Xsig_pred_(0, i);
        double py = Xsig_pred_(1, i);
        double v = Xsig_pred_(2, i);
        double psi = Xsig_pred_(3, i);
        double psiDot = Xsig_pred_(4, i);

        //check for zeros
        if (fabs(px) < 0.001) {
            px = 0.001;
        }
        if (fabs(py) < 0.001) {
            py = 0.001;
        }
        
        // Convert into measurement space
        double rho = sqrt(px * px + py * py);
        double phi = atan2(py, px);
        double rhoDot = (px * cos(psi) * v + py * sin(psi) * v) / rho;
        // Fill in Zsig matrix
        Zsig(0, i) = rho;
        Zsig(1, i) = phi;
        Zsig(2, i) = rhoDot;
    }

    //calculate mean predicted measurement
    z_pred = Zsig * weights_;

    //calculate measurement covariance matrix S
    for (int i = 0; i < 2 * n_aug_ + 1; i++) {
        //residual
        VectorXd z_diff = Zsig.col(i) - z_pred;
        
        //angle normalization
        while (z_diff(1)> M_PI) z_diff(1)-=2.*M_PI;
        while (z_diff(1)<-M_PI) z_diff(1)+=2.*M_PI;
        
        S = S + weights_(i) * z_diff * z_diff.transpose();

    }
    S = S + R_radar_;
    
    // Step 5: Update State and convariance matrix, a-posteriori
    //calculate cross correlation matrix
    MatrixXd Tc = MatrixXd(n_x_, n_z);
    Tc.fill(0.0);
    
    for (int i = 0; i < 2 * n_aug_ + 1; i++) {
        //residual
        VectorXd z_diff = Zsig.col(i) - z_pred;
        
        //angle normalization
        while (z_diff(1)> M_PI) z_diff(1)-=2.*M_PI;
        while (z_diff(1)<-M_PI) z_diff(1)+=2.*M_PI;
        
        // state difference
        VectorXd x_diff = Xsig_pred_.col(i) - x_;
        
        //angle normalization
        while (x_diff(3)> M_PI) x_diff(3)-=2.*M_PI;
        while (x_diff(3)<-M_PI) x_diff(3)+=2.*M_PI;
        
        Tc += weights_(i) * x_diff * z_diff.transpose();
    }
    
    //calculate Kalman gain K;
    MatrixXd K = Tc * S.inverse();
    
    //update state mean and covariance matrix
    VectorXd z_diff = z - z_pred;
    
    //angle normalization
    while (z_diff(1)> M_PI) z_diff(1)-=2.*M_PI;
    while (z_diff(1)<-M_PI) z_diff(1)+=2.*M_PI;
    
    x_ += K * z_diff;
    P_ -= K * S * K.transpose();
    //NIS Update
    NIS_radar_ = z_diff.transpose() * S.inverse() * z_diff;
    
    // Output results
    cout << "x = " << x_ << endl;
    cout << "P = " << P_ << endl;
    cout << "Radar update ends" << endl;
    cout << endl;
}




