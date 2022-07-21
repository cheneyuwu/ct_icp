#pragma once

#include <fstream>

#include "steam.hpp"
#include "steam_icp/odometry.hpp"

namespace steam_icp {

class SteamOdometry2 : public Odometry {
 public:
  enum class STEAM_LOSS_FUNC { L2, DCS, CAUCHY, GM };

  struct Options : public Odometry::Options {
    // sensor vehicle transformation
    Eigen::Matrix<double, 4, 4> T_sr = Eigen::Matrix<double, 4, 4>::Identity();
    // trajectory
    Eigen::Matrix<double, 6, 6> qc_inv = Eigen::Matrix<double, 6, 6>::Identity();
    int num_extra_states = 0;
    //
    bool add_prev_state = false;
    int num_extra_prev_states = 0;
    bool lock_prev_pose = false;
    bool lock_prev_vel = false;
    bool prev_pose_as_prior = false;
    bool prev_vel_as_prior = false;
    //
    int no_prev_state_iters = 0;
    bool association_after_adding_prev_state = true;
    // velocity prior (no side slipping)
    bool use_vp = false;
    Eigen::Matrix<double, 6, 6> vp_cov = Eigen::Matrix<double, 6, 6>::Identity();
    // p2p
    int p2p_initial_iters = 0;
    double p2p_initial_max_dist = 0.3;
    double p2p_refined_max_dist = 0.3;
    STEAM_LOSS_FUNC p2p_loss_func = STEAM_LOSS_FUNC::L2;
    double p2p_loss_sigma = 1.0;
    // radial velocity
    bool use_rv = false;
    bool merge_p2p_rv = false;
    double rv_cov_inv = 1.0;
    double rv_loss_threshold = 1.0;
    // optimization
    bool verbose = false;
    int max_iterations = 1;
    unsigned int num_threads = 1;

    //
    bool delay_adding_points = false;
  };

  SteamOdometry2(const Options &options);

  RegistrationSummary registerFrame(const std::vector<Point3D> &frame) override;

 private:
  void initializeMotion(int index_frame);
  std::vector<Point3D> initializeFrame(int index_frame, const std::vector<Point3D> &const_frame);
  void updateMap(int index_frame, int update_frame);
  void icp(int index_frame, std::vector<Point3D> &keypoints, RegistrationSummary &summary);

 private:
  const Options options_;

  // steam variables
  steam::se3::SE3StateVar::Ptr T_sr_var_ = nullptr;  // robot to sensor transformation as a steam variable

  // trajectory variables
  struct TrajectoryVar {
    TrajectoryVar(const steam::traj::Time &t, const steam::se3::SE3StateVar::Ptr &T,
                  const steam::vspace::VSpaceStateVar<6>::Ptr &w)
        : time(t), T_rm(T), w_mr_inr(w) {}
    steam::traj::Time time;
    steam::se3::SE3StateVar::Ptr T_rm;
    steam::vspace::VSpaceStateVar<6>::Ptr w_mr_inr;
  };
  std::vector<TrajectoryVar> trajectory_vars_;
  steam::traj::const_vel::Interface::Ptr full_steam_trajectory_ = nullptr;

  std::ofstream pose_debug_file_;
  std::ofstream velocity_debug_file_;

  STEAM_ICP_REGISTER_ODOMETRY("STEAM2", SteamOdometry2);
};

}  // namespace steam_icp