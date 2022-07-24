#include "steam_icp/odometry/steam_icp.hpp"

#include <iomanip>
#include <random>

#include <glog/logging.h>

#include "steam.hpp"

#include "steam_icp/utils/stopwatch.hpp"

namespace steam_icp {

namespace {

/** \brief Basic solver interface */
class GaussNewtonIterator {
 public:
  GaussNewtonIterator(steam::Problem &problem) : problem_(problem), state_vector_(problem.getStateVector()) {}

  /** \brief Perform one iteration */
  void iterate() {
    // The 'left-hand-side' of the Gauss-Newton problem, generally known as the
    // approximate Hessian matrix (note we only store the upper-triangular
    // elements)
    Eigen::SparseMatrix<double> approximate_hessian;
    // The 'right-hand-side' of the Gauss-Newton problem, generally known as the
    // gradient vector
    Eigen::VectorXd gradient_vector;
    // Construct system of equations
    problem_.buildGaussNewtonTerms(approximate_hessian, gradient_vector);
    // Solve system
    // Perform a Cholesky factorization of the approximate Hessian matrix
    // Check if the pattern has been initialized
    if (!pattern_initialized_) {
      hessian_solver_.analyzePattern(approximate_hessian);
      pattern_initialized_ = true;
    }

    // Perform a Cholesky factorization of the approximate Hessian matrix
    hessian_solver_.factorize(approximate_hessian);
    if (hessian_solver_.info() != Eigen::Success) throw std::runtime_error("Eigen LLT decomposition failed.");

    // Solve
    Eigen::VectorXd perturbation = hessian_solver_.solve(gradient_vector);

    // Apply update
    state_vector_.lock()->update(perturbation);
  }

 private:
  /** \brief Reference to optimization problem */
  steam::Problem &problem_;
  /** \brief Collection of state variables */
  const steam::StateVector::WeakPtr state_vector_;

  Eigen::SimplicialLLT<Eigen::SparseMatrix<double>, Eigen::Upper> hessian_solver_;
  bool pattern_initialized_ = false;
};

inline double AngularDistance(const Eigen::Matrix3d &rota, const Eigen::Matrix3d &rotb) {
  double norm = ((rota * rotb.transpose()).trace() - 1) / 2;
  norm = std::acos(norm) * 180 / M_PI;
  return norm;
}

/* -------------------------------------------------------------------------------------------------------------- */
// Subsample to keep one random point in every voxel of the current frame
void sub_sample_frame(std::vector<Point3D> &frame, double size_voxel) {
  std::unordered_map<Voxel, std::vector<Point3D>> grid;
  for (int i = 0; i < (int)frame.size(); i++) {
    auto kx = static_cast<short>(frame[i].pt[0] / size_voxel);
    auto ky = static_cast<short>(frame[i].pt[1] / size_voxel);
    auto kz = static_cast<short>(frame[i].pt[2] / size_voxel);
    grid[Voxel(kx, ky, kz)].push_back(frame[i]);
  }
  frame.resize(0);
  int step = 0;  // to take one random point inside each voxel (but with identical results when lunching the SLAM a
                 // second time)
  for (const auto &n : grid) {
    if (n.second.size() > 0) {
      // frame.push_back(n.second[step % (int)n.second.size()]);
      frame.push_back(n.second[0]);
      step++;
    }
  }
}

/* -------------------------------------------------------------------------------------------------------------- */
void grid_sampling(const std::vector<Point3D> &frame, std::vector<Point3D> &keypoints, double size_voxel_subsampling) {
  keypoints.resize(0);
  std::vector<Point3D> frame_sub;
  frame_sub.resize(frame.size());
  for (int i = 0; i < (int)frame_sub.size(); i++) {
    frame_sub[i] = frame[i];
  }
  sub_sample_frame(frame_sub, size_voxel_subsampling);
  keypoints.reserve(frame_sub.size());
  for (int i = 0; i < (int)frame_sub.size(); i++) {
    keypoints.push_back(frame_sub[i]);
  }
}

/* -------------------------------------------------------------------------------------------------------------- */

struct Neighborhood {
  Eigen::Vector3d center = Eigen::Vector3d::Zero();
  Eigen::Vector3d normal = Eigen::Vector3d::Zero();
  Eigen::Matrix3d covariance = Eigen::Matrix3d::Identity();
  double a2D = 1.0;  // Planarity coefficient
};
// Computes normal and planarity coefficient
Neighborhood compute_neighborhood_distribution(const ArrayVector3d &points) {
  Neighborhood neighborhood;
  // Compute the normals
  Eigen::Vector3d barycenter(Eigen::Vector3d(0, 0, 0));
  for (auto &point : points) {
    barycenter += point;
  }
  barycenter /= (double)points.size();
  neighborhood.center = barycenter;

  Eigen::Matrix3d covariance_Matrix(Eigen::Matrix3d::Zero());
  for (auto &point : points) {
    for (int k = 0; k < 3; ++k)
      for (int l = k; l < 3; ++l) covariance_Matrix(k, l) += (point(k) - barycenter(k)) * (point(l) - barycenter(l));
  }
  covariance_Matrix(1, 0) = covariance_Matrix(0, 1);
  covariance_Matrix(2, 0) = covariance_Matrix(0, 2);
  covariance_Matrix(2, 1) = covariance_Matrix(1, 2);
  neighborhood.covariance = covariance_Matrix;
  Eigen::SelfAdjointEigenSolver<Eigen::Matrix3d> es(covariance_Matrix);
  Eigen::Vector3d normal(es.eigenvectors().col(0).normalized());
  neighborhood.normal = normal;

  // Compute planarity from the eigen values
  double sigma_1 = sqrt(std::abs(es.eigenvalues()[2]));  // Be careful, the eigenvalues are not correct with the
                                                         // iterative way to compute the covariance matrix
  double sigma_2 = sqrt(std::abs(es.eigenvalues()[1]));
  double sigma_3 = sqrt(std::abs(es.eigenvalues()[0]));
  neighborhood.a2D = (sigma_2 - sigma_3) / sigma_1;

  if (neighborhood.a2D != neighborhood.a2D) {
    LOG(ERROR) << "FOUND NAN!!!";
    throw std::runtime_error("error");
  }

  return neighborhood;
}

}  // namespace

SteamOdometry::SteamOdometry(const Options &options) : Odometry(options), options_(options) {
  // iniitalize steam vars
  T_sr_var_ = steam::se3::SE3StateVar::MakeShared(lgmath::se3::Transformation(options_.T_sr));
  T_sr_var_->locked() = true;
}

SteamOdometry::~SteamOdometry() {
  using namespace steam::traj;

  std::ofstream trajectory_file;
  auto now = std::chrono::system_clock::now();
  auto utc = std::chrono::duration_cast<std::chrono::seconds>(now.time_since_epoch()).count();
  trajectory_file.open(options_.debug_path + "/trajectory_" + std::to_string(utc) + ".txt", std::ios::out);
  // trajectory_file.open(options_.debug_path + "/trajectory.txt", std::ios::out);

  LOG(INFO) << "Building full trajectory." << std::endl;
  auto full_trajectory = steam::traj::const_vel::Interface::MakeShared(options_.qc_inv);
  for (auto &var : trajectory_vars_) {
    full_trajectory->add(var.time, var.T_rm, var.w_mr_inr);
  }

  LOG(INFO) << "Dumping trajectory." << std::endl;
  double begin_time = trajectory_.front().begin_timestamp;
  double end_time = trajectory_.back().end_timestamp;
  double dt = 0.01;
  for (double time = begin_time; time <= end_time; time += dt) {
    Time steam_time(time);
    //
    const auto T_rm = full_trajectory->getPoseInterpolator(steam_time)->evaluate().matrix();
    const auto w_mr_inr = full_trajectory->getVelocityInterpolator(steam_time)->evaluate();
    //
    trajectory_file << (0.0) << " " << steam_time.nanosecs() << " ";
    trajectory_file << T_rm(0, 0) << " " << T_rm(0, 1) << " " << T_rm(0, 2) << " " << T_rm(0, 3) << " ";
    trajectory_file << T_rm(1, 0) << " " << T_rm(1, 1) << " " << T_rm(1, 2) << " " << T_rm(1, 3) << " ";
    trajectory_file << T_rm(2, 0) << " " << T_rm(2, 1) << " " << T_rm(2, 2) << " " << T_rm(2, 3) << " ";
    trajectory_file << T_rm(3, 0) << " " << T_rm(3, 1) << " " << T_rm(3, 2) << " " << T_rm(3, 3) << " ";
    trajectory_file << w_mr_inr(0) << " " << w_mr_inr(1) << " " << w_mr_inr(2) << " " << w_mr_inr(3) << " ";
    trajectory_file << w_mr_inr(4) << " " << w_mr_inr(5) << std::endl;
  }
  LOG(INFO) << "Dumping trajectory. - DONE" << std::endl;
}

Trajectory SteamOdometry::trajectory() {
  LOG(INFO) << "Building full trajectory." << std::endl;
  auto full_trajectory = steam::traj::const_vel::Interface::MakeShared(options_.qc_inv);
  for (auto &var : trajectory_vars_) {
    full_trajectory->add(var.time, var.T_rm, var.w_mr_inr);
  }

  LOG(INFO) << "Updating trajectory." << std::endl;
  using namespace steam::se3;
  using namespace steam::traj;
  for (auto &frame : trajectory_) {
    Time begin_steam_time(frame.begin_timestamp);
    const auto begin_T_mr = inverse(full_trajectory->getPoseInterpolator(begin_steam_time))->evaluate().matrix();
    const auto begin_T_ms = begin_T_mr * options_.T_sr.inverse();
    frame.begin_R = begin_T_ms.block<3, 3>(0, 0);
    frame.begin_t = begin_T_ms.block<3, 1>(0, 3);

    Time end_steam_time(frame.end_timestamp);
    const auto end_T_mr = inverse(full_trajectory->getPoseInterpolator(end_steam_time))->evaluate().matrix();
    const auto end_T_ms = end_T_mr * options_.T_sr.inverse();
    frame.end_R = end_T_ms.block<3, 3>(0, 0);
    frame.end_t = end_T_ms.block<3, 1>(0, 3);
  }

  return trajectory_;
}

auto SteamOdometry::registerFrame(const std::vector<Point3D> &const_frame) -> RegistrationSummary {
  RegistrationSummary summary;

  // add a new frame
  int index_frame = trajectory_.size();
  trajectory_.emplace_back();

  //
  initializeTimestamp(index_frame, const_frame);

  //
  initializeMotion(index_frame);

  //
  auto frame = initializeFrame(index_frame, const_frame);

  //
  if (index_frame > 0) {
    double sample_voxel_size =
        index_frame < options_.init_num_frames ? options_.init_sample_voxel_size : options_.sample_voxel_size;

    // downsample
    std::vector<Point3D> keypoints;
    grid_sampling(frame, keypoints, sample_voxel_size);

    // icp
    summary.success = icp(index_frame, keypoints);
    summary.keypoints = keypoints;
    if (!summary.success) return summary;
  } else {
    using namespace steam;
    using namespace steam::se3;
    using namespace steam::vspace;
    using namespace steam::traj;

    // initial state
    lgmath::se3::Transformation T_rm;
    Eigen::Matrix<double, 6, 1> w_mr_inr = Eigen::Matrix<double, 6, 1>::Zero();

    // initialize frame (the beginning of the trajectory)
    const double begin_time = trajectory_[index_frame].begin_timestamp;
    Time begin_steam_time(begin_time);
    const auto begin_T_rm_var = SE3StateVar::MakeShared(T_rm);
    const auto begin_w_mr_inr_var = VSpaceStateVar<6>::MakeShared(w_mr_inr);
    trajectory_vars_.emplace_back(begin_steam_time, begin_T_rm_var, begin_w_mr_inr_var);

    // the end of current scan (this is the first state that could be optimized)
    const double end_time = trajectory_[index_frame].end_timestamp;
    Time end_steam_time(end_time);
    const auto end_T_rm_var = SE3StateVar::MakeShared(T_rm);
    const auto end_w_mr_inr_var = VSpaceStateVar<6>::MakeShared(w_mr_inr);
    trajectory_vars_.emplace_back(end_steam_time, end_T_rm_var, end_w_mr_inr_var);

    trajectory_[index_frame].end_T_rm_cov = Eigen::Matrix<double, 6, 6>::Identity() * 1e-4;
    trajectory_[index_frame].end_w_mr_inr_cov = Eigen::Matrix<double, 6, 6>::Identity() * 1e-4;
    trajectory_[index_frame].end_state_cov = Eigen::Matrix<double, 12, 12>::Identity() * 1e-4;

    summary.success = true;
  }
  trajectory_[index_frame].points = frame;

  // add points
  if (index_frame == 0 || (options_.delay_adding_points == 0)) {
    updateMap(index_frame, index_frame);
  } else if (index_frame > 1) {
    updateMap(index_frame, index_frame - 1);
  }

  summary.corrected_points = frame;

#if false
  // correct all points
  summary.all_corrected_points = const_frame;
  auto q_begin = Eigen::Quaterniond(trajectory_[index_frame].begin_R);
  auto q_end = Eigen::Quaterniond(trajectory_[index_frame].end_R);
  Eigen::Vector3d t_begin = trajectory_[index_frame].begin_t;
  Eigen::Vector3d t_end = trajectory_[index_frame].end_t;
  for (auto &point : summary.all_corrected_points) {
    double alpha_timestamp = point.alpha_timestamp;
    Eigen::Matrix3d R = q_begin.slerp(alpha_timestamp, q_end).normalized().toRotationMatrix();
    Eigen::Vector3d t = (1.0 - alpha_timestamp) * t_begin + alpha_timestamp * t_end;
    point.pt = R * point.raw_pt + t;
  }
#endif

  summary.R_ms = trajectory_[index_frame].end_R;
  summary.t_ms = trajectory_[index_frame].end_t;

  return summary;
}

void SteamOdometry::initializeTimestamp(int index_frame, const std::vector<Point3D> &const_frame) {
  double min_timestamp = std::numeric_limits<double>::max();
  double max_timestamp = std::numeric_limits<double>::min();
  for (const auto &point : const_frame) {
    if (point.timestamp > max_timestamp) max_timestamp = point.timestamp;
    if (point.timestamp < min_timestamp) min_timestamp = point.timestamp;
  }
  trajectory_[index_frame].begin_timestamp = min_timestamp;
  trajectory_[index_frame].end_timestamp = max_timestamp;
}

void SteamOdometry::initializeMotion(int index_frame) {
  if (index_frame <= 1) {
    // Initialize first pose at Identity
    trajectory_[index_frame].begin_R = Eigen::MatrixXd::Identity(3, 3);
    trajectory_[index_frame].begin_t = Eigen::Vector3d(0., 0., 0.);
    trajectory_[index_frame].end_R = Eigen::MatrixXd::Identity(3, 3);
    trajectory_[index_frame].end_t = Eigen::Vector3d(0., 0., 0.);
  } else {
    // Different regimen for the second frame due to the bootstrapped elasticity
    Eigen::Matrix3d R_next_end = trajectory_[index_frame - 1].end_R * trajectory_[index_frame - 2].end_R.inverse() *
                                 trajectory_[index_frame - 1].end_R;
    Eigen::Vector3d t_next_end = trajectory_[index_frame - 1].end_t +
                                 trajectory_[index_frame - 1].end_R * trajectory_[index_frame - 2].end_R.inverse() *
                                     (trajectory_[index_frame - 1].end_t - trajectory_[index_frame - 2].end_t);

    trajectory_[index_frame].begin_R = trajectory_[index_frame - 1].end_R;
    trajectory_[index_frame].begin_t = trajectory_[index_frame - 1].end_t;
    trajectory_[index_frame].end_R = R_next_end;
    trajectory_[index_frame].end_t = t_next_end;
  }
}

std::vector<Point3D> SteamOdometry::initializeFrame(int index_frame, const std::vector<Point3D> &const_frame) {
  std::vector<Point3D> frame(const_frame);
  double sample_size = index_frame < options_.init_num_frames ? options_.init_voxel_size : options_.voxel_size;
  std::mt19937_64 g;
  std::shuffle(frame.begin(), frame.end(), g);
  // Subsample the scan with voxels taking one random in every voxel
  sub_sample_frame(frame, sample_size);
  std::shuffle(frame.begin(), frame.end(), g);

  // initialize points
  auto q_begin = Eigen::Quaterniond(trajectory_[index_frame].begin_R);
  auto q_end = Eigen::Quaterniond(trajectory_[index_frame].end_R);
  Eigen::Vector3d t_begin = trajectory_[index_frame].begin_t;
  Eigen::Vector3d t_end = trajectory_[index_frame].end_t;
  for (auto &point : frame) {
    double alpha_timestamp = point.alpha_timestamp;
    Eigen::Matrix3d R = q_begin.slerp(alpha_timestamp, q_end).normalized().toRotationMatrix();
    Eigen::Vector3d t = (1.0 - alpha_timestamp) * t_begin + alpha_timestamp * t_end;
    //
    point.pt = R * point.raw_pt + t;
  }

  return frame;
}

void SteamOdometry::updateMap(int index_frame, int update_frame) {
  const double kSizeVoxelMap = options_.size_voxel_map;
  const double kMinDistancePoints = options_.min_distance_points;
  const int kMaxNumPointsInVoxel = options_.max_num_points_in_voxel;

  // update frame
  auto &frame = trajectory_[update_frame].points;
#if false
  auto q_begin = Eigen::Quaterniond(trajectory_[update_frame].begin_R);
  auto q_end = Eigen::Quaterniond(trajectory_[update_frame].end_R);
  Eigen::Vector3d t_begin = trajectory_[update_frame].begin_t;
  Eigen::Vector3d t_end = trajectory_[update_frame].end_t;
  for (auto &point : frame) {
    // modifies the world point of the frame based on the raw_pt
    double alpha_timestamp = point.alpha_timestamp;
    Eigen::Matrix3d R = q_begin.slerp(alpha_timestamp, q_end).normalized().toRotationMatrix();
    Eigen::Vector3d t = (1.0 - alpha_timestamp) * t_begin + alpha_timestamp * t_end;
    //
    point.pt = R * point.raw_pt + t;
  }
#else
  using namespace steam::se3;
  using namespace steam::traj;

  const auto update_trajectory = const_vel::Interface::MakeShared(options_.qc_inv);
  const auto &prev_var = trajectory_vars_.at(update_frame);
  update_trajectory->add(prev_var.time, prev_var.T_rm, prev_var.w_mr_inr);
  const auto &curr_var = trajectory_vars_.at(update_frame + 1);
  if ((prev_var.time != Time(trajectory_[update_frame - 1].end_timestamp)) &&
      (prev_var.time != Time(trajectory_[update_frame].begin_timestamp)))
    throw std::runtime_error{"previous end timestamp mismatch"};
  if (curr_var.time != Time(trajectory_[update_frame].end_timestamp))
    throw std::runtime_error{"current end timestamp mismatch"};
  update_trajectory->add(curr_var.time, curr_var.T_rm, curr_var.w_mr_inr);

  for (auto &point : frame) {
    const double query_time = point.timestamp;

    const auto T_rm_intp_eval = update_trajectory->getPoseInterpolator(Time(query_time));
    const auto T_ms_intp_eval = inverse(compose(T_sr_var_, T_rm_intp_eval));

    Eigen::Matrix4d T_ms = T_ms_intp_eval->evaluate().matrix();
    Eigen::Matrix3d R = T_ms.block<3, 3>(0, 0);
    Eigen::Vector3d t = T_ms.block<3, 1>(0, 3);
    //
    point.pt = R * point.raw_pt + t;
  }
#endif

  map_.add(frame, kSizeVoxelMap, kMaxNumPointsInVoxel, kMinDistancePoints);
  frame.clear();

  // remove points
  const double kMaxDistance = options_.max_distance;
  const Eigen::Vector3d location = trajectory_[index_frame].end_t;
  map_.remove(location, kMaxDistance);
}

bool SteamOdometry::icp(int index_frame, std::vector<Point3D> &keypoints) {
  using namespace steam;
  using namespace steam::se3;
  using namespace steam::traj;
  using namespace steam::vspace;

  bool icp_success = true;

  ///
  const auto steam_trajectory = const_vel::Interface::MakeShared(options_.qc_inv);
  std::vector<StateVarBase::Ptr> steam_state_vars;
  std::vector<BaseCostTerm::ConstPtr> prior_cost_terms;

  /// use previous trajectory to initialize steam state variables
  LOG(INFO) << "[CT_ICP_STEAM] prev scan end time: " << trajectory_[index_frame - 1].end_timestamp << std::endl;
  const double prev_time = trajectory_[index_frame - 1].end_timestamp;
  if (trajectory_vars_.back().time != Time(static_cast<double>(prev_time)))
    throw std::runtime_error{"missing previous scan end variable"};
  Time prev_steam_time = trajectory_vars_.back().time;
  lgmath::se3::Transformation prev_T_rm = trajectory_vars_.back().T_rm->value();
  Eigen::Matrix<double, 6, 1> prev_w_mr_inr = trajectory_vars_.back().w_mr_inr->value();
  Eigen::Matrix<double, 6, 6> prev_T_rm_cov = trajectory_[index_frame - 1].end_T_rm_cov;
  Eigen::Matrix<double, 6, 6> prev_w_mr_inr_cov = trajectory_[index_frame - 1].end_w_mr_inr_cov;
  Eigen::Matrix<double, 12, 12> prev_state_cov = trajectory_[index_frame - 1].end_state_cov;
  const auto prev_T_rm_var = trajectory_vars_.back().T_rm;
  const auto prev_w_mr_inr_var = trajectory_vars_.back().w_mr_inr;
  steam_trajectory->add(prev_steam_time, prev_T_rm_var, prev_w_mr_inr_var);
  steam_state_vars.emplace_back(prev_T_rm_var);
  steam_state_vars.emplace_back(prev_w_mr_inr_var);

  /// New state for this frame
  LOG(INFO) << "[CT_ICP_STEAM] curr scan end time: " << trajectory_[index_frame].end_timestamp << std::endl;
  LOG(INFO) << "[CT_ICP_STEAM] total num new states: " << (options_.num_extra_states + 1) << std::endl;
  const double curr_time = trajectory_[index_frame].end_timestamp;
  const int num_states = options_.num_extra_states + 1;
  const double time_diff = (curr_time - prev_time) / static_cast<double>(num_states);
  std::vector<double> knot_times;
  knot_times.reserve(num_states);
  for (int i = 0; i < options_.num_extra_states; ++i) {
    knot_times.emplace_back(prev_time + (double)(i + 1) * time_diff);
  }
  knot_times.emplace_back(curr_time);

  for (size_t i = 0; i < knot_times.size(); ++i) {
    double knot_time = knot_times[i];
    Time knot_steam_time(knot_time);
    //
    const Eigen::Matrix<double, 6, 1> xi_mr_inr_odo((knot_steam_time - prev_steam_time).seconds() * prev_w_mr_inr);
    const auto knot_T_rm = lgmath::se3::Transformation(xi_mr_inr_odo) * prev_T_rm;
    const auto T_rm_var = SE3StateVar::MakeShared(knot_T_rm);
    //
    const auto w_mr_inr_var = VSpaceStateVar<6>::MakeShared(prev_w_mr_inr);
    //
    steam_trajectory->add(knot_steam_time, T_rm_var, w_mr_inr_var);
    steam_state_vars.emplace_back(T_rm_var);
    steam_state_vars.emplace_back(w_mr_inr_var);
    //
    if (options_.use_vp) {
      const auto error_func = vspace_error<6>(w_mr_inr_var, Eigen::Matrix<double, 6, 1>::Zero());
      const auto noise_model = StaticNoiseModel<6>::MakeShared(options_.vp_cov);
      const auto loss_func = std::make_shared<L2LossFunc>();
      prior_cost_terms.emplace_back(WeightedLeastSqCostTerm<6>::MakeShared(error_func, noise_model, loss_func));
    }
    // cache the end state in full steam trajectory because it will be used again
    if (i == (knot_times.size() - 1)) {
      trajectory_vars_.emplace_back(knot_steam_time, T_rm_var, w_mr_inr_var);
    }
  }

  // Get evaluator for query points
  std::vector<Evaluable<const_vel::Interface::PoseType>::ConstPtr> T_ms_intp_eval_vec;
  std::vector<Evaluable<const_vel::Interface::VelocityType>::ConstPtr> w_ms_ins_intp_eval_vec;
  T_ms_intp_eval_vec.reserve(keypoints.size());
  for (const auto &keypoint : keypoints) {
    const auto query_time = keypoint.timestamp;
    // pose
    const auto T_rm_intp_eval = steam_trajectory->getPoseInterpolator(Time(query_time));
    const auto T_ms_intp_eval = inverse(compose(T_sr_var_, T_rm_intp_eval));
    T_ms_intp_eval_vec.emplace_back(T_ms_intp_eval);
    // velocity
    const auto w_mr_inr_intp_eval = steam_trajectory->getVelocityInterpolator(Time(query_time));
    const auto w_ms_ins_intp_eval = compose_velocity(T_sr_var_, w_mr_inr_intp_eval);
    w_ms_ins_intp_eval_vec.emplace_back(w_ms_ins_intp_eval);
  }

  // For the 50 first frames, visit 2 voxels
  const short nb_voxels_visited = index_frame < options_.init_num_frames ? 2 : 1;
  int number_keypoints_used = 0;
  const int kMinNumNeighbors = options_.min_number_neighbors;

  // timers
  std::vector<std::pair<std::string, std::unique_ptr<Stopwatch<>>>> timer;
  timer.emplace_back("Association .................... ", std::make_unique<Stopwatch<>>(false));
  timer.emplace_back("Optimization ................... ", std::make_unique<Stopwatch<>>(false));
  timer.emplace_back("Update Transform ............... ", std::make_unique<Stopwatch<>>(false));
  timer.emplace_back("Alignment ...................... ", std::make_unique<Stopwatch<>>(false));
  std::vector<std::pair<std::string, std::unique_ptr<Stopwatch<>>>> inner_timer;
  inner_timer.emplace_back("Search Neighbors ............. ", std::make_unique<Stopwatch<>>(false));
  inner_timer.emplace_back("Compute Normal ............... ", std::make_unique<Stopwatch<>>(false));
  inner_timer.emplace_back("Add Cost Term ................ ", std::make_unique<Stopwatch<>>(false));
  bool innerloop_time = (options_.num_threads == 1);

  //
  int num_iter_icp = index_frame < options_.init_num_frames ? 15 : options_.num_iters_icp;
  num_iter_icp += options_.no_prev_state_iters;
  int ready_to_add_prev_state = 0;  // 0=not ready, 1=ready, 2=already added
  for (int iter(0); iter < num_iter_icp; iter++) {
    if (!options_.add_prev_state) ready_to_add_prev_state = 2;  // assume already added
    if (iter >= options_.no_prev_state_iters && ready_to_add_prev_state == 0) ready_to_add_prev_state = 1;
    LOG_IF(INFO, (ready_to_add_prev_state == 1))
        << "Iteration " << iter << " with ready_to_add_prev_state set to 1" << std::endl;

#if true
    if (options_.add_prev_state && ready_to_add_prev_state == 1) {
      if (prev_time < curr_time) {
        LOG(INFO) << "[CT_ICP_STEAM] The end of last scan < end of current scan with dt=" << std::setprecision(8)
                  << std::fixed << (curr_time - prev_time) << std::endl;
        // lock
        if (options_.lock_prev_pose) prev_T_rm_var->locked() = true;
        if (options_.lock_prev_vel) prev_w_mr_inr_var->locked() = true;
        // prior
        if (options_.prev_pose_as_prior && options_.prev_vel_as_prior)
          steam_trajectory->addStatePrior(prev_steam_time, prev_T_rm, prev_w_mr_inr, prev_state_cov);
        else if (options_.prev_pose_as_prior)
          steam_trajectory->addPosePrior(prev_steam_time, prev_T_rm, prev_T_rm_cov);
        else if (options_.prev_vel_as_prior)
          steam_trajectory->addVelocityPrior(prev_steam_time, prev_w_mr_inr, prev_w_mr_inr_cov);
      } else {
        LOG(ERROR) << "[CT_ICP_STEAM] The end of last scan > end of current scan with frame=" << index_frame
                   << ", dt=" << std::setprecision(8) << std::fixed << (curr_time - prev_time) << std::endl;
        // throw std::runtime_error("[CT_ICP_STEAM] The end of last scan > beginning of current scan - not possible!");
      }
      ready_to_add_prev_state = 2;
    }
#endif
    number_keypoints_used = 0;

    // initialize problem
    OptimizationProblem problem(/* num_threads */ options_.num_threads);

    // add variables
    for (const auto &var : steam_state_vars) problem.addStateVariable(var);

    // add prior cost terms
    steam_trajectory->addPriorCostTerms(problem);
    for (const auto &prior_cost_term : prior_cost_terms) problem.addCostTerm(prior_cost_term);

    timer[0].second->start();

#pragma omp parallel for num_threads(options_.num_threads)
    for (int i = 0; i < (int)keypoints.size(); i++) {
      const auto &keypoint = keypoints[i];
      const auto &pt_keypoint = keypoint.pt;

      if (innerloop_time) inner_timer[0].second->start();

      // Neighborhood search
      ArrayVector3d vector_neighbors =
          map_.searchNeighbors(pt_keypoint, nb_voxels_visited, options_.size_voxel_map, options_.max_number_neighbors);

      if (innerloop_time) inner_timer[0].second->stop();

      if ((int)vector_neighbors.size() < kMinNumNeighbors) {
        continue;
      }

      if (innerloop_time) inner_timer[1].second->start();

      // Compute normals from neighbors
      const auto neighborhood = compute_neighborhood_distribution(vector_neighbors);
      const double planarity_weight = neighborhood.a2D;
      const auto &normal = neighborhood.normal;

      const double weight = planarity_weight * planarity_weight;

      Eigen::Vector3d closest_pt_normal = weight * normal;

      Eigen::Vector3d closest_point = vector_neighbors[0];

      double dist_to_plane = normal[0] * (pt_keypoint[0] - closest_point[0]) +
                             normal[1] * (pt_keypoint[1] - closest_point[1]) +
                             normal[2] * (pt_keypoint[2] - closest_point[2]);
      dist_to_plane = std::abs(dist_to_plane);

      if (innerloop_time) inner_timer[1].second->stop();

      if (innerloop_time) inner_timer[2].second->start();

      double max_dist_to_plane =
          (iter >= options_.p2p_initial_iters ? options_.p2p_refined_max_dist : options_.p2p_initial_max_dist);
      bool use_p2p = (dist_to_plane < max_dist_to_plane);
      if (use_p2p) {
        /// \note query and reference point
        ///   const auto qry_pt = keypoint.raw_pt;
        ///   const auto ref_pt = closest_point;
        if (options_.use_rv && options_.merge_p2p_rv) {
          Eigen::Matrix4d W = Eigen::Matrix4d::Identity();
          W.block<3, 3>(0, 0) =
              (closest_pt_normal * closest_pt_normal.transpose() + 1e-5 * Eigen::Matrix3d::Identity());
          W.block<1, 1>(3, 3) = options_.rv_cov_inv * Eigen::Matrix<double, 1, 1>::Identity();
          const auto noise_model = StaticNoiseModel<4>::MakeShared(W, NoiseType::INFORMATION);

          const auto &T_ms_intp_eval = T_ms_intp_eval_vec[i];
          const auto &w_ms_ins_intp_eval = w_ms_ins_intp_eval_vec[i];
          const auto p2p_error = p2p::p2pError(T_ms_intp_eval, closest_point, keypoint.raw_pt);
          const auto rv_error = p2p::radialVelError(w_ms_ins_intp_eval, keypoint.raw_pt, keypoint.radial_velocity);
          const auto error_func = p2p::p2prvError(p2p_error, rv_error);

          // const auto loss_func = L2LossFunc::MakeShared(); /// \todo what loss threshold to use???
          const auto loss_func = GemanMcClureLossFunc::MakeShared(options_.rv_loss_threshold);

          const auto cost = WeightedLeastSqCostTerm<4>::MakeShared(error_func, noise_model, loss_func);

#pragma omp critical(odometry_cost_term)
          {
            problem.addCostTerm(cost);

            number_keypoints_used++;
          }

        } else {
          Eigen::Matrix3d W = (closest_pt_normal * closest_pt_normal.transpose() + 1e-5 * Eigen::Matrix3d::Identity());
          const auto noise_model = StaticNoiseModel<3>::MakeShared(W, NoiseType::INFORMATION);

          const auto &T_ms_intp_eval = T_ms_intp_eval_vec[i];
          const auto error_func = p2p::p2pError(T_ms_intp_eval, closest_point, keypoint.raw_pt);

          const auto loss_func = [this]() -> BaseLossFunc::Ptr {
            switch (options_.p2p_loss_func) {
              case STEAM_LOSS_FUNC::L2:
                return L2LossFunc::MakeShared();
              case STEAM_LOSS_FUNC::DCS:
                return DcsLossFunc::MakeShared(options_.p2p_loss_sigma);
              case STEAM_LOSS_FUNC::CAUCHY:
                return CauchyLossFunc::MakeShared(options_.p2p_loss_sigma);
              case STEAM_LOSS_FUNC::GM:
                return GemanMcClureLossFunc::MakeShared(options_.p2p_loss_sigma);
              default:
                return nullptr;
            }
            return nullptr;
          }();

          const auto cost = WeightedLeastSqCostTerm<3>::MakeShared(error_func, noise_model, loss_func);

#pragma omp critical(odometry_cost_term)
          {
            problem.addCostTerm(cost);

            number_keypoints_used++;
          }
        }
      }

      if (options_.use_rv && ((!use_p2p) || (use_p2p && !options_.merge_p2p_rv))) {
        Eigen::Matrix<double, 1, 1> W = options_.rv_cov_inv * Eigen::Matrix<double, 1, 1>::Identity();
        const auto noise_model = StaticNoiseModel<1>::MakeShared(W, NoiseType::INFORMATION);

        const auto &w_ms_ins_intp_eval = w_ms_ins_intp_eval_vec[i];
        const auto error_func = p2p::radialVelError(w_ms_ins_intp_eval, keypoint.raw_pt, keypoint.radial_velocity);

        const auto loss_func = GemanMcClureLossFunc::MakeShared(options_.rv_loss_threshold);

        const auto cost = WeightedLeastSqCostTerm<1>::MakeShared(error_func, noise_model, loss_func);

#pragma omp critical(odometry_cost_term)
        { problem.addCostTerm(cost); }
      }

      if (innerloop_time) inner_timer[2].second->stop();
    }

    timer[0].second->stop();

    if (number_keypoints_used < 100) {
      LOG(ERROR) << "[CT_ICP]Error : not enough keypoints selected in ct-icp !" << std::endl;
      LOG(ERROR) << "[CT_ICP]Number_of_residuals : " << number_keypoints_used << std::endl;
      icp_success = false;
      break;
    }

    timer[1].second->start();

    // Solve
    GaussNewtonSolver::Params params;
    params.verbose = options_.verbose;
    if (options_.add_prev_state && (ready_to_add_prev_state == 2) && (!options_.association_after_adding_prev_state)) {
      LOG(INFO) << "Changing maxIteration to 20 due to no re-association." << std::endl;
      params.max_iterations = 20;
    } else {
      params.max_iterations = (unsigned int)options_.max_iterations;
    }
    GaussNewtonSolver solver(problem, params);
    solver.optimize();
    Covariance covariance(solver);

    timer[1].second->stop();

    timer[2].second->start();

    // Update (changes trajectory data)
    double diff_trans = 0, diff_rot = 0;

    auto &current_estimate = trajectory_[index_frame];

    Time begin_steam_time(static_cast<double>(trajectory_[index_frame].begin_timestamp));
    const auto begin_T_mr = inverse(steam_trajectory->getPoseInterpolator(begin_steam_time))->evaluate().matrix();
    const auto begin_T_ms = begin_T_mr * options_.T_sr.inverse();
    diff_trans += (current_estimate.begin_t - begin_T_ms.block<3, 1>(0, 3)).norm();
    diff_rot += AngularDistance(current_estimate.begin_R, begin_T_ms.block<3, 3>(0, 0));
    current_estimate.begin_R = begin_T_ms.block<3, 3>(0, 0);
    current_estimate.begin_t = begin_T_ms.block<3, 1>(0, 3);

    Time end_steam_time(static_cast<double>(trajectory_[index_frame].end_timestamp));
    const auto end_T_mr = inverse(steam_trajectory->getPoseInterpolator(end_steam_time))->evaluate().matrix();
    const auto end_T_ms = end_T_mr * options_.T_sr.inverse();
    diff_trans += (current_estimate.end_t - end_T_ms.block<3, 1>(0, 3)).norm();
    diff_rot += AngularDistance(current_estimate.end_R, end_T_ms.block<3, 3>(0, 0));
    current_estimate.end_R = end_T_ms.block<3, 3>(0, 0);
    current_estimate.end_t = end_T_ms.block<3, 1>(0, 3);

    Eigen::MatrixXd curr_end_state_cov = steam_trajectory->getCovariance(covariance, end_steam_time);
    current_estimate.end_T_rm_cov = curr_end_state_cov.block<6, 6>(0, 0);
    current_estimate.end_w_mr_inr_cov = curr_end_state_cov.block<6, 6>(6, 6);
    current_estimate.end_state_cov = curr_end_state_cov;

    timer[2].second->stop();

    timer[3].second->start();

    // Update keypoints
#pragma omp parallel for num_threads(options_.num_threads)
    for (int i = 0; i < (int)keypoints.size(); i++) {
      auto &keypoint = keypoints[i];
      const auto &T_ms_intp_eval = T_ms_intp_eval_vec[i];

      const auto T_ms = T_ms_intp_eval->evaluate().matrix();
      keypoint.pt = T_ms.block<3, 3>(0, 0) * keypoint.raw_pt + T_ms.block<3, 1>(0, 3);
    }

    timer[3].second->stop();

    if ((index_frame > 1) && iter >= options_.p2p_initial_iters &&
        (diff_rot < options_.threshold_orientation_norm && diff_trans < options_.threshold_translation_norm)) {
      if (options_.debug_print) {
        LOG(INFO) << "CT_ICP: Finished with N=" << iter << " ICP iterations" << std::endl;
      }

      if (ready_to_add_prev_state == 0)
        ready_to_add_prev_state = 1;
      else
        break;
    }

    if (options_.add_prev_state && ready_to_add_prev_state == 2 && (!options_.association_after_adding_prev_state))
      break;
  }
  LOG(INFO) << "Number of keypoints used in CT-ICP : " << number_keypoints_used << std::endl;

  /// Debug print
  if (options_.debug_print) {
    for (size_t i = 0; i < timer.size(); i++)
      LOG(INFO) << "Elapsed " << timer[i].first << *(timer[i].second) << std::endl;
    for (size_t i = 0; i < inner_timer.size(); i++)
      LOG(INFO) << "Elapsed (Inner Loop) " << inner_timer[i].first << *(inner_timer[i].second) << std::endl;
    LOG(INFO) << "Number iterations CT-ICP : " << options_.num_iters_icp << std::endl;
    LOG(INFO) << "Translation Begin: " << trajectory_[index_frame].begin_t.transpose() << std::endl;
    LOG(INFO) << "Translation End: " << trajectory_[index_frame].end_t.transpose() << std::endl;
  }

  return icp_success;
}

}  // namespace steam_icp
