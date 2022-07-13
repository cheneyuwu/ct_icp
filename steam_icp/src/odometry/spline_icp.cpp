#include "steam_icp/odometry/spline_icp.hpp"

#include <iomanip>
#include <random>

#include <glog/logging.h>

#include "steam.hpp"

#include "steam_icp/utils/stopwatch.hpp"

namespace steam_icp {

namespace {

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

}  // namespace

auto SplineOdometry::registerFrame(const std::vector<Point3D> &const_frame) -> RegistrationSummary {
  RegistrationSummary summary;

  // add a new frame
  int index_frame = trajectory_.size();
  trajectory_.emplace_back();

  //
  auto frame = initializeFrame(index_frame, const_frame);

  double sample_voxel_size =
      index_frame < options_.init_num_frames ? options_.init_sample_voxel_size : options_.sample_voxel_size;

  // downsample
  std::vector<Point3D> keypoints;
  grid_sampling(frame, keypoints, sample_voxel_size);
  summary.sample_size = (int)keypoints.size();

  // icp
  icp(index_frame, keypoints, summary);
  summary.keypoints = keypoints;
  summary.frame = trajectory_[index_frame];
  if (!summary.success) return summary;

  return summary;
}

std::vector<Point3D> SplineOdometry::initializeFrame(int index_frame, const std::vector<Point3D> &const_frame) {
  /// PREPROCESS THE INITIAL FRAME
  double sample_size = index_frame < options_.init_num_frames ? options_.init_voxel_size : options_.voxel_size;
  std::vector<Point3D> frame(const_frame);

  std::mt19937_64 g;
  std::shuffle(frame.begin(), frame.end(), g);
  // Subsample the scan with voxels taking one random in every voxel
  sub_sample_frame(frame, sample_size);
  std::shuffle(frame.begin(), frame.end(), g);

  double min_timestamp = std::numeric_limits<double>::max();
  double max_timestamp = std::numeric_limits<double>::min();
  for (auto &point : frame) {
    point.index_frame = index_frame;
    if (point.timestamp > max_timestamp) max_timestamp = point.timestamp;
    if (point.timestamp < min_timestamp) min_timestamp = point.timestamp;
  }

  trajectory_[index_frame].begin_timestamp = min_timestamp;
  trajectory_[index_frame].end_timestamp = max_timestamp;

  return frame;
}

void SplineOdometry::icp(int index_frame, std::vector<Point3D> &keypoints, RegistrationSummary &summary) {
  using namespace steam;
  using namespace steam::se3;
  using namespace steam::traj;
  using namespace steam::vspace;

  // timers
  std::vector<std::pair<std::string, std::unique_ptr<Stopwatch<>>>> timer;
  timer.emplace_back("Instantiation .................. ", std::make_unique<Stopwatch<>>(false));
  timer.emplace_back("Optimization ................... ", std::make_unique<Stopwatch<>>(false));

  /// Create robot to sensor transform variable, fixed.
  const auto T_sr_var = SE3StateVar::MakeShared(lgmath::se3::Transformation(options_.T_sr));
  T_sr_var->locked() = true;

  // initialize problem
  OptimizationProblem problem(/* num_threads */ options_.num_threads);

  timer[0].second->start();

  // side slipping constraint
  for (double t = 0.0; t <= 1.0; t += 0.1) {
    const auto query_time = trajectory_[index_frame].begin_timestamp +
                            t * (trajectory_[index_frame].end_timestamp - trajectory_[index_frame].begin_timestamp);
    const auto w_mr_inr_intp_eval = steam_trajectory_->getVelocityInterpolator(Time(query_time));
    const auto error_func = vspace_error<6>(w_mr_inr_intp_eval, Eigen::Matrix<double, 6, 1>::Zero());
    const auto noise_model = StaticNoiseModel<6>::MakeShared(options_.vp_cov);
    const auto loss_func = std::make_shared<L2LossFunc>();
    const auto cost = WeightedLeastSqCostTerm<6>::MakeShared(error_func, noise_model, loss_func);
    problem.addCostTerm(cost);
  }

  // Get evaluator for query points
  std::vector<Evaluable<const_vel::Interface::VelocityType>::ConstPtr> w_ms_ins_intp_eval_vec;
  w_ms_ins_intp_eval_vec.reserve(keypoints.size());
  for (const auto &keypoint : keypoints) {
    const auto query_time =
        trajectory_[index_frame].begin_timestamp +
        keypoint.alpha_timestamp * (trajectory_[index_frame].end_timestamp - trajectory_[index_frame].begin_timestamp);
    // velocity
    const auto w_mr_inr_intp_eval = steam_trajectory_->getVelocityInterpolator(Time(query_time));
    const auto w_ms_ins_intp_eval = compose_velocity(T_sr_var, w_mr_inr_intp_eval);
    w_ms_ins_intp_eval_vec.emplace_back(w_ms_ins_intp_eval);
  }

  // add velocity cost terms
  Eigen::Matrix<double, 1, 1> W = 1.0 * Eigen::Matrix<double, 1, 1>::Identity();
  const auto noise_model = StaticNoiseModel<1>::MakeShared(W, NoiseType::INFORMATION);
#pragma omp parallel for num_threads(options_.num_threads)
  for (int i = 0; i < (int)keypoints.size(); i++) {
    const auto &keypoint = keypoints[i];

    const auto &w_ms_ins_intp_eval = w_ms_ins_intp_eval_vec[i];
    const auto error_func = p2p::radialVelError(w_ms_ins_intp_eval, keypoint.raw_pt, keypoint.radial_velocity);

    const auto loss_func = [this]() -> BaseLossFunc::Ptr {
      switch (options_.rv_loss_func) {
        case STEAM_LOSS_FUNC::L2:
          return L2LossFunc::MakeShared();
        case STEAM_LOSS_FUNC::DCS:
          return DcsLossFunc::MakeShared(options_.rv_loss_threshold);
        case STEAM_LOSS_FUNC::CAUCHY:
          return CauchyLossFunc::MakeShared(options_.rv_loss_threshold);
        case STEAM_LOSS_FUNC::GM:
          return GemanMcClureLossFunc::MakeShared(options_.rv_loss_threshold);
        default:
          return nullptr;
      }
      return nullptr;
    }();

    const auto cost = WeightedLeastSqCostTerm<1>::MakeShared(error_func, noise_model, loss_func);
#pragma omp critical(odometry_cost_term)
    { problem.addCostTerm(cost); }
  }

  //
  const double begin_timestamp = trajectory_[index_frame].begin_timestamp;
  steam_trajectory_->setActiveWindow(Time(begin_timestamp));

  // add variables
  steam_trajectory_->addStateVariables(problem);

  // add prior cost terms
  steam_trajectory_->addPriorCostTerms(problem);

  timer[0].second->stop();

  timer[1].second->start();

  // Solve
  using SolverType = VanillaGaussNewtonSolver;
  SolverType::Params params;
  params.verbose = options_.verbose;
  params.maxIterations = (unsigned int)options_.max_iterations;
  SolverType solver(&problem, params);
  solver.optimize();

  timer[1].second->stop();

  /// Debug print
  if (options_.debug_print) {
    for (size_t i = 0; i < timer.size(); i++)
      LOG(INFO) << "Elapsed " << timer[i].first << *(timer[i].second) << std::endl;
  }

  if (options_.debug_print) {
    const double begin_timestamp = trajectory_[index_frame].begin_timestamp;
    const double end_timestamp = trajectory_[index_frame].end_timestamp;
    const int num_states = 10;
    const double time_diff = (end_timestamp - begin_timestamp) / (static_cast<double>(num_states) - 1.0);
    for (int i = 0; i < num_states; ++i) {
      Time query_time(static_cast<double>(begin_timestamp + (double)i * time_diff));
      velocity_query_times_.push_back(query_time);
    }
  }
}

}  // namespace steam_icp
