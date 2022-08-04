#include "steam_icp/odometry/elastic_icp.hpp"

#include <iomanip>
#include <random>

#include <glog/logging.h>

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

ElasticOdometry::ElasticOdometry(const Options &options) : Odometry(options), options_(options) {}

ElasticOdometry::~ElasticOdometry() {}

Trajectory ElasticOdometry::trajectory() { return trajectory_; }

auto ElasticOdometry::registerFrame(const std::vector<Point3D> &const_frame) -> RegistrationSummary {
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
    summary.success = true;
  }
  trajectory_[index_frame].points = frame;

  // add points
  updateMap(index_frame, index_frame);

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

void ElasticOdometry::initializeTimestamp(int index_frame, const std::vector<Point3D> &const_frame) {
  double min_timestamp = std::numeric_limits<double>::max();
  double max_timestamp = std::numeric_limits<double>::min();
  for (const auto &point : const_frame) {
    if (point.timestamp > max_timestamp) max_timestamp = point.timestamp;
    if (point.timestamp < min_timestamp) min_timestamp = point.timestamp;
  }
  trajectory_[index_frame].begin_timestamp = min_timestamp;
  trajectory_[index_frame].end_timestamp = max_timestamp;
}

void ElasticOdometry::initializeMotion(int index_frame) {
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

std::vector<Point3D> ElasticOdometry::initializeFrame(int index_frame, const std::vector<Point3D> &const_frame) {
  std::vector<Point3D> frame(const_frame);

  double sample_size = index_frame < options_.init_num_frames ? options_.init_voxel_size : options_.voxel_size;
  std::mt19937_64 g;
  std::shuffle(frame.begin(), frame.end(), g);
  // Subsample the scan with voxels taking one random in every voxel
  sub_sample_frame(frame, sample_size);
  std::shuffle(frame.begin(), frame.end(), g);

  // No elastic ICP for first frame because no initialization of ego-motion
  if (index_frame == 1) {
    for (auto &point : frame) point.alpha_timestamp = 1.0;
  }

  // initialize points
  if (index_frame > 1) {
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
  }

  return frame;
}

void ElasticOdometry::updateMap(int index_frame, int update_frame) {
  const double kSizeVoxelMap = options_.size_voxel_map;
  const double kMinDistancePoints = options_.min_distance_points;
  const int kMaxNumPointsInVoxel = options_.max_num_points_in_voxel;

  // update frame
  auto &frame = trajectory_[update_frame].points;

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

  map_.add(frame, kSizeVoxelMap, kMaxNumPointsInVoxel, kMinDistancePoints);
  frame.clear();

  // remove points
  const double kMaxDistance = options_.max_distance;
  const Eigen::Vector3d location = trajectory_[index_frame].end_t;
  map_.remove(location, kMaxDistance);
}

bool ElasticOdometry::icp(int index_frame, std::vector<Point3D> &keypoints) {
  bool icp_success = true;

  // Optimization with Traj constraints
  double ALPHA_C = options_.beta_location_consistency;  // 0.001;
  double ALPHA_E = options_.beta_constant_velocity;     // 0.001; // no ego (0.0) is not working

  // For the 50 first frames, visit 2 voxels
  const short nb_voxels_visited = index_frame < options_.init_num_frames ? 2 : 1;
  int number_keypoints_used = 0;
  const int kMinNumNeighbors = options_.min_number_neighbors;

  using AType = Eigen::Matrix<double, 12, 12>;
  using bType = Eigen::Matrix<double, 12, 1>;
  AType A;
  bType b;

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
  for (int iter(0); iter < num_iter_icp; iter++) {
    A = Eigen::MatrixXd::Zero(12, 12);
    b = Eigen::VectorXd::Zero(12);

    number_keypoints_used = 0;
    double total_scalar = 0;
    double mean_scalar = 0.0;

    timer[0].second->start();

#pragma omp parallel for num_threads(options_.num_threads)
    for (int i = 0; i < (int)keypoints.size(); i++) {
      const auto &keypoint = keypoints[i];
      const auto &pt_keypoint = keypoint.pt;
      const double alpha_timestamp = keypoint.alpha_timestamp;

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
      auto neighborhood = compute_neighborhood_distribution(vector_neighbors);
      double planarity_weight = neighborhood.a2D;
      auto &normal = neighborhood.normal;

      if (normal.dot(trajectory_[index_frame].begin_t - pt_keypoint) < 0) {
        normal = -1.0 * normal;
      }

      const double weight = planarity_weight * planarity_weight;

      Eigen::Vector3d closest_pt_normal = weight * normal;

      Eigen::Vector3d closest_point = vector_neighbors[0];

      double dist_to_plane = normal[0] * (pt_keypoint[0] - closest_point[0]) +
                             normal[1] * (pt_keypoint[1] - closest_point[1]) +
                             normal[2] * (pt_keypoint[2] - closest_point[2]);
      dist_to_plane = std::abs(dist_to_plane);

      if (innerloop_time) inner_timer[1].second->stop();

      if (innerloop_time) inner_timer[2].second->start();

      if (dist_to_plane < options_.max_dist_to_plane) {
        double scalar = closest_pt_normal[0] * (pt_keypoint[0] - closest_point[0]) +
                        closest_pt_normal[1] * (pt_keypoint[1] - closest_point[1]) +
                        closest_pt_normal[2] * (pt_keypoint[2] - closest_point[2]);

        Eigen::Vector3d frame_idx_previous_origin_begin = trajectory_[index_frame].begin_R * keypoint.raw_pt;
        Eigen::Vector3d frame_idx_previous_origin_end = trajectory_[index_frame].end_R * keypoint.raw_pt;

        double cbx = (1 - alpha_timestamp) * (frame_idx_previous_origin_begin[1] * closest_pt_normal[2] -
                                              frame_idx_previous_origin_begin[2] * closest_pt_normal[1]);
        double cby = (1 - alpha_timestamp) * (frame_idx_previous_origin_begin[2] * closest_pt_normal[0] -
                                              frame_idx_previous_origin_begin[0] * closest_pt_normal[2]);
        double cbz = (1 - alpha_timestamp) * (frame_idx_previous_origin_begin[0] * closest_pt_normal[1] -
                                              frame_idx_previous_origin_begin[1] * closest_pt_normal[0]);

        double nbx = (1 - alpha_timestamp) * closest_pt_normal[0];
        double nby = (1 - alpha_timestamp) * closest_pt_normal[1];
        double nbz = (1 - alpha_timestamp) * closest_pt_normal[2];

        double cex = (alpha_timestamp) * (frame_idx_previous_origin_end[1] * closest_pt_normal[2] -
                                          frame_idx_previous_origin_end[2] * closest_pt_normal[1]);
        double cey = (alpha_timestamp) * (frame_idx_previous_origin_end[2] * closest_pt_normal[0] -
                                          frame_idx_previous_origin_end[0] * closest_pt_normal[2]);
        double cez = (alpha_timestamp) * (frame_idx_previous_origin_end[0] * closest_pt_normal[1] -
                                          frame_idx_previous_origin_end[1] * closest_pt_normal[0]);

        double nex = (alpha_timestamp)*closest_pt_normal[0];
        double ney = (alpha_timestamp)*closest_pt_normal[1];
        double nez = (alpha_timestamp)*closest_pt_normal[2];

        Eigen::VectorXd u(12);
        u << cbx, cby, cbz, nbx, nby, nbz, cex, cey, cez, nex, ney, nez;

#pragma omp critical(odometry_cost_term)
        {
          for (int i = 0; i < 12; i++) {
            for (int j = 0; j < 12; j++) {
              A(i, j) = A(i, j) + u[i] * u[j];
            }
            b(i) = b(i) - u[i] * scalar;
          }

          total_scalar = total_scalar + scalar * scalar;
          mean_scalar = mean_scalar + fabs(scalar);
          number_keypoints_used++;
        }
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

    // Normalize equation
    for (int i(0); i < 12; i++) {
      for (int j(0); j < 12; j++) {
        A(i, j) = A(i, j) / number_keypoints_used;
      }
      b(i) = b(i) / number_keypoints_used;
    }

    // Add constraints in trajectory
    if (index_frame > 1)  // no constraints for frame_index == 1
    {
      Eigen::Vector3d diff_traj = trajectory_[index_frame].begin_t - trajectory_[index_frame - 1].end_t;
      A(3, 3) = A(3, 3) + ALPHA_C;
      A(4, 4) = A(4, 4) + ALPHA_C;
      A(5, 5) = A(5, 5) + ALPHA_C;
      b(3) = b(3) - ALPHA_C * diff_traj(0);
      b(4) = b(4) - ALPHA_C * diff_traj(1);
      b(5) = b(5) - ALPHA_C * diff_traj(2);

      Eigen::Vector3d diff_ego = trajectory_[index_frame].end_t - trajectory_[index_frame].begin_t -
                                 trajectory_[index_frame - 1].end_t + trajectory_[index_frame - 1].begin_t;
      A(9, 9) = A(9, 9) + ALPHA_E;
      A(10, 10) = A(10, 10) + ALPHA_E;
      A(11, 11) = A(11, 11) + ALPHA_E;
      b(9) = b(9) - ALPHA_E * diff_ego(0);
      b(10) = b(10) - ALPHA_E * diff_ego(1);
      b(11) = b(11) - ALPHA_E * diff_ego(2);
    }

    // Solve
    Eigen::VectorXd x_bundle = A.ldlt().solve(b);

    double alpha_begin = x_bundle(0);
    double beta_begin = x_bundle(1);
    double gamma_begin = x_bundle(2);
    Eigen::Matrix3d rotation_begin;
    rotation_begin(0, 0) = cos(gamma_begin) * cos(beta_begin);
    rotation_begin(0, 1) = -sin(gamma_begin) * cos(alpha_begin) + cos(gamma_begin) * sin(beta_begin) * sin(alpha_begin);
    rotation_begin(0, 2) = sin(gamma_begin) * sin(alpha_begin) + cos(gamma_begin) * sin(beta_begin) * cos(alpha_begin);
    rotation_begin(1, 0) = sin(gamma_begin) * cos(beta_begin);
    rotation_begin(1, 1) = cos(gamma_begin) * cos(alpha_begin) + sin(gamma_begin) * sin(beta_begin) * sin(alpha_begin);
    rotation_begin(1, 2) = -cos(gamma_begin) * sin(alpha_begin) + sin(gamma_begin) * sin(beta_begin) * cos(alpha_begin);
    rotation_begin(2, 0) = -sin(beta_begin);
    rotation_begin(2, 1) = cos(beta_begin) * sin(alpha_begin);
    rotation_begin(2, 2) = cos(beta_begin) * cos(alpha_begin);
    Eigen::Vector3d translation_begin = Eigen::Vector3d(x_bundle(3), x_bundle(4), x_bundle(5));

    double alpha_end = x_bundle(6);
    double beta_end = x_bundle(7);
    double gamma_end = x_bundle(8);
    Eigen::Matrix3d rotation_end;
    rotation_end(0, 0) = cos(gamma_end) * cos(beta_end);
    rotation_end(0, 1) = -sin(gamma_end) * cos(alpha_end) + cos(gamma_end) * sin(beta_end) * sin(alpha_end);
    rotation_end(0, 2) = sin(gamma_end) * sin(alpha_end) + cos(gamma_end) * sin(beta_end) * cos(alpha_end);
    rotation_end(1, 0) = sin(gamma_end) * cos(beta_end);
    rotation_end(1, 1) = cos(gamma_end) * cos(alpha_end) + sin(gamma_end) * sin(beta_end) * sin(alpha_end);
    rotation_end(1, 2) = -cos(gamma_end) * sin(alpha_end) + sin(gamma_end) * sin(beta_end) * cos(alpha_end);
    rotation_end(2, 0) = -sin(beta_end);
    rotation_end(2, 1) = cos(beta_end) * sin(alpha_end);
    rotation_end(2, 2) = cos(beta_end) * cos(alpha_end);
    Eigen::Vector3d translation_end = Eigen::Vector3d(x_bundle(9), x_bundle(10), x_bundle(11));

    timer[1].second->stop();

    timer[2].second->start();

    // Update (changes trajectory data)
    trajectory_[index_frame].begin_R = rotation_begin * trajectory_[index_frame].begin_R;
    trajectory_[index_frame].begin_t = trajectory_[index_frame].begin_t + translation_begin;
    trajectory_[index_frame].end_R = rotation_end * trajectory_[index_frame].end_R;
    trajectory_[index_frame].end_t = trajectory_[index_frame].end_t + translation_end;

    timer[2].second->stop();

    timer[3].second->start();

    // Update keypoints
#pragma omp parallel for num_threads(options_.num_threads)
    for (int i = 0; i < (int)keypoints.size(); i++) {
      auto &keypoint = keypoints[i];
      Eigen::Quaterniond q_begin = Eigen::Quaterniond(trajectory_[index_frame].begin_R);
      Eigen::Quaterniond q_end = Eigen::Quaterniond(trajectory_[index_frame].end_R);
      Eigen::Vector3d t_begin = trajectory_[index_frame].begin_t;
      Eigen::Vector3d t_end = trajectory_[index_frame].end_t;
      double alpha_timestamp = keypoint.alpha_timestamp;
      Eigen::Quaterniond q = q_begin.slerp(alpha_timestamp, q_end);
      q.normalize();
      Eigen::Matrix3d R = q.toRotationMatrix();
      Eigen::Vector3d t = (1.0 - alpha_timestamp) * t_begin + alpha_timestamp * t_end;
      keypoint.pt = R * keypoint.raw_pt + t;
    }

    timer[3].second->stop();

    if ((index_frame > 1) && (x_bundle.norm() < options_.convergence_threshold)) {
      if (options_.debug_print) {
        LOG(INFO) << "CT_ICP: Finished with N=" << iter << " ICP iterations" << std::endl;
      }
      break;
    }
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
