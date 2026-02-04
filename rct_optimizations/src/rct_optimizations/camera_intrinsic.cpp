// Include the camera intrinsic calibration problem/solution definitions.
#include "rct_optimizations/experimental/camera_intrinsic.h"

// Include Ceres math utilities for projection and transforms.
#include <rct_optimizations/ceres_math_utilities.h>
// Include covariance analysis helpers for reporting uncertainty.
#include <rct_optimizations/covariance_analysis.h>
// Include conversion helpers between Eigen and internal Pose6d types.
#include <rct_optimizations/eigen_conversions.h>
// Include the PnP solver used to seed initial target poses.
#include <rct_optimizations/pnp.h>

// Include Ceres for nonlinear least squares optimization.
#include <ceres/ceres.h>

// Start an anonymous namespace to keep file-local helpers private.
namespace
{
// Template wrapper that provides read-only access to intrinsic parameters.
template <typename T>
struct CalibCameraIntrinsics
{
  // Raw pointer to the intrinsic parameter array.
  const T* data;

  // Construct from the raw parameter pointer.
  CalibCameraIntrinsics(const T* data) : data(data) {}

  // Access focal length in x.
  const T& fx() const { return data[0]; }
  // Access focal length in y.
  const T& fy() const { return data[1]; }
  // Access principal point x.
  const T& cx() const { return data[2]; }
  // Access principal point y.
  const T& cy() const { return data[3]; }

  // Access radial distortion k1.
  const T& k1() const { return data[4]; }
  // Access radial distortion k2.
  const T& k2() const { return data[5]; }
  // Access tangential distortion p1.
  const T& p1() const { return data[6]; }
  // Access tangential distortion p2.
  const T& p2() const { return data[7]; }
  // Access radial distortion k3.
  const T& k3() const { return data[8]; }

  // Return the total number of intrinsic parameters in this model.
  constexpr static std::size_t size() { return 9; }
};

// Template wrapper that provides mutable access to intrinsic parameters.
template <typename T>
struct MutableCalibCameraIntrinsics
{
  // Raw pointer to the intrinsic parameter array.
  T* data;

  // Construct from the raw parameter pointer.
  MutableCalibCameraIntrinsics(T* data) : data(data) {}

  // Read-only accessors for focal length and principal point.
  const T& fx() const { return data[0]; }
  const T& fy() const { return data[1]; }
  const T& cx() const { return data[2]; }
  const T& cy() const { return data[3]; }

  // Read-only accessors for distortion coefficients.
  const T& k1() const { return data[4]; }
  const T& k2() const { return data[5]; }
  const T& p1() const { return data[6]; }
  const T& p2() const { return data[7]; }
  const T& k3() const { return data[8]; }

  // Mutable accessors for focal length and principal point.
  T& fx() { return data[0]; }
  T& fy() { return data[1]; }
  T& cx() { return data[2]; }
  T& cy() { return data[3]; }

  // Mutable accessors for distortion coefficients.
  T& k1() { return data[4]; }
  T& k2() { return data[5]; }
  T& p1() { return data[6]; }
  T& p2() { return data[7]; }
  T& k3() { return data[8]; }

  // Return the total number of intrinsic parameters in this model.
  constexpr static std::size_t size() { return 9; }
};

// Project a 3D point in camera coordinates into the image with distortion.
template <typename T>
void projectPoints2(const T* const camera_intr, const T* const pt_in_camera, T* pt_in_image)
{
  // Extract X coordinate in camera frame.
  T xp1 = pt_in_camera[0];
  // Extract Y coordinate in camera frame.
  T yp1 = pt_in_camera[1];
  // Extract Z coordinate in camera frame.
  T zp1 = pt_in_camera[2];

  // Wrap intrinsic parameters for convenient access.
  CalibCameraIntrinsics<T> intr(camera_intr);

  // Scale into the normalized image plane by depth.
  T xp;
  // Scale into the normalized image plane by depth.
  T yp;
  // Check for zero depth to avoid division by zero.
  if (zp1 == T(0))  // Avoid dividing by zero.
  {
    // Use unnormalized X if depth is zero.
    xp = xp1;
    // Use unnormalized Y if depth is zero.
    yp = yp1;
  }
  else
  {
    // Normalize X by depth.
    xp = xp1 / zp1;
    // Normalize Y by depth.
    yp = yp1 / zp1;
  }

  // Temporary variables for distortion model (x squared).
  T xp2 = xp * xp;   // x^2
  // Temporary variables for distortion model (y squared).
  T yp2 = yp * yp;   // y^2
  // Compute radius squared.
  T r2 = xp2 + yp2;  // r^2 radius squared
  // Compute r^4.
  T r4 = r2 * r2;    // r^4
  // Compute r^6.
  T r6 = r2 * r4;    // r^6

  // Apply radial and tangential distortion to x.
  T xpp = xp + intr.k1() * r2 * xp           // 2nd order term
          + intr.k2() * r4 * xp              // 4th order term
          + intr.k3() * r6 * xp              // 6th order term
          + intr.p2() * (r2 + T(2.0) * xp2)  // tangential
          + intr.p1() * xp * yp * T(2.0);    // other tangential term

  // Apply radial and tangential distortion to y.
  T ypp = yp + intr.k1() * r2 * yp           // 2nd order term
          + intr.k2() * r4 * yp              // 4th order term
          + intr.k3() * r6 * yp              // 6th order term
          + intr.p1() * (r2 + T(2.0) * yp2)  // tangential term
          + intr.p2() * xp * yp * T(2.0);    // other tangential term

  // Convert normalized coordinates to pixel coordinates using intrinsics.
  pt_in_image[0] = intr.fx() * xpp + intr.cx();
  // Convert normalized coordinates to pixel coordinates using intrinsics.
  pt_in_image[1] = intr.fy() * ypp + intr.cy();
}

// Solve a PnP problem to estimate an initial target pose for an image set.
static rct_optimizations::Pose6d solvePnP(const rct_optimizations::CameraIntrinsics& intr,
                                          const rct_optimizations::Correspondence2D3D::Set& obs,
                                          const rct_optimizations::Pose6d& guess)
{
  // Pull the namespace into scope for readability.
  using namespace rct_optimizations;

  // Create a PnP problem definition.
  PnPProblem problem;
  // Seed PnP with the current pose guess.
  problem.camera_to_target_guess = poseCalToEigen(guess);
  // Assign camera intrinsics.
  problem.intr = intr;
  // Assign 2D-3D correspondences.
  problem.correspondences = obs;

  // Solve the PnP problem.
  PnPResult result = optimize(problem);

  // If PnP fails, throw so the caller can skip this observation.
  if (!result.converged)
    throw std::runtime_error("unable to solve PnP sub-problem");

  // Convert the solution into the internal Pose6d type.
  return poseEigenToCal(result.camera_to_target);
}

// Cost functor for intrinsic calibration residuals.
class IntrinsicCostFunction
{
public:
  // Construct with a 3D target point and its observed 2D pixel.
  IntrinsicCostFunction(const Eigen::Vector3d& in_target, const Eigen::Vector2d& in_image)
    : in_target_(in_target), in_image_(in_image)
  {
  }

  // Compute residuals given target pose and camera intrinsics.
  template <typename T>
  bool operator()(const T* const target_pose, const T* const camera_intr, T* const residual) const
  {
    // Angle-axis rotation for target pose.
    const T* target_angle_axis = target_pose + 0;
    // Translation for target pose.
    const T* target_position = target_pose + 3;

    // Prepare the 3D target point in the target frame.
    T target_pt[3];
    // Copy x component.
    target_pt[0] = T(in_target_(0));
    // Copy y component.
    target_pt[1] = T(in_target_(1));
    // Copy z component.
    target_pt[2] = T(in_target_(2));

    // Allocate space for the point in the camera frame.
    T camera_point[3];  // Point in camera coordinates
    // Transform the target point into the camera frame.
    rct_optimizations::transformPoint(target_angle_axis, target_position, target_pt, camera_point);

    // Allocate space for the projected pixel.
    T xy_image[2];
    // Project the camera-frame point into the image using intrinsics.
    projectPoints2(camera_intr, camera_point, xy_image);

    // Residual for x pixel coordinate.
    residual[0] = xy_image[0] - in_image_.x();
    // Residual for y pixel coordinate.
    residual[1] = xy_image[1] - in_image_.y();

    // Indicate success to Ceres.
    return true;
  }

private:
  // Store the 3D target point.
  Eigen::Vector3d in_target_;
  // Store the observed 2D pixel.
  Eigen::Vector2d in_image_;
};

}  // namespace

// Provide a default initial pose guess for PnP seeding.
static rct_optimizations::Pose6d guessInitialPose()
{
  // Place the target half a meter in front of the camera and flip around X.
  Eigen::Isometry3d guess = Eigen::Translation3d(0, 0, 0.5) * Eigen::AngleAxisd(M_PI, Eigen::Vector3d::UnitX());
  // Convert the Eigen transform to the internal Pose6d type.
  return rct_optimizations::poseEigenToCal(guess);
}

// Solve the intrinsic calibration optimization problem.
rct_optimizations::IntrinsicEstimationResult
rct_optimizations::optimize(const rct_optimizations::IntrinsicEstimationProblem& params)
{
  // Prepare data structure for the camera parameters to optimize.
  std::array<double, CalibCameraIntrinsics<double>::size()> internal_intrinsics_data;
  // Initialize all parameters to zero before seeding.
  for (int i = 0; i < 9; ++i)
    internal_intrinsics_data[i] = 0.0;

  // Wrap the mutable intrinsics array for convenient access.
  MutableCalibCameraIntrinsics<double> internal_intrinsics(internal_intrinsics_data.data());
  // Seed focal length x.
  internal_intrinsics.fx() = params.intrinsics_guess.fx();
  // Seed focal length y.
  internal_intrinsics.fy() = params.intrinsics_guess.fy();
  // Seed principal point x.
  internal_intrinsics.cx() = params.intrinsics_guess.cx();
  // Seed principal point y.
  internal_intrinsics.cy() = params.intrinsics_guess.cy();

  // Prepare space for the target poses to estimate (1 for each observation set).
  std::vector<Pose6d> internal_poses(params.image_observations.size());
  // Track indices of observation sets with valid PnP solutions.
  std::vector<std::size_t> valid_idx;

  // All of the target poses are seeded to be "in front of" and "looking at" the camera.
  for (std::size_t i = 0; i < params.image_observations.size(); ++i)
  {
    try
    {
      // Use PnP to get an initial pose estimate for this image set.
      internal_poses[i] = solvePnP(params.intrinsics_guess, params.image_observations[i], guessInitialPose());
      // Record this observation set as valid.
      valid_idx.push_back(i);
    }
    catch (const std::exception&)
    {
      // Report PnP failure to the console.
      std::cout << "PnP failed for image " << i << std::endl;
      // Skip this observation set.
      continue;
    }
  }

  // Instantiate the Ceres problem.
  ceres::Problem problem;

  // Create a set of cost functions for each observation set.
  for (std::size_t i : valid_idx)
  {
    // Create a cost for each 2D -> 3D image correspondence.
    for (std::size_t j = 0; j < params.image_observations[i].size(); ++j)
    {
      // Reference the 3D target point.
      const auto& point_in_target = params.image_observations[i][j].in_target;
      // Reference the observed 2D pixel.
      const auto& point_in_image = params.image_observations[i][j].in_image;

      // Allocate Ceres data structures - ownership is taken by the ceres
      // Problem data structure.
      auto* cost_fn = new IntrinsicCostFunction(point_in_target, point_in_image);

      // Wrap the cost functor with AutoDiff (2 residuals, 6 pose params, 9 intrinsics).
      auto* cost_block = new ceres::AutoDiffCostFunction<IntrinsicCostFunction, 2, 6, 9>(cost_fn);

      // Add the residual block tied to the target pose and intrinsic parameters.
      problem.AddResidualBlock(cost_block, NULL, internal_poses[i].values.data(), internal_intrinsics_data.data());
    }
  }

  // Track parameter blocks for covariance computation.
  std::vector<const double*> param_blocks;
  // Track labels for each parameter block.
  std::map<const double*, std::vector<std::string>> param_labels;

  // Add the intrinsics parameter block to the list.
  param_blocks.emplace_back(internal_intrinsics_data.data());
  // Attach labels for the intrinsic parameters.
  param_labels[internal_intrinsics_data.data()] =
      std::vector<std::string>(params.labels_intrinsic_params.begin(), params.labels_intrinsic_params.end());

  // Add each pose parameter block to the list.
  for (std::size_t i = 0; i < internal_poses.size(); i++)
  {
    // Register the pose parameter block pointer.
    param_blocks.emplace_back(internal_poses[i].values.data());
    // Create a label list for this pose.
    std::vector<std::string> labels;
    // compose labels poseN_x, etc.
    for (auto label_extr : params.labels_isometry3d)
    {
      // Build per-pose labels with an index suffix.
      labels.push_back(params.label_extr + std::to_string(i) + "_" + label_extr);
    }
    // Store labels for this pose parameter block.
    param_labels[internal_poses[i].values.data()] = labels;
  }

  // Solve.
  ceres::Solver::Options options;
  // Allow a large number of iterations for convergence.
  options.max_num_iterations = 1000;
  // Capture solve summary details.
  ceres::Solver::Summary summary;
  // Run the Ceres solver.
  ceres::Solve(options, &problem, &summary);

  // Package results.
  IntrinsicEstimationResult result;
  // Indicate convergence status.
  result.converged = summary.termination_type == ceres::CONVERGENCE;

  // Copy optimized intrinsics back into the result structure.
  result.intrinsics.fx() = internal_intrinsics.fx();
  result.intrinsics.fy() = internal_intrinsics.fy();
  result.intrinsics.cx() = internal_intrinsics.cx();
  result.intrinsics.cy() = internal_intrinsics.cy();

  // Copy optimized distortion coefficients back into the result structure.
  result.distortions[0] = internal_intrinsics_data[4];
  result.distortions[1] = internal_intrinsics_data[5];
  result.distortions[2] = internal_intrinsics_data[6];
  result.distortions[3] = internal_intrinsics_data[7];
  result.distortions[4] = internal_intrinsics_data[8];

  // Compute the average initial cost per residual.
  result.initial_cost_per_obs = summary.initial_cost / summary.num_residuals;
  // Compute the average final cost per residual.
  result.final_cost_per_obs = summary.final_cost / summary.num_residuals;

  // Compute covariance for parameters using the configured labels.
  result.covariance = rct_optimizations::computeCovariance(problem, param_blocks, param_labels);

  // Return the completed result.
  return result;
}
