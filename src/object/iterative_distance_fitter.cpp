/*********************************************************************
 * Software License Agreement (BSD License)
 *
 *  Copyright (c) 2009, Willow Garage, Inc.
 *  All rights reserved.
 *
 *  Redistribution and use in source and binary forms, with or without
 *  modification, are permitted provided that the following conditions
 *  are met:
 *
 *   * Redistributions of source code must retain the above copyright
 *     notice, this list of conditions and the following disclaimer.
 *   * Redistributions in binary form must reproduce the above
 *     copyright notice, this list of conditions and the following
 *     disclaimer in the documentation and/or other materials provided
 *     with the distribution.
 *   * Neither the name of the Willow Garage nor the names of its
 *     contributors may be used to endorse or promote products derived
 *     from this software without specific prior written permission.
 *
 *  THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS
 *  "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT
 *  LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS
 *  FOR A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE
 *  COPYRIGHT OWNER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT,
 *  INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING,
 *  BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
 *  LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
 *  CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT
 *  LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN
 *  ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
 *  POSSIBILITY OF SUCH DAMAGE.
 *********************************************************************/

// Author(s): Marius Muja, Matei Ciocarlie and Romain Thibaux

#include <tabletop_object_detector/iterative_distance_fitter.h>

#include <boost/bind.hpp>
#include <Eigen/Core>
#include <Eigen/SVD>
#include <eigen_conversions/eigen_msg.h>
#include <shape_tools/shape_to_marker.h>

namespace tabletop_object_detector
{

//------------------------- Implementation follows ----------------------------------------

/*! Computes the point at the bottom of the point cloud vertical to the center of
 *  gravity. This is the point where the table supports the object.
 */
cv::Point3f IterativeTranslationFitter::centerOfSupport(const std::vector<cv::Vec3f>& cloud) const
{
  cv::Point3f center;
  center.x = center.y = center.z = 0;
  if (cloud.empty()) {
    return center;
  }
  for (unsigned int i = 0; i < cloud.size(); ++i) {
    center.x += cloud[i][0];
    center.y += cloud[i][1];
  }
  center.x /= cloud.size();
  center.y /= cloud.size();
  return center;
}


double IterativeTranslationFitter::getFitScoreAndGradient(const std::vector<cv::Vec3f>& cloud,
    const cv::Point3f& location, cv::Point3f& vector,
    boost::function<double(double)> kernel) const
{
  double inlier_count = 0;

  vector.x = 0;
  vector.y = 0;
  vector.z = 0;
  int cnt = 0;

  for (size_t i = 0; i < cloud.size(); i++) {
    double wx = cloud[i][0] - location.x;
    double wy = cloud[i][1] - location.y;
    double wz = cloud[i][2] - location.z;

    int x, y, z;
    double val = truncate_value_;
    if (distance_voxel_grid_->worldToGrid(wx, wy, wz, x, y, z)) {
      const distance_field::PropDistanceFieldVoxel& voxel = distance_voxel_grid_->getCell(x, y, z);
      double cx, cy, cz;
      if (voxel.closest_point_[0] != distance_field::PropDistanceFieldVoxel::UNINITIALIZED) {
        distance_voxel_grid_->gridToWorld(voxel.closest_point_[0],
                                          voxel.closest_point_[1],
                                          voxel.closest_point_[2],
                                          cx, cy, cz);
        val = distance_voxel_grid_->getDistance(x, y, z);
        double weight = kernel(val);
        vector.x += weight * (cx - wx);
        vector.y += weight * (cy - wy);
        vector.z += weight * (cz - wz);

        inlier_count += weight;
      }
    }
  }

  if (inlier_count != 0) {
    vector.x /=  inlier_count;
    vector.y /=  inlier_count;
    vector.z /=  inlier_count;
  }

  return inlier_count / cloud.size();
}


visualization_msgs::Marker IterativeTranslationFitter::createClusterMarker(const EigenSTL::vector_Vector3d & cluster, int id,
        const geometry_msgs::Pose & cloud_pose, const Eigen::Affine3d & icp_transform) const
{
    visualization_msgs::Marker marker;
    if(id >= 42000) {
        id -= 42000;
        marker.ns = "icp_clusters_transformed";
        marker.color.g = 1.0;
        marker.type = visualization_msgs::Marker::POINTS;
        for(int i = 0; i < cluster.size(); i++) {
            geometry_msgs::Point pt;
            pt.x = cluster[i].x();
            pt.y = cluster[i].y();
            pt.z = cluster[i].z();
            marker.points.push_back(pt);
        }
        marker.scale.x = 0.01;
        marker.scale.y = 0.01;
        marker.scale.z = 0.01;
    } else {
        marker.scale.x = 1.0;
        marker.scale.y = 1.0;
        marker.scale.z = 1.0;
        marker.ns = "icp_clusters";
        shape_tools::constructMarkerFromShape(mesh_, marker); // assumes this was initialized from mesh
    }
    marker.header.frame_id = "head_mount_kinect_rgb_optical_frame"; // TODO figure this out, probably incoming msg frame, do we have that???
    marker.id = id;
    marker.action = visualization_msgs::Marker::ADD;
    Eigen::Affine3d cloudPoseEigen;
    tf::poseMsgToEigen(cloud_pose, cloudPoseEigen);
    Eigen::Affine3d fullPose = cloudPoseEigen * icp_transform;
    tf::poseEigenToMsg(fullPose, marker.pose);
    marker.color.a = 0.5;
    marker.color.r = id/100./10.0;
    marker.color.b = 0.5 + id%100/10.0;
    if(marker.color.b > 1.0)
        marker.color.b = 1.0;
        return marker;
}

/// Transform cloud by transform and compute a match score.
/**
 * \param [in] distance_score_kernel a scoring function that given the distance produces a value
 * between 0 and 1, where 1 is best match distance (i.e. inlier)
 */
double IterativeTranslationFitter::applyTransformAndcomputeScore(EigenSTL::vector_Vector3d & cloud,
        const Eigen::Affine3d & transform,
        boost::function<double(double)> distance_score_kernel) const
{
    double inlier_count = 0;
    for(size_t i = 0; i < cloud.size(); i++) {
        // 1. apply transform
        Eigen::Vector3d & cloudPt = cloud[i];
        cloudPt = (transform * cloudPt).eval();
        // 2. compute score for transformed point
        int x, y, z;
        double val = truncate_value_;
        if(distance_voxel_grid_->worldToGrid(cloudPt.x(), cloudPt.y(), cloudPt.z(), x, y, z)) {
            const distance_field::PropDistanceFieldVoxel& voxel = distance_voxel_grid_->getCell(x, y, z);
            if(voxel.closest_point_[0] != distance_field::PropDistanceFieldVoxel::UNINITIALIZED) {
                val = distance_voxel_grid_->getDistance(x, y, z);
                double weight = distance_score_kernel(val);
                inlier_count += weight;
            }
        }
    }
    return inlier_count/cloud.size();
}

Eigen::Vector3d computeCenterOfMass(const EigenSTL::vector_Vector3d & cloud)
{
    Eigen::Vector3d cloudMu = Eigen::Vector3d(0.0, 0.0, 0.0);
    for(size_t i = 0; i < cloud.size(); i++) {
        const Eigen::Vector3d & cloudPt = cloud[i];
        cloudMu += cloudPt;
    }
    cloudMu /= cloud.size();
    return cloudMu;
}

// TODO if we throw out outliers here, they should also be handled in computeCenterOfMass!
Eigen::Matrix3d IterativeTranslationFitter::computeW(
        const EigenSTL::vector_Vector3d & cloud, const Eigen::Vector3d & cloudMu,
        const Eigen::Vector3d & distance_voxel_grid_Mu,
        boost::function<double(double)> distance_score_kernel) const
{
    Eigen::Matrix3d W = Eigen::Matrix3d::Zero();
    for(size_t i = 0; i < cloud.size(); i++) {
        const Eigen::Vector3d & cloudPt = cloud[i];

        // need to find the closest point 
        int x, y, z;
        double val = truncate_value_;
        if(distance_voxel_grid_->worldToGrid(cloudPt.x(), cloudPt.y(), cloudPt.z(), x, y, z)) {
            const distance_field::PropDistanceFieldVoxel& voxel = distance_voxel_grid_->getCell(x, y, z);
            double cx, cy, cz;
            if (voxel.closest_point_[0] != distance_field::PropDistanceFieldVoxel::UNINITIALIZED) {
                distance_voxel_grid_->gridToWorld(voxel.closest_point_[0],
                        voxel.closest_point_[1],
                        voxel.closest_point_[2],
                        cx, cy, cz);
            }
            Eigen::Vector3d modelPt(cx, cy, cz);    // grid point corresponce
            modelPt -= distance_voxel_grid_Mu;;
            Eigen::Vector3d cloudPtX = cloudPt - cloudMu;

            val = distance_voxel_grid_->getDistance(x, y, z);
            double weight = distance_score_kernel(val);
            weight = (val < 0.01) ? 1 : 0;
            //weight = 1.0;
            W += weight * modelPt * cloudPtX.transpose();
        }
    }
    return W;
}

Eigen::Affine3d computeLocalTransform(const Eigen::Matrix3d & W, const Eigen::Vector3d cloudMu,
        const Eigen::Vector3d distance_voxel_grid_Mu)
{
    Eigen::JacobiSVD<Eigen::Matrix3d> svd(W, Eigen::ComputeFullU | Eigen::ComputeFullV);
    Eigen::Matrix3d R = svd.matrixU() * svd.matrixV().transpose();
    Eigen::Vector3d t = distance_voxel_grid_Mu - R * cloudMu;
    Eigen::Affine3d localTrans = Eigen::Affine3d::Identity();
    localTrans.linear() = R;
    localTrans.translation() = t;
    //localTrans.translation().z() = 0.0;
    return localTrans;
}


/*! Iterates over the inner loop of \a getFitScoreAndGradient, then moves in the direction
 *  of the computed gradient. Does this until the score stops decreasing.
 *
 *  The fit is initialized to the centroid of the cloud along x and y, and 0 along z. This
 *  assumes that the meshes are all such that the origin is at the bottom, with the z axis
 *  pointing up. It also assumes the points have been translated to a coordinate frame
 *  where z=0 is the table plane.
 *
 *  For the same reason. there is no iteration done along z at all.
 */
ModelFitInfo IterativeTranslationFitter::fitPointCloud(const std::vector<cv::Vec3f>& cloud,
    cv::flann::Index &search, double min_object_score, const geometry_msgs::Pose & cloud_pose) const
{
  if (cloud.empty()) {
    //ROS_ERROR("Attempt to fit model to empty point cloud");
    geometry_msgs::Pose bogus_pose;
    return ModelFitInfo(model_id_, bogus_pose, 0.0);
  }
  static ros::NodeHandle nh;
  static ros::Publisher pubMarker = nh.advertise<visualization_msgs::MarkerArray>("object_detection_marker", 4);

  // compute center of point cloud
  cv::Point3f center = centerOfSupport(cloud);

  //boost::function<double(double)> kernel = boost::bind(huberKernel, clipping_, _1);
  boost::function<double(double)> kernel = boost::bind(huberKernel, 0.001, _1);
  const int max_iterations = 1000;
  int iter = 0;
  double score = 0;
  const double EPS = 0.0;//01;

  // transform is the transformation to move the model/distance_voxel_grid to the input point cloud
  // i.e. the resulting pose of the model
  Eigen::Affine3d transform = Eigen::Affine3d::Identity();
  // As we can't transform the distance_voxel_grid_, we transform the input cloud towards the model
  // Thus, transformedCloud is transform^-1 * cloud
  EigenSTL::vector_Vector3d transformedCloud;
  EigenSTL::vector_Vector3d rawCloud;
  transformedCloud.reserve(cloud.size());
  rawCloud.reserve(cloud.size());
  // TODO use as initial guess
  //transform.translation().x() = center.x;
  //transform.translation().y() = center.y;
  for(size_t i = 0; i < cloud.size(); i++) {
      rawCloud.push_back(Eigen::Vector3d(
                  cloud[i][0], cloud[i][1], cloud[i][2]));
      transformedCloud.push_back(Eigen::Vector3d(
                  cloud[i][0], cloud[i][1], cloud[i][2]));
  }

  printf("fitPointCloud for model %d\n", model_id_);
#if 1
  visualization_msgs::MarkerArray ma;
  int id = 0;
  ma.markers.push_back(createClusterMarker(rawCloud, model_id_*100 + id++, cloud_pose, transform.inverse()));
  ma.markers.push_back(createClusterMarker(transformedCloud, 42000 + model_id_*100 + id++, cloud_pose, transform));
  do {
    Eigen::Vector3d cloudMu = computeCenterOfMass(transformedCloud);
    Eigen::Vector3d distance_voxel_grid_Mu = distance_field_points_center_of_mass_;

    Eigen::Matrix3d W = computeW(transformedCloud, cloudMu, distance_voxel_grid_Mu, kernel);

    // localTrans is the transform that transforms the cloud towards the distance_voxel_grid_
    Eigen::Affine3d localTrans = computeLocalTransform(W, cloudMu, distance_voxel_grid_Mu);

    // TODO make 2d ICP

    // For the computations distance_voxel_grid_ was the fixed "cloud", while the
    // cloud was transformed. As transform is the transformation to move the model/grid
    // to the cloud we apply the inverse
    Eigen::Affine3d newTransform = transform * localTrans.inverse();
    transform = newTransform;

    double newScore = applyTransformAndcomputeScore(transformedCloud, localTrans, kernel);
    if(iter < 10 || newScore > score + EPS)
        score = newScore;
    else
        break;

    geometry_msgs::Pose pose;
    tf::poseEigenToMsg(transform, pose);
    ROS_INFO_STREAM(pose);
    printf("ICP score: %f\n", score);

    ma.markers.push_back(createClusterMarker(rawCloud, model_id_*100 + id++, cloud_pose, transform));
    ma.markers.push_back(createClusterMarker(transformedCloud, 42000 + model_id_*100 + id++, cloud_pose, Eigen::Affine3d::Identity()));
  } while (++iter < max_iterations);
  pubMarker.publish(ma);
#endif

  //cv::Point3f location = center;
  //cv::Point3f vector;

  //do {
  //  double new_score = getFitScoreAndGradient(cloud, location, vector, kernel);
  //  if (new_score > score + EPS) {
  //    score = new_score;
  //    location.x -= vector.x;
  //    location.y -= vector.y;
  //  } else
  //    break;
  //} while (++iter < max_iterations);


  //if (iter == max_iterations) {
  //  //ROS_WARN("Maximum iterations reached in model fitter");
  //}

  geometry_msgs::Pose pose;
  tf::poseEigenToMsg(transform, pose);
  //pose.position.x = location.x;
  //pose.position.y = location.y;
  //pose.position.z = location.z;
  //pose.orientation.x = 0;
  //pose.orientation.y = 0;
  //pose.orientation.z = 0;
  //pose.orientation.w = 1;

  // evaluating the model score is cost-intensive and since the model_score <= 1 ->
  // if score already below min_object_score, then set to 0 and stop further evaluation!
  //if (score > min_object_score) {
  //  //double model_score = getModelFitScore(cloud, location, kernel, search);
  //  //// since for waterthight model only 50% of the points are visible at max, we weight the model_score only half.
  //  //score *= sqrt(model_score);
  //} else
  //  score = 0;

  return ModelFitInfo(model_id_, pose, score);
}

double IterativeTranslationFitter::getModelFitScore(const std::vector<cv::Vec3f>& cloud, const cv::Point3f& position,
    boost::function<double(double)> kernel,
    cv::flann::Index &search) const
{
  double inlier_count = 0;
  std::vector<int> indices(1);
  std::vector<float> distances(1);
  cv::Mat_<float> points(1, 3);
  for (std::vector<cv::Point3f>::const_iterator mIt = model_points_.begin(); mIt != model_points_.end(); ++mIt) {
    points(0, 0) = mIt->x + position.x;
    points(0, 1) = mIt->y + position.y;
    points(0, 2) = mIt->z + position.z;

    search.knnSearch(points, indices, distances, 1);
    inlier_count += kernel(sqrt(distances[0]));
  }
  return inlier_count / model_points_.size();
}

} //namespace
