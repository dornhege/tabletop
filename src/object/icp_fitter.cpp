/*********************************************************************
 * Software License Agreement (BSD License)
 *
 *  Copyright (c) 2009, Willow Garage, Inc.
 *  Copyright (c) 2015, University of Freiburg
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

// Author(s): Christian Dornhege, Marius Muja, Matei Ciocarlie and Romain Thibaux

#include <tabletop_object_detector/icp_fitter.h>

#include <boost/bind.hpp>
#include <Eigen/Core>
#include <Eigen/SVD>
#include <eigen_conversions/eigen_msg.h>
#include <geometric_shapes/shape_to_marker.h>
#include <pcl/point_types.h>
#include <pcl/filters/voxel_grid.h>

/* Possible improvements:
 * - For convergence: try 45 deg as second initial guess to prevent 90 deg flipping
 * - For numerical stability: maybe every X steps fix transform.linear = transform.rotation?
 * - For performance: Sort out data types to prevent conversion/copying of point clouds
 */

namespace tabletop_object_detector
{

/*! Computes the point at the bottom of the point cloud vertical to the center of
 *  gravity. This is the point where the table supports the object.
 */
// from iterative_translation_fitter.cpp
cv::Point3f IcpFitter::centerOfSupport(const std::vector<cv::Vec3f>& cloud) const
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

visualization_msgs::Marker IcpFitter::createMeshMarker(int id,
        const geometry_msgs::Pose & cloud_pose, const Eigen::Affine3d & icp_transform,
        const std::string & ns) const
{
    visualization_msgs::Marker marker;
    marker.scale.x = 1.0;
    marker.scale.y = 1.0;
    marker.scale.z = 1.0;
    geometric_shapes::constructMarkerFromShape(mesh_, marker); // assumes this was initialized from mesh
    marker.header.frame_id = sensor_frame_id_;
    marker.id = id;
    marker.ns = ns;
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
    if(ns == "final") {
        marker.color.r = 1.0;
        marker.color.g = 1.0;
        marker.color.b = 0.0;
        marker.color.a = 0.8;
    }
    return marker;
}

visualization_msgs::Marker IcpFitter::createClusterMarker(const EigenSTL::vector_Vector3d & cluster, int id,
        const geometry_msgs::Pose & cloud_pose,
        const std::string & ns) const
{
    visualization_msgs::Marker marker;
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
    marker.header.frame_id = sensor_frame_id_;
    marker.id = id;
    marker.ns = ns;
    marker.action = visualization_msgs::Marker::ADD;
    marker.pose = cloud_pose;
    marker.color.a = 0.5;
    marker.color.r = id/100./10.0;
    marker.color.b = 0.5 + id%100/10.0;
    if(marker.color.b > 1.0)
        marker.color.b = 1.0;
    if(marker.ns == "model_fit_cloud") {
        marker.color.r = 1.0;
        marker.color.g = 0.0;
        marker.color.b = 1.0;
        marker.color.a = 0.8;
    }
    return marker;
}

/// Transform cloud by transform and compute a match score.
/**
 * \param [in] distance_score_kernel a scoring function that given the distance produces a value
 * between 0 and 1, where 1 is best match distance (i.e. inlier)
 */
double IcpFitter::applyTransformAndcomputeScore(EigenSTL::vector_Vector3d & cloud,
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

void IcpFitter::computeMus(const EigenSTL::vector_Vector3d & cloud,
        Eigen::Vector3d & cloudMu, Eigen::Vector3d & distance_voxel_grid_Mu,
        boost::function<double(double)> distance_selection_kernel) const
{
    cloudMu = Eigen::Vector3d(0.0, 0.0, 0.0);
    distance_voxel_grid_Mu = Eigen::Vector3d(0.0, 0.0, 0.0);
    double numCorrs = 0.0;
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
                Eigen::Vector3d modelPt(cx, cy, cz);    // grid point corresponce

                val = distance_voxel_grid_->getDistance(x, y, z);
                double weight = distance_selection_kernel(val);
                cloudMu += weight * cloudPt;
                distance_voxel_grid_Mu += weight * modelPt;
                numCorrs += weight;
            }
        }
    }
    if(numCorrs > 0.0) {
        cloudMu /= numCorrs;
        distance_voxel_grid_Mu /= numCorrs;
    }
}

Eigen::Matrix3d IcpFitter::computeW(
        const EigenSTL::vector_Vector3d & cloud, const Eigen::Vector3d & cloudMu,
        const Eigen::Vector3d & distance_voxel_grid_Mu,
        boost::function<double(double)> distance_selection_kernel) const
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
                Eigen::Vector3d modelPt(cx, cy, cz);    // grid point corresponce
                modelPt -= distance_voxel_grid_Mu;;
                Eigen::Vector3d cloudPtX = cloudPt - cloudMu;
                if(!use_3d_icp_) {
                    modelPt.z() = 0;
                    cloudPtX.z() = 0;
                }

                val = distance_voxel_grid_->getDistance(x, y, z);
                double weight = distance_selection_kernel(val);
                W += weight * modelPt * cloudPtX.transpose();
            }
        }
    }
    return W;
}

Eigen::Affine3d IcpFitter::computeLocalTransform(const Eigen::Matrix3d & W, const Eigen::Vector3d cloudMu,
        const Eigen::Vector3d distance_voxel_grid_Mu) const
{
    Eigen::JacobiSVD<Eigen::Matrix3d> svd(W, Eigen::ComputeFullU | Eigen::ComputeFullV);
    Eigen::Matrix3d R = svd.matrixU() * svd.matrixV().transpose();
    if(!use_3d_icp_) {
        // super nasty hack
        double angZ = atan2(R(0, 1), R(0, 0));
        R(0, 0) = cos(angZ);
        R(1, 1) = cos(angZ);
        R(0, 1) = sin(angZ);
        R(1, 0) = -sin(angZ);
        R(0, 2) = 0.0;
        R(1, 2) = 0.0;
        R(2, 0) = 0.0;
        R(2, 1) = 0.0;
        R(2, 2) = 1.0;
    }
    Eigen::Vector3d t = distance_voxel_grid_Mu - R * cloudMu;
    Eigen::Affine3d localTrans = Eigen::Affine3d::Identity();
    localTrans.linear() = R;
    localTrans.translation() = t;
    return localTrans;
}

boost::function<double(double)> IcpFitter::buildKernel(const std::string & kernel_type, double kernel_dist)
{
  boost::function<double(double)> kernel;
  if(kernel_type == "huber") {
      kernel = boost::bind(huberKernel, kernel_dist, _1);
  } else if(kernel_type == "selection") {
      kernel = boost::bind(selectionKernel, kernel_dist, _1);
  } else {
      ROS_ERROR("Unknown kernel: %s", kernel_type.c_str());
  }
  return kernel;
}

/*! Performs standard iterative closest points (ICP).
 *
 *  The fit is initialized to the centroid of the cloud along x and y, and 0 along z. This
 *  assumes that the meshes are all such that the origin is at the bottom, with the z axis
 *  pointing up. It also assumes the points have been translated to a coordinate frame
 *  where z=0 is the table plane.
 *
 *  For the same reason. there is no iteration done along z at all.
 */
ModelFitInfo IcpFitter::fitPointCloud(const std::vector<cv::Vec3f>& cloud,
        const geometry_msgs::Pose & cloud_pose,
        cv::flann::Index &search, double min_object_score) const
{
  if(cloud.empty()) {
    //ROS_ERROR("Attempt to fit model to empty point cloud");
    geometry_msgs::Pose bogus_pose;
    return ModelFitInfo(model_id_, bogus_pose, 0.0);
  }

  boost::function<double(double)> inlier_kernel = buildKernel(inlier_kernel_, inlier_dist_);
  boost::function<double(double)> outlier_kernel = buildKernel(outlier_kernel_, outlier_dist_);

  pcl::PointCloud<pcl::PointXYZ>::Ptr inputCloud(new pcl::PointCloud<pcl::PointXYZ>());
  for(size_t i = 0; i < cloud.size(); i++) {
      inputCloud->push_back(pcl::PointXYZ(
                  cloud[i][0], cloud[i][1], cloud[i][2]));
  }
  pcl::PointCloud<pcl::PointXYZ> downsampledCloud;
  // Downsample cloud
  pcl::VoxelGrid<pcl::PointXYZ> vg;
  vg.setInputCloud(inputCloud);
  vg.setLeafSize(downsample_leaf_size_, downsample_leaf_size_, downsample_leaf_size_);
  vg.filter(downsampledCloud);

  ROS_DEBUG("Downsampled input: from %zu to %zu points.", cloud.size(), downsampledCloud.size());

  // transform is the transformation to move the model/distance_voxel_grid to the input point cloud
  // i.e. the resulting pose of the model
  Eigen::Affine3d transform = Eigen::Affine3d::Identity();
  // As we can't transform the distance_voxel_grid_, we transform the input cloud towards the model
  // Thus, transformedCloud is transform^-1 * cloud
  EigenSTL::vector_Vector3d transformedCloud;
  transformedCloud.reserve(downsampledCloud.size());
  // use center as initial guess
  cv::Point3f center = centerOfSupport(cloud);
  transform.translation().x() = center.x;
  transform.translation().y() = center.y;
  for(size_t i = 0; i < downsampledCloud.size(); i++) {
      transformedCloud.push_back(Eigen::Vector3d(
                  downsampledCloud[i].x - center.x, downsampledCloud[i].y - center.y, downsampledCloud[i].z));
  }

  ROS_INFO("fitPointCloud for model %d", model_id_);
  int id = 0;
  double score = 0;
  visualization_msgs::MarkerArray ma;
  if(pubMarker.getNumSubscribers() > 0) {
      ma.markers.push_back(createMeshMarker(model_id_*100 + id++, cloud_pose, transform, "icp"));
      ma.markers.push_back(createClusterMarker(transformedCloud, model_id_*100 + id, cloud_pose, "icp_transformed"));
  }
  int iteration = 0;
  do {
    Eigen::Vector3d cloudMu;
    Eigen::Vector3d distance_voxel_grid_Mu;
    computeMus(transformedCloud, cloudMu, distance_voxel_grid_Mu, outlier_kernel);
    if(!use_3d_icp_) {
        cloudMu.z() = 0;
        distance_voxel_grid_Mu.z() = 0;
    }

    Eigen::Matrix3d W = computeW(transformedCloud, cloudMu, distance_voxel_grid_Mu, outlier_kernel);

    // localTrans is the transform that transforms the cloud towards the distance_voxel_grid_
    Eigen::Affine3d localTrans = computeLocalTransform(W, cloudMu, distance_voxel_grid_Mu);

    // For the computations distance_voxel_grid_ was the fixed "cloud", while the
    // cloud was transformed. As transform is the transformation to move the model/grid
    // to the cloud we apply the inverse
    Eigen::Affine3d newTransform = transform * localTrans.inverse();
    transform = newTransform;

    double newScore = applyTransformAndcomputeScore(transformedCloud, localTrans, inlier_kernel);
    if(iteration < min_iterations_ || newScore > score + min_iteration_improvement_)
        score = newScore;
    else
        break;

    ROS_DEBUG("i: %d: ICP score: %f", iteration, score);

    if(pubMarker.getNumSubscribers() > 0) {
        ma.markers.push_back(createMeshMarker(model_id_*100 + id++, cloud_pose, transform, "icp"));
        ma.markers.push_back(createClusterMarker(transformedCloud, model_id_*100 + id, cloud_pose, "icp_transformed"));
    }
  } while (++iteration < max_iterations_);

  if(pubMarker.getNumSubscribers() > 0) {
      ma.markers.push_back(createMeshMarker(12345, cloud_pose, transform, "final"));
      pubMarker.publish(ma);
  }

  geometry_msgs::Pose pose;
  tf::poseEigenToMsg(transform, pose);
  ROS_DEBUG_STREAM("Final pose: " << pose);

  // evaluating the model score is cost-intensive and since the model_score <= 1 ->
  // if score already below min_object_score, then set to 0 and stop further evaluation!
  if (score > min_object_score) {
    double model_score = getModelFitScore(cloud, transform, inlier_kernel, search, cloud_pose);
    //// since for waterthight model only 50% of the points are visible at max, we weight the model_score only half.
    score *= sqrt(model_score);
    ROS_DEBUG("Model_score: %f Final score: %f", model_score, score);
  } else
      score = 0;

  return ModelFitInfo(model_id_, pose, score);
}

// from iterative_translation_fitter.cpp
double IcpFitter::getModelFitScore(const std::vector<cv::Vec3f>& cloud,
        const Eigen::Affine3d & pose,
    boost::function<double(double)> kernel,
    cv::flann::Index &search, const geometry_msgs::Pose & cloud_pose) const
{
  double inlier_count = 0;
  std::vector<int> indices(1);
  std::vector<float> distances(1);
  cv::Mat_<float> points(1, 3);
  EigenSTL::vector_Vector3d model_cloud;
  ROS_DEBUG("%s: Model_surface_points: %zu", __func__, model_surface_points_.size());
  for (std::vector<cv::Point3f>::const_iterator mIt = model_surface_points_.begin(); mIt != model_surface_points_.end(); ++mIt) {
      Eigen::Vector3d pt(mIt->x, mIt->y, mIt->z);
      Eigen::Vector3d transformedPoint = pose * pt;
      model_cloud.push_back(transformedPoint);
    points(0, 0) = transformedPoint.x();
    points(0, 1) = transformedPoint.y();
    points(0, 2) = transformedPoint.z();

    search.knnSearch(points, indices, distances, 1);
    inlier_count += kernel(sqrt(distances[0]));
  }
  if(pubMarker.getNumSubscribers() > 0) {
      visualization_msgs::MarkerArray ma;
      ma.markers.push_back(createClusterMarker(model_cloud, 12346, cloud_pose, "model_fit_cloud"));
      pubMarker.publish(ma);
  }
  return inlier_count / model_surface_points_.size();
}

} //namespace
