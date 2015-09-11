/*
 * Software License Agreement (BSD License)
 *
 *  Copyright (c) 2012, Willow Garage, Inc.
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
 *   * Neither the name of Willow Garage, Inc. nor the names of its
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
 *
 */

#ifndef TABLETOP_OBJECT_DETECTOR_H_
#define TABLETOP_OBJECT_DETECTOR_H_

// Author(s): Marius Muja and Matei Ciocarlie

#include <tabletop_object_detector/exhaustive_fit_detector.h>
#include <tabletop_object_detector/iterative_distance_fitter.h>
#include <tabletop_object_detector/icp_fitter.h>

#include <string>

#include <opencv2/flann/flann.hpp>
#include <visualization_msgs/MarkerArray.h>
#include <ros/ros.h>

namespace tabletop_object_detector
{
class TabletopObjectRecognizer
{
  private:
    //! The instance of the detector used for all detecting tasks
    ExhaustiveFitDetector detector_;

    //! The threshold for merging two models that were fit very close to each other
    double fit_merge_threshold_;
    ros::Publisher pubMarker_;
    std::string fitter_type_;

    double getConfidence (double score) const
    {
      return (1.0 - (1.0 - score) * (1.0 - score));
    }
    double getScore(double confidence) const
    {
        return 1.0 - sqrt(1 - confidence);
    }
  public:
    //! Subscribes to and advertises topics; initializes fitter
    TabletopObjectRecognizer()
    {
      detector_ = ExhaustiveFitDetector();
      //initialize operational flags
      fit_merge_threshold_ = 0.02;

      ros::NodeHandle nh;
      pubMarker_ = nh.advertise<visualization_msgs::MarkerArray>("object_detection_marker", 1);

      ros::NodeHandle nhPriv("~");
      nhPriv.param("fitter_type", fitter_type_, std::string("iterative_translation"));
    }

    //! Empty stub
    ~TabletopObjectRecognizer()
    {
    }

    void
    clearObjects()
    {
      detector_.clearObjects();
    }

    void
    addObject(int model_id, const shape_msgs::Mesh & mesh)
    {
        if(fitter_type_ == "iterative_translation")
            detector_.addObject<IterativeTranslationFitter>(model_id, mesh);
        else if(fitter_type_ == "icp")
            detector_.addObject<IcpFitter>(model_id, mesh);
        else
            ROS_WARN("%s: Unknown fitter type: %s", __PRETTY_FUNCTION__, fitter_type_.c_str());
    }

    /** Structure used a return type for objectDetection */
    struct TabletopResult
    {
      geometry_msgs::Pose pose_;
      float confidence_;
      int object_id_;
      std::vector<cv::Vec3f> cloud_;
      size_t cloud_index_;
    };

    visualization_msgs::Marker createClusterMarker(const std::vector<cv::Vec3f> & cluster, int id,
            const geometry_msgs::Pose & cluster_pose)
    {
        visualization_msgs::Marker marker;
        marker.header.frame_id = "head_mount_kinect_rgb_optical_frame"; // TODO figure this out, probably incoming msg frame, do we have that???
        marker.ns = "clusters";
        marker.id = id;
        marker.type = visualization_msgs::Marker::POINTS;
        marker.action = visualization_msgs::Marker::ADD;
        marker.pose = cluster_pose;
        marker.scale.x = 0.01;
        marker.scale.y = 0.01;
        marker.scale.z = 0.01;
        marker.color.a = 1.0;
        marker.color.r = 0.5 + 0.5 * drand48();
        marker.color.g = 0.5 + 0.5 * drand48();
        marker.color.b = 0.5 + 0.5 * drand48();
        for(int i = 0; i < cluster.size(); i++) {
            geometry_msgs::Point pt;
            pt.x = cluster[i][0];
            pt.y = cluster[i][1];
            pt.z = cluster[i][2];
            marker.points.push_back(pt);
        }
        return marker;
    }

    /*! Performs the detection on each of the clusters, and populates the returned message.
     */
    void
    objectDetection(std::vector<std::vector<cv::Vec3f> > &clusters,
            const std::vector<geometry_msgs::Pose> & cluster_poses,
            float confidence_cutoff,
            bool perform_fit_merge, std::vector<TabletopResult > &results)
    {
      //do the model fitting part
      std::vector<size_t> cluster_model_indices;
      std::vector<std::vector<ModelFitInfo> > raw_fit_results(clusters.size());
      std::vector<cv::flann::Index> search(clusters.size());
      cluster_model_indices.resize(clusters.size(), -1);
      int num_models = 1;
      visualization_msgs::MarkerArray ma;
      for (size_t i = 0; i < clusters.size(); i++)
      {
        cluster_model_indices[i] = i;
        cv::Mat features = cv::Mat(clusters[i]).reshape(1);
        search[i].build(features, cv::flann::KDTreeIndexParams());

        raw_fit_results[i] = detector_.fitBestModels(clusters[i], cluster_poses[i],
                std::max(1, num_models), search[i], getScore(confidence_cutoff));
        ma.markers.push_back(createClusterMarker(clusters[i], i, cluster_poses[i]));
      }
      pubMarker_.publish(ma);

      //merge models that were fit very close to each other
      if (perform_fit_merge)
      {
        size_t i = 0;
        while (i < clusters.size())
        {
          //if cluster i has already been merged continue
          if (cluster_model_indices[i] != (int) i || raw_fit_results.at(i).empty())
          {
            i++;
            continue;
          }

          size_t j;
          for (j = i + 1; j < clusters.size(); j++)
          {
            //if cluster j has already been merged continue
            if (cluster_model_indices[j] != (int) j)
              continue;
            //if there are no fits, merge based on cluster vs. fit
//            if (raw_fit_results.at(j).empty())
//            {
//              if (fitClusterDistance<typename pcl::PointCloud<PointType> >(raw_fit_results.at(i).at(0), *clusters[j])
//                  < fit_merge_threshold_)
//                break;
//              else
//                continue;
//            }
            //else merge based on fits
            if (!raw_fit_results.at(j).empty() && fitDistance(raw_fit_results.at(i).at(0), raw_fit_results.at(j).at(0)) < fit_merge_threshold_)
              break;
          }          
          if (j < clusters.size())
          {
            //merge cluster j into i
            clusters[i].insert(clusters[i].end(), clusters[j].begin(), clusters[j].end());
            //delete fits for cluster j so we ignore it from now on
            raw_fit_results.at(j).clear();
            //fits for cluster j now point at fit for cluster i
            cluster_model_indices[j] = i;
            //refit cluster i
            raw_fit_results.at(i) = detector_.fitBestModels(clusters[i], cluster_poses[i],
                    std::max(1, num_models), search[i], getScore(confidence_cutoff));
          }
          else
          {
            i++;
          }
        }
      }

      // Merge clusters together
      for (size_t i = 0; i < cluster_model_indices.size(); i++)
      {
        if ((cluster_model_indices[i] != int(i)) || (raw_fit_results[i].empty()))
          continue;

        double confidence = getConfidence (raw_fit_results[i][0].getScore());

        if (confidence < confidence_cutoff)
          continue;

        TabletopResult result;
        result.object_id_ = raw_fit_results[i][0].getModelId();
        result.pose_ = raw_fit_results[i][0].getPose();
        result.confidence_ = confidence;
        result.cloud_ = clusters[i];
        result.cloud_index_ = i;

        results.push_back(result);
      }
    }

    //-------------------- Misc -------------------

    //! Helper function that returns the distance along the plane between two fit models
    double
    fitDistance(const ModelFitInfo &m1, const ModelFitInfo &m2)
    {
      double dx = m1.getPose().position.x - m2.getPose().position.x;
      double dy = m1.getPose().position.y - m2.getPose().position.y;
      double d = dx * dx + dy * dy;
      return sqrt(d);
    }

    template<class PointCloudType>
    double
    fitClusterDistance(const ModelFitInfo &m, const PointCloudType &cluster)
    {
      double dist = 100.0 * 100.0;
      double mx = m.getPose().position.x;
      double my = m.getPose().position.y;
      for (size_t i = 0; i < cluster.points.size(); i++)
      {
        double dx = cluster.points[i].x - mx;
        double dy = cluster.points[i].y - my;
        double d = dx * dx + dy * dy;
        dist = std::min(d, dist);
      }
      return sqrt(dist);
    }
  };
}

#endif /* TABLETOP_OBJECT_DETECTOR_H_ */
