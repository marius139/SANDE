#include "ros/ros.h"

//Libraries for  openCV
#include <sstream>
#include <vector>
#include <math.h>

#include <opencv2/opencv.hpp>
#include <opencv2/core.hpp>

#include <image_transport/image_transport.h>
#include <cv_bridge/cv_bridge.h>
#include <sensor_msgs/image_encodings.h>
#include <tf/transform_listener.h>
#include <geometry_msgs/PoseWithCovarianceStamped.h>

#include "people_msgs/People.h"

#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/ml.hpp>

#include "opencv2/xfeatures2d.hpp"
#include "opencv2/xfeatures2d/nonfree.hpp"

#include <pcl/ModelCoefficients.h>
#include <pcl/point_types.h>
#include <pcl/io/pcd_io.h>
#include <pcl/filters/extract_indices.h>
#include <pcl/filters/voxel_grid.h>
#include <pcl/features/normal_3d.h>
#include <pcl/kdtree/kdtree.h>
#include <pcl/sample_consensus/method_types.h>
#include <pcl/sample_consensus/model_types.h>
#include <pcl/segmentation/sac_segmentation.h>
#include <pcl/segmentation/extract_clusters.h>
#include <pcl/common/transforms.h>
#include <pcl/common/common.h>
#include <pcl/filters/passthrough.h>


#define OPENCV_ENABLE_NONFREE

using namespace std;
using namespace cv;

//Path to working directory for testing data
const std::string desPath = "/media/ros/ES/Train/";

//Path to resources
const std::string resources = "/home/ros/MiR_ws/src/simple_navigation_goals/src/resources/";
const int nrMatches = 10;

cv_bridge::CvImagePtr cv_ptr;
cv_bridge::CvImagePtr cv_ptrDep;
double curXpose;
double curYpose;
double curAngle;

unsigned long int  sequenceID = 0;
double totalTime = 0;

//Plane segmentation
const float k1 = 0.264516f;
const float k2 = -0.839907f;
const float k3 = 0.911925f;
const float p1 = -0.001992f;
const float p2 = 0.001437f;
const float fx = 529.215081f;
const float fy = 525.563936f;
const float cx = 328.942720f;
const float cy = 267.480682f;
const int img_width = 640;
const int img_height = 480;

const int square_size = 21; //31

//Plane planeSegmentation algorithm
void planeSegment(){


  ros::Duration dur;
  double timeSpent;
  ros::Time startTime;

  for (int idx = 0; idx < 1133; idx++){

   std::stringstream sImg, sCloud, sFinal;
   std::cout<<"Iteration "<<idx<<std::endl;

   //Path to get where to get point cloud testing images
   sCloud << desPath+"cloud/cloud_" << idx << ".pcd";
   //Path to where to save after plane exstraction
   sFinal << desPath+"result_image/rgb_" << idx << ".png";

  if(idx<10){
    sImg << desPath+"rgb/seq0_000"<<idx<<"_0.ppm";
  } else if (idx == 10 || idx < 100) {
    sImg << desPath+"rgb/seq0_00"<<idx<<"_0.ppm";
  } else if (idx == 100 || idx < 1000) {
    sImg << desPath+"rgb/seq0_0"<<idx<<"_0.ppm";
  } else if (idx == 1000 || idx < 10000) {
    sImg << desPath+"rgb/seq0_"<<idx<<"_0.ppm";
  }

    startTime = ros::Time::now();

   pcl::PointCloud<pcl::PointXYZ>::Ptr cloud_NaN (new pcl::PointCloud<pcl::PointXYZ>);
   if (pcl::io::loadPCDFile<pcl::PointXYZ> (sCloud.str(), *cloud_NaN) == -1) //* load the file
   {
     ROS_INFO("Couldn't read file test_pcd.pcd");
   }

   pcl::PointCloud<pcl::PointXYZ>::Ptr cloud_long (new pcl::PointCloud<pcl::PointXYZ>);
   std::vector<int> indices_NaN;
   pcl::removeNaNFromPointCloud(*cloud_NaN, *cloud_long, indices_NaN);
  std::cout << "cloud_long has " << cloud_long -> points.size () << " data points." << std::endl;

   // Create the filtering object
   pcl::PointCloud<pcl::PointXYZ>::Ptr cloud (new pcl::PointCloud<pcl::PointXYZ>);
   pcl::PassThrough<pcl::PointXYZ> pass;
   pass.setInputCloud (cloud_long);
   pass.setFilterFieldName ("z");
   pass.setFilterLimits (0.0, 5.0);
   pass.filter (*cloud);
   std::cout << "cloud has " << cloud -> points.size () << " data points." << std::endl;

   pcl::PCDWriter writer;
   writer.write<pcl::PointXYZ> ("cloud.pcd", *cloud, false);

   pcl::PointCloud<pcl::PointXYZ>::Ptr cloud_filtered (new pcl::PointCloud<pcl::PointXYZ>);
   pcl::VoxelGrid<pcl::PointXYZ> vg;
   vg.setInputCloud (cloud);
   vg.setLeafSize (0.03f, 0.03f, 0.03f); //(0.0175f, 0.0175f, 0.0175f) //(0.04f, 0.04f, 0.04f)
   vg.filter (*cloud_filtered);
   std::cout << "Filtered pointcloud has: " << cloud_filtered->points.size ()  << " data points." << std::endl;
   writer.write<pcl::PointXYZ> ("cloud_filtered.pcd", *cloud_filtered, false);

   // Create the segmentation object for the planar model and set all the parameters
   pcl::SACSegmentation<pcl::PointXYZ> seg;
   pcl::PointIndices::Ptr inliers (new pcl::PointIndices);
   pcl::ModelCoefficients::Ptr coefficients (new pcl::ModelCoefficients);
   pcl::PointCloud<pcl::PointXYZ>::Ptr cloud_plane (new pcl::PointCloud<pcl::PointXYZ> ());
   seg.setOptimizeCoefficients (true); //true
   seg.setModelType (pcl::SACMODEL_PLANE); //pcl::SACMODEL_PLANE
   seg.setMethodType (pcl::SAC_RANSAC); //pcl::SAC_RANSAC
   seg.setMaxIterations (1000); //100
   seg.setDistanceThreshold (0.2); //0.005f

   int i=0, nr_points = (int) cloud_filtered->points.size ();
   pcl::PointCloud<pcl::PointXYZ>::Ptr cloud_f (new pcl::PointCloud<pcl::PointXYZ>);
   while (cloud_filtered->points.size () > 0.3 * nr_points) //0.3
   {
     // Segment the largest planar component from the remaining cloud
     seg.setInputCloud (cloud_filtered);
     seg.segment (*inliers, *coefficients);
     if (inliers->indices.size () == 0)
     {
       //std::cout << "Could not estimate a planar model for the given dataset." << std::endl;
       break;
     }
     // Extract the planar inliers from the input cloud
     pcl::ExtractIndices<pcl::PointXYZ> extract;
     extract.setInputCloud (cloud_filtered);
     extract.setIndices (inliers);
     extract.setNegative (false);
     // Get the points associated with the planar surface
     extract.filter (*cloud_plane);
     // Remove the planar inliers, extract the rest
     extract.setNegative (true);
     extract.filter (*cloud_f);
     *cloud_filtered = *cloud_f;
   }

   // Creating the KdTree object for the search method of the extraction
   //std::cout<<"KDTree START ";
   pcl::search::KdTree<pcl::PointXYZ>::Ptr tree (new pcl::search::KdTree<pcl::PointXYZ>);
   //std::cout<<"KDTree SETUP ";
   tree->setInputCloud (cloud_filtered);
   std::vector<pcl::PointIndices> cluster_indices;
   pcl::EuclideanClusterExtraction<pcl::PointXYZ> ec;
   ec.setClusterTolerance (0.2f);//0.05 // 0.018 0.02 2cm
   ec.setMinClusterSize (5); //5
   ec.setMaxClusterSize (50000);
   ec.setSearchMethod (tree);
   ec.setInputCloud (cloud_filtered);
   //std::cout<<"KDTree EXTRACT ";
   ec.extract (cluster_indices);
   //std::cout<<"KDTree SUCCESS"<<std::endl;
   //size of cluster_indices equals to the number of cluster objects

   pcl::PointCloud<pcl::PointXYZ> cloud_cluster_complete;
   for (std::vector<pcl::PointIndices>::const_iterator it = cluster_indices.begin (); it != cluster_indices.end (); ++it)
   {
     pcl::PointCloud<pcl::PointXYZ>::Ptr cloud_cluster (new pcl::PointCloud<pcl::PointXYZ>);
     for (std::vector<int>::const_iterator pit = it->indices.begin (); pit != it->indices.end (); ++pit)
     cloud_cluster->points.push_back (cloud_filtered->points[*pit]);
     cloud_cluster->width = cloud_cluster->points.size ();
     cloud_cluster->height = 1;
     cloud_cluster->is_dense = true;

     cloud_cluster_complete += *cloud_cluster;
   }

   std::cout << "PointCloud representing the complete Cluster has " << cloud_cluster_complete.points.size() << " data points." << std::endl;
   writer.write<pcl::PointXYZ> ("cloud_cluster_complete.pcd", cloud_cluster_complete, false);

   std::vector<float> Xim, Yim;
   float u,v,xP,yP, zP ,xPP,yPP,r, rx, ry, k;
   for (size_t i = 0; i < cloud_cluster_complete.points.size(); i++) {
     xP = cloud_cluster_complete.points[i].x;
     yP = cloud_cluster_complete.points[i].y - 0.1f;
     zP = cloud_cluster_complete.points[i].z;
     xPP = xP/zP;
     yPP = yP/zP;
     u = fx * xPP + cx;
     v = fy * yPP + cy;

     if(u < img_width && v < img_height && u >= 0 && v >= 0) {
       Xim.push_back(u);
       Yim.push_back(v);
     }

     if(u > img_width) std::cout<< "u Out of range " << u << std::endl;
     if(v > img_height) std::cout<< "v Out of range" << v << std::endl;
   }



   std::cout<<"Size of the Xim vector is " << Xim.size() <<std::endl;
   std::cout<<"Size of the Yim vector is " << Yim.size() <<std::endl;

   cv::Mat image = cv::Mat::zeros(cv::Size(img_width, img_height), CV_8UC1); //CV_8UC3
   std::cout << "Image has  rows " << image.rows << " and cols " << image.cols << std::endl;
   std::vector<int> compression_params;
   compression_params.push_back(CV_IMWRITE_PNG_COMPRESSION);
   compression_params.push_back(9);

   for (size_t i = 0; i < Yim.size(); i++) {
     image.at<uchar>(Yim[i],Xim[i]) = 255;

     for (size_t j = 1 ; j < square_size; j++) {
       for (size_t k = 1 ; k < square_size; k++) {
         if(Yim[i] + j <= img_height-1 && Xim[i] + k <= img_width-1) image.at<uchar>(Yim[i]+j,Xim[i]+k) = 255;
         if(Yim[i] - j >= 0 && Xim[i] - k >= 0) image.at<uchar>(Yim[i]-j,Xim[i]-k) = 255;
       }
     }
   }

  cv::Mat colorImage = cv::imread(sImg.str(), -1);
  //std::cout << "Image has width " << colorImage.cols << " and height " << colorImage.rows <<std::endl;

  if(colorImage.empty()!=1){
    for( int i = 0; i < colorImage.rows; ++i){
      for(int j = 0; j< colorImage.cols; ++j){
        if(image.at<uchar>(i,j) != 255) colorImage.at<cv::Vec3b>(i,j) = cv::Vec3b(0,0,0 );
      }
    }

    dur = ros::Time::now() - startTime;
    timeSpent = dur.toSec();
    totalTime += timeSpent;

    ROS_INFO("Time spent: %f", timeSpent);

    try {
         cv::imwrite(sFinal.str(), colorImage, compression_params);
         //std::cout<<"PictureColor was saved"<<std::endl;
     }
     catch (std::runtime_error& ex) {
         fprintf(stderr, "Exception converting image to PNG format: %s\n", ex.what());
         exit(2);
     }
  }

  }

}

//Detect humans with hog
void hogDetect(){

	//Folder to get test image from
    std::string From = "data_rot0/";
	//Folder to save output in
    std::string To = "test/";

    ros::Duration dur;
    double timeSpent;
    ros::Time startTime;

     Mat currentImg;
     Mat img;
     /// Set up the pedestrian detector --> let us take the default one
     HOGDescriptor hog;
     hog.setSVMDetector(HOGDescriptor::getDefaultPeopleDetector());

     /// Set  up tracking vector
     vector<Point> track;


        for(int i=0; i<1133; i++){


          std::ostringstream k;
          k << "Original_rgbR_"<<i<<".png";
          std::string imagePath(k.str());

          //ROS_INFO("nr: %i", i);

          //img = imread(desPath+From+imagePath, 1);
          img = imread(desPath+From+"Original_rgbR_12.png", 1);
          while(img.empty()){
            ROS_INFO("waiting...");
          }

         //cv::resize(currentImg, img, cv::Size(960,1280));
         //img = currentImg;


         /// groupThreshold (set groupThreshold to 0 to turn off the grouping completely).

         startTime = ros::Time::now();

         vector<Rect> found;
         vector<double> weights;
         double scaleFactor = 1.05;
         Size winStride=Size(4,4);
         Size padding=Size(8,8);
         int groupThreshold = 2;

         hog.detectMultiScale(img, found, 0,  winStride, padding, scaleFactor, groupThreshold);

         /// draw detections and store location
         for( size_t z = 0; z < found.size(); z++ )
         {
             Rect r = found[z];
             rectangle(img, found[z], cv::Scalar(0,0,255), 3);

         }

         dur = ros::Time::now() - startTime;
         timeSpent = dur.toSec();
         totalTime += timeSpent;

         //ROS_INFO("Time spent: %f", timeSpent);

         std::ostringstream k_rot;
         k_rot << "0rgb_"<<i<<".png";
         std::string imagePathRot(k_rot.str());

         //ROS_INFO("pathRot: %s", imagePathRot.c_str());

         imwrite(desPath+To+imagePathRot, img);


         /// Show
         //namedWindow("person",WINDOW_AUTOSIZE);
         //resizeWindow("person", 600,600);
         imshow("person", img);
         waitKey(0);
     }


}

int main(int argc, char** argv){

  ros::init(argc, argv, "human_detect");
  ros::NodeHandle nh1;
  ROS_INFO("Version: %i.%i", CV_MAJOR_VERSION,CV_MINOR_VERSION);
  //Subscribe to camera rgb image
  ros::Subscriber image_sub_ = nh1.subscribe("/camera/color/image_raw", 1,imageCallback);
  //Subscribe to camera depth
  ros::Subscriber depth_sub_ = nh1.subscribe("/camera/depth/image_rect_raw", 1,depthCallback);
  //Subscribe to the mir platform current position in the map
  ros::Subscriber sub_pose = nh1.subscribe<geometry_msgs::PoseWithCovarianceStamped>("/amcl_pose", 10, moveBaseCallback);
  //People message publisher
  ros::Publisher pubCV = nh1.advertise<people_msgs::People> ("/personPosition", 100);

  std::vector<float> mapPos;

  ros::Duration dur;
  double timeSpent;
  ros::Time startTime;

  //Test run
  //planeSegment();
  hogDetect();

  //ROS_INFO("TOTAL Time spent: %f", totalTime);
  //ROS_INFO("Avg Time spent: %f", totalTime/1132);

}
