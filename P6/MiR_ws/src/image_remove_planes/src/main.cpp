#include <ros/ros.h>
#include <pcl_ros/point_cloud.h>
#include <pcl_ros/transforms.h>
#include <boost/foreach.hpp>

#include <pcl/ModelCoefficients.h>
#include <pcl/point_types.h>
#include <pcl/io/pcd_io.h>
#include <pcl/filters/extract_indices.h>
#include <pcl/filters/voxel_grid.h>
#include <pcl/filters/passthrough.h>
#include <pcl/features/normal_3d.h>
#include <pcl/kdtree/kdtree.h>
#include <pcl/sample_consensus/method_types.h>
#include <pcl/sample_consensus/model_types.h>
#include <pcl/segmentation/sac_segmentation.h>
#include <pcl/segmentation/extract_clusters.h>
#include <pcl/common/transforms.h>
#include <pcl/common/common.h>

#include <opencv2/core.hpp>
#include <image_transport/image_transport.h>
#include <cv_bridge/cv_bridge.h>
#include <sensor_msgs/image_encodings.h>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/highgui/highgui.hpp>

//Values fx, fy, cx, cy, img_height, img_width are unique for given resolution and has to be changed if the user wishes to use different resolution.
//The values can be obtained from the topic /camera/aligned_depth_to_color/camera_info. The values bellow, commented out, are for the resolution 640x480.
/*const float fx = 615.718017578125f;
const float cx = 321.9320373535156f;
const float fy = 615.8450317382812f;
const float cy = 230.76663208007812f;
const int img_height = 480;
const int img_width = 640;*/

const float fx = 923.5769653320312f;
const float cx = 642.8980102539062f;
const float fy = 923.767578125f;
const float cy = 346.14996337890625f;
const int img_height = 720;
const int img_width = 1280;

//Parameter square_size describes the size of the burned area around each pixel after transforming the cluster pointcloud points into the pixel locations.
//The higher the value, the bigger BLOBs are in the BLOB picture.
const int square_size = 21;

class remove_planes {
  public:
   remove_planes();
   void callback(const sensor_msgs::PointCloud2::ConstPtr& msg);
   void imageCallback(const sensor_msgs::ImageConstPtr& img_msg);
   cv::Mat getImage();

  private:
   ros::NodeHandle nh;
   ros::NodeHandle nh_im;
   ros::Publisher pub;
   ros::Publisher pub_image;
   ros::Subscriber sub;
   ros::Subscriber sub_image;
   cv_bridge::CvImagePtr cv_ptr;
};

remove_planes::remove_planes(){
  //sub = nh.subscribe<sensor_msgs::PointCloud2>("/camera/depth/color/points", 1, &remove_planes::callback, this);
  sub = nh.subscribe<sensor_msgs::PointCloud2>("/camera/depth_registered/points", 10, &remove_planes::callback, this);
  sub_image = nh_im.subscribe<sensor_msgs::Image>("/camera/color/image_raw", 10, &remove_planes::imageCallback, this);
  pub = nh.advertise<sensor_msgs::PointCloud2>("pointcloud_final", 1);
  pub_image = nh_im.advertise<sensor_msgs::Image>("image_final", 1);
}

void remove_planes::imageCallback(const sensor_msgs::ImageConstPtr& img_msg){
  std::cout<<"\n ImageCallback START  ";
  try
  {
    cv_ptr = cv_bridge::toCvCopy(*img_msg, sensor_msgs::image_encodings::BGR8);
    std::cout<<"ImageCallback SUCCESS "<<std::endl;
  }
  catch (cv_bridge::Exception& e)
  {
    ROS_ERROR("cv_bridge exception: %s", e.what());
    return;
  }
}

cv::Mat remove_planes::getImage() {
  std::cout<<"getImage START ";
  cv::Mat img = cv::Mat(100,100, CV_8UC3); //RGB

    if(cv_ptr){
      std::cout<<"getImage IF ";
      img = cv::Mat(cv_ptr->image.rows, cv_ptr->image.cols, CV_8UC3);
      img = cv_ptr->image;
      cv_ptr.reset();
      std::cout<<"getImage SUCCESS "<<std::endl;
      return img;
    }
}

void remove_planes::callback(const sensor_msgs::PointCloud2::ConstPtr& msg) {
  std::cout<<"\n START   START   START   START   START"<<std::endl;
  pcl::PCLPointCloud2::Ptr cloud2 (new pcl::PCLPointCloud2);
  pcl_conversions::toPCL(*msg,*cloud2);

  pcl::PointCloud<pcl::PointXYZ>::Ptr cloud_NaN (new pcl::PointCloud<pcl::PointXYZ>);
  pcl::fromPCLPointCloud2 (*cloud2, *cloud_NaN);
  std::cout << "cloud_NaN has " << cloud_NaN->points.size() << " data points." << std::endl;

  pcl::PointCloud<pcl::PointXYZ>::Ptr cloud_long (new pcl::PointCloud<pcl::PointXYZ>);
  std::vector<int> indices_NaN;
  pcl::removeNaNFromPointCloud(*cloud_NaN, *cloud_long, indices_NaN);
  std::cout << "cloud_long has " << cloud_long -> points.size () << " data points." << std::endl;

  pcl::PCDWriter writer;
  writer.write<pcl::PointXYZ> ("cloud_NaN.pcd", *cloud_NaN, false);
  writer.write<pcl::PointXYZ> ("cloud_long.pcd", *cloud_long, false);

  // Create the filtering object
  pcl::PointCloud<pcl::PointXYZ>::Ptr cloud (new pcl::PointCloud<pcl::PointXYZ>);
  pcl::PassThrough<pcl::PointXYZ> pass;
  pass.setInputCloud (cloud_long);
  pass.setFilterFieldName ("z");
  pass.setFilterLimits (0.0, 5.0);
  pass.filter (*cloud);
  writer.write<pcl::PointXYZ> ("cloud.pcd", *cloud, false);
  std::cout << "cloud has " << cloud -> points.size () << " data points." << std::endl;

  pcl::PointCloud<pcl::PointXYZ>::Ptr cloud_filtered (new pcl::PointCloud<pcl::PointXYZ>);
  pcl::VoxelGrid<pcl::PointXYZ> vg;
  vg.setInputCloud (cloud);
  vg.setLeafSize (0.03f, 0.03f, 0.03f); //(0.01f, 0.01f, 0.01f)
  vg.filter (*cloud_filtered);
  std::cout << "Downsampled pointcloud has: " << cloud_filtered->points.size ()  << " data points." << std::endl;
  writer.write<pcl::PointXYZ> ("cloud_filtered.pcd", *cloud_filtered, false);

  // Create the segmentation object for the planar model and set all the parameters
  pcl::SACSegmentation<pcl::PointXYZ> seg;
  pcl::PointIndices::Ptr inliers (new pcl::PointIndices);
  pcl::ModelCoefficients::Ptr coefficients (new pcl::ModelCoefficients);
  pcl::PointCloud<pcl::PointXYZ>::Ptr cloud_plane (new pcl::PointCloud<pcl::PointXYZ> ());
  seg.setOptimizeCoefficients (true);
  seg.setModelType (pcl::SACMODEL_PLANE);
  seg.setMethodType (pcl::SAC_RANSAC);
  seg.setMaxIterations (1000);
  seg.setDistanceThreshold (0.03);

  int i=0, nr_points = (int) cloud_filtered->points.size ();
  pcl::PointCloud<pcl::PointXYZ>::Ptr cloud_f (new pcl::PointCloud<pcl::PointXYZ>);
  while (cloud_filtered->points.size () > 0.3 * nr_points) //0.3
  {
    // Segment the largest planar component from the remaining cloud
    seg.setInputCloud (cloud_filtered);
    seg.segment (*inliers, *coefficients);
    if (inliers->indices.size () == 0)
    {
      std::cout << "Could not estimate a planar model for the given dataset." << std::endl;
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
  pcl::search::KdTree<pcl::PointXYZ>::Ptr tree (new pcl::search::KdTree<pcl::PointXYZ>);
  tree->setInputCloud (cloud_filtered);
  std::vector<pcl::PointIndices> cluster_indices;
  pcl::EuclideanClusterExtraction<pcl::PointXYZ> ec;
  ec.setClusterTolerance (0.2f); //0.2f
  ec.setMinClusterSize (5); //5
  ec.setMaxClusterSize (50000);
  ec.setSearchMethod (tree);
  ec.setInputCloud (cloud_filtered);
  ec.extract (cluster_indices);

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

  std::cout << "PointCloud representing the all clusters has " << cloud_cluster_complete.points.size() << " data points." << std::endl;
  writer.write<pcl::PointXYZ> ("cloud_cluster_complete.pcd", cloud_cluster_complete, false);

  std::vector<float> Xim, Yim;
  float u,v,x,y;
  for (size_t i = 0; i < cloud_cluster_complete.points.size(); i++) {
    x = cloud_cluster_complete.points[i].x/cloud_cluster_complete.points[i].z;
    y = cloud_cluster_complete.points[i].y/cloud_cluster_complete.points[i].z;
    u = fx * x + cx;
    v = fy * y + cy;

    Xim.push_back(u);
    Yim.push_back(v);

    if(u > img_width) std::cout<< "u Out of range" << std::endl;
    if(v > img_height) std::cout<< "v Out of range" << std::endl;
  }
  std::cout<<"Size of the Xim vector is " << Xim.size() <<std::endl;
  std::cout<<"Size of the Yim vector is " << Yim.size() <<std::endl;

  cv::Mat image = cv::Mat::zeros(cv::Size(img_width, img_height), CV_8UC1); //CV_8UC3
  std::cout << "Image has  rows " << image.rows << " and cols " << image.cols << std::endl;
  std::vector<int> compression_params;
  compression_params.push_back(CV_IMWRITE_PNG_COMPRESSION);
  compression_params.push_back(9);

  for (size_t i = 0; i < cloud_cluster_complete.points.size(); i++) {
    image.at<uchar>(Yim[i],Xim[i]) = 255;

    for (size_t j = 1 ; j < square_size; j++) {
      for (size_t k = 1 ; k < square_size; k++) {
        if(Yim[i] + j <= img_height-1 && Xim[i] + k <= img_width-1) image.at<uchar>(Yim[i]+j,Xim[i]+k) = 255;
        if(Yim[i] - j >= 0 && Xim[i] - k >= 0) image.at<uchar>(Yim[i]-j,Xim[i]-k) = 255;
      }
    }
  }

  try {
       cv::imwrite("pictureBLOB.png", image, compression_params);
       std::cout<<"PictureBLOB was saved"<<std::endl;
   }
   catch (std::runtime_error& ex) {
       fprintf(stderr, "Exception converting image to PNG format: %s\n", ex.what());
       exit(2);
   }

  cv::Mat colorImage = getImage();
  std::cout << "colorImage has  rows " << colorImage.rows << " and cols " << colorImage.cols << std::endl;

  if(colorImage.empty()!=1){
    for( int i = 0; i < colorImage.rows; ++i){
      for(int j = 0; j< colorImage.cols; ++j){
        if(image.at<uchar>(i,j) != 255) colorImage.at<cv::Vec3b>(i,j) = cv::Vec3b(0,0,0 );
      }
    }

    try {
         cv::imwrite("pictureColor.png", colorImage, compression_params);
         std::cout<<"PictureColor was saved"<<std::endl;
     }
     catch (std::runtime_error& ex) {
         fprintf(stderr, "Exception converting image to PNG format: %s\n", ex.what());
         exit(2);
     }
  }

  cv_bridge::CvImage out_Image;
  out_Image.header.frame_id   = "camera_color_optical_frame"; // Same timestamp and tf frame as input image
  out_Image.header.stamp = ros::Time::now();
  out_Image.encoding = sensor_msgs::image_encodings::BGR8; // Or whatever
  out_Image.image = colorImage; // Your cv::Mat
  pub_image.publish(out_Image.toImageMsg());
  exit(3);
}


int main (int argc, char** argv) {
  ros::init(argc, argv, "pcl");

  remove_planes remove_planes_object;
  ros::spin();

return 0;
}
