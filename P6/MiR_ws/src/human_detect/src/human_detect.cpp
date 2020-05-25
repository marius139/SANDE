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


#define OPENCV_ENABLE_NONFREE

using namespace std;
using namespace cv;

//Path to folder to store descriptors
const std::string desPath = "/media/ros/ES/Train/";

//Path to resources
const std::string resources = "/home/ros/MiR_ws/src/simple_navigation_goals/src/resources/";
const int nrMatches = 10;

cv_bridge::CvImagePtr cv_ptr;
cv_bridge::CvImagePtr cv_ptrDep;
double curXpose;
double curYpose;
double curAngle;

double totalTime = 0;

void imageCallback(const sensor_msgs::ImageConstPtr msg){
  try
  {
    cv_ptr = cv_bridge::toCvCopy(*msg, sensor_msgs::image_encodings::BGR8);
  }
  catch (cv_bridge::Exception& e)
  {
    ROS_ERROR("cv_bridge exception: %s", e.what());
    return;
  }
}

void depthCallback(const sensor_msgs::ImageConstPtr msg){
  try
  {
    cv_ptrDep = cv_bridge::toCvCopy(*msg, sensor_msgs::image_encodings::TYPE_16UC1);
  }
  catch (cv_bridge::Exception& e)
  {
    ROS_ERROR("cv_bridge exception: %s", e.what());
    return;
  }
}

cv::Mat getImage() {
  cv::Mat img = cv::Mat(100,100, CV_8UC3); //RGB
  ros::Time tid = ros::Time::now()+ros::Duration(5.0);
  while (ros::ok()) {
    if(ros::Time::now()>tid){
      ROS_INFO("Did not get image. Abandonning");
      return img;
    }
    ros::spinOnce();
    if(cv_ptr){
      img = cv::Mat(cv_ptr->image.rows, cv_ptr->image.cols, CV_8UC3);
      img = cv_ptr->image;
      cv_ptr.reset();
      return img;
    }
  }
  return img;
}

cv::Mat getDepthImage() {
  cv::Mat imgD = cv::Mat(100,100, CV_16UC1); //Depth
  ros::Time tidD = ros::Time::now()+ros::Duration(5.0);
  while (ros::ok()) {
    if(ros::Time::now()>tidD){
      ROS_INFO("Did not get image. Abandonning");
      return imgD;
    }
    ros::spinOnce();
    if(cv_ptrDep){
      imgD = cv::Mat(cv_ptrDep->image.rows, cv_ptrDep->image.cols, CV_16UC1);
      imgD = cv_ptrDep->image;
      cv_ptrDep.reset();
      return imgD;
    }
  }
  return imgD;
}

void moveBaseCallback(const geometry_msgs::PoseWithCovarianceStamped msg){
  tfScalar yaw,pitch,roll;
  curXpose = msg.pose.pose.position.x;
  curYpose = msg.pose.pose.position.y;

  tf::Quaternion quat;
  quat = tf::Quaternion(msg.pose.pose.orientation.x,msg.pose.pose.orientation.y,msg.pose.pose.orientation.z,msg.pose.pose.orientation.w);
  tf::Matrix3x3 mat(quat);
  mat.getEulerYPR(yaw,pitch,roll);
  ROS_INFO("yaw: %f, pitch: %f, roll: %f", yaw,pitch,roll);
  curAngle = yaw;
}

void hogDetect(){

     Mat currentImg;
     Mat img;
     /// Set up the pedestrian detector --> let us take the default one
     HOGDescriptor hog;
     hog.setSVMDetector(HOGDescriptor::getDefaultPeopleDetector());

          currentImg = getImage();
          while(currentImg.empty()){
            ROS_INFO("waiting...");
          }

         cv::resize(currentImg, img, cv::Size(360,640));

         ///image, 4 of rectangles, hit threshold, win stride, padding, scale, group threshold

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

         /// Show
         //namedWindow("person",WINDOW_AUTOSIZE);
         //resizeWindow("person", 600,600);
         //imshow("person", img);
         //waitKey(0);
     }


unsigned long int  sequenceID = 0;
void sendPeople(std::vector<float> mapPos, int PersonID, ros::Publisher pub){

  people_msgs::People peopleMsg;

  sequenceID++;
  peopleMsg.header.seq = sequenceID;
  peopleMsg.header.frame_id = "map";
  peopleMsg.header.stamp = ros::Time::now();

  peopleMsg.people[PersonID].name = "person1";

  peopleMsg.people[PersonID].position.x = mapPos[0];
  peopleMsg.people[PersonID].position.y = mapPos[1];
  peopleMsg.people[PersonID].position.z = 0;

  peopleMsg.people[PersonID].velocity.x = 0;
  peopleMsg.people[PersonID].velocity.y = 0;
  peopleMsg.people[PersonID].velocity.z = 0;

  peopleMsg.people[PersonID].reliability = 1;

  peopleMsg.people[PersonID].tagnames[0] = "person";
  peopleMsg.people[PersonID].tags[0] = "person";

  pub.publish(peopleMsg);

}

std::vector<float> calTransform(){

  //apply transform from camera to person, return position of person on global map

  std::vector<float> positionGlobalMap;
  positionGlobalMap.push_back(0);
  positionGlobalMap.push_back(0);
  return positionGlobalMap;
  }

int main(int argc, char** argv){

  ros::init(argc, argv, "human_detect");
  ros::NodeHandle nh1;
  ROS_INFO("Version: %i.%i", CV_MAJOR_VERSION,CV_MINOR_VERSION);
  //Subscribe to camera rgb image
  ros::Subscriber image_sub_ = nh1.subscribe("image_final", 1,imageCallback);
  //Subscribe to the mir platform current position in the map
  ros::Subscriber sub_pose = nh1.subscribe<geometry_msgs::PoseWithCovarianceStamped>("/amcl_pose", 10, moveBaseCallback);
  //People message publisher
  ros::Publisher pubCV = nh1.advertise<people_msgs::People> ("/People", 100);

  std::vector<float> mapPos;

  ros::Duration dur;
  double timeSpent;
  ros::Time startTime;


  while(ros::ok()){

    hogDetect();
    mapPos = calTransform();
    sendPeople(mapPos, 1, pubCV);
    ros::spinOnce();
    ROS_INFO("Running...");
  }

}
