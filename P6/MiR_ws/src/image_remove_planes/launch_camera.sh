#!/bin/bash
source devel/setup.bash
roslaunch realsense2_camera rs_rgbd.launch depth_height:=720 depth_width:=1280 depth_fps:=6 color_height:=720 color_width:=1280 color_fps:=6



