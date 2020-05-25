Before the package is launched, the realsense node must be started. This is done via ./launch_camera.sh script. Within this script, a user
can also adjust the resolution and FPS of the individual streams. 

After launching the realsense node, a warning may appear saying "Frames didn't arrived within 5 seconds". This does not seem to have any effect on the performance of the package.

Finally, the package is launched via the command rosrun image_remove_planes image_remove_planes. If the package is launched successfully, you should see in your terminal "ImageCallback START ImageCallback SUCCESS". The program will automatically loop until the user ends it. 