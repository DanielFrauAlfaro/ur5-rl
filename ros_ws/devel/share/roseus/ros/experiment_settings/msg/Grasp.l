;; Auto-generated. Do not edit!


(when (boundp 'experiment_settings::Grasp)
  (if (not (find-package "EXPERIMENT_SETTINGS"))
    (make-package "EXPERIMENT_SETTINGS"))
  (shadow 'Grasp (find-package "EXPERIMENT_SETTINGS")))
(unless (find-package "EXPERIMENT_SETTINGS::GRASP")
  (make-package "EXPERIMENT_SETTINGS::GRASP"))

(in-package "ROS")
;;//! \htmlinclude Grasp.msg.html
(if (not (find-package "STD_MSGS"))
  (ros::roseus-add-msgs "std_msgs"))


(defclass experiment_settings::Grasp
  :super ros::object
  :slots (_header _bestGrasp _ranking _bestPose ))

(defmethod experiment_settings::Grasp
  (:init
   (&key
    ((:header __header) (instance std_msgs::Header :init))
    ((:bestGrasp __bestGrasp) (instance experiment_settings::GraspEvoContacts :init))
    ((:ranking __ranking) 0.0)
    ((:bestPose __bestPose) (instance experiment_settings::GraspEvoPose :init))
    )
   (send-super :init)
   (setq _header __header)
   (setq _bestGrasp __bestGrasp)
   (setq _ranking (float __ranking))
   (setq _bestPose __bestPose)
   self)
  (:header
   (&rest __header)
   (if (keywordp (car __header))
       (send* _header __header)
     (progn
       (if __header (setq _header (car __header)))
       _header)))
  (:bestGrasp
   (&rest __bestGrasp)
   (if (keywordp (car __bestGrasp))
       (send* _bestGrasp __bestGrasp)
     (progn
       (if __bestGrasp (setq _bestGrasp (car __bestGrasp)))
       _bestGrasp)))
  (:ranking
   (&optional __ranking)
   (if __ranking (setq _ranking __ranking)) _ranking)
  (:bestPose
   (&rest __bestPose)
   (if (keywordp (car __bestPose))
       (send* _bestPose __bestPose)
     (progn
       (if __bestPose (setq _bestPose (car __bestPose)))
       _bestPose)))
  (:serialization-length
   ()
   (+
    ;; std_msgs/Header _header
    (send _header :serialization-length)
    ;; experiment_settings/GraspEvoContacts _bestGrasp
    (send _bestGrasp :serialization-length)
    ;; float32 _ranking
    4
    ;; experiment_settings/GraspEvoPose _bestPose
    (send _bestPose :serialization-length)
    ))
  (:serialize
   (&optional strm)
   (let ((s (if strm strm
              (make-string-output-stream (send self :serialization-length)))))
     ;; std_msgs/Header _header
       (send _header :serialize s)
     ;; experiment_settings/GraspEvoContacts _bestGrasp
       (send _bestGrasp :serialize s)
     ;; float32 _ranking
       (sys::poke _ranking (send s :buffer) (send s :count) :float) (incf (stream-count s) 4)
     ;; experiment_settings/GraspEvoPose _bestPose
       (send _bestPose :serialize s)
     ;;
     (if (null strm) (get-output-stream-string s))))
  (:deserialize
   (buf &optional (ptr- 0))
   ;; std_msgs/Header _header
     (send _header :deserialize buf ptr-) (incf ptr- (send _header :serialization-length))
   ;; experiment_settings/GraspEvoContacts _bestGrasp
     (send _bestGrasp :deserialize buf ptr-) (incf ptr- (send _bestGrasp :serialization-length))
   ;; float32 _ranking
     (setq _ranking (sys::peek buf ptr- :float)) (incf ptr- 4)
   ;; experiment_settings/GraspEvoPose _bestPose
     (send _bestPose :deserialize buf ptr-) (incf ptr- (send _bestPose :serialization-length))
   ;;
   self)
  )

(setf (get experiment_settings::Grasp :md5sum-) "07436bacd6c0331b872013638b94fe36")
(setf (get experiment_settings::Grasp :datatype-) "experiment_settings/Grasp")
(setf (get experiment_settings::Grasp :definition-)
      "Header header
GraspEvoContacts bestGrasp
float32 ranking
GraspEvoPose bestPose

================================================================================
MSG: std_msgs/Header
# Standard metadata for higher-level stamped data types.
# This is generally used to communicate timestamped data 
# in a particular coordinate frame.
# 
# sequence ID: consecutively increasing ID 
uint32 seq
#Two-integer timestamp that is expressed as:
# * stamp.sec: seconds (stamp_secs) since epoch (in Python the variable is called 'secs')
# * stamp.nsec: nanoseconds since stamp_secs (in Python the variable is called 'nsecs')
# time-handling sugar is provided by the client library
time stamp
#Frame this data is associated with
string frame_id

================================================================================
MSG: experiment_settings/GraspEvoContacts
Header header
sensor_msgs/PointCloud2 graspContactPoints

================================================================================
MSG: sensor_msgs/PointCloud2
# This message holds a collection of N-dimensional points, which may
# contain additional information such as normals, intensity, etc. The
# point data is stored as a binary blob, its layout described by the
# contents of the \"fields\" array.

# The point cloud data may be organized 2d (image-like) or 1d
# (unordered). Point clouds organized as 2d images may be produced by
# camera depth sensors such as stereo or time-of-flight.

# Time of sensor data acquisition, and the coordinate frame ID (for 3d
# points).
Header header

# 2D structure of the point cloud. If the cloud is unordered, height is
# 1 and width is the length of the point cloud.
uint32 height
uint32 width

# Describes the channels and their layout in the binary data blob.
PointField[] fields

bool    is_bigendian # Is this data bigendian?
uint32  point_step   # Length of a point in bytes
uint32  row_step     # Length of a row in bytes
uint8[] data         # Actual point data, size is (row_step*height)

bool is_dense        # True if there are no invalid points

================================================================================
MSG: sensor_msgs/PointField
# This message holds the description of one point entry in the
# PointCloud2 message format.
uint8 INT8    = 1
uint8 UINT8   = 2
uint8 INT16   = 3
uint8 UINT16  = 4
uint8 INT32   = 5
uint8 UINT32  = 6
uint8 FLOAT32 = 7
uint8 FLOAT64 = 8

string name      # Name of field
uint32 offset    # Offset from start of point struct
uint8  datatype  # Datatype enumeration, see above
uint32 count     # How many elements in the field

================================================================================
MSG: experiment_settings/GraspEvoPose
Header header
geometry_msgs/Vector3[] graspPosePoints
geometry_msgs/Pose midPointPose

================================================================================
MSG: geometry_msgs/Vector3
# This represents a vector in free space. 
# It is only meant to represent a direction. Therefore, it does not
# make sense to apply a translation to it (e.g., when applying a 
# generic rigid transformation to a Vector3, tf2 will only apply the
# rotation). If you want your data to be translatable too, use the
# geometry_msgs/Point message instead.

float64 x
float64 y
float64 z
================================================================================
MSG: geometry_msgs/Pose
# A representation of pose in free space, composed of position and orientation. 
Point position
Quaternion orientation

================================================================================
MSG: geometry_msgs/Point
# This contains the position of a point in free space
float64 x
float64 y
float64 z

================================================================================
MSG: geometry_msgs/Quaternion
# This represents an orientation in free space in quaternion form.

float64 x
float64 y
float64 z
float64 w

")



(provide :experiment_settings/Grasp "07436bacd6c0331b872013638b94fe36")


