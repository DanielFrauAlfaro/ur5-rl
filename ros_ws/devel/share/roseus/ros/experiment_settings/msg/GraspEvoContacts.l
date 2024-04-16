;; Auto-generated. Do not edit!


(when (boundp 'experiment_settings::GraspEvoContacts)
  (if (not (find-package "EXPERIMENT_SETTINGS"))
    (make-package "EXPERIMENT_SETTINGS"))
  (shadow 'GraspEvoContacts (find-package "EXPERIMENT_SETTINGS")))
(unless (find-package "EXPERIMENT_SETTINGS::GRASPEVOCONTACTS")
  (make-package "EXPERIMENT_SETTINGS::GRASPEVOCONTACTS"))

(in-package "ROS")
;;//! \htmlinclude GraspEvoContacts.msg.html
(if (not (find-package "SENSOR_MSGS"))
  (ros::roseus-add-msgs "sensor_msgs"))
(if (not (find-package "STD_MSGS"))
  (ros::roseus-add-msgs "std_msgs"))


(defclass experiment_settings::GraspEvoContacts
  :super ros::object
  :slots (_header _graspContactPoints ))

(defmethod experiment_settings::GraspEvoContacts
  (:init
   (&key
    ((:header __header) (instance std_msgs::Header :init))
    ((:graspContactPoints __graspContactPoints) (instance sensor_msgs::PointCloud2 :init))
    )
   (send-super :init)
   (setq _header __header)
   (setq _graspContactPoints __graspContactPoints)
   self)
  (:header
   (&rest __header)
   (if (keywordp (car __header))
       (send* _header __header)
     (progn
       (if __header (setq _header (car __header)))
       _header)))
  (:graspContactPoints
   (&rest __graspContactPoints)
   (if (keywordp (car __graspContactPoints))
       (send* _graspContactPoints __graspContactPoints)
     (progn
       (if __graspContactPoints (setq _graspContactPoints (car __graspContactPoints)))
       _graspContactPoints)))
  (:serialization-length
   ()
   (+
    ;; std_msgs/Header _header
    (send _header :serialization-length)
    ;; sensor_msgs/PointCloud2 _graspContactPoints
    (send _graspContactPoints :serialization-length)
    ))
  (:serialize
   (&optional strm)
   (let ((s (if strm strm
              (make-string-output-stream (send self :serialization-length)))))
     ;; std_msgs/Header _header
       (send _header :serialize s)
     ;; sensor_msgs/PointCloud2 _graspContactPoints
       (send _graspContactPoints :serialize s)
     ;;
     (if (null strm) (get-output-stream-string s))))
  (:deserialize
   (buf &optional (ptr- 0))
   ;; std_msgs/Header _header
     (send _header :deserialize buf ptr-) (incf ptr- (send _header :serialization-length))
   ;; sensor_msgs/PointCloud2 _graspContactPoints
     (send _graspContactPoints :deserialize buf ptr-) (incf ptr- (send _graspContactPoints :serialization-length))
   ;;
   self)
  )

(setf (get experiment_settings::GraspEvoContacts :md5sum-) "ffb9522e165b26672a4ccd953a52b38b")
(setf (get experiment_settings::GraspEvoContacts :datatype-) "experiment_settings/GraspEvoContacts")
(setf (get experiment_settings::GraspEvoContacts :definition-)
      "Header header
sensor_msgs/PointCloud2 graspContactPoints

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

")



(provide :experiment_settings/GraspEvoContacts "ffb9522e165b26672a4ccd953a52b38b")


