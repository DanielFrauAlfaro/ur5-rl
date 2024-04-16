;; Auto-generated. Do not edit!


(when (boundp 'experiment_settings::GraspEvoPose)
  (if (not (find-package "EXPERIMENT_SETTINGS"))
    (make-package "EXPERIMENT_SETTINGS"))
  (shadow 'GraspEvoPose (find-package "EXPERIMENT_SETTINGS")))
(unless (find-package "EXPERIMENT_SETTINGS::GRASPEVOPOSE")
  (make-package "EXPERIMENT_SETTINGS::GRASPEVOPOSE"))

(in-package "ROS")
;;//! \htmlinclude GraspEvoPose.msg.html
(if (not (find-package "GEOMETRY_MSGS"))
  (ros::roseus-add-msgs "geometry_msgs"))
(if (not (find-package "STD_MSGS"))
  (ros::roseus-add-msgs "std_msgs"))


(defclass experiment_settings::GraspEvoPose
  :super ros::object
  :slots (_header _graspPosePoints _midPointPose ))

(defmethod experiment_settings::GraspEvoPose
  (:init
   (&key
    ((:header __header) (instance std_msgs::Header :init))
    ((:graspPosePoints __graspPosePoints) ())
    ((:midPointPose __midPointPose) (instance geometry_msgs::Pose :init))
    )
   (send-super :init)
   (setq _header __header)
   (setq _graspPosePoints __graspPosePoints)
   (setq _midPointPose __midPointPose)
   self)
  (:header
   (&rest __header)
   (if (keywordp (car __header))
       (send* _header __header)
     (progn
       (if __header (setq _header (car __header)))
       _header)))
  (:graspPosePoints
   (&rest __graspPosePoints)
   (if (keywordp (car __graspPosePoints))
       (send* _graspPosePoints __graspPosePoints)
     (progn
       (if __graspPosePoints (setq _graspPosePoints (car __graspPosePoints)))
       _graspPosePoints)))
  (:midPointPose
   (&rest __midPointPose)
   (if (keywordp (car __midPointPose))
       (send* _midPointPose __midPointPose)
     (progn
       (if __midPointPose (setq _midPointPose (car __midPointPose)))
       _midPointPose)))
  (:serialization-length
   ()
   (+
    ;; std_msgs/Header _header
    (send _header :serialization-length)
    ;; geometry_msgs/Vector3[] _graspPosePoints
    (apply #'+ (send-all _graspPosePoints :serialization-length)) 4
    ;; geometry_msgs/Pose _midPointPose
    (send _midPointPose :serialization-length)
    ))
  (:serialize
   (&optional strm)
   (let ((s (if strm strm
              (make-string-output-stream (send self :serialization-length)))))
     ;; std_msgs/Header _header
       (send _header :serialize s)
     ;; geometry_msgs/Vector3[] _graspPosePoints
     (write-long (length _graspPosePoints) s)
     (dolist (elem _graspPosePoints)
       (send elem :serialize s)
       )
     ;; geometry_msgs/Pose _midPointPose
       (send _midPointPose :serialize s)
     ;;
     (if (null strm) (get-output-stream-string s))))
  (:deserialize
   (buf &optional (ptr- 0))
   ;; std_msgs/Header _header
     (send _header :deserialize buf ptr-) (incf ptr- (send _header :serialization-length))
   ;; geometry_msgs/Vector3[] _graspPosePoints
   (let (n)
     (setq n (sys::peek buf ptr- :integer)) (incf ptr- 4)
     (setq _graspPosePoints (let (r) (dotimes (i n) (push (instance geometry_msgs::Vector3 :init) r)) r))
     (dolist (elem- _graspPosePoints)
     (send elem- :deserialize buf ptr-) (incf ptr- (send elem- :serialization-length))
     ))
   ;; geometry_msgs/Pose _midPointPose
     (send _midPointPose :deserialize buf ptr-) (incf ptr- (send _midPointPose :serialization-length))
   ;;
   self)
  )

(setf (get experiment_settings::GraspEvoPose :md5sum-) "edfddb404e31fac0fe15cbe5e0286026")
(setf (get experiment_settings::GraspEvoPose :datatype-) "experiment_settings/GraspEvoPose")
(setf (get experiment_settings::GraspEvoPose :definition-)
      "Header header
geometry_msgs/Vector3[] graspPosePoints
geometry_msgs/Pose midPointPose

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



(provide :experiment_settings/GraspEvoPose "edfddb404e31fac0fe15cbe5e0286026")


