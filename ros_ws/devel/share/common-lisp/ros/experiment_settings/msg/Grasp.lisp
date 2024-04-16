; Auto-generated. Do not edit!


(cl:in-package experiment_settings-msg)


;//! \htmlinclude Grasp.msg.html

(cl:defclass <Grasp> (roslisp-msg-protocol:ros-message)
  ((header
    :reader header
    :initarg :header
    :type std_msgs-msg:Header
    :initform (cl:make-instance 'std_msgs-msg:Header))
   (bestGrasp
    :reader bestGrasp
    :initarg :bestGrasp
    :type experiment_settings-msg:GraspEvoContacts
    :initform (cl:make-instance 'experiment_settings-msg:GraspEvoContacts))
   (ranking
    :reader ranking
    :initarg :ranking
    :type cl:float
    :initform 0.0)
   (bestPose
    :reader bestPose
    :initarg :bestPose
    :type experiment_settings-msg:GraspEvoPose
    :initform (cl:make-instance 'experiment_settings-msg:GraspEvoPose)))
)

(cl:defclass Grasp (<Grasp>)
  ())

(cl:defmethod cl:initialize-instance :after ((m <Grasp>) cl:&rest args)
  (cl:declare (cl:ignorable args))
  (cl:unless (cl:typep m 'Grasp)
    (roslisp-msg-protocol:msg-deprecation-warning "using old message class name experiment_settings-msg:<Grasp> is deprecated: use experiment_settings-msg:Grasp instead.")))

(cl:ensure-generic-function 'header-val :lambda-list '(m))
(cl:defmethod header-val ((m <Grasp>))
  (roslisp-msg-protocol:msg-deprecation-warning "Using old-style slot reader experiment_settings-msg:header-val is deprecated.  Use experiment_settings-msg:header instead.")
  (header m))

(cl:ensure-generic-function 'bestGrasp-val :lambda-list '(m))
(cl:defmethod bestGrasp-val ((m <Grasp>))
  (roslisp-msg-protocol:msg-deprecation-warning "Using old-style slot reader experiment_settings-msg:bestGrasp-val is deprecated.  Use experiment_settings-msg:bestGrasp instead.")
  (bestGrasp m))

(cl:ensure-generic-function 'ranking-val :lambda-list '(m))
(cl:defmethod ranking-val ((m <Grasp>))
  (roslisp-msg-protocol:msg-deprecation-warning "Using old-style slot reader experiment_settings-msg:ranking-val is deprecated.  Use experiment_settings-msg:ranking instead.")
  (ranking m))

(cl:ensure-generic-function 'bestPose-val :lambda-list '(m))
(cl:defmethod bestPose-val ((m <Grasp>))
  (roslisp-msg-protocol:msg-deprecation-warning "Using old-style slot reader experiment_settings-msg:bestPose-val is deprecated.  Use experiment_settings-msg:bestPose instead.")
  (bestPose m))
(cl:defmethod roslisp-msg-protocol:serialize ((msg <Grasp>) ostream)
  "Serializes a message object of type '<Grasp>"
  (roslisp-msg-protocol:serialize (cl:slot-value msg 'header) ostream)
  (roslisp-msg-protocol:serialize (cl:slot-value msg 'bestGrasp) ostream)
  (cl:let ((bits (roslisp-utils:encode-single-float-bits (cl:slot-value msg 'ranking))))
    (cl:write-byte (cl:ldb (cl:byte 8 0) bits) ostream)
    (cl:write-byte (cl:ldb (cl:byte 8 8) bits) ostream)
    (cl:write-byte (cl:ldb (cl:byte 8 16) bits) ostream)
    (cl:write-byte (cl:ldb (cl:byte 8 24) bits) ostream))
  (roslisp-msg-protocol:serialize (cl:slot-value msg 'bestPose) ostream)
)
(cl:defmethod roslisp-msg-protocol:deserialize ((msg <Grasp>) istream)
  "Deserializes a message object of type '<Grasp>"
  (roslisp-msg-protocol:deserialize (cl:slot-value msg 'header) istream)
  (roslisp-msg-protocol:deserialize (cl:slot-value msg 'bestGrasp) istream)
    (cl:let ((bits 0))
      (cl:setf (cl:ldb (cl:byte 8 0) bits) (cl:read-byte istream))
      (cl:setf (cl:ldb (cl:byte 8 8) bits) (cl:read-byte istream))
      (cl:setf (cl:ldb (cl:byte 8 16) bits) (cl:read-byte istream))
      (cl:setf (cl:ldb (cl:byte 8 24) bits) (cl:read-byte istream))
    (cl:setf (cl:slot-value msg 'ranking) (roslisp-utils:decode-single-float-bits bits)))
  (roslisp-msg-protocol:deserialize (cl:slot-value msg 'bestPose) istream)
  msg
)
(cl:defmethod roslisp-msg-protocol:ros-datatype ((msg (cl:eql '<Grasp>)))
  "Returns string type for a message object of type '<Grasp>"
  "experiment_settings/Grasp")
(cl:defmethod roslisp-msg-protocol:ros-datatype ((msg (cl:eql 'Grasp)))
  "Returns string type for a message object of type 'Grasp"
  "experiment_settings/Grasp")
(cl:defmethod roslisp-msg-protocol:md5sum ((type (cl:eql '<Grasp>)))
  "Returns md5sum for a message object of type '<Grasp>"
  "07436bacd6c0331b872013638b94fe36")
(cl:defmethod roslisp-msg-protocol:md5sum ((type (cl:eql 'Grasp)))
  "Returns md5sum for a message object of type 'Grasp"
  "07436bacd6c0331b872013638b94fe36")
(cl:defmethod roslisp-msg-protocol:message-definition ((type (cl:eql '<Grasp>)))
  "Returns full string definition for message of type '<Grasp>"
  (cl:format cl:nil "Header header~%GraspEvoContacts bestGrasp~%float32 ranking~%GraspEvoPose bestPose~%~%================================================================================~%MSG: std_msgs/Header~%# Standard metadata for higher-level stamped data types.~%# This is generally used to communicate timestamped data ~%# in a particular coordinate frame.~%# ~%# sequence ID: consecutively increasing ID ~%uint32 seq~%#Two-integer timestamp that is expressed as:~%# * stamp.sec: seconds (stamp_secs) since epoch (in Python the variable is called 'secs')~%# * stamp.nsec: nanoseconds since stamp_secs (in Python the variable is called 'nsecs')~%# time-handling sugar is provided by the client library~%time stamp~%#Frame this data is associated with~%string frame_id~%~%================================================================================~%MSG: experiment_settings/GraspEvoContacts~%Header header~%sensor_msgs/PointCloud2 graspContactPoints~%~%================================================================================~%MSG: sensor_msgs/PointCloud2~%# This message holds a collection of N-dimensional points, which may~%# contain additional information such as normals, intensity, etc. The~%# point data is stored as a binary blob, its layout described by the~%# contents of the \"fields\" array.~%~%# The point cloud data may be organized 2d (image-like) or 1d~%# (unordered). Point clouds organized as 2d images may be produced by~%# camera depth sensors such as stereo or time-of-flight.~%~%# Time of sensor data acquisition, and the coordinate frame ID (for 3d~%# points).~%Header header~%~%# 2D structure of the point cloud. If the cloud is unordered, height is~%# 1 and width is the length of the point cloud.~%uint32 height~%uint32 width~%~%# Describes the channels and their layout in the binary data blob.~%PointField[] fields~%~%bool    is_bigendian # Is this data bigendian?~%uint32  point_step   # Length of a point in bytes~%uint32  row_step     # Length of a row in bytes~%uint8[] data         # Actual point data, size is (row_step*height)~%~%bool is_dense        # True if there are no invalid points~%~%================================================================================~%MSG: sensor_msgs/PointField~%# This message holds the description of one point entry in the~%# PointCloud2 message format.~%uint8 INT8    = 1~%uint8 UINT8   = 2~%uint8 INT16   = 3~%uint8 UINT16  = 4~%uint8 INT32   = 5~%uint8 UINT32  = 6~%uint8 FLOAT32 = 7~%uint8 FLOAT64 = 8~%~%string name      # Name of field~%uint32 offset    # Offset from start of point struct~%uint8  datatype  # Datatype enumeration, see above~%uint32 count     # How many elements in the field~%~%================================================================================~%MSG: experiment_settings/GraspEvoPose~%Header header~%geometry_msgs/Vector3[] graspPosePoints~%geometry_msgs/Pose midPointPose~%~%================================================================================~%MSG: geometry_msgs/Vector3~%# This represents a vector in free space. ~%# It is only meant to represent a direction. Therefore, it does not~%# make sense to apply a translation to it (e.g., when applying a ~%# generic rigid transformation to a Vector3, tf2 will only apply the~%# rotation). If you want your data to be translatable too, use the~%# geometry_msgs/Point message instead.~%~%float64 x~%float64 y~%float64 z~%================================================================================~%MSG: geometry_msgs/Pose~%# A representation of pose in free space, composed of position and orientation. ~%Point position~%Quaternion orientation~%~%================================================================================~%MSG: geometry_msgs/Point~%# This contains the position of a point in free space~%float64 x~%float64 y~%float64 z~%~%================================================================================~%MSG: geometry_msgs/Quaternion~%# This represents an orientation in free space in quaternion form.~%~%float64 x~%float64 y~%float64 z~%float64 w~%~%~%"))
(cl:defmethod roslisp-msg-protocol:message-definition ((type (cl:eql 'Grasp)))
  "Returns full string definition for message of type 'Grasp"
  (cl:format cl:nil "Header header~%GraspEvoContacts bestGrasp~%float32 ranking~%GraspEvoPose bestPose~%~%================================================================================~%MSG: std_msgs/Header~%# Standard metadata for higher-level stamped data types.~%# This is generally used to communicate timestamped data ~%# in a particular coordinate frame.~%# ~%# sequence ID: consecutively increasing ID ~%uint32 seq~%#Two-integer timestamp that is expressed as:~%# * stamp.sec: seconds (stamp_secs) since epoch (in Python the variable is called 'secs')~%# * stamp.nsec: nanoseconds since stamp_secs (in Python the variable is called 'nsecs')~%# time-handling sugar is provided by the client library~%time stamp~%#Frame this data is associated with~%string frame_id~%~%================================================================================~%MSG: experiment_settings/GraspEvoContacts~%Header header~%sensor_msgs/PointCloud2 graspContactPoints~%~%================================================================================~%MSG: sensor_msgs/PointCloud2~%# This message holds a collection of N-dimensional points, which may~%# contain additional information such as normals, intensity, etc. The~%# point data is stored as a binary blob, its layout described by the~%# contents of the \"fields\" array.~%~%# The point cloud data may be organized 2d (image-like) or 1d~%# (unordered). Point clouds organized as 2d images may be produced by~%# camera depth sensors such as stereo or time-of-flight.~%~%# Time of sensor data acquisition, and the coordinate frame ID (for 3d~%# points).~%Header header~%~%# 2D structure of the point cloud. If the cloud is unordered, height is~%# 1 and width is the length of the point cloud.~%uint32 height~%uint32 width~%~%# Describes the channels and their layout in the binary data blob.~%PointField[] fields~%~%bool    is_bigendian # Is this data bigendian?~%uint32  point_step   # Length of a point in bytes~%uint32  row_step     # Length of a row in bytes~%uint8[] data         # Actual point data, size is (row_step*height)~%~%bool is_dense        # True if there are no invalid points~%~%================================================================================~%MSG: sensor_msgs/PointField~%# This message holds the description of one point entry in the~%# PointCloud2 message format.~%uint8 INT8    = 1~%uint8 UINT8   = 2~%uint8 INT16   = 3~%uint8 UINT16  = 4~%uint8 INT32   = 5~%uint8 UINT32  = 6~%uint8 FLOAT32 = 7~%uint8 FLOAT64 = 8~%~%string name      # Name of field~%uint32 offset    # Offset from start of point struct~%uint8  datatype  # Datatype enumeration, see above~%uint32 count     # How many elements in the field~%~%================================================================================~%MSG: experiment_settings/GraspEvoPose~%Header header~%geometry_msgs/Vector3[] graspPosePoints~%geometry_msgs/Pose midPointPose~%~%================================================================================~%MSG: geometry_msgs/Vector3~%# This represents a vector in free space. ~%# It is only meant to represent a direction. Therefore, it does not~%# make sense to apply a translation to it (e.g., when applying a ~%# generic rigid transformation to a Vector3, tf2 will only apply the~%# rotation). If you want your data to be translatable too, use the~%# geometry_msgs/Point message instead.~%~%float64 x~%float64 y~%float64 z~%================================================================================~%MSG: geometry_msgs/Pose~%# A representation of pose in free space, composed of position and orientation. ~%Point position~%Quaternion orientation~%~%================================================================================~%MSG: geometry_msgs/Point~%# This contains the position of a point in free space~%float64 x~%float64 y~%float64 z~%~%================================================================================~%MSG: geometry_msgs/Quaternion~%# This represents an orientation in free space in quaternion form.~%~%float64 x~%float64 y~%float64 z~%float64 w~%~%~%"))
(cl:defmethod roslisp-msg-protocol:serialization-length ((msg <Grasp>))
  (cl:+ 0
     (roslisp-msg-protocol:serialization-length (cl:slot-value msg 'header))
     (roslisp-msg-protocol:serialization-length (cl:slot-value msg 'bestGrasp))
     4
     (roslisp-msg-protocol:serialization-length (cl:slot-value msg 'bestPose))
))
(cl:defmethod roslisp-msg-protocol:ros-message-to-list ((msg <Grasp>))
  "Converts a ROS message object to a list"
  (cl:list 'Grasp
    (cl:cons ':header (header msg))
    (cl:cons ':bestGrasp (bestGrasp msg))
    (cl:cons ':ranking (ranking msg))
    (cl:cons ':bestPose (bestPose msg))
))
