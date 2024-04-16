; Auto-generated. Do not edit!


(cl:in-package experiment_settings-msg)


;//! \htmlinclude GraspEvoContacts.msg.html

(cl:defclass <GraspEvoContacts> (roslisp-msg-protocol:ros-message)
  ((header
    :reader header
    :initarg :header
    :type std_msgs-msg:Header
    :initform (cl:make-instance 'std_msgs-msg:Header))
   (graspContactPoints
    :reader graspContactPoints
    :initarg :graspContactPoints
    :type sensor_msgs-msg:PointCloud2
    :initform (cl:make-instance 'sensor_msgs-msg:PointCloud2)))
)

(cl:defclass GraspEvoContacts (<GraspEvoContacts>)
  ())

(cl:defmethod cl:initialize-instance :after ((m <GraspEvoContacts>) cl:&rest args)
  (cl:declare (cl:ignorable args))
  (cl:unless (cl:typep m 'GraspEvoContacts)
    (roslisp-msg-protocol:msg-deprecation-warning "using old message class name experiment_settings-msg:<GraspEvoContacts> is deprecated: use experiment_settings-msg:GraspEvoContacts instead.")))

(cl:ensure-generic-function 'header-val :lambda-list '(m))
(cl:defmethod header-val ((m <GraspEvoContacts>))
  (roslisp-msg-protocol:msg-deprecation-warning "Using old-style slot reader experiment_settings-msg:header-val is deprecated.  Use experiment_settings-msg:header instead.")
  (header m))

(cl:ensure-generic-function 'graspContactPoints-val :lambda-list '(m))
(cl:defmethod graspContactPoints-val ((m <GraspEvoContacts>))
  (roslisp-msg-protocol:msg-deprecation-warning "Using old-style slot reader experiment_settings-msg:graspContactPoints-val is deprecated.  Use experiment_settings-msg:graspContactPoints instead.")
  (graspContactPoints m))
(cl:defmethod roslisp-msg-protocol:serialize ((msg <GraspEvoContacts>) ostream)
  "Serializes a message object of type '<GraspEvoContacts>"
  (roslisp-msg-protocol:serialize (cl:slot-value msg 'header) ostream)
  (roslisp-msg-protocol:serialize (cl:slot-value msg 'graspContactPoints) ostream)
)
(cl:defmethod roslisp-msg-protocol:deserialize ((msg <GraspEvoContacts>) istream)
  "Deserializes a message object of type '<GraspEvoContacts>"
  (roslisp-msg-protocol:deserialize (cl:slot-value msg 'header) istream)
  (roslisp-msg-protocol:deserialize (cl:slot-value msg 'graspContactPoints) istream)
  msg
)
(cl:defmethod roslisp-msg-protocol:ros-datatype ((msg (cl:eql '<GraspEvoContacts>)))
  "Returns string type for a message object of type '<GraspEvoContacts>"
  "experiment_settings/GraspEvoContacts")
(cl:defmethod roslisp-msg-protocol:ros-datatype ((msg (cl:eql 'GraspEvoContacts)))
  "Returns string type for a message object of type 'GraspEvoContacts"
  "experiment_settings/GraspEvoContacts")
(cl:defmethod roslisp-msg-protocol:md5sum ((type (cl:eql '<GraspEvoContacts>)))
  "Returns md5sum for a message object of type '<GraspEvoContacts>"
  "ffb9522e165b26672a4ccd953a52b38b")
(cl:defmethod roslisp-msg-protocol:md5sum ((type (cl:eql 'GraspEvoContacts)))
  "Returns md5sum for a message object of type 'GraspEvoContacts"
  "ffb9522e165b26672a4ccd953a52b38b")
(cl:defmethod roslisp-msg-protocol:message-definition ((type (cl:eql '<GraspEvoContacts>)))
  "Returns full string definition for message of type '<GraspEvoContacts>"
  (cl:format cl:nil "Header header~%sensor_msgs/PointCloud2 graspContactPoints~%~%================================================================================~%MSG: std_msgs/Header~%# Standard metadata for higher-level stamped data types.~%# This is generally used to communicate timestamped data ~%# in a particular coordinate frame.~%# ~%# sequence ID: consecutively increasing ID ~%uint32 seq~%#Two-integer timestamp that is expressed as:~%# * stamp.sec: seconds (stamp_secs) since epoch (in Python the variable is called 'secs')~%# * stamp.nsec: nanoseconds since stamp_secs (in Python the variable is called 'nsecs')~%# time-handling sugar is provided by the client library~%time stamp~%#Frame this data is associated with~%string frame_id~%~%================================================================================~%MSG: sensor_msgs/PointCloud2~%# This message holds a collection of N-dimensional points, which may~%# contain additional information such as normals, intensity, etc. The~%# point data is stored as a binary blob, its layout described by the~%# contents of the \"fields\" array.~%~%# The point cloud data may be organized 2d (image-like) or 1d~%# (unordered). Point clouds organized as 2d images may be produced by~%# camera depth sensors such as stereo or time-of-flight.~%~%# Time of sensor data acquisition, and the coordinate frame ID (for 3d~%# points).~%Header header~%~%# 2D structure of the point cloud. If the cloud is unordered, height is~%# 1 and width is the length of the point cloud.~%uint32 height~%uint32 width~%~%# Describes the channels and their layout in the binary data blob.~%PointField[] fields~%~%bool    is_bigendian # Is this data bigendian?~%uint32  point_step   # Length of a point in bytes~%uint32  row_step     # Length of a row in bytes~%uint8[] data         # Actual point data, size is (row_step*height)~%~%bool is_dense        # True if there are no invalid points~%~%================================================================================~%MSG: sensor_msgs/PointField~%# This message holds the description of one point entry in the~%# PointCloud2 message format.~%uint8 INT8    = 1~%uint8 UINT8   = 2~%uint8 INT16   = 3~%uint8 UINT16  = 4~%uint8 INT32   = 5~%uint8 UINT32  = 6~%uint8 FLOAT32 = 7~%uint8 FLOAT64 = 8~%~%string name      # Name of field~%uint32 offset    # Offset from start of point struct~%uint8  datatype  # Datatype enumeration, see above~%uint32 count     # How many elements in the field~%~%~%"))
(cl:defmethod roslisp-msg-protocol:message-definition ((type (cl:eql 'GraspEvoContacts)))
  "Returns full string definition for message of type 'GraspEvoContacts"
  (cl:format cl:nil "Header header~%sensor_msgs/PointCloud2 graspContactPoints~%~%================================================================================~%MSG: std_msgs/Header~%# Standard metadata for higher-level stamped data types.~%# This is generally used to communicate timestamped data ~%# in a particular coordinate frame.~%# ~%# sequence ID: consecutively increasing ID ~%uint32 seq~%#Two-integer timestamp that is expressed as:~%# * stamp.sec: seconds (stamp_secs) since epoch (in Python the variable is called 'secs')~%# * stamp.nsec: nanoseconds since stamp_secs (in Python the variable is called 'nsecs')~%# time-handling sugar is provided by the client library~%time stamp~%#Frame this data is associated with~%string frame_id~%~%================================================================================~%MSG: sensor_msgs/PointCloud2~%# This message holds a collection of N-dimensional points, which may~%# contain additional information such as normals, intensity, etc. The~%# point data is stored as a binary blob, its layout described by the~%# contents of the \"fields\" array.~%~%# The point cloud data may be organized 2d (image-like) or 1d~%# (unordered). Point clouds organized as 2d images may be produced by~%# camera depth sensors such as stereo or time-of-flight.~%~%# Time of sensor data acquisition, and the coordinate frame ID (for 3d~%# points).~%Header header~%~%# 2D structure of the point cloud. If the cloud is unordered, height is~%# 1 and width is the length of the point cloud.~%uint32 height~%uint32 width~%~%# Describes the channels and their layout in the binary data blob.~%PointField[] fields~%~%bool    is_bigendian # Is this data bigendian?~%uint32  point_step   # Length of a point in bytes~%uint32  row_step     # Length of a row in bytes~%uint8[] data         # Actual point data, size is (row_step*height)~%~%bool is_dense        # True if there are no invalid points~%~%================================================================================~%MSG: sensor_msgs/PointField~%# This message holds the description of one point entry in the~%# PointCloud2 message format.~%uint8 INT8    = 1~%uint8 UINT8   = 2~%uint8 INT16   = 3~%uint8 UINT16  = 4~%uint8 INT32   = 5~%uint8 UINT32  = 6~%uint8 FLOAT32 = 7~%uint8 FLOAT64 = 8~%~%string name      # Name of field~%uint32 offset    # Offset from start of point struct~%uint8  datatype  # Datatype enumeration, see above~%uint32 count     # How many elements in the field~%~%~%"))
(cl:defmethod roslisp-msg-protocol:serialization-length ((msg <GraspEvoContacts>))
  (cl:+ 0
     (roslisp-msg-protocol:serialization-length (cl:slot-value msg 'header))
     (roslisp-msg-protocol:serialization-length (cl:slot-value msg 'graspContactPoints))
))
(cl:defmethod roslisp-msg-protocol:ros-message-to-list ((msg <GraspEvoContacts>))
  "Converts a ROS message object to a list"
  (cl:list 'GraspEvoContacts
    (cl:cons ':header (header msg))
    (cl:cons ':graspContactPoints (graspContactPoints msg))
))
