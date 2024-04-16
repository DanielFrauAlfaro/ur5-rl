; Auto-generated. Do not edit!


(cl:in-package experiment_settings-msg)


;//! \htmlinclude GraspEvoPose.msg.html

(cl:defclass <GraspEvoPose> (roslisp-msg-protocol:ros-message)
  ((header
    :reader header
    :initarg :header
    :type std_msgs-msg:Header
    :initform (cl:make-instance 'std_msgs-msg:Header))
   (graspPosePoints
    :reader graspPosePoints
    :initarg :graspPosePoints
    :type (cl:vector geometry_msgs-msg:Vector3)
   :initform (cl:make-array 0 :element-type 'geometry_msgs-msg:Vector3 :initial-element (cl:make-instance 'geometry_msgs-msg:Vector3)))
   (midPointPose
    :reader midPointPose
    :initarg :midPointPose
    :type geometry_msgs-msg:Pose
    :initform (cl:make-instance 'geometry_msgs-msg:Pose)))
)

(cl:defclass GraspEvoPose (<GraspEvoPose>)
  ())

(cl:defmethod cl:initialize-instance :after ((m <GraspEvoPose>) cl:&rest args)
  (cl:declare (cl:ignorable args))
  (cl:unless (cl:typep m 'GraspEvoPose)
    (roslisp-msg-protocol:msg-deprecation-warning "using old message class name experiment_settings-msg:<GraspEvoPose> is deprecated: use experiment_settings-msg:GraspEvoPose instead.")))

(cl:ensure-generic-function 'header-val :lambda-list '(m))
(cl:defmethod header-val ((m <GraspEvoPose>))
  (roslisp-msg-protocol:msg-deprecation-warning "Using old-style slot reader experiment_settings-msg:header-val is deprecated.  Use experiment_settings-msg:header instead.")
  (header m))

(cl:ensure-generic-function 'graspPosePoints-val :lambda-list '(m))
(cl:defmethod graspPosePoints-val ((m <GraspEvoPose>))
  (roslisp-msg-protocol:msg-deprecation-warning "Using old-style slot reader experiment_settings-msg:graspPosePoints-val is deprecated.  Use experiment_settings-msg:graspPosePoints instead.")
  (graspPosePoints m))

(cl:ensure-generic-function 'midPointPose-val :lambda-list '(m))
(cl:defmethod midPointPose-val ((m <GraspEvoPose>))
  (roslisp-msg-protocol:msg-deprecation-warning "Using old-style slot reader experiment_settings-msg:midPointPose-val is deprecated.  Use experiment_settings-msg:midPointPose instead.")
  (midPointPose m))
(cl:defmethod roslisp-msg-protocol:serialize ((msg <GraspEvoPose>) ostream)
  "Serializes a message object of type '<GraspEvoPose>"
  (roslisp-msg-protocol:serialize (cl:slot-value msg 'header) ostream)
  (cl:let ((__ros_arr_len (cl:length (cl:slot-value msg 'graspPosePoints))))
    (cl:write-byte (cl:ldb (cl:byte 8 0) __ros_arr_len) ostream)
    (cl:write-byte (cl:ldb (cl:byte 8 8) __ros_arr_len) ostream)
    (cl:write-byte (cl:ldb (cl:byte 8 16) __ros_arr_len) ostream)
    (cl:write-byte (cl:ldb (cl:byte 8 24) __ros_arr_len) ostream))
  (cl:map cl:nil #'(cl:lambda (ele) (roslisp-msg-protocol:serialize ele ostream))
   (cl:slot-value msg 'graspPosePoints))
  (roslisp-msg-protocol:serialize (cl:slot-value msg 'midPointPose) ostream)
)
(cl:defmethod roslisp-msg-protocol:deserialize ((msg <GraspEvoPose>) istream)
  "Deserializes a message object of type '<GraspEvoPose>"
  (roslisp-msg-protocol:deserialize (cl:slot-value msg 'header) istream)
  (cl:let ((__ros_arr_len 0))
    (cl:setf (cl:ldb (cl:byte 8 0) __ros_arr_len) (cl:read-byte istream))
    (cl:setf (cl:ldb (cl:byte 8 8) __ros_arr_len) (cl:read-byte istream))
    (cl:setf (cl:ldb (cl:byte 8 16) __ros_arr_len) (cl:read-byte istream))
    (cl:setf (cl:ldb (cl:byte 8 24) __ros_arr_len) (cl:read-byte istream))
  (cl:setf (cl:slot-value msg 'graspPosePoints) (cl:make-array __ros_arr_len))
  (cl:let ((vals (cl:slot-value msg 'graspPosePoints)))
    (cl:dotimes (i __ros_arr_len)
    (cl:setf (cl:aref vals i) (cl:make-instance 'geometry_msgs-msg:Vector3))
  (roslisp-msg-protocol:deserialize (cl:aref vals i) istream))))
  (roslisp-msg-protocol:deserialize (cl:slot-value msg 'midPointPose) istream)
  msg
)
(cl:defmethod roslisp-msg-protocol:ros-datatype ((msg (cl:eql '<GraspEvoPose>)))
  "Returns string type for a message object of type '<GraspEvoPose>"
  "experiment_settings/GraspEvoPose")
(cl:defmethod roslisp-msg-protocol:ros-datatype ((msg (cl:eql 'GraspEvoPose)))
  "Returns string type for a message object of type 'GraspEvoPose"
  "experiment_settings/GraspEvoPose")
(cl:defmethod roslisp-msg-protocol:md5sum ((type (cl:eql '<GraspEvoPose>)))
  "Returns md5sum for a message object of type '<GraspEvoPose>"
  "edfddb404e31fac0fe15cbe5e0286026")
(cl:defmethod roslisp-msg-protocol:md5sum ((type (cl:eql 'GraspEvoPose)))
  "Returns md5sum for a message object of type 'GraspEvoPose"
  "edfddb404e31fac0fe15cbe5e0286026")
(cl:defmethod roslisp-msg-protocol:message-definition ((type (cl:eql '<GraspEvoPose>)))
  "Returns full string definition for message of type '<GraspEvoPose>"
  (cl:format cl:nil "Header header~%geometry_msgs/Vector3[] graspPosePoints~%geometry_msgs/Pose midPointPose~%~%================================================================================~%MSG: std_msgs/Header~%# Standard metadata for higher-level stamped data types.~%# This is generally used to communicate timestamped data ~%# in a particular coordinate frame.~%# ~%# sequence ID: consecutively increasing ID ~%uint32 seq~%#Two-integer timestamp that is expressed as:~%# * stamp.sec: seconds (stamp_secs) since epoch (in Python the variable is called 'secs')~%# * stamp.nsec: nanoseconds since stamp_secs (in Python the variable is called 'nsecs')~%# time-handling sugar is provided by the client library~%time stamp~%#Frame this data is associated with~%string frame_id~%~%================================================================================~%MSG: geometry_msgs/Vector3~%# This represents a vector in free space. ~%# It is only meant to represent a direction. Therefore, it does not~%# make sense to apply a translation to it (e.g., when applying a ~%# generic rigid transformation to a Vector3, tf2 will only apply the~%# rotation). If you want your data to be translatable too, use the~%# geometry_msgs/Point message instead.~%~%float64 x~%float64 y~%float64 z~%================================================================================~%MSG: geometry_msgs/Pose~%# A representation of pose in free space, composed of position and orientation. ~%Point position~%Quaternion orientation~%~%================================================================================~%MSG: geometry_msgs/Point~%# This contains the position of a point in free space~%float64 x~%float64 y~%float64 z~%~%================================================================================~%MSG: geometry_msgs/Quaternion~%# This represents an orientation in free space in quaternion form.~%~%float64 x~%float64 y~%float64 z~%float64 w~%~%~%"))
(cl:defmethod roslisp-msg-protocol:message-definition ((type (cl:eql 'GraspEvoPose)))
  "Returns full string definition for message of type 'GraspEvoPose"
  (cl:format cl:nil "Header header~%geometry_msgs/Vector3[] graspPosePoints~%geometry_msgs/Pose midPointPose~%~%================================================================================~%MSG: std_msgs/Header~%# Standard metadata for higher-level stamped data types.~%# This is generally used to communicate timestamped data ~%# in a particular coordinate frame.~%# ~%# sequence ID: consecutively increasing ID ~%uint32 seq~%#Two-integer timestamp that is expressed as:~%# * stamp.sec: seconds (stamp_secs) since epoch (in Python the variable is called 'secs')~%# * stamp.nsec: nanoseconds since stamp_secs (in Python the variable is called 'nsecs')~%# time-handling sugar is provided by the client library~%time stamp~%#Frame this data is associated with~%string frame_id~%~%================================================================================~%MSG: geometry_msgs/Vector3~%# This represents a vector in free space. ~%# It is only meant to represent a direction. Therefore, it does not~%# make sense to apply a translation to it (e.g., when applying a ~%# generic rigid transformation to a Vector3, tf2 will only apply the~%# rotation). If you want your data to be translatable too, use the~%# geometry_msgs/Point message instead.~%~%float64 x~%float64 y~%float64 z~%================================================================================~%MSG: geometry_msgs/Pose~%# A representation of pose in free space, composed of position and orientation. ~%Point position~%Quaternion orientation~%~%================================================================================~%MSG: geometry_msgs/Point~%# This contains the position of a point in free space~%float64 x~%float64 y~%float64 z~%~%================================================================================~%MSG: geometry_msgs/Quaternion~%# This represents an orientation in free space in quaternion form.~%~%float64 x~%float64 y~%float64 z~%float64 w~%~%~%"))
(cl:defmethod roslisp-msg-protocol:serialization-length ((msg <GraspEvoPose>))
  (cl:+ 0
     (roslisp-msg-protocol:serialization-length (cl:slot-value msg 'header))
     4 (cl:reduce #'cl:+ (cl:slot-value msg 'graspPosePoints) :key #'(cl:lambda (ele) (cl:declare (cl:ignorable ele)) (cl:+ (roslisp-msg-protocol:serialization-length ele))))
     (roslisp-msg-protocol:serialization-length (cl:slot-value msg 'midPointPose))
))
(cl:defmethod roslisp-msg-protocol:ros-message-to-list ((msg <GraspEvoPose>))
  "Converts a ROS message object to a list"
  (cl:list 'GraspEvoPose
    (cl:cons ':header (header msg))
    (cl:cons ':graspPosePoints (graspPosePoints msg))
    (cl:cons ':midPointPose (midPointPose msg))
))
