; Auto-generated. Do not edit!


(cl:in-package canbus_interface-srv)


;//! \htmlinclude CanFrameSrv-request.msg.html

(cl:defclass <CanFrameSrv-request> (roslisp-msg-protocol:ros-message)
  ((arbitration_id
    :reader arbitration_id
    :initarg :arbitration_id
    :type cl:integer
    :initform 0)
   (data
    :reader data
    :initarg :data
    :type (cl:vector cl:fixnum)
   :initform (cl:make-array 0 :element-type 'cl:fixnum :initial-element 0)))
)

(cl:defclass CanFrameSrv-request (<CanFrameSrv-request>)
  ())

(cl:defmethod cl:initialize-instance :after ((m <CanFrameSrv-request>) cl:&rest args)
  (cl:declare (cl:ignorable args))
  (cl:unless (cl:typep m 'CanFrameSrv-request)
    (roslisp-msg-protocol:msg-deprecation-warning "using old message class name canbus_interface-srv:<CanFrameSrv-request> is deprecated: use canbus_interface-srv:CanFrameSrv-request instead.")))

(cl:ensure-generic-function 'arbitration_id-val :lambda-list '(m))
(cl:defmethod arbitration_id-val ((m <CanFrameSrv-request>))
  (roslisp-msg-protocol:msg-deprecation-warning "Using old-style slot reader canbus_interface-srv:arbitration_id-val is deprecated.  Use canbus_interface-srv:arbitration_id instead.")
  (arbitration_id m))

(cl:ensure-generic-function 'data-val :lambda-list '(m))
(cl:defmethod data-val ((m <CanFrameSrv-request>))
  (roslisp-msg-protocol:msg-deprecation-warning "Using old-style slot reader canbus_interface-srv:data-val is deprecated.  Use canbus_interface-srv:data instead.")
  (data m))
(cl:defmethod roslisp-msg-protocol:serialize ((msg <CanFrameSrv-request>) ostream)
  "Serializes a message object of type '<CanFrameSrv-request>"
  (cl:write-byte (cl:ldb (cl:byte 8 0) (cl:slot-value msg 'arbitration_id)) ostream)
  (cl:write-byte (cl:ldb (cl:byte 8 8) (cl:slot-value msg 'arbitration_id)) ostream)
  (cl:write-byte (cl:ldb (cl:byte 8 16) (cl:slot-value msg 'arbitration_id)) ostream)
  (cl:write-byte (cl:ldb (cl:byte 8 24) (cl:slot-value msg 'arbitration_id)) ostream)
  (cl:let ((__ros_arr_len (cl:length (cl:slot-value msg 'data))))
    (cl:write-byte (cl:ldb (cl:byte 8 0) __ros_arr_len) ostream)
    (cl:write-byte (cl:ldb (cl:byte 8 8) __ros_arr_len) ostream)
    (cl:write-byte (cl:ldb (cl:byte 8 16) __ros_arr_len) ostream)
    (cl:write-byte (cl:ldb (cl:byte 8 24) __ros_arr_len) ostream))
  (cl:map cl:nil #'(cl:lambda (ele) (cl:write-byte (cl:ldb (cl:byte 8 0) ele) ostream))
   (cl:slot-value msg 'data))
)
(cl:defmethod roslisp-msg-protocol:deserialize ((msg <CanFrameSrv-request>) istream)
  "Deserializes a message object of type '<CanFrameSrv-request>"
    (cl:setf (cl:ldb (cl:byte 8 0) (cl:slot-value msg 'arbitration_id)) (cl:read-byte istream))
    (cl:setf (cl:ldb (cl:byte 8 8) (cl:slot-value msg 'arbitration_id)) (cl:read-byte istream))
    (cl:setf (cl:ldb (cl:byte 8 16) (cl:slot-value msg 'arbitration_id)) (cl:read-byte istream))
    (cl:setf (cl:ldb (cl:byte 8 24) (cl:slot-value msg 'arbitration_id)) (cl:read-byte istream))
  (cl:let ((__ros_arr_len 0))
    (cl:setf (cl:ldb (cl:byte 8 0) __ros_arr_len) (cl:read-byte istream))
    (cl:setf (cl:ldb (cl:byte 8 8) __ros_arr_len) (cl:read-byte istream))
    (cl:setf (cl:ldb (cl:byte 8 16) __ros_arr_len) (cl:read-byte istream))
    (cl:setf (cl:ldb (cl:byte 8 24) __ros_arr_len) (cl:read-byte istream))
  (cl:setf (cl:slot-value msg 'data) (cl:make-array __ros_arr_len))
  (cl:let ((vals (cl:slot-value msg 'data)))
    (cl:dotimes (i __ros_arr_len)
    (cl:setf (cl:ldb (cl:byte 8 0) (cl:aref vals i)) (cl:read-byte istream)))))
  msg
)
(cl:defmethod roslisp-msg-protocol:ros-datatype ((msg (cl:eql '<CanFrameSrv-request>)))
  "Returns string type for a service object of type '<CanFrameSrv-request>"
  "canbus_interface/CanFrameSrvRequest")
(cl:defmethod roslisp-msg-protocol:ros-datatype ((msg (cl:eql 'CanFrameSrv-request)))
  "Returns string type for a service object of type 'CanFrameSrv-request"
  "canbus_interface/CanFrameSrvRequest")
(cl:defmethod roslisp-msg-protocol:md5sum ((type (cl:eql '<CanFrameSrv-request>)))
  "Returns md5sum for a message object of type '<CanFrameSrv-request>"
  "675dd66a5938847259a403df984d1151")
(cl:defmethod roslisp-msg-protocol:md5sum ((type (cl:eql 'CanFrameSrv-request)))
  "Returns md5sum for a message object of type 'CanFrameSrv-request"
  "675dd66a5938847259a403df984d1151")
(cl:defmethod roslisp-msg-protocol:message-definition ((type (cl:eql '<CanFrameSrv-request>)))
  "Returns full string definition for message of type '<CanFrameSrv-request>"
  (cl:format cl:nil "uint32 arbitration_id~%uint8[] data~%~%~%"))
(cl:defmethod roslisp-msg-protocol:message-definition ((type (cl:eql 'CanFrameSrv-request)))
  "Returns full string definition for message of type 'CanFrameSrv-request"
  (cl:format cl:nil "uint32 arbitration_id~%uint8[] data~%~%~%"))
(cl:defmethod roslisp-msg-protocol:serialization-length ((msg <CanFrameSrv-request>))
  (cl:+ 0
     4
     4 (cl:reduce #'cl:+ (cl:slot-value msg 'data) :key #'(cl:lambda (ele) (cl:declare (cl:ignorable ele)) (cl:+ 1)))
))
(cl:defmethod roslisp-msg-protocol:ros-message-to-list ((msg <CanFrameSrv-request>))
  "Converts a ROS message object to a list"
  (cl:list 'CanFrameSrv-request
    (cl:cons ':arbitration_id (arbitration_id msg))
    (cl:cons ':data (data msg))
))
;//! \htmlinclude CanFrameSrv-response.msg.html

(cl:defclass <CanFrameSrv-response> (roslisp-msg-protocol:ros-message)
  ((success
    :reader success
    :initarg :success
    :type cl:boolean
    :initform cl:nil))
)

(cl:defclass CanFrameSrv-response (<CanFrameSrv-response>)
  ())

(cl:defmethod cl:initialize-instance :after ((m <CanFrameSrv-response>) cl:&rest args)
  (cl:declare (cl:ignorable args))
  (cl:unless (cl:typep m 'CanFrameSrv-response)
    (roslisp-msg-protocol:msg-deprecation-warning "using old message class name canbus_interface-srv:<CanFrameSrv-response> is deprecated: use canbus_interface-srv:CanFrameSrv-response instead.")))

(cl:ensure-generic-function 'success-val :lambda-list '(m))
(cl:defmethod success-val ((m <CanFrameSrv-response>))
  (roslisp-msg-protocol:msg-deprecation-warning "Using old-style slot reader canbus_interface-srv:success-val is deprecated.  Use canbus_interface-srv:success instead.")
  (success m))
(cl:defmethod roslisp-msg-protocol:serialize ((msg <CanFrameSrv-response>) ostream)
  "Serializes a message object of type '<CanFrameSrv-response>"
  (cl:write-byte (cl:ldb (cl:byte 8 0) (cl:if (cl:slot-value msg 'success) 1 0)) ostream)
)
(cl:defmethod roslisp-msg-protocol:deserialize ((msg <CanFrameSrv-response>) istream)
  "Deserializes a message object of type '<CanFrameSrv-response>"
    (cl:setf (cl:slot-value msg 'success) (cl:not (cl:zerop (cl:read-byte istream))))
  msg
)
(cl:defmethod roslisp-msg-protocol:ros-datatype ((msg (cl:eql '<CanFrameSrv-response>)))
  "Returns string type for a service object of type '<CanFrameSrv-response>"
  "canbus_interface/CanFrameSrvResponse")
(cl:defmethod roslisp-msg-protocol:ros-datatype ((msg (cl:eql 'CanFrameSrv-response)))
  "Returns string type for a service object of type 'CanFrameSrv-response"
  "canbus_interface/CanFrameSrvResponse")
(cl:defmethod roslisp-msg-protocol:md5sum ((type (cl:eql '<CanFrameSrv-response>)))
  "Returns md5sum for a message object of type '<CanFrameSrv-response>"
  "675dd66a5938847259a403df984d1151")
(cl:defmethod roslisp-msg-protocol:md5sum ((type (cl:eql 'CanFrameSrv-response)))
  "Returns md5sum for a message object of type 'CanFrameSrv-response"
  "675dd66a5938847259a403df984d1151")
(cl:defmethod roslisp-msg-protocol:message-definition ((type (cl:eql '<CanFrameSrv-response>)))
  "Returns full string definition for message of type '<CanFrameSrv-response>"
  (cl:format cl:nil "bool success~%~%~%~%"))
(cl:defmethod roslisp-msg-protocol:message-definition ((type (cl:eql 'CanFrameSrv-response)))
  "Returns full string definition for message of type 'CanFrameSrv-response"
  (cl:format cl:nil "bool success~%~%~%~%"))
(cl:defmethod roslisp-msg-protocol:serialization-length ((msg <CanFrameSrv-response>))
  (cl:+ 0
     1
))
(cl:defmethod roslisp-msg-protocol:ros-message-to-list ((msg <CanFrameSrv-response>))
  "Converts a ROS message object to a list"
  (cl:list 'CanFrameSrv-response
    (cl:cons ':success (success msg))
))
(cl:defmethod roslisp-msg-protocol:service-request-type ((msg (cl:eql 'CanFrameSrv)))
  'CanFrameSrv-request)
(cl:defmethod roslisp-msg-protocol:service-response-type ((msg (cl:eql 'CanFrameSrv)))
  'CanFrameSrv-response)
(cl:defmethod roslisp-msg-protocol:ros-datatype ((msg (cl:eql 'CanFrameSrv)))
  "Returns string type for a service object of type '<CanFrameSrv>"
  "canbus_interface/CanFrameSrv")