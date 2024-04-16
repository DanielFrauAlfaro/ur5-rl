; Auto-generated. Do not edit!


(cl:in-package canbus_interface-msg)


;//! \htmlinclude CanFrame.msg.html

(cl:defclass <CanFrame> (roslisp-msg-protocol:ros-message)
  ((timestamp
    :reader timestamp
    :initarg :timestamp
    :type cl:real
    :initform 0)
   (arbitration_id
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

(cl:defclass CanFrame (<CanFrame>)
  ())

(cl:defmethod cl:initialize-instance :after ((m <CanFrame>) cl:&rest args)
  (cl:declare (cl:ignorable args))
  (cl:unless (cl:typep m 'CanFrame)
    (roslisp-msg-protocol:msg-deprecation-warning "using old message class name canbus_interface-msg:<CanFrame> is deprecated: use canbus_interface-msg:CanFrame instead.")))

(cl:ensure-generic-function 'timestamp-val :lambda-list '(m))
(cl:defmethod timestamp-val ((m <CanFrame>))
  (roslisp-msg-protocol:msg-deprecation-warning "Using old-style slot reader canbus_interface-msg:timestamp-val is deprecated.  Use canbus_interface-msg:timestamp instead.")
  (timestamp m))

(cl:ensure-generic-function 'arbitration_id-val :lambda-list '(m))
(cl:defmethod arbitration_id-val ((m <CanFrame>))
  (roslisp-msg-protocol:msg-deprecation-warning "Using old-style slot reader canbus_interface-msg:arbitration_id-val is deprecated.  Use canbus_interface-msg:arbitration_id instead.")
  (arbitration_id m))

(cl:ensure-generic-function 'data-val :lambda-list '(m))
(cl:defmethod data-val ((m <CanFrame>))
  (roslisp-msg-protocol:msg-deprecation-warning "Using old-style slot reader canbus_interface-msg:data-val is deprecated.  Use canbus_interface-msg:data instead.")
  (data m))
(cl:defmethod roslisp-msg-protocol:serialize ((msg <CanFrame>) ostream)
  "Serializes a message object of type '<CanFrame>"
  (cl:let ((__sec (cl:floor (cl:slot-value msg 'timestamp)))
        (__nsec (cl:round (cl:* 1e9 (cl:- (cl:slot-value msg 'timestamp) (cl:floor (cl:slot-value msg 'timestamp)))))))
    (cl:write-byte (cl:ldb (cl:byte 8 0) __sec) ostream)
    (cl:write-byte (cl:ldb (cl:byte 8 8) __sec) ostream)
    (cl:write-byte (cl:ldb (cl:byte 8 16) __sec) ostream)
    (cl:write-byte (cl:ldb (cl:byte 8 24) __sec) ostream)
    (cl:write-byte (cl:ldb (cl:byte 8 0) __nsec) ostream)
    (cl:write-byte (cl:ldb (cl:byte 8 8) __nsec) ostream)
    (cl:write-byte (cl:ldb (cl:byte 8 16) __nsec) ostream)
    (cl:write-byte (cl:ldb (cl:byte 8 24) __nsec) ostream))
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
(cl:defmethod roslisp-msg-protocol:deserialize ((msg <CanFrame>) istream)
  "Deserializes a message object of type '<CanFrame>"
    (cl:let ((__sec 0) (__nsec 0))
      (cl:setf (cl:ldb (cl:byte 8 0) __sec) (cl:read-byte istream))
      (cl:setf (cl:ldb (cl:byte 8 8) __sec) (cl:read-byte istream))
      (cl:setf (cl:ldb (cl:byte 8 16) __sec) (cl:read-byte istream))
      (cl:setf (cl:ldb (cl:byte 8 24) __sec) (cl:read-byte istream))
      (cl:setf (cl:ldb (cl:byte 8 0) __nsec) (cl:read-byte istream))
      (cl:setf (cl:ldb (cl:byte 8 8) __nsec) (cl:read-byte istream))
      (cl:setf (cl:ldb (cl:byte 8 16) __nsec) (cl:read-byte istream))
      (cl:setf (cl:ldb (cl:byte 8 24) __nsec) (cl:read-byte istream))
      (cl:setf (cl:slot-value msg 'timestamp) (cl:+ (cl:coerce __sec 'cl:double-float) (cl:/ __nsec 1e9))))
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
(cl:defmethod roslisp-msg-protocol:ros-datatype ((msg (cl:eql '<CanFrame>)))
  "Returns string type for a message object of type '<CanFrame>"
  "canbus_interface/CanFrame")
(cl:defmethod roslisp-msg-protocol:ros-datatype ((msg (cl:eql 'CanFrame)))
  "Returns string type for a message object of type 'CanFrame"
  "canbus_interface/CanFrame")
(cl:defmethod roslisp-msg-protocol:md5sum ((type (cl:eql '<CanFrame>)))
  "Returns md5sum for a message object of type '<CanFrame>"
  "2c45261c8ac6bf8f2671904c70614099")
(cl:defmethod roslisp-msg-protocol:md5sum ((type (cl:eql 'CanFrame)))
  "Returns md5sum for a message object of type 'CanFrame"
  "2c45261c8ac6bf8f2671904c70614099")
(cl:defmethod roslisp-msg-protocol:message-definition ((type (cl:eql '<CanFrame>)))
  "Returns full string definition for message of type '<CanFrame>"
  (cl:format cl:nil "time timestamp~%uint32 arbitration_id~%uint8[] data~%~%~%"))
(cl:defmethod roslisp-msg-protocol:message-definition ((type (cl:eql 'CanFrame)))
  "Returns full string definition for message of type 'CanFrame"
  (cl:format cl:nil "time timestamp~%uint32 arbitration_id~%uint8[] data~%~%~%"))
(cl:defmethod roslisp-msg-protocol:serialization-length ((msg <CanFrame>))
  (cl:+ 0
     8
     4
     4 (cl:reduce #'cl:+ (cl:slot-value msg 'data) :key #'(cl:lambda (ele) (cl:declare (cl:ignorable ele)) (cl:+ 1)))
))
(cl:defmethod roslisp-msg-protocol:ros-message-to-list ((msg <CanFrame>))
  "Converts a ROS message object to a list"
  (cl:list 'CanFrame
    (cl:cons ':timestamp (timestamp msg))
    (cl:cons ':arbitration_id (arbitration_id msg))
    (cl:cons ':data (data msg))
))
