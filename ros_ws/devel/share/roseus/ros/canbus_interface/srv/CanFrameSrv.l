;; Auto-generated. Do not edit!


(when (boundp 'canbus_interface::CanFrameSrv)
  (if (not (find-package "CANBUS_INTERFACE"))
    (make-package "CANBUS_INTERFACE"))
  (shadow 'CanFrameSrv (find-package "CANBUS_INTERFACE")))
(unless (find-package "CANBUS_INTERFACE::CANFRAMESRV")
  (make-package "CANBUS_INTERFACE::CANFRAMESRV"))
(unless (find-package "CANBUS_INTERFACE::CANFRAMESRVREQUEST")
  (make-package "CANBUS_INTERFACE::CANFRAMESRVREQUEST"))
(unless (find-package "CANBUS_INTERFACE::CANFRAMESRVRESPONSE")
  (make-package "CANBUS_INTERFACE::CANFRAMESRVRESPONSE"))

(in-package "ROS")





(defclass canbus_interface::CanFrameSrvRequest
  :super ros::object
  :slots (_arbitration_id _data ))

(defmethod canbus_interface::CanFrameSrvRequest
  (:init
   (&key
    ((:arbitration_id __arbitration_id) 0)
    ((:data __data) (make-array 0 :initial-element 0 :element-type :char))
    )
   (send-super :init)
   (setq _arbitration_id (round __arbitration_id))
   (setq _data __data)
   self)
  (:arbitration_id
   (&optional __arbitration_id)
   (if __arbitration_id (setq _arbitration_id __arbitration_id)) _arbitration_id)
  (:data
   (&optional __data)
   (if __data (setq _data __data)) _data)
  (:serialization-length
   ()
   (+
    ;; uint32 _arbitration_id
    4
    ;; uint8[] _data
    (* 1    (length _data)) 4
    ))
  (:serialize
   (&optional strm)
   (let ((s (if strm strm
              (make-string-output-stream (send self :serialization-length)))))
     ;; uint32 _arbitration_id
       (write-long _arbitration_id s)
     ;; uint8[] _data
     (write-long (length _data) s)
     (princ _data s)
     ;;
     (if (null strm) (get-output-stream-string s))))
  (:deserialize
   (buf &optional (ptr- 0))
   ;; uint32 _arbitration_id
     (setq _arbitration_id (sys::peek buf ptr- :integer)) (incf ptr- 4)
   ;; uint8[] _data
   (let ((n (sys::peek buf ptr- :integer))) (incf ptr- 4)
     (setq _data (make-array n :element-type :char))
     (replace _data buf :start2 ptr-) (incf ptr- n))
   ;;
   self)
  )

(defclass canbus_interface::CanFrameSrvResponse
  :super ros::object
  :slots (_success ))

(defmethod canbus_interface::CanFrameSrvResponse
  (:init
   (&key
    ((:success __success) nil)
    )
   (send-super :init)
   (setq _success __success)
   self)
  (:success
   (&optional (__success :null))
   (if (not (eq __success :null)) (setq _success __success)) _success)
  (:serialization-length
   ()
   (+
    ;; bool _success
    1
    ))
  (:serialize
   (&optional strm)
   (let ((s (if strm strm
              (make-string-output-stream (send self :serialization-length)))))
     ;; bool _success
       (if _success (write-byte -1 s) (write-byte 0 s))
     ;;
     (if (null strm) (get-output-stream-string s))))
  (:deserialize
   (buf &optional (ptr- 0))
   ;; bool _success
     (setq _success (not (= 0 (sys::peek buf ptr- :char)))) (incf ptr- 1)
   ;;
   self)
  )

(defclass canbus_interface::CanFrameSrv
  :super ros::object
  :slots ())

(setf (get canbus_interface::CanFrameSrv :md5sum-) "675dd66a5938847259a403df984d1151")
(setf (get canbus_interface::CanFrameSrv :datatype-) "canbus_interface/CanFrameSrv")
(setf (get canbus_interface::CanFrameSrv :request) canbus_interface::CanFrameSrvRequest)
(setf (get canbus_interface::CanFrameSrv :response) canbus_interface::CanFrameSrvResponse)

(defmethod canbus_interface::CanFrameSrvRequest
  (:response () (instance canbus_interface::CanFrameSrvResponse :init)))

(setf (get canbus_interface::CanFrameSrvRequest :md5sum-) "675dd66a5938847259a403df984d1151")
(setf (get canbus_interface::CanFrameSrvRequest :datatype-) "canbus_interface/CanFrameSrvRequest")
(setf (get canbus_interface::CanFrameSrvRequest :definition-)
      "uint32 arbitration_id
uint8[] data
---
bool success

")

(setf (get canbus_interface::CanFrameSrvResponse :md5sum-) "675dd66a5938847259a403df984d1151")
(setf (get canbus_interface::CanFrameSrvResponse :datatype-) "canbus_interface/CanFrameSrvResponse")
(setf (get canbus_interface::CanFrameSrvResponse :definition-)
      "uint32 arbitration_id
uint8[] data
---
bool success

")



(provide :canbus_interface/CanFrameSrv "675dd66a5938847259a403df984d1151")


