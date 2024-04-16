;; Auto-generated. Do not edit!


(when (boundp 'canbus_interface::CanFrame)
  (if (not (find-package "CANBUS_INTERFACE"))
    (make-package "CANBUS_INTERFACE"))
  (shadow 'CanFrame (find-package "CANBUS_INTERFACE")))
(unless (find-package "CANBUS_INTERFACE::CANFRAME")
  (make-package "CANBUS_INTERFACE::CANFRAME"))

(in-package "ROS")
;;//! \htmlinclude CanFrame.msg.html


(defclass canbus_interface::CanFrame
  :super ros::object
  :slots (_timestamp _arbitration_id _data ))

(defmethod canbus_interface::CanFrame
  (:init
   (&key
    ((:timestamp __timestamp) (instance ros::time :init))
    ((:arbitration_id __arbitration_id) 0)
    ((:data __data) (make-array 0 :initial-element 0 :element-type :char))
    )
   (send-super :init)
   (setq _timestamp __timestamp)
   (setq _arbitration_id (round __arbitration_id))
   (setq _data __data)
   self)
  (:timestamp
   (&optional __timestamp)
   (if __timestamp (setq _timestamp __timestamp)) _timestamp)
  (:arbitration_id
   (&optional __arbitration_id)
   (if __arbitration_id (setq _arbitration_id __arbitration_id)) _arbitration_id)
  (:data
   (&optional __data)
   (if __data (setq _data __data)) _data)
  (:serialization-length
   ()
   (+
    ;; time _timestamp
    8
    ;; uint32 _arbitration_id
    4
    ;; uint8[] _data
    (* 1    (length _data)) 4
    ))
  (:serialize
   (&optional strm)
   (let ((s (if strm strm
              (make-string-output-stream (send self :serialization-length)))))
     ;; time _timestamp
       (write-long (send _timestamp :sec) s) (write-long (send _timestamp :nsec) s)
     ;; uint32 _arbitration_id
       (write-long _arbitration_id s)
     ;; uint8[] _data
     (write-long (length _data) s)
     (princ _data s)
     ;;
     (if (null strm) (get-output-stream-string s))))
  (:deserialize
   (buf &optional (ptr- 0))
   ;; time _timestamp
     (send _timestamp :sec (sys::peek buf ptr- :integer)) (incf ptr- 4)  (send _timestamp :nsec (sys::peek buf ptr- :integer)) (incf ptr- 4)
   ;; uint32 _arbitration_id
     (setq _arbitration_id (sys::peek buf ptr- :integer)) (incf ptr- 4)
   ;; uint8[] _data
   (let ((n (sys::peek buf ptr- :integer))) (incf ptr- 4)
     (setq _data (make-array n :element-type :char))
     (replace _data buf :start2 ptr-) (incf ptr- n))
   ;;
   self)
  )

(setf (get canbus_interface::CanFrame :md5sum-) "2c45261c8ac6bf8f2671904c70614099")
(setf (get canbus_interface::CanFrame :datatype-) "canbus_interface/CanFrame")
(setf (get canbus_interface::CanFrame :definition-)
      "time timestamp
uint32 arbitration_id
uint8[] data

")



(provide :canbus_interface/CanFrame "2c45261c8ac6bf8f2671904c70614099")


