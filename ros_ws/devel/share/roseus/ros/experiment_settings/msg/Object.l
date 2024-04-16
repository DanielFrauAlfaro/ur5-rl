;; Auto-generated. Do not edit!


(when (boundp 'experiment_settings::Object)
  (if (not (find-package "EXPERIMENT_SETTINGS"))
    (make-package "EXPERIMENT_SETTINGS"))
  (shadow 'Object (find-package "EXPERIMENT_SETTINGS")))
(unless (find-package "EXPERIMENT_SETTINGS::OBJECT")
  (make-package "EXPERIMENT_SETTINGS::OBJECT"))

(in-package "ROS")
;;//! \htmlinclude Object.msg.html
(if (not (find-package "GEOMETRY_MSGS"))
  (ros::roseus-add-msgs "geometry_msgs"))
(if (not (find-package "STD_MSGS"))
  (ros::roseus-add-msgs "std_msgs"))


(defclass experiment_settings::Object
  :super ros::object
  :slots (_header _chosenObject _pose _orientation ))

(defmethod experiment_settings::Object
  (:init
   (&key
    ((:header __header) (instance std_msgs::Header :init))
    ((:chosenObject __chosenObject) (let (r) (dotimes (i 0) (push "" r)) r))
    ((:pose __pose) ())
    ((:orientation __orientation) ())
    )
   (send-super :init)
   (setq _header __header)
   (setq _chosenObject __chosenObject)
   (setq _pose __pose)
   (setq _orientation __orientation)
   self)
  (:header
   (&rest __header)
   (if (keywordp (car __header))
       (send* _header __header)
     (progn
       (if __header (setq _header (car __header)))
       _header)))
  (:chosenObject
   (&optional __chosenObject)
   (if __chosenObject (setq _chosenObject __chosenObject)) _chosenObject)
  (:pose
   (&rest __pose)
   (if (keywordp (car __pose))
       (send* _pose __pose)
     (progn
       (if __pose (setq _pose (car __pose)))
       _pose)))
  (:orientation
   (&rest __orientation)
   (if (keywordp (car __orientation))
       (send* _orientation __orientation)
     (progn
       (if __orientation (setq _orientation (car __orientation)))
       _orientation)))
  (:serialization-length
   ()
   (+
    ;; std_msgs/Header _header
    (send _header :serialization-length)
    ;; string[] _chosenObject
    (apply #'+ (mapcar #'(lambda (x) (+ 4 (length x))) _chosenObject)) 4
    ;; geometry_msgs/Point[] _pose
    (apply #'+ (send-all _pose :serialization-length)) 4
    ;; geometry_msgs/Point[] _orientation
    (apply #'+ (send-all _orientation :serialization-length)) 4
    ))
  (:serialize
   (&optional strm)
   (let ((s (if strm strm
              (make-string-output-stream (send self :serialization-length)))))
     ;; std_msgs/Header _header
       (send _header :serialize s)
     ;; string[] _chosenObject
     (write-long (length _chosenObject) s)
     (dolist (elem _chosenObject)
       (write-long (length elem) s) (princ elem s)
       )
     ;; geometry_msgs/Point[] _pose
     (write-long (length _pose) s)
     (dolist (elem _pose)
       (send elem :serialize s)
       )
     ;; geometry_msgs/Point[] _orientation
     (write-long (length _orientation) s)
     (dolist (elem _orientation)
       (send elem :serialize s)
       )
     ;;
     (if (null strm) (get-output-stream-string s))))
  (:deserialize
   (buf &optional (ptr- 0))
   ;; std_msgs/Header _header
     (send _header :deserialize buf ptr-) (incf ptr- (send _header :serialization-length))
   ;; string[] _chosenObject
   (let (n)
     (setq n (sys::peek buf ptr- :integer)) (incf ptr- 4)
     (setq _chosenObject (make-list n))
     (dotimes (i n)
     (let (n) (setq n (sys::peek buf ptr- :integer)) (incf ptr- 4) (setf (elt _chosenObject i) (subseq buf ptr- (+ ptr- n))) (incf ptr- n))
     ))
   ;; geometry_msgs/Point[] _pose
   (let (n)
     (setq n (sys::peek buf ptr- :integer)) (incf ptr- 4)
     (setq _pose (let (r) (dotimes (i n) (push (instance geometry_msgs::Point :init) r)) r))
     (dolist (elem- _pose)
     (send elem- :deserialize buf ptr-) (incf ptr- (send elem- :serialization-length))
     ))
   ;; geometry_msgs/Point[] _orientation
   (let (n)
     (setq n (sys::peek buf ptr- :integer)) (incf ptr- 4)
     (setq _orientation (let (r) (dotimes (i n) (push (instance geometry_msgs::Point :init) r)) r))
     (dolist (elem- _orientation)
     (send elem- :deserialize buf ptr-) (incf ptr- (send elem- :serialization-length))
     ))
   ;;
   self)
  )

(setf (get experiment_settings::Object :md5sum-) "6189364a18d4f68fba2b3e2a8b22bfc9")
(setf (get experiment_settings::Object :datatype-) "experiment_settings/Object")
(setf (get experiment_settings::Object :definition-)
      "Header header
string[] chosenObject
geometry_msgs/Point[] pose
geometry_msgs/Point[] orientation

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
MSG: geometry_msgs/Point
# This contains the position of a point in free space
float64 x
float64 y
float64 z

")



(provide :experiment_settings/Object "6189364a18d4f68fba2b3e2a8b22bfc9")


