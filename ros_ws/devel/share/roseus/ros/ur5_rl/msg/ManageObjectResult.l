;; Auto-generated. Do not edit!


(when (boundp 'ur5_rl::ManageObjectResult)
  (if (not (find-package "UR5_RL"))
    (make-package "UR5_RL"))
  (shadow 'ManageObjectResult (find-package "UR5_RL")))
(unless (find-package "UR5_RL::MANAGEOBJECTRESULT")
  (make-package "UR5_RL::MANAGEOBJECTRESULT"))

(in-package "ROS")
;;//! \htmlinclude ManageObjectResult.msg.html


(defclass ur5_rl::ManageObjectResult
  :super ros::object
  :slots (_res ))

(defmethod ur5_rl::ManageObjectResult
  (:init
   (&key
    ((:res __res) 0)
    )
   (send-super :init)
   (setq _res (round __res))
   self)
  (:res
   (&optional __res)
   (if __res (setq _res __res)) _res)
  (:serialization-length
   ()
   (+
    ;; int32 _res
    4
    ))
  (:serialize
   (&optional strm)
   (let ((s (if strm strm
              (make-string-output-stream (send self :serialization-length)))))
     ;; int32 _res
       (write-long _res s)
     ;;
     (if (null strm) (get-output-stream-string s))))
  (:deserialize
   (buf &optional (ptr- 0))
   ;; int32 _res
     (setq _res (sys::peek buf ptr- :integer)) (incf ptr- 4)
   ;;
   self)
  )

(setf (get ur5_rl::ManageObjectResult :md5sum-) "ca16cfbd5443ad97f6cc7ffd6bb67292")
(setf (get ur5_rl::ManageObjectResult :datatype-) "ur5_rl/ManageObjectResult")
(setf (get ur5_rl::ManageObjectResult :definition-)
      "# ====== DO NOT MODIFY! AUTOGENERATED FROM AN ACTION DEFINITION ======
#result definition
int32 res

")



(provide :ur5_rl/ManageObjectResult "ca16cfbd5443ad97f6cc7ffd6bb67292")

