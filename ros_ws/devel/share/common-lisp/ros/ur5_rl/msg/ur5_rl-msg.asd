
(cl:in-package :asdf)

(defsystem "ur5_rl-msg"
  :depends-on (:roslisp-msg-protocol :roslisp-utils :actionlib_msgs-msg
               :std_msgs-msg
)
  :components ((:file "_package")
    (:file "ManageObjectAction" :depends-on ("_package_ManageObjectAction"))
    (:file "_package_ManageObjectAction" :depends-on ("_package"))
    (:file "ManageObjectActionFeedback" :depends-on ("_package_ManageObjectActionFeedback"))
    (:file "_package_ManageObjectActionFeedback" :depends-on ("_package"))
    (:file "ManageObjectActionGoal" :depends-on ("_package_ManageObjectActionGoal"))
    (:file "_package_ManageObjectActionGoal" :depends-on ("_package"))
    (:file "ManageObjectActionResult" :depends-on ("_package_ManageObjectActionResult"))
    (:file "_package_ManageObjectActionResult" :depends-on ("_package"))
    (:file "ManageObjectFeedback" :depends-on ("_package_ManageObjectFeedback"))
    (:file "_package_ManageObjectFeedback" :depends-on ("_package"))
    (:file "ManageObjectGoal" :depends-on ("_package_ManageObjectGoal"))
    (:file "_package_ManageObjectGoal" :depends-on ("_package"))
    (:file "ManageObjectResult" :depends-on ("_package_ManageObjectResult"))
    (:file "_package_ManageObjectResult" :depends-on ("_package"))
  ))