
(cl:in-package :asdf)

(defsystem "canbus_interface-srv"
  :depends-on (:roslisp-msg-protocol :roslisp-utils )
  :components ((:file "_package")
    (:file "CanFrameSrv" :depends-on ("_package_CanFrameSrv"))
    (:file "_package_CanFrameSrv" :depends-on ("_package"))
  ))