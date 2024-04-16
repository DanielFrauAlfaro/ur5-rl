
(cl:in-package :asdf)

(defsystem "canbus_interface-msg"
  :depends-on (:roslisp-msg-protocol :roslisp-utils )
  :components ((:file "_package")
    (:file "CanFrame" :depends-on ("_package_CanFrame"))
    (:file "_package_CanFrame" :depends-on ("_package"))
  ))