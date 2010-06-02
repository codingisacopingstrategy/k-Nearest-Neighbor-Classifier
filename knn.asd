(defpackage :classifier.knn-system (:use :asdf :cl))
(in-package :classifier.knn-system)

(defsystem knn
  :name "kNN"
  :author "Folgert Karsdorp <fbkarsdorp@gmail.com>"
  :version "0.3"
  :maintainer "Folgert Karsdorp <fbkarsdorp@gmail.com>"
  :licence "GNU General Public Licence"
  :description "k-Nearest Neighbor classifier"
  :long-description ""
  :components
  ((:file "packages")
   (:file "classifier-utilities" :depends-on ("packages"))
   (:file "data-set" :depends-on ("classifier-utilities"))
   (:file "scores" :depends-on ("data-set"))
   (:file "prediction" :depends-on ("data-set" "classifier-utilities"))
   (:file "classifier" :depends-on ("classifier-utilities" "scores" "prediction")))
  :depends-on (:split-sequence))