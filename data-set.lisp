(in-package :classifier.knn)

(declaim (optimize (speed 3) (safety 0)))

(defparameter *training-set* nil "The training set")
(defparameter *test-set* nil "The test set")

(defclass data-set ()
  ((data
    :type simple-array
    :initarg :data
    :accessor data
    :initform (error "Must supply data in hash table form.")
    :documentation "The data set as a list of hash tables.")
   (target-attr
    :initarg :target-attr
    :accessor target-attr
    :initform (error "Must supply target attribute.")
    :documentation "The target attribute of the dataset")
   (attributes
    :initarg :attributes
    :accessor attributes
    :initform (error "Supply list of attributes.")
    :documentation "The list of attributes of the dataset.")
   (weights
    :initarg :weights
    :accessor weights
    :initform (error "Could not determine weight of attribute.")
    :documentation "The gain ratios of the attributes in a hashtable.")
   (attribute-weight-order
    :type simple-array
    :initarg :attribute-weight-order
    :accessor attribute-weight-order
    :documentation "Value permuation based on Gain Ratio information.")
   (categories
    :initarg :categories
    :accessor categories
    :initform (error "Must supply list containing categories.")
    :documentation "The categories in the classification task.")
   (data-counts
    :initarg :data-counts
    :accessor data-counts
    :initform (error "The dataset has no items")
    :documentation "The number of items in the data set")))

(defgeneric print-weights (object stream)
  (:documentation "Print the gain ratio for each attribute in the training set."))

(defmethod print-weights ((object data-set) stream)
  (format stream "~&~@(~a~)~20t~@(~a~)~%" 'attribute 'gain-ratio)
  (format stream "--------------------------------~%")
  (dolist (attribute (butlast (attributes object)))
    (format stream "~&~@(~a~):~20t~,9f~%"
	    attribute (gethash attribute (weights object))))
  (format stream "--------------------------------~%~%"))

(defgeneric print-data-information (object stream)
  (:documentation "Print information about the classification task: how many
    datapoints in test set and in training set; how many attributes etc."))

(defmethod print-data-information ((object data-set) stream)
  (with-slots (data-counts attributes) object
    (format stream "~{~&~@(~a~):~25t~d~%~}"
	    (list "Data points" (data-counts object)
		  "Number of attributes" (length (attributes object))
		  "Number of categories" (length (categories object)))))
  (format stream "~%"))

(defun print-info (data-set)
  "Helper function to print information about the dataset."
  (format t "~&Information training set:~%")
  (print-data-information data-set t)
  (print-weights data-set t))

(defmethod initialize-instance :after ((object data-set) &key)
  "Sort the attributes on the basis of their gain ratio in decreasing order.
   This method is called after the main instance of data-set
   has been created."
  (setf (slot-value object 'attribute-weight-order)
	(make-array (hash-table-count (weights object))
		    :initial-contents (mapcar #'(lambda (x) (car x))
		(sort-hash-table (weights object) #'>)))))

(defun value-frequencies (data attr)
  "Calculate the frequencies of the values in the dataset"
  (let ((value-frequency (make-hash-table :test #'eql)))
    (loop for record across data
	 do (if (gethash (svref record attr) value-frequency)
		(incf (gethash (svref record attr) value-frequency) 1.0)
		(setf (gethash (svref record attr) value-frequency) 1.0)))
    value-frequency))

(defun entropy (data target-attr)
  "Calculate the entropy of a category."
  (declare (type vector data))
  (let ((value-frequency (value-frequencies data target-attr))
	(data-entropy 0.0))
    (declare (single-float data-entropy))
    (loop for freq being the hash-values in value-frequency
       do (incf data-entropy
		(the single-float (* (/ (- freq) (length data))
				     (log (/ freq (length data)) 2)))))
    data-entropy))

(defun gain (data attr target-attr)
  "Calculate the information gain of an attribute and the
   split information of this attribute."
  (let ((value-frequency (value-frequencies data attr))
	(subset-entropy 0.0) (split-information 0.0))
    (declare (single-float subset-entropy split-information))
    (loop for val being the hash-key in value-frequency
       do (let ((val-prob
		 (/ (gethash val value-frequency)
		    (loop for v being the hash-values of value-frequency sum v)))
		(data-subset (make-array 1 :fill-pointer 0 :adjustable t)))
	    (declare (single-float val-prob) (type vector data-subset))
	    (loop for record across data
	       when (eql (svref record attr) val)
	       do (vector-push-extend record data-subset))
	    (incf subset-entropy
		  (the single-float (* val-prob (entropy data-subset target-attr))))
	    (incf split-information
		  (the single-float (* val-prob (log val-prob 2))))))
    (list (- (entropy data target-attr) subset-entropy) (- split-information))))

(defun gain-ratio (information-gain-and-split-information)
  "Calculate the Gain Ratio on the basis of the information gain
   and the split information."
  (let ((information-gain (nth 0 information-gain-and-split-information))
	(split-information (nth 1 information-gain-and-split-information)))
    (declare (single-float information-gain split-information))
    (if (= split-information 0.0)
	information-gain
	(/ information-gain split-information))))

(defun setup-data (matrix)
  "Sets up the data set and calculates weights."
  (let* ((data (make-array (length matrix)))
	 (attribute-keys (loop for i from 0 below (length (car matrix))
			    collect i))
	 (target-attr (car (last attribute-keys)))
	 (weight-table (make-hash-table
			:test #'eql :size (length (butlast attribute-keys)))))
    (declare (type simple-array data))
    (dotimes (line (length matrix))
      (let ((record (make-array (length attribute-keys) :element-type 'fixnum
				:initial-contents (elt matrix line))))
	(setf (svref data line) record)))
    (dolist (attribute (butlast attribute-keys))
      (setf (gethash attribute weight-table)
	    (gain-ratio (gain data attribute target-attr))))
    (make-instance 'data-set :data data :target-attr target-attr
		   :weights weight-table :attributes attribute-keys
		   :categories (get-categories matrix) :data-counts (length data))))

(defun test-item (test-string)
  "Set up the attributes for a single test-item"
  (let ((test-list (extract-values test-string))
	(test-hash (make-hash-table :test #'equal))
	(attributes (attributes *training-set*)))
    (if (not (eql (length test-list)
		  (length attributes)))
	(error "The length of the test item differs from those in the training set.")
	(dolist (pair (transpose (list attributes test-list)))
	  (setf (gethash (first pair) test-hash) (second pair))))
    test-hash))

;; data-set.lisp ends here