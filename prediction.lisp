(in-package :classifier.knn)

(declaim (optimize (speed 3) (safety 0)))

(defparameter *voting-function* nil)

(defclass exemplar-features ()
  ((exemplar
    :initarg :exemplar
    :accessor exemplar
    :initform (error "Must supply exemplar!")
    :documentation "A particular exemplar")
   (nearest-neighbors
    :initarg :nearest-neighbors
    :accessor nearest-neighbors
    :initform (error "Supply nearest neighbors")
    :documentation "The set of nearest neighbors of an exemplar")
   (prediction
    :initarg :prediction
    :accessor prediction
    :initform (error "Couldn't resolve prediction...")
    :documentation "The prediction of the exemplar")
   (k-level
    :initarg :k-level
    :accessor k-level
    :initform 1
    :documentation "The levels of k used for the prediction of the exemplar")))

(defmethod print-object ((object exemplar-features) stream)
  (with-slots (exemplar nearest-neighbors prediction k-level) object
    (format stream "~&~@(~a~):~7t" 'exemplar)
    (loop for attr across exemplar do
    	 (format stream "~a," (aref *integer->char* attr)))
    (format stream "~&~@(~a~):~7t~a~%" 'prediction
	    (aref *integer->char* prediction))
    (print-neighbors nearest-neighbors k-level stream)))

(defun print-neighbors (neighbors k stream)
  "Function to pretty-print the neighbors of an exemplar. The neighbors
   are printed at distinct k values."
  (loop for i from 1 to k
     for k-items = (select (where :k i) neighbors)
     do (format stream "\# k=~d, ~d Neighbor(s) at distance:~1@T~,8f~%"
		i (length k-items) (getf (car k-items) :distance))
        (dolist (neighbor k-items)
	  (loop for attr across (getf neighbor :neighbor)
	     do (format stream "~a," (aref *integer->char* attr)))
	  (format stream "~%")))
  (format stream "~%"))

(defparameter *k-distance* nil) ;; moet dit een globale variable blijven?

(defun set-distance (distance k)
  "Put the found distances in an array and sort in increasing order."
  (declare (single-float distance)
	   (fixnum k)
	   (type (vector single-float *) *k-distance*))
  ; only add a new item when its not already in the list.
  (when (<= distance (the single-float
		       (aref *k-distance* (1- (length *k-distance*)))))
    (when (not (find distance *k-distance*))
      (if (= (length *k-distance*) k)
	  (progn (vector-pop *k-distance*)
		 (vector-push distance *k-distance*))
	  (vector-push distance *k-distance*))
      (sort *k-distance* #'<))))

(defun calculate-distance (test-item training-item k weight-order)
  "Return distance between test and training item using Manhattan distance
   and Gain Ratio weighting."
  (declare (type (vector single-float *) *k-distance*)
	   (type simple-array test-item training-item weight-order)
	   (type fixnum k))
  (let ((distance 0.0)
	(compare-distance (aref *k-distance* (1- (length *k-distance*)))))
    (declare (single-float distance compare-distance))
    ; loop over the attributes (columns) in decreasing gain ratio order
    (loop for attr across weight-order never (> distance compare-distance)
      ; when the distance between test and training item is larger than
      ; allowed under a level of k, stop the loop. Otherwise calculate the
      ; distance between two attributes of the test and training item.
       :unless (eql (aref (the (simple-array fixnum *) test-item) attr)
		    (aref (the (simple-array fixnum *) training-item) attr))
       :do (incf distance (the single-float
			   (* 1.0 (the single-float
				    (gethash attr (weights *training-set*))))))
      ; we place the new distance in the *k-distance* list and return the
      ; distance plus the training-item.
       :finally (set-distance distance k)
	        (return (list distance training-item)))))

(defun predict-outcome (nearest-neighbors test-item k)
  "Predict the outcome category of a record in the dataset."
  (let ((prediction-table (make-hash-table :test #'eql))
	(outcome (target-attr *training-set*)))
    ; loop over all nearest neighbors and add frequencies to prediction table.
    (dolist (neighbor-record nearest-neighbors)
      (let* ((neighbor (getf neighbor-record :neighbor))
	     (prediction (svref neighbor outcome)))
	(if (gethash prediction prediction-table)
	    (incf (gethash prediction prediction-table)
		  (funcall *voting-function* (getf neighbor-record :k)))
	    (setf (gethash prediction prediction-table)
		  (funcall *voting-function* (getf neighbor-record :k))))))
    ; push all predictions with their frequencies in list of conses
    (let* ((predictions (sort-hash-table prediction-table #'>))
           ; take the best guess (first in sorted list)
	   (first-guess (nth 0 predictions)))
      ; if there is not only one best guess
      (if (> (count (cdr first-guess)
		    (loop for (prediction . count) in predictions
		       collect count) :test #'eql)
	     1)
	  nil ; return no prediction
	  (car first-guess))))) ; else return best guess

(defun majority-voting (k) 1)

(defun inverse-linear-weighting (k)
  "Inverse linear weighting of neihgbors at different levels of k."
  (let ((d-k (aref *k-distance* (1- (length *k-distance*))))
	(d-j (aref *k-distance* (- k 1)))
	(d-1 (aref *k-distance* 0)))
    (if (= d-j d-1) ; if d-j equals k=1
	1
	(/ (- d-k d-j) (- d-k d-1)))))

(defun inverse-distance-weight (k)
  "Inverse distance weighting of neighbor at different levels of k."
  (/ 1 (+ (aref *k-distance* (- k 1)) ; the distance at level k
	  0.0001))) ; to ensure divisionability
      
(defun neighbor-set (neighbors k-distance)
  "Calculate the set of nearest neighbors for a particular test item."
  (declare (type (vector single-float *) k-distance)
	   (type (vector list *) neighbors))
  (let ((compare-distance (aref k-distance (1- (length k-distance))))
	(neighbor-database nil))
    (declare (single-float compare-distance))
    (loop for neighbor across neighbors
       while (<= (the single-float (car neighbor)) compare-distance)
       do (push (k-record (+ (position (car neighbor) k-distance) 1)
			  (car (last neighbor)) (car neighbor))
		neighbor-database))
    neighbor-database))

(defun distances (test-item &optional (k 1))
  "Return sorted distances between test item and training items"
  (declare (fixnum k))
  (setf *k-distance* (make-array k :fill-pointer 0 :element-type 'single-float))
  (vector-push 1000.0 *k-distance*)
  (let* ((weight-order (attribute-weight-order *training-set*))
	 (distances (make-array k :adjustable t :fill-pointer 0)))
    (declare (type (vector list *) distances)
	     (type simple-array weight-order))
    (dotimes (i (length (data *training-set*)))
      (let ((result (calculate-distance test-item
		     (svref (the simple-array (data *training-set*)) i)
		     k weight-order)))
	(unless (null result)
	  (vector-push-extend result distances))))
    (sort distances #'< :key #'car)))

;; distance calculation and prediction ends here

