(in-package :classifier.knn)

(declaim (optimize (speed 3) (safety 0)))

;(proclaim '(inline data weights))

(defparameter *item-matrix* nil "The main matrix holding the data.")

(defun partition-data (data-set &key (start 0) end)
  "Help function to divide dataset into test and training part."
  (loop for idx from start below (or end (length data-set))
       collect (elt data-set idx)))

(defun divide-into-train-and-test (data-set testing-fraction)
  "Divide a dataset into a test and training set using a defined
   testing fraction"
  (let* ((shuffled (shuffle-list (cdr data-set)))
	 (size (length (cdr data-set)))
	 (train-on (floor (* size (- 1 testing-fraction)))))
    (setf *training-set*
	  (setup-data (partition-data shuffled :start 0 :end train-on)))
    (setf *test-set*
	  (setup-data (partition-data shuffled :start train-on)))))

(defun train-and-test (data-set start end k &optional (test-set nil))
  "Reserve items from START to END for test set; train on remainder."
  (when (null test-set)
    (setf *test-set*
	  (setup-data (partition-data data-set :start start :end end))))
  (setf *training-set*
	(setup-data (nconc (partition-data data-set :end start)
			   (partition-data data-set :start end))))
  (classify-testset *test-set* k))

(defun cross-validation (data-set &key (folds 10) (k 1))
  "Do k-fold cross-validation test and return their mean.
   That is, keep out 1/k of the examples for testing on each of k runs.
   shuffle the examples first."
  (declare (fixnum folds k))
  (let* ((size (length data-set))
	 (shuffled (shuffle-list data-set))
	 (results (make-array folds))
	 (accuracy 0.0))
    (declare (type (simple-array *) results)
	     (single-float accuracy) (fixnum size))
    (dotimes (i folds)
      (setf (svref results i)
       (train-and-test shuffled (floor (* i (/ size folds)))
		       (floor (* (+ i 1) (/ size folds))) k))
      (incf accuracy (the single-float
		       (car (accuracy (svref results i)
				      (length (svref results i)))))))
    (dolist (category (categories *test-set*))
      (let ((category-result (make-array folds))
	    (validation-result nil))
	(declare (type simple-array category-result))
	(dotimes (result folds)
	  (setf (svref category-result result)
		(score (car category) (svref results result))))
	(setf validation-result
	      (return-validation-results category-result category folds))
	(pprint-results validation-result)))
    (format t "Overall accuracy:~20t~,7f~%" (/ accuracy folds))))

(defun leave-one-out (data-set &key (k 1) (print-exemplars nil))
  "Leave one out cross-validation over the dataset."
  (let* ((shuffled (shuffle-list data-set))
	 (results (list)))
    (setf *training-set* (setup-data shuffled))
    (let ((temp-set (make-array (data-counts *training-set*)
				:initial-contents (data *training-set*))))
      (declare (type (simple-array hash-table) temp-set))
      (dotimes (i (length temp-set))
	(setf (data *training-set*)
	      (concatenate 'vector (partition-data temp-set :end i)
			   (partition-data temp-set :start (1+ i))))
	(push (classify (svref temp-set i) k) results))
      (collect-score-and-accuracy results *training-set* :info t
				  :print-exemplars print-exemplars))))

(defun learning-curve (data-set &key (trials 10) (sizes nil) (k 1))
  "Return the learning curve of the classifier over the dataset."
  (declare (fixnum trials k))
  (let ((size-result (list)))
    (when (null sizes)
      (setf sizes (loop for i from 2 to (- (length data-set) 12) by 2
		     collect i)))
    (dolist (size sizes)
      (declare (fixnum size))
      (format t "Test / Train: ~d / ~d~%" size (- (length data-set) size))
      (push (list size (- (length data-set) size)
		  (mean (loop for trail from 1 to trials collect
			     (car (accuracy (train-and-test
					     (shuffle-list data-set)
					     0 size k)
					    size)))))
	    size-result))
    (print-learning-curve size-result)))

(defun accumulative-learning (file test-indexes &key (k 1)
			      (periods 3) (print-exemplars nil))
  "Accumulative learning function. Given a data-set and defined test
   items within this data-set, the algorithm classifies each test item
   on the basis of all previously seen items in the data-set."
  (let ((data-set (open-file file))
	(classification-results nil)
	(time-frames (split-periods test-indexes periods)))
    (do ((i 0 (1+ i)))
	((= i (length test-indexes)))
      (let ((test-item (make-array (length (elt data-set 0))
			:initial-contents (elt data-set (elt test-indexes i)))))
	(setf *training-set*
	      (setup-data (partition-data data-set :end (elt test-indexes i))))
	(push (classify test-item k) classification-results)))
    (progn (scores-within-timeframe time-frames classification-results)
	   (collect-score-and-accuracy
	    classification-results *training-set* :counts (length test-indexes)
	    :print-exemplars print-exemplars))))

(defun split-periods (indexes periods)
  (let* ((start-ends (list 0))
	 (highest-index (car (last indexes)))
	 (steps (floor (/ highest-index periods))))
    (do ((i 0 (+ i steps))
	 (j steps (+ j steps)))
	((> j highest-index))
      (loop for k from 0 below (length indexes)
	 do (when (> (elt indexes k) j) 
	      (return (push k start-ends)))))
    (reverse start-ends)))

(defun scores-within-timeframe (time-frames results)
  (do ((i 0 (1+ i)) (j 1 (1+ j)))
      ((= j (length time-frames)))
    (let ((frame-results (subseq results (elt time-frames i)
				  (elt time-frames j))))
      (collect-score-and-accuracy
       frame-results *training-set* :counts (length frame-results)))))

(defun print-learning-curve (results)
  "Pretty-print the learning-curve results in tabular format."
  (dolist (result results)
    (format t "~&~d~3@T~d~3@T~,7f"
	    (elt result 0) (elt result 1) (elt result 2))))

(defun classify-dataset (file test-fn)
  "Open a single file from which a testing and a training set will be derived."
  (setf *training-set* nil)
  (setf *test-set* nil)
  (let ((data-set (open-file file)))
    (funcall test-fn data-set)))

(defmacro run-classification ((&key (test-fn 'with-test-set) (data-set nil)
				    (test-file nil) (voting 'majority-voting))
				    &rest function-arguments)
  (setf *voting-function* voting)
  (if (eql test-fn 'with-test-set)
      (if (null test-file)
	  (error "No test-file given")
	  `(progn (load-training-file ,data-set)
		  (load-test-file ,test-file)
		  (classify-test-on-trainingfile *test-set* ,@function-arguments)))
      `(,test-fn (open-file ,data-set) ,@function-arguments)))

(defun load-training-file (file)
  "Load the training file."
  (let ((training-file (open-file file)))
    (setf *training-set* (setup-data training-file))))

(defun load-test-file (file)
  "Load the test file."
  (let ((test-file (open-file file)))
    (setf *test-set* (setup-data test-file))))

(defun classify (test-item &optional (k 1))
  "Classify a test-item on the basis of a training set. Calculate the
   distances between the test item and the training items, define the set
   of nearest neighbors and finally predict the outcome of the test item."
  (let* ((results (distances test-item k))
	 ; define the set of nearest neighbors
    	 (nearest-neighbors (neighbor-set results *k-distance*))
	 ; predict the outcome of the test-item
	 (prediction (predict-outcome nearest-neighbors test-item k)))
    ; make instance of exemplar containing the test-item,
    ; the nearest neighbors and the predicted outcome.
    (if (null prediction)
	(classify test-item (+ k 1))
	(make-instance 'exemplar-features
		       :exemplar test-item
		       :nearest-neighbors nearest-neighbors
		       :prediction prediction
		       :k-level k))))

(defun classify-testset (test-items &optional (k 1))
  "Classify a test-set on the basis of a training set."
  (let ((results (loop for test-item across (data test-items)
		       for result = (classify test-item k)
		       unless (null result)
		       collect result)))
    results))

(defun classify-test-on-trainingfile (test-set &key (k 1) (print-exemplars nil))
  "Classify a separate test-file on the basis of a training-file."
  (collect-score-and-accuracy
   (classify-testset test-set k) test-set :info t
   :print-exemplars print-exemplars))


;; classifier.lisp ends here


