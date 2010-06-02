(in-package :classifier.knn)

(declaim (optimize (speed 3) (safety 0)))

(define-condition zero-by-zero-score (error)
  ((argument-1 :initarg argument-1 :reader argument-1)
   (argument-1 :initarg argument-2 :reader argument-2)))

(defun accuracy (results size)
  "Return the overall accuracy of a classification."
  (declare (fixnum size))
  (let ((correct 0)
	(target (target-attr *training-set*)))
    (declare (fixnum correct target))
    (dolist (result results)
      (when (eql (svref (exemplar result) target)
		 (prediction result))
	(incf correct (the fixnum 1))))
    (list (float (/ correct size)) (list correct size))))

(defun precision (true-positives false-positives)
  "Return the precision measure: tp / (tp + fp)."
  (declare (fixnum true-positives false-positives))
  (if (or (/= true-positives 0) (/= false-positives 0))
      (float (/ true-positives (+ true-positives false-positives)))
      (error 'zero-by-zero-score
	     :argument-1 true-positives :argument-2 false-positives)))

(defun recall (true-positives false-negatives)
  "Return the recall measure: tp / (tp + fp)."
  (declare (fixnum true-positives false-negatives))
  (if (or (/= true-positives 0) (/= false-negatives 0))
      (float (/ true-positives (+ true-positives false-negatives)))
      (error 'zero-by-zero-score
	     :argument-1 true-positives :argument-2 false-negatives)))

(defun F-score (precision recall)
  "Return the F-score, i.e. the harmonic mean of the precision
   and the recall."
  (declare (single-float precision recall))
  (if (or (/= precision 0.0) (/= recall 0.0))
      (the single-float (* 2.0  (/ (the single-float (* precision recall))
				   (the single-float (+ precision recall)))))
      (error 'zero-by-zero-score :argument-1 precision :argument-2 recall)))

(defun type-I-and-II-errors (category results)
  "Calculate the type I and type II errors in the classification task:
   true-positives, false-positives, true-negatives and false-negatives."
  (let ((true-positives 0) (false-positives 0)
	(true-negatives 0) (false-negatives 0))
    (declare (fixnum true-positives false-positives
		     true-negatives false-negatives))
    (dolist (result results)
      (let ((outcome-result (svref (exemplar result) (target-attr *training-set*)))
	    (prediction-result (prediction result)))
	(declare (fixnum category prediction-result outcome-result))
	(cond
	  ((eql category outcome-result) ; if condition:
	   (if (eql outcome-result prediction-result)	; and positive result
	       (incf true-positives) ; we have a true-positive
	       (incf false-negatives))) ; else, we have a false-negative
	  ((eql category prediction-result) ; if not condition & positive result
	   (incf false-positives)) ; we have a false-positive
	  (t (incf true-negatives))))) ; else, true-negative
    (list :true-positives true-positives :false-positives false-positives
	  :true-negatives true-negatives :false-negatives false-negatives)))

(defun score (category results)
  "Calculate scores of the classification task."
  (let* ((total-count 0) (precision 0.0) (recall 0.0) (F-score 0.0)
	 (type-errors (type-I-and-II-errors category results))
	 (true-positives (getf type-errors :true-positives))
	 (true-negatives (getf type-errors :true-negatives))
	 (false-negatives (getf type-errors :false-negatives))
	 (false-positives (getf type-errors :false-positives)))
    (declare (fixnum total-count true-positives true-negatives
		     false-negatives false-positives))
    (setf total-count (the fixnum (+ true-positives false-negatives)))
    (setf precision (handler-case (precision true-positives false-positives)
		      (zero-by-zero-score () 0.0)))
    (setf recall (handler-case (recall true-positives false-negatives)
		   (zero-by-zero-score () 0.0)))
    (setf F-score (handler-case (F-score precision recall)
		    (zero-by-zero-score () 0.0)))
    (list (cons 'category (aref *integer->char* category))
	  (cons 'count total-count)
	  (cons 'true-positives true-positives)
	  (cons 'false-positives false-positives)
	  (cons 'true-negatives true-negatives)
	  (cons 'false-negatives false-negatives)
	  (cons 'precision precision)
	  (cons 'recall recall)
	  (cons 'F-score F-score))))

(defun return-validation-results (category-result category folds)
  "Helper function for CROSS-VALIDATION. Return conses of scores."
  (declare (fixnum folds))
  (loop for result across category-result
     summing (assoc-cdr 'true-positives  result) into true-positives
     summing (assoc-cdr 'false-negatives result) into false-negatives
     summing (assoc-cdr 'true-negatives  result) into true-negatives
     summing (assoc-cdr 'false-positives result) into false-positives
     summing (assoc-cdr 'precision       result) into precision
     summing (assoc-cdr 'recall          result) into recall
     summing (assoc-cdr 'F-score         result) into F-score
     summing (assoc-cdr 'count           result) into total-count
     finally (return
	       (list (cons 'category (aref *integer->char* (car category)))
		     (cons 'count total-count)
		     (cons 'true-positives true-positives)
		     (cons 'false-positives false-positives)
		     (cons 'true-negatives true-negatives)
		     (cons 'false-negatives false-negatives)
		     (cons 'precision (the single-float (/ precision folds)))
		     (cons 'recall (the single-float (/ recall folds)))
		     (cons 'F-score (the single-float (/ F-score folds)))))))

(defun print-scores (results data-set)
  "Pretty print the scores of the classification in tabulated format. "
  (print-line 80)
  (format t "~& ~,6T~{~a ~,7T~} "
	  '(counts tp fp tn fn precision recall F-score))
  (print-line 80)
  (dolist (category (categories data-set))
    (let ((result (score (car category) results)))
      (format t "~&~a:~,6T" (cdr (assoc 'category result)))
      (loop for (label . counts) in result
	 unless (equal label 'category)
	 do (format t "~:[~d~;~,7f~]~,7T"
		    (typep counts 'single-float) counts))))
  (print-line 80))

(defun collect-score-and-accuracy (results data-set &key (counts nil)
				   (info nil) (print-exemplars nil))
  "Collect the scores and the accuracy of the classification task."
  (when (null counts)
    (setf counts (data-counts data-set)))
  (unless (null print-exemplars)
    (print-exemplars-and-NN results print-exemplars))
  (unless (null info)
    (print-info data-set))
  (print-scores results data-set)
  (format t "~&~@(~a~):~20t~{~,7f ~}~%"
 	    "Overall accuracy" (accuracy results counts)))

(defun pprint-results (results)
  (loop for (label . counts) in results
       do (if (typep counts 'integer)
	      (format t "~&~@(~a~):~20t~d~%" label counts)
	      (format t "~&~@(~a~):~20t~,7f~%" label counts)))
  (format t "~%"))

;; FIXME: INBOUWEN
(defun pprint-results-2 (results)
  (loop for (label . counts) in results
       do (format t "~&~@(~a~):~20t~:[~d~;~,7f~]~%"
		  label (typep counts 'single-float) counts)
  (format t "~%")))

;; scores.lisp ends here