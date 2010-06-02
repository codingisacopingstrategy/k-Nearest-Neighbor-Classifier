(in-package :classifier.knn)

(declaim (optimize (speed 3) (safety 0)))

(defun transpose (lst)
  "Zip function."
  (apply #'mapcar #'list lst))

(defun assoc-cdr (item alist)
  "Return last element of association list."
  (cdr (assoc item alist)))

(defun sort-hash-table (ht sort-fn &optional &key (by :values))
  "Sort a hash table on its keys or values"
  (let ((sorted-entries nil))
    (maphash #'(lambda (k v) (push (cons k v) sorted-entries)) ht)
    (let ((sort-key #'car))
      (when (equal by :values)
	(setf sort-key #'cdr))
      (sort sorted-entries sort-fn :key sort-key))))

(defun nshuffle-list (lst)
  "Shuffle a vector in place."
  (loop for idx downfrom (1- (length lst)) to 1
       for other = (random (1+ idx))
       do (unless (= idx other)
	    (rotatef (elt lst idx) (elt lst other))))
  lst)

(defun shuffle-list (lst)
  "Return a shuffled copy of lst"
  (nshuffle-list (copy-seq lst)))

(defun pprint-hash-table (hash-table)
  "Print the values of a hash-table."
  (loop for value being the hash-value in hash-table
       collect value))

(defmacro matrix-element (i j matrix)
  "Return element of matrix at position (i,j)"
  `(nth (1- ,j) (nth (1- ,i) ,matrix)))

(defun matrix-rows (matrix)
  "Return number of rows in the matrix"
  (length matrix))

(defun matrix-row (n matrix)
  "Return a specific row in the matrix"
  (if (or (> n (matrix-rows matrix))
	  (< n 1))
      (error "Invalid row."))
  (list (copy-list (nth (1- n) matrix))))
	   
(defun matrix-columns (matrix)
  "Return number of columns in the matrix"
  (length (first matrix)))

(defun matrix-column (n matrix)
  "Return the nth column of a matrix (i.e. list of lists)"
  (if (or (> n (matrix-columns matrix))
	  (< n 1))
      (error "Invalid column."))
  (mapcar #'(lambda (r) (list (nth (1- n) r))) matrix))

(defun matrix-minor (row column matrix)
  "Return a matrix without a particular row or column."
  (let ((result (copy-tree matrix)))
    (if (eql row 1)
	(pop result)
	(setf (cdr (nthcdr (- row 2) result)) (nthcdr row result)))
    (dotimes (current-row-index (length result))
      (let ((current-row (nth current-row-index result)))
	(if (eql column 1)
	    (pop (nth current-row-index result))
	    (setf (cdr (nthcdr (- column 2) current-row))
		  (nthcdr column current-row)))))
    result))

(defun extract-values (exemplar &optional (sep #\COMMA))
  "Split sequence on the basis of preset separator"
  (string->integers (split-sequence:SPLIT-SEQUENCE sep exemplar)))

(defun open-file (file)
  (with-open-file (in file :external-format :iso-8859-1)
    (loop for line = (read-line in nil)
	 while line collect (extract-values line))))

(defun print-exemplars-and-NN (exemplars stream)
  "Write the results from the classification to a file."
  (if (symbolp stream)
      (dolist (exemplar exemplars)
	(print-object exemplar stream))
      (with-open-file (out stream
			   :direction :output
			   :if-exists :supersede
			   :if-does-not-exist :create
			   :external-format :iso-8859-1)
	(dolist (exemplar exemplars)
	  (print-object exemplar out)))))

(defun get-attributes (data)
  "The list of attribute names in the data set."
  (car (matrix-row 1 data)))

(defun category-label (exemplar)
  "The category label of an exemplar"
  (last exemplar))

(defun exemplar-values (exemplar)
  "The values of an exemplar"
  (butlast exemplar))

(defun number-of-categories (data-set)
  "How many categories are there in the data set?"
  (length (delete-duplicates
	   (matrix-column (matrix-columns data-set) data-set) :test #'equal)))

(defun get-categories (data-set)
  "What are the target categories?"
  (delete-duplicates
   (matrix-column (matrix-columns data-set) data-set) :test #'equal))

; from Peter Norvig's PAIP
(defun memo (fn name key test)
  "Return a memo-function of fn."
  (let ((table (make-hash-table :test test)))
    (setf (get name 'memo) table)
    #'(lambda (&rest args)
	(let ((k (funcall key args)))
	  (multiple-value-bind (val found-p)
	      (gethash k table)
	    (if found-p
		val
		(setf (gethash k table) (apply fn args))))))))

(defun memoize (fn-name &key (key #'first) (test #'eql))
  "Replace fn-name's global definition with a memoized version."
  (setf (symbol-function fn-name)
	(memo (symbol-function fn-name) fn-name key test)))

(defun clear-memoize (fn-name)
  "Clear the hash table from a memo function."
  (let ((table (get fn-name 'memo)))
    (when table (clrhash table))))

(defmacro defun-memo (fn args &body body)
  "Define a memoized function."
  `(memoize (defun ,fn ,args . ,body)))

;; functions for association lists quering

(defun make-comparisons-expr (field value)
  `(equal (getf item ,field) ,value))

(defun make-comparisons-list (fields)
  (loop while fields
       collecting (make-comparisons-expr (pop fields) (pop fields))))

(defmacro where (&rest clauses)
  `#'(lambda (item) (and ,@(make-comparisons-list clauses))))

(defun select (selector-fn database)
  (remove-if-not selector-fn database))

(defun k-record (k neighbor distance)
  "Add a nearest neighbor to the database with its k value,
   the item itself and its distance to the test item"
  (list :k k :neighbor neighbor :distance distance))

(defun sum (numbers &optional (key #'identity))
  "Add up all the numbers; if KEY is given, apply it to each number first."
  (if (null numbers)
      0
      (+ (funcall key (first numbers)) (sum (rest numbers) key))))

(defun mean (numbers)
  "Numerical average (mean) of a list of numbers."
  (/ (sum numbers) (length numbers)))

(defun hash-print (h &optional (stream t)) 
  "Prints a hash table line by line."
  (maphash #'(lambda (key val) (format stream "~&~A:~10T ~A" key val)) h)
  h)

(defparameter *integer->char* (make-array 0 :adjustable t :fill-pointer 0))

(defun char->integer (char)
  "Convert a character into an integer and save the character
   on the index equal to the integer in the array."
  (when (equal (position char *integer->char* :test #'equal) nil)
    (vector-push-extend char *integer->char*))
  (position char *integer->char* :test #'equal))

(defun string->integers (string)
  "Convert a list of strings into a list of integers"
  (let ((result-string (copy-list string)))
    (dotimes (element (length string))
      (setf (elt result-string element) (char->integer (elt string element))))
    result-string))

(defun integers->string (integer-list)
  (let ((result-string (make-array (length integer-list)
			:adjustable t :fill-pointer 0)))
    (loop for item across integer-list
       do (vector-push-extend (aref *integer->char* item) result-string))
    result-string))

(defun print-line (length)
  "print line of certain length"
  (dotimes (i length)
    (format t "-")))