Vintager Homepage
=================
**Vintager** (Python) is designed to combine crowdsourcing annotation and automatic object detection
for **Optical Music Recogntion**.

We provide utilities from score file reading/conversion to symbol annotating and detection. More specifically,
**Vintager** contains API's to perform the following functionalities:

================  ============================================================
List Type         Examples (syntax in the `text source <cheatsheet.txt>`_)
================  ============================================================
I/O               Read PDF and convert to cv2.image type, stored in pdfReader.images list
Preprocessing     1. items use any variation of "1.", "A)", and "(i)"
                  #. also auto-enumerated
Definition list   Term is flush-left : optional classifier
                      Definition is indented, no blank line between
Field list        :field name: field body
Option list       -o  at least 2 spaces between option & description
================  ============================================================

utilities to transform printed music scores in pdf format to
gray-scale JPEG image and rescale it to a constant spatium (staff height = 30).



detect various types of music symbols and display
their associated bounding boxes. The detected symbols will also be saved into a database which will then be fed
into web-based application for the users to verify. User-labeled symbols will, in return, be used by **Vintager** to
re-train the symbol models. This loop will incrementally improve the performance of our detectors and create annotated
music symbol dataset at the same time.

Symbol Classes of interest: bar line, solid note head, open note head, whole note, flat, sharp, natural, treble clef,
alto clef, bass clef.

Symbol Detectors:
1, pixel features + linear svm
2, HOG features + linear svm
3, Convolutional Neural Network

Annotation format:
Each symbol is associated with a bounding box of a constant size.
Annotation file contains a list of symbols, each having one name and its center location.
symbolAnnotator class reads annotation files, convert
the annotation into symbol objects and display them on the image.

Todo: evaluation, database

Requirements
============
Python2.7

OpenCV

TensorFlow

Scikit-learn

API Documentation
=================
`link`_

.. _link: http://liang-chen.github.io/Vintager
