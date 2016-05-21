Vintager Homepage
=================
**Vintager** (Python) is designed to combine crowdsourcing annotation and automatic object detection
for **Optical Music Recogntion**.

We provide utilities from score file reading/conversion to symbol annotating and detection. More specifically,
**Vintager** provides API's to perform various functionalities as follows:

================  ============================================================
Type              Coverage
================  ============================================================
I/O               Read PDF and convert it to cv2.image type, stored in pdfReader.images list
Preprocessing     1. Staff detection
                  #. Page rescaling according to detected staff and ideal spatium
Symbol Annotator    Given annotation list, display associated bounding box on the image or crop and store symbol instances
Model Training    1. Pixel feature + linear Support Vector Machine
                  #. HOG feature + linear Support Vector Machine
                  #. Convolutional Neural Network
Symbol Detection  Given test image, detect target symbols
================  ============================================================

The symbols that are currently of interest to us:

=========== =========   ====    =====   ======= =============== ==============  ==========  ========
treble clef bass clef   flat    sharp   natural solid note head open note head  whole note  bar line
=========== =========   ====    =====   ======= =============== ==============  ==========  ========

The above list may be extended for a full-fledged OMR system in the future.

Annotation Format
=================
Annotation file contains a list of symbols, each having one name and its center location.
Each symbol is associated with a bounding box of a type-dependent size.
symbolAnnotator class reads annotation files, convert
the annotation into symbol objects and display them on the image.

Todo
====
* Evaluation
* The detected symbols need to be saved in a database which will later be fed into some web-based application for the users to verify. User-labeled symbols will, in return, be used by **Vintager** to re-train the symbol models. This loop will incrementally improve the performance of our detectors and create annotated
music symbol dataset at the same time.

Requirements
============
* Python 2.7
* OpenCV 3.1.0
* Scikit-learn 0.17.1
* TensorFlow 0.8

API Documentation
=================
Check the documentation `here`_.

.. _here: http://liang-chen.github.io/Vintager/API.html

LICENSE
=======
See the license `file`_.

.. _file: https://github.com/liang-chen/Vintager/blob/master/LICENSE

