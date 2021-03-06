
"""
Detect music symbols via different detectors
1, Pixel + Multiclass Linear SVM
2, HOG + Muticlass Linear SVM
3, Convolutional Neural Network
"""

import cv2
import json
import numpy as np
from sklearn.externals import joblib
from globv import *
from feature import hog, pixel_vec
from utils import create_symbol_with_upper_left_corner, get_sub_im, get_bounded_sub_im
from symbol import LOC
from cnn import load_cnn_classifier, detect_cnn, detect_cnn_proba


class DetectorOption:
    """
    Detector Option Class
    """
    def __init__(self, name):
        """
        Initialize a detector option instance

        :param name: name of the detector
        :type name: string
        """
        self._name = name
        self._name_parse = name.split('_')

    def load_extractor(self):
        if len(self._name_parse) != 2:
            return None
        if self._name_parse[0] == 'hog':
            return hog
        elif self._name_parse[0] == 'pixel':
            return pixel_vec
        else:
            return None

    def load_classifier(self):
        if len(self._name_parse) == 1: #load cnn
            return load_cnn_classifier()
        elif len(self._name_parse) == 2 and self._name_parse[1] == 'svm': #load svm
            model_dir = '../models/'
            try:
                cls = joblib.load(model_dir + self._name + '.pkl')
                return cls
            except Exception:
                print Exception
        else:
            return None

    def model_name(self):
        return self._name_parse[-1]


class SymbolDetector:
    """
    Symbol Detector Class
    """

    def __init__(self, option):
        """
        Initialize a symbol detector

        :param option: input detector option
        :type option: DetectorOption()
        """
        self._extractor = option.load_extractor()
        self._model = option.load_classifier()
        self._name = option.model_name()

    def classify_image(self, im):
        """
        return label and score
        :param im:
        :type im:
        :return:
        :rtype:
        """
        if self._name == "svm":
            feature = self._extractor(im)
            feature = feature.reshape(-1, feature.size)
            return (self._model.predict(feature), self._model.predict_proba(feature)[0])
        elif self._name == "cnn":
            return (detect_cnn(im, self._model), detect_cnn_proba(im, self._model)[0])

    def detect(self, im, label, mode):
        """
        Detect a certain symbol (given by the label) on the image

        :param im: input image
        :type im: cv2.image
        :param label: target symbol label
        :type label: string
        :param mode: interatively show the result or not
        :type mode: string
        """
        tot_rows, tot_cols = im.shape[:2]
        [rows, cols] = symbol_label_parms[label]
        detected = []
        step_size = 2
        for i in range(0, tot_rows - rows, step_size):
            for j in range(0, tot_cols - cols, step_size):
                loc = LOC(j,i)
                sym = create_symbol_with_upper_left_corner(label, loc)
                sub_im = get_sub_im(im, sym)
                if self.classify_image(sub_im)[0] == label:
                    detected.append((i,j))

        if mode == "show":
            rgb_im = cv2.cvtColor(im, cv2.COLOR_GRAY2RGB)
            font = cv2.FONT_HERSHEY_SIMPLEX
            for (i, j) in detected:
                cv2.rectangle(rgb_im, (j, i), (j + cols, i + rows), (0, 0, 255), 2)
                cv2.putText(rgb_im, label, (j, i - 10), font, 0.4, (0, 0, 255), 1)
            cv2.imwrite(label+".jpg", rgb_im)
            cv2.imshow("image", rgb_im)
            cv2.waitKey(0)
        elif mode == "noshow":
            return
        else:
            raise Exception

    def detect_all(self, im, mode):
        """
        Detect all the labels on the target image

        :param im: input image
        :type im: cv2.image
        :param mode: interactively show the results or not
        :type mode: string
        """
        tot_rows, tot_cols = im.shape[:2]
        detected = []
        step_size = 1

        if self._name == "svm":
            cls = self._model.classes_
        else:
            cls = symbol_label_list

        stream = []

        for i in xrange(20, tot_rows, step_size):
            for j in xrange(0, tot_cols, step_size):
                for label in symbol_label_parms.keys():
                    [rows, cols] = symbol_label_parms[label]
                    if i + rows >= tot_rows or j + cols >= tot_cols:
                        continue
                    loc = LOC(j, i)
                    sym = create_symbol_with_upper_left_corner(label, loc)


                    #sub_im = get_sub_im(im, sym)
                    #if self.classify_image(sub_im)[0] == label:
                    #    detected.append((self.classify_image(sub_im)[1][cls.index(label)],
                    #                     i, j, label))

                    # how to get comparable score here???
                    sub_im = get_sub_im(im, sym)
                    feature = self._extractor(sub_im)
                    feature = feature.reshape(-1, feature.size)
                    temp_prob = self._model.predict_proba(feature)[0]

                    if label == 'background' and temp_prob[0] >= 0.9:
                        #print cls
                        #print len(cls), len(temp_prob)
                        break

                    if cls[np.argmax(temp_prob)] != "background":
                        #print cls[np.argmax(temp_prob)]
                        rgb_sub_im = get_bounded_sub_im(im, sym)
                        if rgb_sub_im is None:
                             continue
                        img_file_name = 'test_' + str(i) + '_' + str(j) + '.jpg'
                        cv2.imwrite('results/' + img_file_name, rgb_sub_im)
                        temp_dict = {}
                        temp_dict["name"] = img_file_name
                        temp_dict["src"] = "test.pdf"
                        temp_dict["page"] = 1
                        temp_dict["scale"] = 1.0
                        temp_dict["algorithm"] = "hog_svm"
                        temp_dict["x"] = j
                        temp_dict["dx"] = cols
                        temp_dict["y"] = i
                        temp_dict["dy"] = rows

                        label_dict = [{"name": cls[i], "probability": temp_prob[i]} for i in xrange(len(cls))]
                        temp_dict["label"] = label_dict
                        stream.append(temp_dict)

                        print len(stream)
                        if len(stream) > 150:
                            print "here"
                            with open('results/database.json', 'w') as file:
                                json.dump(stream, file)
                            return

        print len(stream)
        with open('results/database.json', 'w') as file:
            json.dump(stream, file)
        return

        #suppressed = [(i,j,label) for (prob,i,j,label) in detected if label is not "background"]
        ##Non-Maxima Suppression
        detected.sort(key=lambda tup: tup[0], reverse=True)
        hashed = np.zeros((tot_rows, tot_cols), dtype=bool)
        suppressed = []
        for (score, i, j, label) in detected:
            if score is None or hashed[i][j]:
                continue
            suppressed.append((i,j,label))
            print i,j,score

            hashed[i][j] = True
            # if label == "background":
            #     hashed[i][j] = True
            # else:
            #     hashed[max(0, i - default_symbol_rows / 2): min(i + default_symbol_rows / 2, tot_rows - 1), max(0, j - default_symbol_cols / 2): min(j + default_symbol_cols / 2, tot_cols - 1)] = True

        if mode == "show":
            rgb_im = cv2.cvtColor(im, cv2.COLOR_GRAY2RGB)
            font = cv2.FONT_HERSHEY_SIMPLEX

            for (i, j, label) in suppressed:
                if label == "background":
                    continue
                [rows, cols] = symbol_label_parms[label]
                rows = int(rows)
                cols = int(cols)
                cv2.rectangle(rgb_im, (j, i), (j + cols, i + rows), (0, 0, 255), 2)
                cv2.putText(rgb_im, label, (j, i - 10), font, 0.4, (0, 0, 255), 1)

            cv2.imwrite('detected.jpg', rgb_im)
            cv2.imshow("image", rgb_im)
            cv2.waitKey(0)
        elif mode == "noshow":
            return
        else:
            raise Exception