
"""
Gloabel Variables
shared parameter throughout the entire program
"""

from math import ceil

#: Height of a staff. Score needs to be rescaled according to this constant.
staff_height = 30.0

#: Space unit used to measure the size of bounding box.
space_unit = staff_height/4.0

#: Dictionary that takes symbol types and their corresponding bounding box sizes [rows, cols].
symbol_label_parms = {
    "bar_line": map(lambda x: int(x*space_unit), [1,5]),
    "treble_clef": map(lambda x: int(x*space_unit), [10,4]),
    "bass_clef": map(lambda x: int(x*space_unit), [10,4]),
    "alto_clef": map(lambda x: int(x*space_unit), [10,4]),
    "flat": map(lambda x: int(x*space_unit), [3,3]),
    "sharp": map(lambda x: int(x*space_unit), [3,3]),
    "natural": map(lambda x: int(x*space_unit), [3,3]),
    "solid_note_head": map(lambda x: int(x*space_unit), [3,3]),
    "open_note_head": map(lambda x: int(x*space_unit), [3,3]),
    "whole_note_head": map(lambda x: int(x*space_unit), [3,3]),
    "background": map(lambda x: int(x*space_unit), [3,3])
}

#: keep the order of appereace in symbol dictionary
symbol_label_list = ["bar_line", "treble_clef", "bass_clef", "alto_clef","flat", "sharp", "natural", "solid_note_head", "open_note_head", "whole_note_head", "background"]

#: Use this number of rows if the symbol type if unknown.
default_symbol_rows = int(3*space_unit)

#: Use this number of columns if the symbol type if unknown.
default_symbol_cols = int(3*space_unit)

#:Unified feature vector length.
uni_feature_len = int(ceil(40*space_unit*space_unit))

#: Rescale sub-image to this unified size for feature extraction or classification.
#uni_size = (int(5*space_unit), int(5*space_unit))
uni_size = (64, 64)

#
uni_image_pixels = uni_size[0]*uni_size[1]

#
cnn_model_path = "../models/cnn_model"
