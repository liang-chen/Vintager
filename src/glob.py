
# global parameters
# better to be written in file

from math import ceil

staff_height = 30.0
space_unit = staff_height/4.0
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
default_symbol_rows = int(3*space_unit)
default_symbol_cols = int(3*space_unit)

uni_feature_len = int(ceil(40*space_unit*space_unit))

uni_size = (int(5*space_unit), int(5*space_unit))