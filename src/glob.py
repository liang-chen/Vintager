
# shared parameters
# better to be written in file

from math import ceil

staff_height = 30.0
space_unit = staff_height/4.0
symbol_label_parms = {
    "bar_line":[space_unit, 5*space_unit],
    "treble_clef": [10*space_unit, 4*space_unit],
    "bass_clef": [10*space_unit, 4*space_unit],
    "alto_clef": [10*space_unit, 4*space_unit],
    "flat": [3*space_unit, 3*space_unit],
    "sharp": [3*space_unit, 3*space_unit],
    "natural": [3*space_unit, 3*space_unit],
    "solid_note_head": [3*space_unit, 3*space_unit],
    "open_note_head": [3*space_unit, 3*space_unit],
    "whole_note_head": [3*space_unit, 3*space_unit],
    "background": [3*space_unit, 3*space_unit]
}
default_symbol_rows = int(3*space_unit)
default_symbol_cols = int(3*space_unit)

uni_feature_len = int(ceil(40*space_unit*space_unit))

uni_size = (int(5*space_unit), int(5*space_unit))