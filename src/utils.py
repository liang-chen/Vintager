
# utilities

from glob import symbol_label_parms
from symbol import Symbol, BBox


def create_symbol(label, loc):
    if label not in symbol_label_parms.keys():
        return None

    [rows, cols] = symbol_label_parms[label]
    bbox = BBox(loc, rows, cols)
    return Symbol(label, bbox)