
"""
Utilities
"""

from globv import symbol_label_parms
from symbol import Symbol


def create_symbol(label, loc):
    """
    Create a symbol object with its label and location (center)

    :param label: name of the symbol
    :type label: string
    :param loc: center location of the symbol
    :type loc: LOC
    :return: sym
    :rtype: Symbol
    """
    if label not in symbol_label_parms.keys():
        return None

    [rows, cols] = symbol_label_parms[label]
    sym = Symbol(label, None)
    sym.set_bbox(loc, rows, cols)
    return sym