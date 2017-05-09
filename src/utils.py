PHN_LOOKUP_TABLE = ['aa', 'ae', 'ah', 'ao', 'aw', 'ax', 'ax-h', 'axr', 'ay', 'b', 'bcl', 'ch', 'd', 'dcl', 'dh', 'dx',
                    'eh', 'el', 'em', 'en', 'eng', 'epi', 'er', 'ey', 'f', 'g', 'gcl', 'h#', 'hh', 'hv', 'ih', 'ix',
                    'iy', 'jh', 'k', 'kcl', 'l', 'm', 'n', 'ng', 'nx', 'ow', 'oy', 'p', 'pau', 'pcl', 'q', 'r', 's',
                    'sh', 't', 'tcl', 'th', 'uh', 'uw', 'ux', 'v', 'w', 'y', 'z', 'zh']

CHA_LOOKUP_TABLE = [' ', 'a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j', 'k', 'l', 'm', 'n', 'o', 'p', 'q', 'r', 's',
                    't', 'u', 'v', 'w', 'x', 'y', 'z', "'", ]


def label_indices_to_characters(labels, label_type='phn'):
    if label_type == 'phn':
        lookup = PHN_LOOKUP_TABLE
        chars = ' '.join([lookup[l] for l in labels])
    elif label_type == 'cha':
        lookup = CHA_LOOKUP_TABLE
        chars = ''.join([lookup[l] for l in labels])
    else:
        raise ValueError('Label type {!r} is not supported.'.format(label_type))

    return chars
