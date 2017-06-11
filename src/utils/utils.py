from .chr_idx_mapping import PHN_LOOKUP_TABLE, PHN_MAPPING


def phn_map_func(ori_phn_number):
    return PHN_LOOKUP_TABLE.index(PHN_MAPPING[PHN_LOOKUP_TABLE[ori_phn_number]])
