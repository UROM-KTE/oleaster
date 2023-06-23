import tensorflow

import box_converters


def compute_intersection_over_union(first_box, second_box):
    first_boxes_corners = box_converters.convert_to_corners(first_box)
    second_boxes_corners = box_converters.convert_to_corners(second_box)
