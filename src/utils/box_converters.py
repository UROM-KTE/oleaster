import tensorflow

"""Bounding boxes can be represented in multiple ways, the formats used in this project:
    - coordinates of corners: [x_min, y_min, x_max, y_max]
    - storing the coordinates of center and the box dimensions: [x_center, y_center, width, height]
"""


def swap_xy(boxes):
    """Swaps order the of x and y coordinates of the boxes.

    Arguments:
      boxes: A tensor with shape `(num_boxes, 4)` representing bounding boxes.

    Returns:
      swapped box with shape same as that of box.
    """
    return tensorflow.stack([boxes[:, 1], boxes[:, 0], boxes[:, 3], boxes[:, 2]], axis=-1)


def convert_to_xy_width_height(boxes):
    """Changes the box format to center, width and height.

    Arguments:
      boxes: A tensor of rank 2 or higher with a shape of `(..., num_boxes, 4)`
        representing bounding boxes where each box is of the format
        `[xmin, ymin, xmax, ymax]`.

    Returns:
      converted box with shape same as that of box.
    """
    return tensorflow.concat(
        [(boxes[..., :2] + boxes[..., 2:]) / 2.0, boxes[..., 2:] - boxes[..., :2]], axis=-1)


def convert_to_corners(boxes):
    """Changes the box format to corner coordinates

    Arguments:
        boxes: A tensor of rank 2 or higher with a shape of `(..., num_boxes, 4)`
            representing bounding box where each box is of the format
            `[x, y, width, height]`.
    Returns:
        converted box with shape same as that of box.
    """
    return tensorflow.concat(
        [boxes[..., :2] - boxes[..., 2:] / 2.0, boxes[..., :2] + boxes[..., 2:] / 2.0], axis=-1)
