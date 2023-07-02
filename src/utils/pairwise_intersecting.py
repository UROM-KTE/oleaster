import numpy
import tensorflow
from matplotlib import pyplot

import box_converters


def compute_intersection_over_union(first_boxes, second_boxes):
    """Computes pairwise IOU matrix for given two sets of boxes

        Arguments:
          first_boxes: A tensor with shape `(N, 4)` representing bounding boxes
            where each box is of the format `[x, y, width, height]`.
          second_boxes: A tensor with shape `(M, 4)` representing bounding boxes
            where each box is of the format `[x, y, width, height]`.

        Returns:
          pairwise IOU matrix with shape `(N, M)`, where the value at ith row
            jth column holds the IOU between ith box and jth box from
            boxes1 and boxes2 respectively.
    """
    first_boxes_corners = box_converters.convert_to_corners(first_boxes)
    second_boxes_corners = box_converters.convert_to_corners(second_boxes)
    left_up = tensorflow.maximum(first_boxes_corners[:, None, :2], second_boxes_corners[second_boxes_corners[:, :2]])
    right_down = tensorflow.minimum(first_boxes_corners[:, None, :2], second_boxes_corners[second_boxes_corners[:, :2]])
    intersection = tensorflow.maximum(0.0, right_down - left_up)
    intersection_area = intersection[:, :, 0] * intersection[:, :, 1]
    first_boxes_area = first_boxes[:, 2] * first_boxes[:, 3]
    second_boxes_area = second_boxes[:, 2] * second_boxes[:, 3]
    union_area = tensorflow.maximum(
        first_boxes_area[:, None] + second_boxes_area - intersection_area, 1e-8
    )
    return tensorflow.clip_by_value(intersection_area / union_area, 0.0, 1.0)


def visualize_detections(
        image, boxes, classes, scores, figure_size=(7, 7), linewidth=1, color=[0, 0, 1]
):
    image = numpy.array(image, dtype=numpy.uint8)
    pyplot.figure(figsize=figure_size)
    pyplot.axis("off")
    pyplot.imshow(image)
    ax = pyplot.gca()
    for box, _cls, score in zip(boxes, classes, scores):
        text = '{}: {:.2f}'.format(_cls, score)
        x1, y1, x2, y2 = box
        width, height = x2 - x1, y2 - y1
        patch = pyplot.Rectangle(
            (x1, y1), width, height, fill=False, edgecolor=color, linewidth=linewidth
        )
        ax.add_patch(patch)
        ax.text(
            x1,
            y1,
            text,
            bbox={'facecolor': color, 'alpha': 0.4},
            clip_box=ax.clipbox,
            clip_on=True
        )
    pyplot.show()
    return ax

