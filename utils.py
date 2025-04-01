def scale_bboxes(bboxes, new_size, original_size=640):
    """Scale multiple bounding boxes to a new image size.

    Args:
        bboxes (list of tuples): List of (x_min, y_min, x_max, y_max) bounding boxes.
        new_size (tuple): (new_width, new_height) to scale the bboxes to.
        original_size (int): Original image size (default is 640x640).

    Returns:
        list of tuples: Scaled bounding boxes.
    """
    new_w, new_h = new_size
    scale_x = new_w / original_size
    scale_y = new_h / original_size

    return [
        (
            int(x_min * scale_x),
            int(y_min * scale_y),
            int(x_max * scale_x),
            int(y_max * scale_y)
        )
        for x_min, y_min, x_max, y_max in bboxes
    ]
