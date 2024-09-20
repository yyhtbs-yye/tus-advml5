import numpy as np

def get_label_boxes(label):
    tlbr_boxes = []  # Use tlbr notation for clarity
    classes = []
    annotations = label['annotation']['object']
    for obj in annotations:
        top = int(obj['bndbox']['ymin'])
        left = int(obj['bndbox']['xmin'])
        bottom = int(obj['bndbox']['ymax'])
        right = int(obj['bndbox']['xmax'])
        cls = obj['name']
        tlbr_boxes.append(np.array([top, left, bottom, right], dtype=np.int32))
        classes.append(cls)
    return tlbr_boxes, classes
