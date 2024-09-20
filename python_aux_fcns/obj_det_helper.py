def iou(boxA_tlbr, boxB_tlbr):
    # Determine the coordinates of the intersection rectangle
    top = max(boxA_tlbr[0], boxB_tlbr[0])
    left = max(boxA_tlbr[1], boxB_tlbr[1])
    bottom = min(boxA_tlbr[2], boxB_tlbr[2])
    right = min(boxA_tlbr[3], boxB_tlbr[3])

    # Compute the area of intersection
    intersectionArea = max(0, right - left) * max(0, bottom - top)

    # Compute the area of both the prediction and ground-truth rectangles
    boxAArea = (boxA_tlbr[3] - boxA_tlbr[1]) * (boxA_tlbr[2] - boxA_tlbr[0])
    boxBArea = (boxB_tlbr[3] - boxB_tlbr[1]) * (boxB_tlbr[2] - boxB_tlbr[0])

    iou = intersectionArea / float(boxAArea + boxBArea - intersectionArea)
    return iou
