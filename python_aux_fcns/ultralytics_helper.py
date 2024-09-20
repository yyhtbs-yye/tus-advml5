import cv2

def draw_boxes(image, result, names, color=(255, 0, 0), thickness=1):
  
    image = image.copy()
  
    if isinstance(result, list):
        print('''Error: Arg.0 "result" should not be a list!''')
        return image
    for box in result.boxes:
        xyxy = box.xyxy.to("cpu").view(-1)
        class_output = names[int(box.cls.cpu().numpy())]
    
        cv2.rectangle(image, (int(xyxy[0]), int(xyxy[1])), (int(xyxy[2]), int(xyxy[3])), color, thickness)
        cv2.putText(image, class_output, (int(xyxy[0]), int(xyxy[1]) - 2*thickness), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color=color)
    return image

def crop_and_resize_boxes(image, tlbr_boxes, target_size=(224, 224)):
    cropped_resized_images = []
    for tlbr_box in tlbr_boxes:
        cropped_img = image[tlbr_box[0]:tlbr_box[2], tlbr_box[1]:tlbr_box[3]]
        resized_img = cv2.resize(cropped_img, target_size)
        cropped_resized_images.append(resized_img)

    return cropped_resized_images
