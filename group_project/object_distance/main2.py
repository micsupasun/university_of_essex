import cv2
import numpy as np

# Distance constants
KNOWN_DISTANCE = 20  # INCHES
MOBILE_WIDTH = 21  # INCHES
MOBILE_HEIGHT = 4  # INCHES
PERSON_WIDTH = 20  # INCHES
CUP_WIDTH = 4.0  # INCHES
CHAIR_WIDTH = 11  # INCHES
BOOK_WIDTH = 4.0  # INCHES

# Object detector constant
CONFIDENCE_THRESHOLD = 0.4
NMS_THRESHOLD = 0.3

# Colors for BBOX
COLORS = [(255, 0, 0), (255, 0, 255), (0, 255, 255), (255, 255, 0), (0, 255, 0), (255, 0, 0)]
GREEN = (0, 255, 0)
BLACK = (0, 0, 0)
WHITE = (255, 255, 255)
# Defining fonts
FONTS = cv2.FONT_HERSHEY_COMPLEX

# Load COCO Dataset
classNames = []
classFile = '/Users/supasunkhumpraphan/Desktop/work_realv2/essex_university/computer_visionv7/object_detection_v3/coco.names'
with open(classFile, 'rt') as f:
    classNames = f.read().rstrip('\n').split('\n')

# Load SSD Algorithm and trained weights
configPath = '/Users/supasunkhumpraphan/Desktop/work_realv2/essex_university/computer_visionv7/object_detection_v3/ssd_mobilenet_v3_large_coco_2020_01_14.pbtxt'
# configPath = '/Users/supasunkhumpraphan/Desktop/work_realv2/essex_university/computer_visionv7/object_detection_v3/labelmap.pbtxt'

weightsPath = '/Users/supasunkhumpraphan/Desktop/work_realv2/essex_university/computer_visionv7/object_detection_v3/frozen_inference_graph.pb'

net = cv2.dnn_DetectionModel(weightsPath, configPath)
net.setInputSize(320, 320)
net.setInputScale(1.0 / 127.5)
net.setInputMean((127.5, 127.5, 127.5))
net.setInputSwapRB(True)


# Object Detection Function
def object_detector(img):
    classes, scores, boxes = net.detect(img, CONFIDENCE_THRESHOLD)
    # Creating empty list to add objects data
    data_list = []
    for (classid, score, box) in zip(classes, scores, boxes):
        # define color of each, object based on its class id
        color = COLORS[int(classid) % len(COLORS)]

        # label = "%s : %f" % (classNames[classid[0]], score)
        # draw rectangle on and label on object
        cv2.rectangle(img, box, color, 2)
        cv2.rectangle(img, (box[0] - 1, box[1] - 28), (box[0] + 150, box[1]), color, -1)
        cv2.putText(img, classNames[classid - 1], (box[0], box[1] - 10), FONTS, 0.5, (255, 255, 255), 1)

        # getting the data
        # 1: class name  2: object width in pixels, 3: position where have to draw text(distance)
        if classid == 77:  # cellphone class id
            data_list.append([classNames[classid - 1], box[2], (box[0], box[1] - 2)])
        elif classid == 1:  # person class id
            data_list.append([classNames[classid - 1], box[2], (box[0], box[1] - 2)])
        elif classid == 62:  # chair class id
            data_list.append([classNames[classid - 1], box[2], (box[0], box[1] - 2)])
        elif classid == 47:  # cup class id
            data_list.append([classNames[classid - 1], box[2], (box[0], box[1] - 2)])
        elif classid == 84:  # book class id
            data_list.append([classNames[classid - 1], box[2], (box[0], box[1] - 2)])

        # returning list containing the object data.
    return data_list


# Getting Focal Length
def focal_length_finder(measured_distance, real_width, width_in_rf):
    focal_length = (width_in_rf * measured_distance) / real_width

    return focal_length


# Getting Distance
def distance_finder(focal_length, real_object_width, width_in_frame):
    distance = (real_object_width * focal_length) / width_in_frame

    return distance


# Reading the reference image from dir
ref_mobile = cv2.imread('/Users/supasunkhumpraphan/Desktop/work_realv2/essex_university/computer_visionv7/object_detection_v3/ReferenceImages/phone.jpeg') # not done
ref_person = cv2.imread('/Users/supasunkhumpraphan/Desktop/work_realv2/essex_university/computer_visionv7/object_detection_v3/ReferenceImages/person.jpeg')
ref_chair = cv2.imread('/Users/supasunkhumpraphan/Desktop/work_realv2/essex_university/computer_visionv7/object_detection_v3/ReferenceImages/chair.jpeg') # not done
ref_cup = cv2.imread('/Users/supasunkhumpraphan/Desktop/work_realv2/essex_university/computer_visionv7/object_detection_v3/ReferenceImages/cup.jepg') # not done
ref_book = cv2.imread('/Users/supasunkhumpraphan/Desktop/work_realv2/essex_university/computer_visionv7/object_detection_v3/ReferenceImages/book.jpeg')

mobile_data = object_detector(ref_mobile)
print('mobile_data',mobile_data)
if mobile_data == []:
    pass
else:
    mobile_width_in_rf = mobile_data[0][1]


try:
    person_data = object_detector(ref_person)
    person_width_in_rf = person_data[0][1]
except:
    pass
try:
    chair_data = object_detector(ref_chair)
    chair_width_in_rf = chair_data[0][1]
except:
    pass

try:
    cup_data = object_detector(ref_cup)
    cup_width_in_rf = cup_data[0][1]
except:
    pass
try:
    book_data = object_detector(ref_book)
    book_width_in_rf = book_data[0][1]
except:
    pass

# print(f"Mobile width in pixel: {mobile_width_in_rf}")

# Getting focal length
try:
    focal_mobile = focal_length_finder(KNOWN_DISTANCE, MOBILE_WIDTH, mobile_width_in_rf)
except:
    pass
try:
    focal_person = focal_length_finder(KNOWN_DISTANCE, PERSON_WIDTH, person_width_in_rf)
except:
    pass
try:
    focal_chair = focal_length_finder(KNOWN_DISTANCE, CHAIR_WIDTH, chair_width_in_rf)
except:
    pass
try:
    focal_cup = focal_length_finder(KNOWN_DISTANCE, CUP_WIDTH, cup_width_in_rf)
except:
    pass
try:
    focal_book = focal_length_finder(KNOWN_DISTANCE, BOOK_WIDTH, book_width_in_rf)
except:
    pass

cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    # frame = cv2.imread('ReferenceImages/image9.jpg')

    data = object_detector(frame)
    for d in data:
        try:
            if d[0] == 'cell phone':
                distance = distance_finder(focal_mobile, MOBILE_WIDTH, d[1])
                x, y = d[2]
            elif d[0] == 'person':
                distance = distance_finder(focal_person, PERSON_WIDTH, d[1])
                x, y = d[2]
            elif d[0] == 'chair':
                distance = distance_finder(focal_person, CHAIR_WIDTH, d[1])
                x, y = d[2]
            elif d[0] == 'cup':
                distance = distance_finder(focal_person, CUP_WIDTH, d[1])
                x, y = d[2]
            elif d[0] == 'book':
                distance = distance_finder(focal_person, BOOK_WIDTH, d[1])
                x, y = d[2]
            cv2.rectangle(frame, (x - 1, y - 3), (x + 150, y + 23), BLACK, -1)
            cv2.putText(frame, f'Dist: {round(distance, 2)} inch', (x + 5, y + 13), FONTS, 0.48, GREEN, 1)
        except:
            pass
    cv2.imshow('frame', frame)

    key = cv2.waitKey(1)
    if key == ord('q'):
        break
cv2.destroyAllWindows()
cap.release()
