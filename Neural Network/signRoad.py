# import the necessary packages
# import tensorflow as tf
import numpy as np
import cv2
from tensorflow.keras.models import load_model
from matplotlib import pyplot as plt

# read Image ###############3
img = cv2.imread('sign1.png')

# load model ###############3
model = load_model("CNN_model/signs_cnn.h5")
print(model.summary())
threshold = 0.75         # PROBABLITY THRESHOLD

# Image preprocessing function ###############3
def grayscale(img):
    img = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    return img
def equalize(img):
    img =cv2.equalizeHist(img)
    return img
def preprocessing(img):
    img = grayscale(img)
    img = equalize(img)
    img = img/255
    return img

def getCalssName(classNo):
    if   classNo == 0: return 'Speed Limit 20 km/h'
    elif classNo == 1: return 'Speed Limit 30 km/h'
    elif classNo == 2: return 'Speed Limit 50 km/h'
    elif classNo == 3: return 'Speed Limit 60 km/h'
    elif classNo == 4: return 'Speed Limit 70 km/h'
    elif classNo == 5: return 'Speed Limit 80 km/h'
    elif classNo == 6: return 'End of Speed Limit 80 km/h'
    elif classNo == 7: return 'Speed Limit 100 km/h'
    elif classNo == 8: return 'Speed Limit 120 km/h'
    elif classNo == 9: return 'No passing'
    elif classNo == 10: return 'No passing for vechiles over 3.5 metric tons'
    elif classNo == 11: return 'Right-of-way at the next intersection'
    elif classNo == 12: return 'Priority road'
    elif classNo == 13: return 'Yield'
    elif classNo == 14: return 'Stop'
    elif classNo == 15: return 'No vechiles'
    elif classNo == 16: return 'Vechiles over 3.5 metric tons prohibited'
    elif classNo == 17: return 'No entry'
    elif classNo == 18: return 'General caution'
    elif classNo == 19: return 'Dangerous curve to the left'
    elif classNo == 20: return 'Dangerous curve to the right'
    elif classNo == 21: return 'Double curve'
    elif classNo == 22: return 'Bumpy road'
    elif classNo == 23: return 'Slippery road'
    elif classNo == 24: return 'Road narrows on the right'
    elif classNo == 25: return 'Road work'
    elif classNo == 26: return 'Traffic signals'
    elif classNo == 27: return 'Pedestrians'
    elif classNo == 28: return 'Children crossing'
    elif classNo == 29: return 'Bicycles crossing'
    elif classNo == 30: return 'Beware of ice/snow'
    elif classNo == 31: return 'Wild animals crossing'
    elif classNo == 32: return 'End of all speed and passing limits'
    elif classNo == 33: return 'Turn right ahead'
    elif classNo == 34: return 'Turn left ahead'
    elif classNo == 35: return 'Ahead only'
    elif classNo == 36: return 'Go straight or right'
    elif classNo == 37: return 'Go straight or left'
    elif classNo == 38: return 'Keep right'
    elif classNo == 39: return 'Keep left'
    elif classNo == 40: return 'Roundabout mandatory'
    elif classNo == 41: return 'End of no passing'
    elif classNo == 42: return 'End of no passing by vechiles over 3.5 metric tons'

# Convert to graycsale ###############3
img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# Blur the image for better edge detection ###############3
img_blur = cv2.GaussianBlur(img_gray, (3,3), 0)

# Cany Edge Detection method ###############3
th1 = 100  # lower threshold 50
th2 = 200  # upper threshold 100
edges = cv2.Canny(img_blur, th1, th2)

# Find contours ###############3
contours, hierarchy = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)#cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
cv2.drawContours(img_blur, contours, -1, (0,255,0), 3)
# print(type(contours))

# find the biggest countour (c) by the area ###############3
c = max(contours, key = cv2.contourArea)
x,y,w,h = cv2.boundingRect(c)

# crop image at given contour ###############3
cropped_contour= img[y-10:y+h+10, x-10:x+w+10]
# print(contours.index(c))

# PROCESS IMAGE
img2 = np.asarray(cropped_contour)
img2 = cv2.resize(img2, (32, 32))
img2 = preprocessing(img2)
cv2.imshow("Processed Image", img2)
img2 = img2.reshape(1, 32, 32, 1)

# PREDICT IMAGE
predictions = model.predict(img2)
classIndex = model.predict_classes(img2)
probabilityValue =np.amax(predictions)
if probabilityValue > threshold:
    print(classIndex,getCalssName(classIndex))

# Show the output image
# cv2.imshow('Output', cropped_contour)

# draw the biggest contour (c) in green
# cv2.rectangle(img,(x,y),(x+w,y+h),(0,255,0),2)


# cv2.imshow('road',img)


cv2.waitKey(0)
cv2.destroyAllWindows()



# crop non signed region
# idx = contours.index(c)
# mask = np.zeros_like(img) # Create mask where white is what we want, black otherwise
# cv2.drawContours(mask, contours, idx, 255, -1) # Draw filled contour in mask
# out = np.zeros_like(img) # Extract out the object and place into output image
# out[mask == 255] = img[mask == 255]

# Now crop
# (y, x) = np.where(mask == 255)
# (topy, topx) = (np.min(y), np.min(x))
# (bottomy, bottomx) = (np.max(y), np.max(x))
# out = out[topy:bottomy+1, topx:bottomx+1]
