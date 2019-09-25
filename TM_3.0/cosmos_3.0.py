''' 
Author: Esteban J. Chaves, PhD
Observatorio Vulcanológico y Sismológico de Costa Rica
OVSICORI-UNA
'''


import cv2
import numpy as np
import matplotlib.pyplot as plt
cv2.namedWindow("Detected", cv2.WINDOW_NORMAL) 

# 1. Reading the Original Photo
Data='ww2.jpg'
# 2. Reading the template
template='Waldo.png'
# 3. Define the name of the output figure
out='Detections.png'

# Method to be used for the template matching
method=cv2.TM_CCOEFF_NORMED

# Threshold or similatity 1 = 100 % similar
threshold=0.30 # Similarity = 0.45 % 


img_rgb = cv2.imread(Data)
img_gray = cv2.cvtColor(img_rgb, cv2.COLOR_BGR2GRAY)
template = cv2.imread(template, 0)

# 4. Computing the dimension of the template
w, h = template.shape[::-1]
print("\nTemplate dimensions: Width = " + str(w) + ", Height = " + str(h))

# 5. Template matching computation
res = cv2.matchTemplate(img_gray, template, method)

loc = np.where(res >= threshold)

if not loc:
	print("Sorry, there are no matches in this image\n, maybe try a different threshold...")

detections =[]
for pt in zip(*loc[::-1]):
	cv2.rectangle(img_rgb, pt, (pt[0] + w, pt[1] + h), (255,255,255), 2)
	detections.append(pt)

detections = np.asarray(detections)
print("Detections found = %s " % str(detections.shape[0]))

   
imS = cv2.resize(img_rgb, (1000, 960))
cv2.imwrite(out,imS)





