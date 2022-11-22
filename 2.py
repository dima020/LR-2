import numpy as np
import cv2 as cv
import matplotlib.pyplot as plt


img1 = cv.imread('input/image_3.jpg',cv.IMREAD_GRAYSCALE)          # queryImage
img2 = cv.imread('input/image_3_template.jpg',cv.IMREAD_GRAYSCALE) # trainImage

titles = ["original", "template"]
images = [img1,img2]
for i in range(2):
    plt.subplot(1,2,i+1),plt.imshow(cv.cvtColor(img1[i], cv.COLOR_BGR2RGB))
    plt.title(titles[i])
    plt.xticks([]),plt.yticks([])
plt.show()
# Initiate ORB detector

orb = cv.ORB_create()
# find the keypoints and descriptors with ORB
kp1, des1 = orb.detectAndCompute(img1,None)
kp2, des2 = orb.detectAndCompute(img2,None)

 #create BFMatcher object
bf = cv.BFMatcher(cv.NORM_HAMMING, crossCheck=True)
# Match descriptors.
matches = bf.match(des1,des2)
# Sort them in the order of their distance.
matches = sorted(matches, key = lambda x:x.distance)
# Draw first 10 matches.
img3 = cv.drawMatches(img1,kp1,img2,kp2,matches[:10],None,flags=cv.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)
plt.imshow(img3),plt.show()
#print(cv.KeyPoint_convert(kp2))

# Записываем первые 10 соответсвий с эталона и исходного изображения
src_pts = np.float32([kp1[m.queryIdx].pt for m in matches[:10]]).reshape(-1, 1, 2)
dst_pts = np.float32([kp2[m.trainIdx].pt for m in matches[:10]]).reshape(-1, 1, 2)

# Рассчитываем матрицу гомографии
h, _ = cv.findHomography(src_pts, dst_pts)

# Трансформируем исходное изображение, используя полученную гомографию
# im_out = cv2.warpPerspective(template, h, (image.shape[1],image.shape[0]))


rectangle_pts = np.float32([[0, 0],
                            [0, img2.shape[0]],
                            [img2.shape[1], 0],
                            [img2.shape[1], img2.shape[0]]]).reshape(-1, 1, 2)

rectangle_pts = cv.perspectiveTransform(rectangle_pts, h).reshape(-1, 2)

# Находим углы рамки
left_top = [img1.shape[1], img1.shape[0]]
right_bottom = [0, 0]

for point in rectangle_pts:
    # Находим левую верхнюю точку рамки
    if point[0] < left_top[0]:
        left_top[0] = point[0] if point[0] >= 0 else 0
    if point[1] < left_top[1]:
        left_top[1] = point[1] if point[1] >= 0 else 0

    # Находим правую нижнюю точку рамки
    if point[0] > right_bottom[0]:
        right_bottom[0] = point[0] if point[0] <= img1.shape[1] else img1.shape[1]
    if point[1] > right_bottom[1]:
        right_bottom[1] = point[1] if point[1] <= img1.shape[0] else img1.shape[0]

# Чтобы рамку было видно у края изображения:
left_top = [int(left_top[0]), int(left_top[1])]
if left_top[0] == 0:
    left_top[0] += 10
if left_top[1] == 0:
    left_top[1] += 10

right_bottom = [int(right_bottom[0]), int(right_bottom[1])]
if right_bottom[0] == img1.shape[1]:
    right_bottom[0] -= 10
if right_bottom[1] == img1.shape[0]:
    right_bottom[1] -= 10

# Чертим рамку
img3 = img1.copy()
cv.rectangle(img3, left_top, right_bottom, 255, 10)

# Вывод mathplotlib
plt.imshow(cv.cvtColor(img3, cv.COLOR_BGR2RGB))
plt.show()