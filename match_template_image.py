import cv2

image = cv2.imread('input/image_1.jpg')
image_copy = image.copy()
image_copy = cv2.cvtColor(image_copy, cv2.COLOR_BGR2GRAY)

template = cv2.imread('input/image_1_template.jpg', 0)
w, h = template.shape[::-1]

result = cv2.matchTemplate(image_copy, template, cv2.TM_CCOEFF_NORMED)
min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(result)

x1, y1 = max_loc

x2, y2 = (x1 + w, y1 + h)
cv2.rectangle(image, (x1, y1), (x2, y2), (0, 0, 255), 2)
cv2.imshow('Result', image)

cv2.normalize(result, result, 0, 1, cv2.NORM_MINMAX, -1 )
#cv2.imshow('Detected point', result)
cv2.waitKey(0)
cv2.imwrite('outputs/image_result.jpg', image)
#cv2.imwrite('outputs/template_result.jpg')