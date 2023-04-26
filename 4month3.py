import cv2
import numpy as np

# Load ảnh xám
img_gray = cv2.imread('img_gray.jpg', cv2.IMREAD_GRAYSCALE)

# Định nghĩa ma trận Robert
robert = np.array([[-1, 0], [0, 1]]) 

# Áp dụng ma trận Robert lên ảnh
edge = cv2.filter2D(img_gray, -1, robert)
# Hiển thị ảnh gốc
cv2_imshow(img_gray)
# Hiển thị kết quả
cv2_imshow(edge)