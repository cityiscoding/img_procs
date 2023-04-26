import cv2
import numpy as np

# Đọc ảnh đầu vào
img = cv2.imread(r'C:\Users\Admin\PycharmProjects\xulyanh\source\gx2.jpg', cv2.IMREAD_GRAYSCALE)

# Kích thước của ảnh
height, width = img.shape[:2]

# Khởi tạo ma trận kết quả
result = np.zeros((height, width), np.uint8)

# Ma trận lọc (kernel)
kernel = np.array([[-1,-1,-1],
                   [-1, 9,-1],
                   [-1,-1,-1]])

# Duyệt qua từng pixel trong ảnh
for i in range(1, height - 1):
    for j in range(1, width - 1):
        # Áp dụng ma trận lọc vào từng phần tử của ảnh
        sum = 0
        for x in range(-1, 2):
            for y in range(-1, 2):
                sum += kernel[x + 1, y + 1] * img[i + x, j + y]
        result[i, j] = np.clip(sum, 0, 255)


cv2.imshow('Result', result)
cv2.waitKey(0)
cv2.destroyAllWindows()
