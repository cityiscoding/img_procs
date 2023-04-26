import cv2

# Đọc ảnh đầu vào
img = cv2.imread('input_image.jpg')

# Áp dụng giải thuật k láng giềng gần nhất để làm sạch ảnh và giảm nhiễu
denoised_img = cv2.fastNlMeansDenoisingColored(img, None, 10, 10, 7, 21)

# Chuyển đổi không gian màu sang không gian màu LAB
lab_img = cv2.cvtColor(denoised_img, cv2.COLOR_BGR2LAB)

# Phân đoạn các đối tượng trong ảnh bằng phân đoạn theo màu sắc
lower_red = (0, 100, 100)
upper_red = (20, 255, 255)
mask = cv2.inRange(lab_img, lower_red, upper_red)

# Tô màu các đối tượng đã phân đoạn
contours, _ = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
cv2.drawContours(img, contours, -1, (0, 255, 0), 2)
cv2.fillPoly(img, contours, (0, 255, 0))

# Hiển thị ảnh kết quả
cv2.imshow
