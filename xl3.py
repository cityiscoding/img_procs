# import cv2
# import numpy as np
#
# # Load ảnh xám
# img_gray = cv2.imread(r'C:\Project_Work\pythonProject\images\pic2.jpg', cv2.IMREAD_GRAYSCALE)
#
# # Định nghĩa ma trận Robert
# # robert = np.array([[-1, 0], [0, 1]])
# # prewitt = np.array([[-1, -1, -1], [0, 0, 0] ,[1, 1, 1]])
#
# sobel = np.array([[-1, -2, -1], [0, 0, 0] ,[1, 2, 1], [-1, 0 , 1], [-2 , 0, 2], [-1 , 0 , 1]])
#
# # Áp dụng ma trận Robert lên ảnh
# # edge = cv2.filter2D(img_gray, -1, sobel)
# for i in range(3):
#     for j in range(3):
#         sobel[i][j] *= 1/8
# edge = cv2.filter2D(img_gray, -1, sobel)
# # Hiển thị ảnh gốc
# cv2.imshow('Original Image', img_gray)
# # Hiển thị kết quả
# cv2.imshow('Edge Detection', edge)
#
# cv2.waitKey(0)
# cv2.destroyAllWindows()
# import cv2
# import numpy as np
#
# def load_image_gray(image_path):
#     """
#     Hàm tải ảnh xám từ đường dẫn `image_path`.
#     """
#     return cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
#
# def prewitt_filter(kernel_size=3):
#     """
#     Hàm trả về ma trận Prewitt theo kích thước `kernel_size`.
#     """
#     kernel = np.zeros((kernel_size, kernel_size))
#     kernel[:, :kernel_size//2] = -1
#     kernel[:, kernel_size//2+1:] = 1
#     return kernel
#
# def apply_filter(image, kernel):
#     """
#     Hàm áp dụng ma trận `kernel` lên ảnh `image`.
#     """
#     return cv2.filter2D(image, -1, kernel)
#
# def show_image(image, window_name):
#     """
#     Hàm hiển thị ảnh `image` trong cửa sổ có tên `window_name`.
#     """
#     cv2.imshow(window_name, image)
#     cv2.waitKey(0)
#     cv2.destroyAllWindows()
#
# if __name__ == '__main__':
#     # Tải ảnh xám
#     img_gray = load_image_gray(r'C:\Project_Work\pythonProject\images\pic.jpg')
#
#     # Định nghĩa ma trận Prewitt
#     prewitt = prewitt_filter()
#
#     # Áp dụng ma trận Prewitt lên ảnh
#     edge = apply_filter(img_gray, prewitt)
#
#     # Hiển thị ảnh gốc và kết quả
#     show_image(img_gray, 'Original Image')
#     show_image(edge, 'Edge Detection')

# import cv2
# import numpy as np
#
# # Load ảnh xám
# img_gray = cv2.imread(r'C:\Project_Work\pythonProject\images\pic2.jpg', cv2.IMREAD_GRAYSCALE)
#
# # Định nghĩa ma trận Sobel
# sobel_x = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]])
# sobel_y = np.array([[-1, -2, -1], [0, 0, 0], [1, 2, 1]])
#
# # Tính gradient theo trục x và y
# grad_x = cv2.filter2D(img_gray, -1, sobel_x)
# grad_y = cv2.filter2D(img_gray, -1, sobel_y)
#
# # Kết hợp gradient x và y với hàm addWeighted
# edge = cv2.addWeighted(grad_x, 0.5, grad_y, 0.5, 0)
#
# # Hiển thị ảnh gốc
# cv2.imshow('Original Image', img_gray)
#
# # Hiển thị kết quả
# cv2.imshow('Edge Detection', edge)
#
# cv2.waitKey(0)
# cv2.destroyAllWindows()


# import cv2
# import numpy as np
#
# # Load ảnh xám
# anh_xam = cv2.imread(r'C:\Project_Work\pythonProject\images\pic2.jpg', cv2.IMREAD_GRAYSCALE)
#
# # Ma trận  kirsh
# kirsh = np.array([[-3 , -3 , 5], [-3 , 0 , 5], [-3 ,-3 , 5]])
# kirsh2 = np.array([[-3 , 5 , 5], [-3, 0 , 5], [-3, -3, -3]])
# kirsh3 = np.array([[5 ,5 , 5], [-3, 0 , -3], [-3, -3, -3]])
# kirsh4 = np.array([[5 ,5 , -3], [5, 0 , -3], [-3, -3, -3]])
# kirsh5 = np.array([[5 ,-3 , -3], [5, 0 , -3], [5, -3, -3]])
# kirsh6 = np.array([[-3 ,-3 , -3], [5, 0 , -3], [5, 5, -3]])
# kirsh7 = np.array([[-3 ,-3 , -3], [-3, 0 , -3], [5, 5, 5]])
# kirsh8 = np.array([[-3 ,-3 , 5], [-3, 0 , 5], [-3, 5, 5]])
#
# # Nhân ma trận kirsh
# anh1 = cv2.filter2D(anh_xam, -1, kirsh)
# anh2 = cv2.filter2D(anh_xam, -1, kirsh2)
# anh3 = cv2.filter2D(anh_xam, -1, kirsh3)
# anh4 = cv2.filter2D(anh_xam, -1, kirsh4)
# anh5 = cv2.filter2D(anh_xam, -1, kirsh5)
# anh6 = cv2.filter2D(anh_xam, -1, kirsh6)
# anh7 = cv2.filter2D(anh_xam, -1, kirsh7)
# anh8 = cv2.filter2D(anh_xam, -1, kirsh8)
#
# # Tạo một danh sách chứa tất cả các ảnh
# danh_sach_anh = [anh1, anh2, anh3, anh4, anh5, anh6, anh7, anh8]
#
# # Tìm tấm ảnh có biên độ điểm lớn nhất
# max_edge = None
# for anh in danh_sach_anh:
#     abs_anh = cv2.convertScaleAbs(anh)
# # Tìm giá trị cạnh lớn nhất
#     edge = np.max(abs_anh)
# if max_edge is None or edge > max_edge:
#     max_edge = edge
#     final_img = abs_anh
#
# # Hiển thị ảnh có biên độ từng điểm lớn nhất
#
# cv2.imshow('ảnh gốc', anh_xam)
# cv2.imshow('Ảnh cuối cùng', final_img)
# cv2.waitKey(0)
# cv2.destroyAllWindows()


import cv2
#
# # Load ảnh vào biến image
# image = cv2.imread(r'C:\Project_Work\pythonProject\images\pic2.jpg')
#
# # Chuyển ảnh sang grayscale
# gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
#
# # Áp dụng toán tử Laplacian
# laplacian = cv2.Laplacian(gray, cv2.CV_64F)
#
# # Hiển thị kết quả
# cv2.imshow('Original Image', image)
# cv2.imshow('Laplacian Filtered Image', laplacian)
# cv2.waitKey(0)
# cv2.destroyAllWindows()

# import cv2
# import numpy as np
#
# # Load ảnh xám
# img_gray = cv2.imread(r'C:\Project_Work\pythonProject\images\pic2.jpg', cv2.IMREAD_GRAYSCALE)
#
# # Định nghĩa ma trận  laplacian
# # robert = np.array([[0, 1, 0], [1, -4 , 1],[0, 1, 0]])
# robert = np.array([[1, 1, 1], [1, -8 , 1],[1, 1, 1]])
# # Áp dụng ma trận  laplacian lên ảnh
# edge = cv2.filter2D(img_gray, -1, robert)
# # Hiển thị ảnh gốc
# cv2.imshow('Original Image', img_gray)
# # Hiển thị kết quả
# cv2.imshow('',edge)
# cv2.waitKey(0)
# cv2.destroyAllWindows()

# import cv2
# import numpy as np
#
# # Đọc ảnh
# img = cv2.imread(r'C:\Project_Work\pythonProject\images\pic2.jpg')
#
# # Tạo kernel sử dụng nhân chập Mexican Hat
# kernel_size = 5
# sigma = 1.0
# kernel = np.zeros((kernel_size, kernel_size))
# for i in range(kernel_size):
#     for j in range(kernel_size):
#         x, y = i - kernel_size // 2, j - kernel_size // 2
#         kernel[i, j] = (1 - (x ** 2 + y ** 2) / (2 * sigma ** 2)) * np.exp(-(x ** 2 + y ** 2) / (2 * sigma ** 2))
# kernel = kernel / np.sum(kernel)
#
# # Áp dụng bộ lọc Gaussian với nhân chập Mexican Hat lên ảnh
# blurred = cv2.filter2D(img, -1, kernel)
#
# # Hiển thị ảnh gốc và ảnh sau khi làm mờ
# cv2.imshow('Original Image', img)
# cv2.imshow('Blurred Image', blurred)
# cv2.waitKey(0)
# cv2.destroyAllWindows()

''''''# import cv2
# import numpy as np
#
# # Đọc ảnh
# img = cv2.imread(r'C:\Project_Work\pythonProject\images\pic2.jpg')
#
# # Áp dụng mặt nạ DOG với các tham số khác nhau
# ksize1 = (3, 3)
# ksize2 = (7, 7)
# sigma1 = 1.0
# sigma2 = 2.0
# gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
# gaussian1 = cv2.GaussianBlur(gray, ksize1, sigma1)
# gaussian2 = cv2.GaussianBlur(gray, ksize2, sigma2)
# dog = gaussian1 - gaussian2
#
# # Sử dụng mặt nạ dò cạnh để phát hiện các cạnh trên ảnh
# edges = cv2.Canny(dog, 50, 500)
#
# # Hiển thị ảnh gốc và ảnh kết quả
# cv2.imshow('Original Image', img)
# cv2.imshow('DOG Edges', edges)
# cv2.waitKey(0)
# cv2.destroyAllWindows()


# import cv2
#
# # Đọc ảnh cần xử lý
# img = cv2.imread(r'C:\Project_Work\pythonProject\images\pic2.jpg')
#
# # Áp dụng bộ lọc Canny
# edges = cv2.Canny(img, 100, 200)
#
# # Hiển thị ảnh gốc và ảnh sau khi xử lý
# cv2.imshow('Original Image', img)
# cv2.imshow('Canny Edges', edges)
#
# # Chờ nhấn phím bất kỳ để đóng cửa sổ
# cv2.waitKey(0)
# cv2.destroyAllWindows()

# 4/4

# import cv2
# import numpy as np
#
# img = cv2.imread(r'C:\Project_Work\pythonProject\images\anhnhieu3.jpg')
#
# # lấy kích thước ảnh
# height, width, _ = img.shape
#
# # tạo một ma trận con 3x3 với giá trị 1/9
# kernel = np.ones((3, 3), np.float32) / 9
#
# # tạo một ảnh mới với kích thước bằng với ảnh gốc
# filtered_img = np.zeros((height, width, 3), np.uint8)
#
# # áp dụng bộ lọc trung bình
# for i in range(1, height - 1):
#     for j in range(1, width - 1):
#         for k in range(3):
#             sum_value = 0
#             for x in range(-1, 2):
#                 for y in range(-1, 2):
#                     sum_value += img[i+x][j+y][k] * kernel[x+1][y+1]
#             filtered_img[i][j][k] = sum_value
#
# # hiển thị ảnh gốc và ảnh đã được xử lý
# cv2.imshow('Original Image', img)
# cv2.imshow('Filtered Image', filtered_img)
#
# # chờ người dùng nhấn phím bất kỳ để thoát
# cv2.waitKey(0)
# cv2.destroyAllWindows()

# import cv2
# import numpy as np
# import math
#
# img = cv2.imread(r'C:\Project_Work\pythonProject\images\anhnhieu3.jpg')
#
# # lấy kích thước ảnh
# height, width, _ = img.shape
#
# # tạo một ma trận con 3x3 với giá trị 1/9
# kernel = np.ones((3, 3), np.float32)
#
# # tạo một ảnh mới với kích thước bằng với ảnh gốc
# filtered_img = np.zeros((height, width, 3), np.uint8)
#
# # áp dụng bộ lọc trung bình
# for i in range(1, height - 1):
#     for j in range(1, width - 1):
#         for k in range(3):
#             sum_value = 1
#             for x in range(-1, 2):
#                 for y in range(-1, 2):
#                     sum_value *= img[i+x][j+y][k] * kernel[x+1][y+1]
#             filtered_img[i][j][k] = pow(sum_value,1/9)
#
# # hiển thị ảnh gốc và ảnh đã được xử lý
# cv2.imshow('Original Image', img)
# cv2.imshow('Filtered Image', filtered_img)
#
# # chờ người dùng nhấn phím bất kỳ để thoát
# cv2.waitKey(0)
# cv2.destroyAllWindows()

# import cv2
# import numpy as np
# img = cv2.imread(r'C:\Project_Work\pythonProject\images\anhnhieu3.jpg')
#
# # lấy kích thước ảnh
# height, width, _ = img.shape
#
# # tạo một ma trận con 3x3 với giá trị 1
# kernel_size = 3
# kernel = np.ones((kernel_size, kernel_size), np.float32)
#
# # tạo một ảnh mới với kích thước bằng với ảnh gốc
# filtered_img = np.zeros((height, width, 3), np.uint8)
#
# # áp dụng bộ lọc harmonic mean filter
# for i in range(1, height - 1):
#     for j in range(1, width - 1):
#         for k in range(3):
#             sum_value = 0
#             for x in range(-1, 2):
#                 for y in range(-1, 2):
#                     if img[i+x][j+y][k] != 0:
#                         sum_value += 1.0 / img[i+x][j+y][k]
#             if sum_value != 0:
#                 filtered_img[i][j][k] = kernel_size**2 / sum_value
#             else:
#                 filtered_img[i][j][k] = img[i][j][k]
#
# # hiển thị ảnh gốc và ảnh đã được xử lý
# cv2.imshow('Original Image', img)
# cv2.imshow('Filtered Image', filtered_img)
#
# # chờ người dùng nhấn phím bất kỳ để thoát
# cv2.waitKey(0)
#
# cv2.destroyAllWindows()


# contra-harmonic mean filter
# import cv2
# import numpy as np
#
# img = cv2.imread(r'C:\Project_Work\pythonProject\images\anhnhieu3.jpg')
#
# # lấy kích thước ảnh
# height, width, _ = img.shape
#
# # tạo một ma trận con 3x3 với giá trị 1
# kernel_size = 3
# kernel = np.ones((kernel_size, kernel_size), np.float32)
#
# # tạo một ảnh mới với kích thước bằng với ảnh gốc
# filtered_img = np.zeros((height, width, 3), np.uint8)
#
# # áp dụng bộ lọc contra-harmonic mean filter với q = -1.5
# q = -1
# for i in range(1, height - 1):
#     for j in range(1, width - 1):
#         for k in range(3):
#             numerator = 0
#             denominator = 0.0
#             for x in range(-1, 2):
#                 for y in range(-1, 2):
#                     if img[i+x][j+y][k] != 0:
#                         numerator += (img[i+x][j+y][k] ** (q+1))
#                         denominator += (float(img[i+x][j+y][k]) ** q)
#             if denominator != 0:
#                 filtered_img[i][j][k] = numerator / denominator
#             else:
#                 filtered_img[i][j][k] = img[i][j][k]
#
# # hiển thị ảnh gốc và ảnh đã được xử lý
# cv2.imshow('Original Image', img)
# cv2.imshow('Filtered Image', filtered_img)

# chờ người dùng nhấn phím bất kỳ để thoát
# cv2.waitKey(0)
# cv2.destroyAllWindows()

# Phép lọc midpoint là một phương pháp lọc không phụ thuộc vào cường độ đỉnh của ảnh để loại bỏ nhiễu.
# import cv2
# import numpy as np
# img = cv2.imread(r'C:\Project_Work\pythonProject\images\anhnhieu3.jpg')
#
# # lấy kích thước ảnh
# height, width, _ = img.shape
#
# # tạo một ma trận con 3x3 với giá trị 1
# kernel_size = 3
# kernel = np.ones((kernel_size, kernel_size), np.float32)
# #max min
#
# # tạo một ảnh mới với kích thước bằng với ảnh gốc
# filtered_img = np.zeros((height, width, 3), np.float64)
#
# # áp dụng bộ lọc midpoint  với q = -1.5
#
# q = -1.5
# for i in range(1, height - 1):
#     for j in range(1, width - 1):
#         for k in range(3):
#             max_value = np.float64(img[i-1][j-1][k])
#             min_value = np.float64(img[i-1][j-1][k])
#             for x in range(-1, 2):
#                 for y in range(-1, 2):
#                     if np.float64(img[i+x][j+y][k]) > max_value:
#                         max_value = np.float64(img[i+x][j+y][k])
#                     if np.float64(img[i+x][j+y][k]) < min_value:
#                         min_value = np.float64(img[i+x][j+y][k])
#             # tính giá trị mới cho mỗi pixel trong ảnh đầu ra sau khi áp dụng bộ lọc midpoint
#             filtered_img[i][j][k] = (1 + q) * (max_value + min_value) / (2 + 2 * q)
# # hiển thị ảnh gốc và ảnh đã được xử lý
# cv2.imshow('Original Image', img)
# cv2.imshow('Filtered Image', filtered_img.astype(np.uint8))
# cv2.waitKey(0)

# Kỹ thuật #erosion thường được sử dụng để loại bỏ các đối tượng nhỏ hoặc các chi tiết không quan trọng trong ảnh.
# Quá trình này di chuyển một bộ lọc (kernel) qua ảnh và loại bỏ các pixel nằm trong vùng kernel. Quá trình này có thể giúp loại bỏ
# các đối tượng không cần thiết như các đốm, nhiễu và viền của các đối tượng lớn hơn. Kết quả là ảnh sẽ trở nên mịn hơn và các đối
# tượng sẽ được thu nhỏ.
#Kỹ thuật dilation, ngược lại, thường được sử dụng để phóng to và nổi bật các đối tượng trong ảnh. Quá trình này cũng di chuyển
# một bộ lọc (kernel) qua ảnh nhưng lần này là để tăng cường các pixel nằm trong vùng kernel. Quá trình này có thể giúp tăng cường
# đường viền và làm nổi bật các chi tiết trong ảnh. Kết quả là ảnh sẽ trở nên rõ nét hơn và các đối tượng sẽ được phóng to.
# import cv2
# import numpy as np
#
# # đọc ảnh và chuyển sang ảnh xám
# img = cv2.imread(r'C:\Project_Work\pythonProject\images\l.jpg', 0)
#
# if img is not None:
#     # kernel
#     kernel = np.ones((5,5),np.uint8)
#
#     # erosion
#     erosion = cv2.erode(img, kernel, iterations = 1)
#
#     # dilation
#     dilation = cv2.dilate(img, kernel, iterations = 1)
#
#     # hiển thị ảnh gốc và ảnh xử lý
#     cv2.imshow('Original Image', img)
#     cv2.imshow('Erosion', erosion)
#     cv2.imshow('Dilation', dilation)
#
#     cv2.waitKey(0)
#     cv2.destroyAllWindows()
# else:
#     print("Failed to load image.")

#closing
# opening

# Bộ lọc closing: Bộ lọc này được sử dụng để đóng các lỗ trống và loại bỏ các vùng nhỏ bên trong các đối tượng trắng trên ảnh.
# Kết quả là các đối tượng trắng trên ảnh được tăng cường và các đường viền trở nên rõ ràng hơn. Bộ lọc closing thường được sử dụng
# để giảm nhiễu trên các đối tượng trắng trên ảnh và cải thiện chất lượng ảnh.
# Bộ lọc opening: Bộ lọc này được sử dụng để loại bỏ các đối tượng đen nhỏ bên trong các đối tượng trắng trên ảnh và làm giảm kích thước
# của các đối tượng trắng. Kết quả là các đối tượng trắng trên ảnh được tách biệt rõ ràng hơn và các đường viền trở nên mịn màng hơn.
# Bộ lọc opening thường được sử dụng để loại bỏ các nhiễu đen nhỏ hoặc các đối tượng đen bên trong các đối tượng trắng trên ảnh.
# import cv2
import numpy as np

# Đọc ảnh vào
img = cv2.imread(r'C:\Project_Work\pythonProject\images\l.jpg', cv2.IMREAD_GRAYSCALE)

# Khởi tạo kernel cho bộ lọc
kernel = np.ones((5,5),np.uint8)

# Áp dụng bộ lọc closing trên ảnh
closing = cv2.morphologyEx(img, cv2.MORPH_CLOSE, kernel)

# Áp dụng bộ lọc opening trên ảnh
opening = cv2.morphologyEx(img, cv2.MORPH_OPEN, kernel)

# Hiển thị ảnh gốc, ảnh đã xử lý bằng bộ lọc closing và opening
cv2.imshow('Original', img)
cv2.imshow('Closing', closing)
cv2.imshow('Opening', opening)

# Chờ nhấn phím bất kỳ để thoát
cv2.waitKey(0)
cv2.destroyAllWindows()
