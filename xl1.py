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
# import numpy as np
#
# # Đọc ảnh vào
# img = cv2.imread(r'C:\Project_Work\pythonProject\images\l.jpg', cv2.IMREAD_GRAYSCALE)
#
# # Khởi tạo kernel cho bộ lọc
# kernel = np.ones((5,5),np.uint8)
#
# # Áp dụng bộ lọc closing trên ảnh
# closing = cv2.morphologyEx(img, cv2.MORPH_CLOSE, kernel)
#
# # Áp dụng bộ lọc opening trên ảnh
# opening = cv2.morphologyEx(img, cv2.MORPH_OPEN, kernel)
#
# # Hiển thị ảnh gốc, ảnh đã xử lý bằng bộ lọc closing và opening
# cv2.imshow('Original', img)
# cv2.imshow('Closing', closing)
# cv2.imshow('Opening', opening)
#
# # Chờ nhấn phím bất kỳ để thoát
# cv2.waitKey(0)
# cv2.destroyAllWindows()


# 4/4 buoi chieu
# import cv2
# img = cv2.imread(r'C:\Project_Work\pythonProject\images\tachen1.webp')
#
# # chuyển đổi ảnh sang ảnh grayscale
# gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
#
# # áp dụng phương pháp Otsu để tách đối tượng và nền trong ảnh grayscale
# _, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU) #Phép toán thresholding nhị phân kết hợp với phương pháp Otsu.
# _, thresh1 = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
# _, thresh2 = cv2.threshold(gray, 0, 255, cv2.THRESH_TOZERO + cv2.THRESH_OTSU)
# _, thresh3 = cv2.threshold(gray, 0, 255, cv2.THRESH_TOZERO_INV + cv2.THRESH_OTSU)
# # hiển thị ảnh kết quả
# cv2.imshow('BINARY_INV', thresh)
# cv2.imshow('BINARY', thresh1)
# cv2.imshow('TOZERO', thresh2)
# cv2.imshow('TOZERO_INV', thresh2)
# cv2.imshow('anh goc', img)
# cv2.waitKey(0)

# Mean-shift segmentation là một phương pháp phân vùng ảnh không cần thiết định trước số lượng vùng cần tách ra và cũng không
# cần thiết định trước kích thước hay hình dạng của các vùng này. Trong OpenCV, phương pháp này được cài đặt trong hàm
# cv2.pyrMeanShiftFiltering().
# import cv2
#
# # Đọc ảnh
# img = cv2.imread(r"C:\Project_Work\pythonProject\images\tachen1.webp")
#
# # Chuyển ảnh sang không gian màu L*a*b để làm cho phương pháp mean-shift segmentation hoạt động tốt hơn
# lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
#
# # Áp dụng mean-shift segmentation
# shifted = cv2.pyrMeanShiftFiltering(lab, 20, 45)
#
# # Chuyển ảnh về không gian màu RGB để hiển thị
# result = cv2.cvtColor(shifted, cv2.COLOR_LAB2BGR)
#
# # Hiển thị ảnh kết quả
# cv2.imshow("goc" , img)
# cv2.imshow("Result", result)
# cv2.waitKey(0)
# cv2.destroyAllWindows()

# import cv2
# import sklearn
# from numpy import where
# from sklearn.datasets import make_classification
# from matplotlib import pyplot
# # Đọc ảnh
#
# X, y = make_classification(n_samples=1000, n_features=2, n_informative=2, n_redundant=0, n_clusters_per_class=1, random_state=4)
# for class_value in range(2):
#   row_ix = where(y == class_value)
# pyplot.scatter(X[row_ix, 0], X[row_ix, 1])
# pyplot.show()

# import cv2
# import numpy as np
# from matplotlib import pyplot as plt
#
# # Matplotlib không thể hiển thị hình ảnh trực tiếp trong môi trường phiên bản không đồ họa. Để sửa lỗi này, ta có thể thay đổi
# # cách Matplotlib hiển thị hình ảnh từ "interactive" sang "inline", bằng cách thêm đoạn code
# import matplotlib
# matplotlib.use('Agg')
#
# # Đọc ảnh
# img = cv2.imread(r'C:\Project_Work\pythonProject\images\tachen1.webp')
#
# # Chuyển đổi ảnh thành mảng numpy và chia mỗi giá trị điểm ảnh cho 255 để nằm trong khoảng [0, 1]
# img_array = np.array(img) / 255.0
#
# # Tạo một mảng 2D có số hàng và số cột tương ứng với chiều rộng và chiều cao của ảnh Sau đó,
# # các chỉ số hàng và cột được tạo thành các mảng đối xứng và được xếp chồng lên nhau để tạo thành một mảng 2D chứa tất cả các
# # điểm ảnh trong ảnh.
# h, w = img_array.shape[:2]
# xx, yy = np.mgrid[0:h, 0:w]
# coordinates = np.column_stack((xx.ravel(), yy.ravel()))
#
# # Tạo một danh sách các điểm ảnh trong ảnh bằng cách chuyển đổi mảng 3 chiều chứa giá trị RGB của từng
# # điểm ảnh thành một danh sách các điểm ảnh 2 chiều
# pixels = img_array.reshape((h * w, 3))
#
# # Dòng này tạo ra một đồ thị 2 chiều và trực quan hóa danh sách các điểm ảnh trong ảnh bằng phương thức scatter của đối tượng ax.
# # Màu của từng điểm ảnh được xác định bằng giá trị màu tương ứng với điểm ảnh đó.
# fig, ax = plt.subplots()
# ax.scatter(coordinates[:, 0], coordinates[:, 1], c=pixels)
#
# # Thêm dòng này để lưu hình ảnh vào file PNG thay vì hiển thị trực tiếp
# plt.savefig('output.png')
#
# # Thay vì sử dụng plt.show(), ta sẽ lưu hình ảnh vào file PNG và thông báo cho người dùng biết
# print('Hình ảnh đã được lưu vào file output.png')


import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import KMeans
import warnings
# warnings.filterwarnings("ignore", category=MatplotlibDeprecationWarning): Bỏ qua các cảnh báo liên quan đến thư viện
# MatplotlibDeprecationWarnin
from matplotlib import MatplotlibDeprecationWarning
warnings.filterwarnings("ignore", category=MatplotlibDeprecationWarning)
# Đặt chế độ mặc định cho seaborn để trực quan hóa dữ liệu.
sns.set()

# Load data
raw_data = pd.read_csv(r'C:\Project_Work\pythonProject\images\Countries_exercise.csv')
data = raw_data.copy() # Tạo một bản sao của dữ liệu gốc để thay đổi dữ liệu mà không ảnh hưởng đến dữ liệu gốc.

# Plot data
# Tạo một hình ảnh (figure) mới và trả về một đối tượng hình ảnh (figure) và một đối tượng trục (axis).
fig, ax = plt.subplots()
# Vẽ biểu đồ phân tán của dữ liệu trên trục tọa độ Longitude và Latitude.
ax.scatter(data['Longitude'], data['Latitude'])
# Đặt giới hạn trục x trong khoảng -180 đến 180.
ax.set_xlim(-180, 180)
# Đặt giới hạn trục y trong khoảng -90 đến 90.
ax.set_ylim(-90, 90)
plt.show()

# Apply clustering
# Lấy các giá trị của cột thứ hai và thứ ba trong bảng dữ liệu để sử dụng cho phân cụm.
x = data.iloc[:, 1:3]
# Khởi tạo một đối tượng KMeans với 5 nhóm và 10 lần khởi tạo khác nhau.
kmeans = KMeans(n_clusters=5, n_init=10)

kmeans.fit(x) # Thực hiện thuật toán KMeans để phân cụm dữ liệu.
identified_clusters = kmeans.predict(x)

# Plot clustered data
fig, ax = plt.subplots()
scatter = ax.scatter(data['Longitude'], data['Latitude'], c=identified_clusters, cmap='rainbow')
ax.set_xlim(-180, 180)
ax.set_ylim(-90, 90)
plt.show()


