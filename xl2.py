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


