# viết chương trình xử lý ảnh bằng pycharm cv2.
# - Tích chập giữa f(M1*N1) và nhân h(M2*N2) có thể tạo ra các ma trận có kích thước như sau, tùy thuộc vào kiểu chập.
# - Giữ nguyên kích thước: M1 *N1 (same convolution)
# -Tăng kích thước: (M1 +M2 -1)*(N1+N2-1) (full convolution)
# -Giảm kích thước: (M1-M2) + 1)*(N1-N2+1) (valid convolution)

#1/25 LÀ LOW PASS FILLTER THI

# import cv2
# import numpy as np

# img = cv2.imread(r"C:\Users\Admin\PycharmProjects\xulyanh\source\me.jpg",)

# kernel = np.array([[0, 0, 0],
#                     [0, 1, 0],
#                     [0, 0, 0]], np.float32)

# tạo kernel bỘ LỌC GAUSS
# kernel = np.array([[0,0,0,5,0,0,0],
#                    [0,5,18,32,18,5,0],
#                    [0,18,64,100,64,18,0],
#                    [5,32,100,100,100,32,5],
#                    [0,18,64,256,64,18,0],
#                    [0,5,18,32,18,5,0],
#                    [0,0,0,5,0,0,0]], np.float32)
#
# # Xác định bộ lọc
# filter_size = 3
# h = np.ones((filter_size, filter_size), np.float32) / (filter_size * filter_size)
#
# # - Giữ nguyên kích thước:
# result_same = cv2.filter2D(img, -1, h)
# # -Tăng kích thước:
# result_full = cv2.filter2D(img, -1, h)
# # -Giảm kích thước:
# result_valid = cv2.filter2D(img, -1, h)
#
# # Kết hợp hình ảnh với kernel
# result_kernel = cv2.filter2D(img, -1, kernel)

#show
# cv2.imshow("Anh goc", img)
# cv2.imshow("1 (Same Convolution)", result_same)
# cv2.imshow("2 (Full Convolution)", result_full)
# cv2.imshow("3 (Valid Convolution)", result_valid)

# save source
# cv2.imwrite(r"C:\Users\Admin\PycharmProjects\xulyanh\source\nopbai\Anhgocme.jpg", img)
# cv2.imwrite(r"C:\Users\Admin\PycharmProjects\xulyanh\source\nopbai\result_same.jpg", result_same)
# cv2.imwrite(r"C:\Users\Admin\PycharmProjects\xulyanh\source\nopbai\result_full.jpg", result_full)
# cv2.imwrite(r"C:\Users\Admin\PycharmProjects\xulyanh\source\nopbai\result_valid.jpg", result_valid)
#
# cv2.waitKey(0)
# cv2.destroyAllWindows()


# python median filter, blur filter picture

# import cv2
# import numpy as np
#
# img = cv2.imread(r'C:\Users\Admin\PycharmProjects\xulyanh\source\flower.jpg')
#
# # Xử lý bằng Median
# median = cv2.medianBlur(img, 5)
#
# # Xử lý bằng Blur
# blur = cv2.GaussianBlur(img, (5, 5), 0)
#
#
# # cv2.imshow('Original Image', img)
# # cv2.imshow('Median Filter', median)
# # cv2.imshow('Blur Filter', blur)
#
# cv2.imwrite(r"C:\Users\Admin\PycharmProjects\xulyanh\source\nopbai\Anhgoc.jpg", img)
# cv2.imwrite(r"C:\Users\Admin\PycharmProjects\xulyanh\source\nopbai\median.jpg", median)
# cv2.imwrite(r"C:\Users\Admin\PycharmProjects\xulyanh\source\nopbai\blur.jpg", blur)
#
# cv2.waitKey(0)
# cv2.destroyAllWindows()

# Bộ lọc kuwahara
import numpy as np
import cv2

def kuwahara_filter(img, kernel_size):
    # Khởi tạo một mảng trống để lưu kết quả
    out = np.zeros_like(img)
    # Lấy kích thước hình ảnh
    height, width = img.shape[:2]
    # Tạo các kernel
    kernels = []
    for i in range(4):
        kernel = np.zeros((kernel_size, kernel_size), np.float32)
        for x in range(kernel_size):
            for y in range(kernel_size):
                dx = x - int(kernel_size / 2)
                dy = y - int(kernel_size / 2)
                if i == 0:
                    if dx >= 0 and dy >= 0:
                        kernel[x, y] = 1.0 / (dx + dy + 1)
                elif i == 1:
                    if dx >= 0 and dy <= 0:
                        kernel[x, y] = 1.0 / (dx - dy + 1)
                elif i == 2:
                    if dx <= 0 and dy <= 0:
                        kernel[x, y] = 1.0 / (-dx - dy + 1)
                elif i == 3:
                    if dx <= 0 and dy >= 0:
                        kernel[x, y] = 1.0 / (-dx + dy + 1)
        kernel = kernel / np.sum(kernel)
        kernels.append(kernel)
    # Áp dụng bộ lọc
    for x in range(int(kernel_size / 2), height - int(kernel_size / 2)):
        for y in range(int(kernel_size / 2), width - int(kernel_size / 2)):
            patch = img[x - int(kernel_size / 2):x + int(kernel_size / 2) + 1,
                        y - int(kernel_size / 2):y + int(kernel_size / 2) + 1]
            stats = []
            for i in range(4):
                stat = cv2.filter2D(patch, -1, kernels[i])
                stats.append(stat)
            stats = np.asarray(stats)
            stat_mean = np.mean(stats, axis=0)
            stat_variance = np.var(stats, axis=0)
            min_variance = np.min(stat_variance)
            index = np.argmin(stat_variance)
            out[x, y] = stat_mean[index]
    return out
