# import cv2
#
# img = cv2.imread(r"C:\Users\Admin\PycharmProjects\pythonProject\hi.jpg")
#
# cv2.imshow("anh goc: ", img)
#
# [w, h] = img.shape[:2]
# for i in range(w):
#     for j in range(h):
#         img[i][j] = 255 - img[i][j]
#
# cv2.imshow(' anh am ban: ', img)
# cv2.waitKey(0)


# import cv2
# import numpy as np
#
# img = cv2.imread(r"C:\Users\Admin\PycharmProjects\pythonProject\trangden.jpg")
#
# cv2.imshow("anh goc: ", img)
# cv2.waitKey(4)
#
# [w, h] = img.shape[:2]
#
# max_value = np.max(img)
# c = 100 / np.log(1 + max_value)
#
# for i in range(w):
#     for j in range(h):
#         r = img[i][j]
#         s = c * np.log(1 + r)
#         img[i][j] = s
#
# img2 = cv2.imwrite("anhbiendoilog.jpg", img)


# hieu cinh gamma cong thuc Power-law: s =c.r**y
# import cv2
# import numpy as np
#
# img = cv2.imread(r'C:\Users\Admin\PycharmProjects\pythonProject\gammalow2.jpg')
#
# def adjust_gamma(image, gamma=1.0):
#     invGamma = 1.0 / gamma
#     table = (np.power(np.linspace(0, 1, 256), invGamma) * 255).astype('uint8')
#     return cv2.LUT(image, table)
#
# gamma = 1.5
# adjusted = adjust_gamma(img, gamma)
#
# cv2.imshow("Original", img)
# cv2.imshow("Adjusted", adjusted)
# cv2.waitKey(0)
# cv2.destroyAllWindows()

#chuyển ảnh màu thành ảnh XÁM = CÁCH LÁY TRUNG BÌNH 3 MÀU
# from PIL import Image
#
# img = Image.open(r"C:\Users\Admin\PycharmProjects\pythonProject\hi.jpg")
#
# # Chuyen doi sang anh xam
# gray_img = Image.new("L", img.size)  # Tao anh xam moi cung kich thuoc voi anh ban dau
#
# for x in range(img.width):
#     for y in range(img.height):
#         r, g, b = img.getpixel((x, y))  # Lay gia tri pixel (r, g, b)
#         gray_value = int((r + g + b) / 3)  # Tinh gia tri trung binh cua 3 mau
#         gray_img.putpixel((x, y), gray_value)  # Gan gia tri xam cho moi pixel
#
# gray_img.show("output_image.jpg")

#can bang histogram

import numpy as np
import cv2 as cv
from matplotlib import pyplot as plt

img = cv.imread(r'C:\Users\Admin\PycharmProjects\pythonProject\hi2.jpg', cv.IMREAD_GRAYSCALE)
assert img is not None, "file could not be read, check with os.path.exists()"
hist,bins = np.histogram(img.flatten(),256,[0,256])
cdf = hist.cumsum()
cdf_normalized = cdf * float(hist.max()) / cdf.max()
plt.plot(cdf_normalized, color = 'b')
plt.hist(img.flatten(),256,[0,256], color = 'r')
plt.xlim([0,256])
plt.legend(('cdf','histogram'), loc = 'upper left')
cv.imshow("Original", img)
plt.show()
cv.waitKey(0)
cv.destroyAllWindows()












