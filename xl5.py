# import cv2
# import numpy as np
#
# # đọc ảnh và chuyển sang grayscale
# dog_img = cv2.imread(r"C:\Project_Work\Pycharm\Baitap4\bt10_4\src\anhconcho.jpg")
# dog_gray = cv2.cvtColor(dog_img, cv2.COLOR_BGR2GRAY)
#
# cat_img = cv2.imread(r"C:\Project_Work\Pycharm\Baitap4\bt10_4\src\anhconmeo.jpg")
# cat_gray = cv2.cvtColor(cat_img, cv2.COLOR_BGR2GRAY)
#
# bird_img = cv2.imread(r"C:\Project_Work\Pycharm\Baitap4\bt10_4\src\anhconchim.jpg")
# bird_gray = cv2.cvtColor(bird_img, cv2.COLOR_BGR2GRAY)
#
# # áp dụng giải thuật K láng giềng gần nhất
# K = 5 # số láng giềng gần nhất
#
# dog_result = cv2.Laplacian(dog_gray, cv2.CV_32F)
# dog_result = cv2.convertScaleAbs(dog_result)
# for i in range(K):
#     _, binary = cv2.threshold(dog_result, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
#     dog_result = cv2.bitwise_and(dog_gray, binary)
#
# cat_result = cv2.Laplacian(cat_gray, cv2.CV_32F)
# cat_result = cv2.convertScaleAbs(cat_result)
# for i in range(K):
#     _, binary = cv2.threshold(cat_result, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
#     cat_result = cv2.bitwise_and(cat_gray, binary)
#
# bird_result = cv2.Laplacian(bird_gray, cv2.CV_32F)
# bird_result = cv2.convertScaleAbs(bird_result)
# for i in range(K):
#     _, binary = cv2.threshold(bird_result, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
#     bird_result = cv2.bitwise_and(bird_gray, binary)
#
# # phân đoạn ảnh và hiển thị kết quả
# dog_label = np.zeros((dog_img.shape[0], dog_img.shape[1], 3), dtype=np.uint8)
# dog_label[dog_result > 0] = [0, 0, 255] # màu đỏ
#
# cat_label = np.zeros((cat_img.shape[0], cat_img.shape[1], 3), dtype=np.uint8)
# cat_label[cat_result > 0] = [0, 255, 0] # màu xanh lá
#
# bird_label = np.zeros((bird_img.shape[0], bird_img.shape[1], 3), dtype=np.uint8)
# bird_label[bird_result > 0] = [255, 0, 0] # màu xanh dương
# cv2.imshow("Dog Result", dog_label)
# cv2.imshow("Cat Result", cat_label)
# cv2.imshow("Bird Result", bird_label)
# cv2.waitKey(0)
# cv2.destroyAllWindows()