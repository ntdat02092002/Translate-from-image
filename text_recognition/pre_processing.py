from PIL import Image, ImageEnhance
import numpy as np
import cv2


def run(img):
    # Rescaling
    # Tính toán kích thước mới dựa trên tỉ lệ chiều rộng của ảnh
    scale_percent = 200 / img.shape[0]
    width = int(img.shape[1] * scale_percent)
    height = int(img.shape[0] * scale_percent)
    dim = (width, height)
    # Scale ảnh
    resized_img = cv2.resize(img, dim, interpolation=cv2.INTER_AREA)


    dilation_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3,3))
    dilated_img = cv2.dilate(resized_img, dilation_kernel, iterations=1)

    gray_img = cv2.cvtColor(dilated_img, cv2.COLOR_BGR2GRAY)
    # Làm rõ ảnh bằng histogram equalization
    equalized_img = cv2.equalizeHist(gray_img)
    # Chuyển đổi lại sang ảnh màu
    enhanced_img = cv2.cvtColor(equalized_img, cv2.COLOR_GRAY2BGR)

    cv2.imshow("word", enhanced_img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    return enhanced_img
