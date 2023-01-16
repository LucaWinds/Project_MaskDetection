#사용하지 않습니다!!!

import os
import numpy as np
from PIL import Image

image_path_Masked = './Masked/'
image_path_WithoutMasked = './WithoutMasked/'
image_path_IncorrectedMasked = './IncorrectedMasked/'

image_path = image_path_Masked

img_list = os.listdir(image_path) #디렉토리 내 모든 파일 불러오기
print ("img_list: {}".format(img_list))

img_list_np = []


for i in img_list:
    img = Image.open(image_path + i)
    img = img.resize((32,32))
    img_array = np.array(img)
    img_array = img_array.astype(float)
    img_list_np.append(img_array)
    print(i, " 추가 완료 - 구조:", img_array.shape) # 불러온 이미지의 차원 확인 (세로X가로X색)


img_np = np.array(img_list_np) #리스트를 numpy로 변환
print(img_np.shape)

np.save('Masked.npy',img_np)