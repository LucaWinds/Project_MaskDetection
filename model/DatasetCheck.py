import os

# 기본 경로
base_dir = '..\dataset\data'


train_dir = os.path.join(base_dir, 'train')
validation_dir = os.path.join(base_dir, 'validation')
print(train_dir)
print(validation_dir)

# 훈련에 사용되는 이미지 경로
train_Masked_dir = os.path.join(train_dir, 'Masked')
train_WithoutMasked_dir = os.path.join(train_dir, 'WithoutMasked')
print(train_Masked_dir)
print(train_WithoutMasked_dir)

# 테스트에 사용되는 이미지 경로
validation_Masked_dir = os.path.join(validation_dir, 'Masked')
validation_WithoutMasked_dir = os.path.join(validation_dir, 'WithoutMasked')
print(train_Masked_dir)
print(train_WithoutMasked_dir)

# 파일 이름 확인하기
train_Masked_fnames = os.listdir( train_Masked_dir )
train_WithoutMasked_fnames = os.listdir( train_WithoutMasked_dir )
print(train_Masked_fnames[:5])
print(train_WithoutMasked_fnames[:5])

# 파일 개수 확인하기
print("Total training Masked Images : ", len(os.listdir( train_Masked_dir )))
print("Total training WithoutMasked Images : ", len(os.listdir( train_WithoutMasked_dir )))
print("Total validation Masked Images : ", len(os.listdir( validation_Masked_dir)))
print("Total validation WithoutMasked Images : ", len(os.listdir( validation_WithoutMasked_dir)))


import matplotlib.image as mpimg
import matplotlib.pyplot as plt

nrows, ncols = 4, 4
pic_index = 0

fig = plt.gcf()
fig.set_size_inches(ncols*3, nrows*3)

pic_index+=8

next_Masked_pix = [os.path.join(train_Masked_dir, fname)
                for fname in train_Masked_fnames[ pic_index-8:pic_index]]

next_WithoutMasked_pix = [os.path.join(train_WithoutMasked_dir, fname)
                for fname in train_WithoutMasked_fnames[ pic_index-8:pic_index]]

for i, img_path in enumerate(next_Masked_pix+next_WithoutMasked_pix):
  sp = plt.subplot(nrows, ncols, i + 1)
  sp.axis('Off')
  img = mpimg.imread(img_path)
  plt.imshow(img)

plt.show()