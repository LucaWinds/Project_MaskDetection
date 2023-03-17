import tensorflow as tf
import numpy as np
import keras.utils as image

def reimg(path):
    img1 = image.load_img(path, target_size = (150, 150))
    x = image.img_to_array(img1)
#    x = np.expand_dims(x, axis=0)
    x = x.reshape(-1, 150, 150, 3)
#    images = np.vstack([x])/255.
    images = x.astype(np.float32)/255.0

    return images

model = tf.keras.models.load_model('../model/maskmodel.h5')

#마스크 씀
img = reimg('./ex/ex1.png')
res = model.predict(img)
print(img)
print(res)

if res < 0.5:
    print("Masked")
else:
    print("WithoutMasked")


#마스크 안씀
img = reimg('./ex/ex2.jpg')
res = model.predict(img)

print(res)

if res < 0.5:
    print("Masked")
else:
    print("WithoutMasked")

#마스크 씀
img = reimg('./ex/ex3.png')
res = model.predict(img)

print(res)

if res < 0.5:
    print("Masked")
else:
    print("WithoutMasked")

#마스크 안씀
img = reimg('./ex/ex4.png')
res = model.predict(img)

print(res)

if res < 0.5:
    print("Masked")
else:
    print("WithoutMasked")

# 나의 사진 ㅠㅠ

#마스크 씀
img = reimg('./ex/www.png')
res = model.predict(img)

print(res)

if res < 0.5:
    print("Masked")
else:
    print("WithoutMasked")

#마스크 안씀
img = reimg('./ex/www2.png')
res = model.predict(img)

print(res)

if res < 0.5:
    print("Masked")
else:
    print("WithoutMasked")