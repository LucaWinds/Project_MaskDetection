from PIL import Image
import os

print(os.listdir('./data/train/Masked/'))

for image in os.listdir('./data/train/Masked/'):
  im = Image.open(image)
  # If is png image
  if im.format == 'PNG':
    # and is not RGBA
    if im.mode != 'RGBA':
      im.convert("RGBA").save(f"{image}2.png")

for image in os.listdir('./data/train/WithoutMasked/'):
  im = Image.open(image)
  # If is png image
  if im.format == 'PNG':
    # and is not RGBA
    if im.mode != 'RGBA':
      im.convert("RGBA").save(f"{image}2.png")

for image in os.listdir('./data/validation/Masked/'):
  im = Image.open(image)
  # If is png image
  if im.format == 'PNG':
    # and is not RGBA
    if im.mode != 'RGBA':
      im.convert("RGBA").save(f"{image}2.png")

for image in os.listdir('./data/validation/WithoutMasked/'):
  im = Image.open(image)
  # If is png image
  if im.format == 'PNG':
    # and is not RGBA
    if im.mode != 'RGBA':
      im.convert("RGBA").save(f"{image}2.png")
