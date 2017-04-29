import ikrdata
import numpy as np
from PIL import Image

image_size = 80

i_train, labels = ikrdata.get_filenames_extension(ikrdata.TRAIN_DIR, ikrdata.GLOB_FACES)
i_dev, labels = ikrdata.get_filenames_extension(ikrdata.DEV_DIR, ikrdata.GLOB_FACES)
images = i_train + i_dev

image_number = len(images)

avgface = np.zeros((image_size, image_size), np.float)

for im in images:
    pil_img = Image.open(im).convert('L')
    img = np.array(pil_img, dtype=np.float)
    avgface = avgface + img/image_number

avgface = np.array(np.round(avgface), dtype=np.uint8)
out = Image.fromarray(avgface, mode='L')
out.save("data/avgface.png")