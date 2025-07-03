from PIL import Image
import numpy as np
import imageio

# 1. 读原始 trimap
mask = imageio.imread('dataset/annotations/trimaps/Abyssinian_1.png')  # e.g. shape = (360,500)
binary_pet = (mask == 1).astype(np.uint8)  # 0/1 二值

# 2. 乘 255 → 0/255 灰度
pet_gray = (binary_pet * 255).astype(np.uint8)  # 数值范围是 0 或 255

# 3. 直接用 PIL 保存成 PNG，保持原始 (360,500) 不变
Image.fromarray(pet_gray, mode='L')\
     .save('inference_results/unet/Abyssinian_1_GT.png')