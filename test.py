import cv2
import train
import numpy as np
from PIL import Image
import torch
import datetime



comp = cv2.imread('/media/zhengmana/mn/zero/leftcompletion/01_0_0000000000.png')
comp = comp / 255.0
comp = np.array([comp])
rela = cv2.imread('/media/zhengmana/mn/zero/rightcompletion/01_0_0000000000.png')
rela = rela / 255.0
rela = np.array([rela])
mask = cv2.imread('/media/zhengmana/mn/zero/maskleft/01_0_0000000000.png', 0)
mask = np.array(mask)
offset = cv2.imread('/media/zhengmana/mn/zero/leftdisp/01_0_0000000000.png', 0)

cv2.imwrite('shicha.png',offset)
offset = np.array([offset, offset])

ret, binary = cv2.threshold(mask, 127, 255, cv2.THRESH_BINARY)
_, contours, hierarchy = cv2.findContours(binary, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
# print(contours[0].shape)

x1 = min(contours[0][:, :, 0])[0]   # 10
y1 = min(contours[0][:, :, 1])[0]
x2 = max(contours[0][:, :, 0])[0]
y2 = max(contours[0][:, :, 1])[0]


mask_bbox = np.array([[x1, y1, x2, y2], [x1, y1, x2, y2]], dtype=np.int32)
b_bbox = mask_bbox
comp = comp.transpose(0, 3, 1, 2)
comp = np.tile(comp, (2, 1, 1, 1))
rela = rela.transpose(0, 3, 1, 2)
rela = np.tile(rela, (2, 1, 1, 1))

comp = torch.from_numpy(comp).float().cuda()
rela = torch.from_numpy(rela).float().cuda()
mask_bbox = torch.from_numpy(mask_bbox).float().cuda()
offset = torch.from_numpy(offset).float().cuda()

block_num = 7

starttime = datetime.datetime.now()
loss1, img = train.calc_loss(comp,rela, block_num, mask_bbox, offset)
endtime = datetime.datetime.now()

print (endtime - starttime).seconds
img = img * 255.0
img = cv2.imwrite('./fakeleft.png',img)
# img = cv2.imread('/home/zhengmana/Desktop/stereo image inpainting/fake_left/1.png')
# img = np.array(img, dtype=np.uint8)
# range_w = int((b_bbox[0][2] - b_bbox[0][0]) / block_num)
# range_h = int((b_bbox[0][3] - b_bbox[0][1]) / block_num)
# x1 = b_bbox[0][0]
# y1 = b_bbox[0][1]
# for i in range(1, block_num+1):
#     for j in range(1, block_num+1):
#         cv2.rectangle(img, (x1, y1), (x1 + i * range_w, y1 + j * range_h), (0, 255, 0), thickness=2)
# cv2.imwrite('zuotu.png', img)
print(loss1.cpu().numpy())



