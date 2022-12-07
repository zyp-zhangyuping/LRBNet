'''
This file is used to save the output image
'''

from platform import platform
import torch.utils.data
from utils.parser import get_parser_with_args
from utils.helpers import get_test_loaders, initialize_metrics
import os
from tqdm import tqdm
import cv2
import numpy as np
np.set_printoptions(threshold=np.inf) 
if not os.path.exists('./output-20221030-xviewalldata-cutmix-2-CONV'):
    os.mkdir('./output-20221030-xviewalldata-cutmix-2-CONV')

parser, metadata = get_parser_with_args()
opt = parser.parse_args()

dev = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
# dev = torch.device( 'cpu')
test_loader = get_test_loaders(opt, batch_size=1)

path = './tmp-20221030-xviewalldata-cutmix-2-CONV-0.001/checkpoint_epoch_40.pt'   # the path of the model
model = torch.load(path)

model.eval()
index_img = 0
test_metrics = initialize_metrics()
# with torch.no_grad():
#     tbar = tqdm(test_loader)
#     for batch_img1, batch_img2, labels in tbar:

#         batch_img1 = batch_img1.float().to(dev)
#         batch_img2 = batch_img2.float().to(dev)
#         labels = labels.long().to(dev)

#         cd_preds = model(batch_img1, batch_img2)

#         cd_preds = cd_preds[-1]
#         _, cd_preds = torch.max(cd_preds, 1)
#         cd_preds = cd_preds.data.cpu().numpy()

#         size = 1024
#         # ####测色效果图
#         b = [[0 for col in range(size)] for row in range(size)]
#         g = [[0 for col in range(size)] for row in range(size)]
#         r = [[0 for col in range(size)] for row in range(size)]

#         plate = cd_preds[0]
#         for i in range(size):
#             for j in range(size):
#                 if plate[i][j] == 0:
#                     b[i][j] = (138)
#                     g[i][j] = (0)
#                     r[i][j] = (1)
#                 elif plate[i][j] == 1:
#                     b[i][j] = (50)
#                     g[i][j] = (179)
#                     r[i][j] = (50)
#                 elif plate[i][j] == 2:
#                     b[i][j] = (189)
#                     g[i][j] = (250)
#                     r[i][j] = (254)
#                 elif plate[i][j] == 3:
#                     b[i][j] = (0)
#                     g[i][j] = (165)
#                     r[i][j] = (253)
#                 elif plate[i][j] == 4:
#                     b[i][j] = (0)
#                     g[i][j] = (41)
#                     r[i][j] = (253)


#         r = np.array(r)
#         g = np.array(g)
#         b = np.array(b)
#         ####彩色效果图输出
#         cd_preds = cv2.merge([b, g, r])
#         # file_path = './output_mixdata_epoch8/' + str(index_img).zfill(5)
#         # cv2.imwrite(file_path + '.png', cd_preds)
#         file_path = './output-20221030-xviewalldata-cutmix-2-CONV-color/' + 'test_' + str(index_img + 1)
#         cv2.imwrite(file_path + '.png', cd_preds)

#         index_img += 1

with torch.no_grad():
    tbar = tqdm(test_loader)
    for batch_img1, batch_img2, labels in tbar:

        batch_img1 = batch_img1.float().to(dev)
        batch_img2 = batch_img2.float().to(dev)
        labels = labels.long().to(dev)
        # print("---------------------------")
        # print(batch_img1.shape)
        cd_preds = model(batch_img1, batch_img2)

        cd_preds = cd_preds[-1]
        _, cd_preds = torch.max(cd_preds, 1)
        cd_preds = cd_preds.data.cpu().numpy()
        # print(cd_preds)
        cd_preds = cd_preds.squeeze() * 50
        # cd_preds = np.expand_dims(cd_preds, 0)
        # print(cd_preds.shape)
        # break
        # file_path = './output_mixdata_epoch8/' + str(index_img).zfill(5)
        # cv2.imwrite(file_path + '.png', cd_preds)
        file_path = './output-20221030-xviewalldata-cutmix-2-CONV/' + 'test_' + str(index_img + 1)
        cv2.imwrite(file_path + '.png', cd_preds)

        index_img += 1