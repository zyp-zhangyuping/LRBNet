# # Kaiyu Li
# # https://github.com/likyoo


# import torch.nn as nn
# import torch

# class conv_block_nested(nn.Module):
#     def __init__(self, in_ch, mid_ch, out_ch):
#         super(conv_block_nested, self).__init__()
#         self.activation = nn.ReLU(inplace=True)
#         self.conv1 = nn.Conv2d(in_ch, mid_ch, kernel_size=3, padding=1, bias=True)
#         self.bn1 = nn.BatchNorm2d(mid_ch)
#         self.conv2 = nn.Conv2d(mid_ch, out_ch, kernel_size=3, padding=1, bias=True)
#         self.bn2 = nn.BatchNorm2d(out_ch)

#     def forward(self, x):
#         x = self.conv1(x)
#         identity = x
#         x = self.bn1(x)
#         x = self.activation(x)

#         x = self.conv2(x)
#         x = self.bn2(x)
#         output = self.activation(x + identity)
#         return output


# class up(nn.Module):
#     def __init__(self, in_ch, bilinear=False):
#         super(up, self).__init__()

#         if bilinear:
#             self.up = nn.Upsample(scale_factor=2,
#                                   mode='bilinear',
#                                   align_corners=True)
#         else:
#             self.up = nn.ConvTranspose2d(in_ch, in_ch, 2, stride=2)

#     def forward(self, x):

#         x = self.up(x)
#         return x


# class ChannelAttention(nn.Module):
#     def __init__(self, in_channels, ratio = 16):
#         super(ChannelAttention, self).__init__()
#         self.avg_pool = nn.AdaptiveAvgPool2d(1)
#         self.max_pool = nn.AdaptiveMaxPool2d(1)
#         self.fc1 = nn.Conv2d(in_channels,in_channels//ratio,1,bias=False)
#         self.relu1 = nn.ReLU()
#         self.fc2 = nn.Conv2d(in_channels//ratio, in_channels,1,bias=False)
#         self.sigmod = nn.Sigmoid()
#     def forward(self,x):
#         avg_out = self.fc2(self.relu1(self.fc1(self.avg_pool(x))))
#         max_out = self.fc2(self.relu1(self.fc1(self.max_pool(x))))
#         out = avg_out + max_out
#         return self.sigmod(out)



# class SNUNet_ECAM(nn.Module):
#     # SNUNet-CD with ECAM
#     def __init__(self, in_ch=3, out_ch=2):
#         super(SNUNet_ECAM, self).__init__()
#         torch.nn.Module.dump_patches = True
#         n1 = 32     # the initial number of channels of feature map
#         filters = [n1, n1 * 2, n1 * 4, n1 * 8, n1 * 16]

#         self.pool = nn.MaxPool2d(kernel_size=2, stride=2)

#         self.conv0_0 = conv_block_nested(in_ch, filters[0], filters[0])
#         self.conv1_0 = conv_block_nested(filters[0], filters[1], filters[1])
#         self.Up1_0 = up(filters[1])
#         self.conv2_0 = conv_block_nested(filters[1], filters[2], filters[2])
#         self.Up2_0 = up(filters[2])
#         self.conv3_0 = conv_block_nested(filters[2], filters[3], filters[3])
#         self.Up3_0 = up(filters[3])
#         self.conv4_0 = conv_block_nested(filters[3], filters[4], filters[4])
#         self.Up4_0 = up(filters[4])

#         self.conv0_1 = conv_block_nested(filters[0] * 2 + filters[1], filters[0], filters[0])
#         self.conv1_1 = conv_block_nested(filters[1] * 2 + filters[2], filters[1], filters[1])
#         self.Up1_1 = up(filters[1])
#         self.conv2_1 = conv_block_nested(filters[2] * 2 + filters[3], filters[2], filters[2])
#         self.Up2_1 = up(filters[2])
#         self.conv3_1 = conv_block_nested(filters[3] * 2 + filters[4], filters[3], filters[3])
#         self.Up3_1 = up(filters[3])

#         self.conv0_2 = conv_block_nested(filters[0] * 3 + filters[1], filters[0], filters[0])
#         self.conv1_2 = conv_block_nested(filters[1] * 3 + filters[2], filters[1], filters[1])
#         self.Up1_2 = up(filters[1])
#         self.conv2_2 = conv_block_nested(filters[2] * 3 + filters[3], filters[2], filters[2])
#         self.Up2_2 = up(filters[2])

#         self.conv0_3 = conv_block_nested(filters[0] * 4 + filters[1], filters[0], filters[0])
#         self.conv1_3 = conv_block_nested(filters[1] * 4 + filters[2], filters[1], filters[1])
#         self.Up1_3 = up(filters[1])

#         self.conv0_4 = conv_block_nested(filters[0] * 5 + filters[1], filters[0], filters[0])

#         self.ca = ChannelAttention(filters[0] * 4, ratio=16)
#         # self.sa = SpatialAttention(3)
#         self.ca1 = ChannelAttention(filters[0], ratio=16 // 4)
#         # self.ca = CBAM(filters[0] * 4, 3,ratio=16)
#         # self.ca1 = CBAM(filters[0], 3,ratio=16 // 4)

#         self.conv_final = nn.Conv2d(filters[0] * 4, out_ch, kernel_size=1)
#         # self.CBAM1 = CBAM(filters[1],3)
#         # self.CBAM2 = CBAM(filters[2],3)
#         # self.CBAM3 = CBAM(filters[3],3)
#         # self.CBAM4 = CBAM(filters[4],3)


#         for m in self.modules():
#             if isinstance(m, nn.Conv2d):
#                 nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
#             elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
#                 nn.init.constant_(m.weight, 1)
#                 nn.init.constant_(m.bias, 0)


#     def forward(self, xA, xB):
#         '''xA'''
#         x0_0A = self.conv0_0(xA)
#         x1_0A = self.conv1_0(self.pool(x0_0A))
#         x2_0A = self.conv2_0(self.pool(x1_0A))
#         x3_0A = self.conv3_0(self.pool(x2_0A))
#         # x4_0A = self.conv4_0(self.pool(x3_0A))
#         '''xB'''
#         x0_0B = self.conv0_0(xB)
#         x1_0B = self.conv1_0(self.pool(x0_0B))
#         x2_0B = self.conv2_0(self.pool(x1_0B))
#         x3_0B = self.conv3_0(self.pool(x2_0B))
#         x4_0B = self.conv4_0(self.pool(x3_0B))


#         # A1 =  self.CBAM1(torch.cat([x0_0A, x0_0B],1))
#         # x0_1 = self.conv0_1(torch.cat([A1, self.Up1_0(x1_0B)], 1))
#         x0_1 = self.conv0_1(torch.cat([x0_0A, x0_0B, self.Up1_0(x1_0B)], 1))

#         # A2 =  self.CBAM2(torch.cat([x1_0A, x1_0B],1))
#         # x1_1 = self.conv1_1(torch.cat([A2, self.Up2_0(x2_0B)], 1))
#         x1_1 = self.conv1_1(torch.cat([x1_0A, x1_0B, self.Up2_0(x2_0B)], 1))
#         # x0_2 = self.conv0_2(torch.cat([A1, x0_1, self.Up1_1(x1_1)], 1))
#         x0_2 = self.conv0_2(torch.cat([x0_0A, x0_0B, x0_1, self.Up1_1(x1_1)], 1))

#         # A3 =  self.CBAM3(torch.cat([x2_0A, x2_0B],1))
#         # x2_1 = self.conv2_1(torch.cat([A3, self.Up3_0(x3_0B)], 1))
#         x2_1 = self.conv2_1(torch.cat([x2_0A, x2_0B, self.Up3_0(x3_0B)], 1))
#         # x1_2 = self.conv1_2(torch.cat([A2, x1_1, self.Up2_1(x2_1)], 1))
#         x1_2 = self.conv1_2(torch.cat([x1_0A, x1_0B, x1_1, self.Up2_1(x2_1)], 1))
#         # x0_3 = self.conv0_3(torch.cat([A1, x0_1, x0_2, self.Up1_2(x1_2)], 1))
#         x0_3 = self.conv0_3(torch.cat([x0_0A, x0_0B, x0_1, x0_2, self.Up1_2(x1_2)], 1))
        
#         # A4 =  self.CBAM4(torch.cat([x3_0A, x3_0B],1))
#         # x3_1 = self.conv3_1(torch.cat([A4, self.Up4_0(x4_0B)], 1))
#         x3_1 = self.conv3_1(torch.cat([x3_0A, x3_0B, self.Up4_0(x4_0B)], 1))
#         # x2_2 = self.conv2_2(torch.cat([A3, x2_1, self.Up3_1(x3_1)], 1))
#         x2_2 = self.conv2_2(torch.cat([x2_0A, x2_0B, x2_1, self.Up3_1(x3_1)], 1))
#         # x1_3 = self.conv1_3(torch.cat([A2, x1_1, x1_2, self.Up2_2(x2_2)], 1))
#         x1_3 = self.conv1_3(torch.cat([x1_0A, x1_0B, x1_1, x1_2, self.Up2_2(x2_2)], 1))
#         x0_4 = self.conv0_4(torch.cat([x0_0A, x0_0B, x0_1, x0_2, x0_3, self.Up1_3(x1_3)], 1))

#         out = torch.cat([x0_1, x0_2, x0_3, x0_4], 1)

#         intra = torch.sum(torch.stack((x0_1, x0_2, x0_3, x0_4)), dim=0)
#         ca1 = self.ca1(intra)
#         out = self.ca(out) * (out + ca1.repeat(1, 4, 1, 1))
#         # out = self.sa(out) * (out + ca1.repeat(1, 4, 1, 1))
#         out = self.conv_final(out)

#         return (out, )
#         # return out, 1
#         # return out

# # ECAM 模块  通道注意模块

# # class SNUNet_ECAM(nn.Module):
# #     # SNUNet-CD with ECAM
# #     def __init__(self, in_ch=3, out_ch=2):
# #         super(SNUNet_ECAM, self).__init__()
# #         torch.nn.Module.dump_patches = True
# #         n1 = 32     # the initial number of channels of feature map 特征图的初始通道数
# #         filters = [n1, n1 * 2, n1 * 4, n1 * 8, n1 * 16] 

# #         self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
# #         self.conv0_0 = conv_block_nested(in_ch, filters[0], filters[0])  
# #         self.conv1_0 = conv_block_nested(filters[0], filters[1], filters[1])  # 融合的过程
# #         # self.cbam1_0b = CBAM(filters[1],3)
# #         self.Up1_0 = up(filters[1])
# #         self.conv2_0 = conv_block_nested(filters[1], filters[2], filters[2])
# #         # self.cbam2_0b = CBAM(filters[2],3)
# #         self.Up2_0 = up(filters[2])
# #         self.conv3_0 = conv_block_nested(filters[2], filters[3], filters[3])
# #         # self.cbam3_0b = CBAM(filters[3],3)
# #         self.Up3_0 = up(filters[3])
# #         self.conv4_0 = conv_block_nested(filters[3], filters[4], filters[4])
# #         # self.cbam4_0b = CBAM(filters[4],3)
# #         self.Up4_0 = up(filters[4])

# #         self.conv0_1 = conv_block_nested(filters[0] * 2 + filters[1], filters[0], filters[0])
# #         self.cbam0_1 = CBAM(filters[0], 3)
# #         self.conv1_1 = conv_block_nested(filters[1] * 2 + filters[2], filters[1], filters[1])
# #         self.cbam1_1 = CBAM(filters[1],3)
# #         self.Up1_1 = up(filters[1])
# #         self.conv2_1 = conv_block_nested(filters[2] * 2 + filters[3], filters[2], filters[2])
# #         self.cbam2_1 = CBAM(filters[2],3)
# #         self.Up2_1 = up(filters[2])
# #         self.conv3_1 = conv_block_nested(filters[3] * 2 + filters[4], filters[3], filters[3])
# #         self.cbam3_1 = CBAM(filters[3],3)
# #         self.Up3_1 = up(filters[3])

# #         self.conv0_2 = conv_block_nested(filters[0] * 3 + filters[1], filters[0], filters[0])
# #         self.cbam0_2 = CBAM(filters[0], 3)
# #         self.conv1_2 = conv_block_nested(filters[1] * 3 + filters[2], filters[1], filters[1])
# #         self.cbam1_2 = CBAM(filters[1],3)
# #         self.Up1_2 = up(filters[1])
# #         self.conv2_2 = conv_block_nested(filters[2] * 3 + filters[3], filters[2], filters[2])
# #         self.cbam2_2 = CBAM(filters[2],3)
# #         self.Up2_2 = up(filters[2])

# #         self.conv0_3 = conv_block_nested(filters[0] * 4 + filters[1], filters[0], filters[0])
# #         self.cbam0_3 = CBAM(filters[0], 3)
# #         self.conv1_3 = conv_block_nested(filters[1] * 4 + filters[2], filters[1], filters[1])
# #         self.cbam1_3 = CBAM(filters[1],3)
# #         self.Up1_3 = up(filters[1])

# #         self.conv0_4 = conv_block_nested(filters[0] * 5 + filters[1], filters[0], filters[0])
# #         self.cbam0_4 = CBAM(filters[0], 3)

# #         self.ca = ChannelAttention(filters[0] * 4, ratio=16)
# #         #定义SAM
# #         # self.sa = SpatialAttention(3)
# #         self.ca1 = ChannelAttention(filters[0], ratio=16 // 4)

# #         self.conv_final = nn.Conv2d(filters[0] * 4, out_ch, kernel_size=1)
# #         # self.final_change = nn.Sequential(
# #         #         ConvRelu(filters[0] * 4, 32),
# #         #         nn.Conv2d(32, out_ch, kernel_size=1)
# #         #     )

# #         for m in self.modules():
# #             if isinstance(m, nn.Conv2d):
# #                 nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
# #             elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
# #                 nn.init.constant_(m.weight, 1)
# #                 nn.init.constant_(m.bias, 0)


# #     def forward(self, xA, xB):
# #         '''xA'''
# #         x0_0A = self.conv0_0(xA)
# #         x1_0A = self.conv1_0(self.pool(x0_0A))
# #         x2_0A = self.conv2_0(self.pool(x1_0A))
# #         x3_0A = self.conv3_0(self.pool(x2_0A))
# #         # x4_0A = self.conv4_0(self.pool(x3_0A))
# #         '''xB'''
# #         x0_0B = self.conv0_0(xB)
# #         x1_0B = self.conv1_0(self.pool(x0_0B))
# #         x2_0B = self.conv2_0(self.pool(x1_0B))
# #         x3_0B = self.conv3_0(self.pool(x2_0B))
# #         x4_0B = self.conv4_0(self.pool(x3_0B))
        
# #         # # x0_1 = self.conv0_1(torch.cat([x0_0A, x0_0B, self.Up1_0(x1_0B)], 1)) #torch.cat 是将两个张量拼接在一起 拼接
# #         # # # x0_1 = self.cbam1_0b(self.conv0_1(torch.cat([x0_0A, x0_0B, self.Up1_0(x1_0B)], 1))) #torch.cat 是将两个张量拼接在一起 拼接
# #         # # x1_1 = self.conv1_1(torch.cat([x1_0A, x1_0B, self.Up2_0(x2_0B)], 1))
# #         # # x0_2 = self.conv0_2(torch.cat([x0_0A, x0_0B, x0_1, self.Up1_1(self.cbam1_1(x1_1))], 1))

        
# #         # # x2_1 = self.conv2_1(torch.cat([x2_0A, x2_0B, self.Up3_0(x3_0B)], 1))
# #         # # x1_2 = self.conv1_2(torch.cat([x1_0A, x1_0B, x1_1, self.Up2_1(self.cbam2_1(x2_1))], 1))
# #         # # x0_3 = self.conv0_3(torch.cat([x0_0A, x0_0B, x0_1, x0_2, self.Up1_2(self.cbam1_2(x1_2))], 1))

# #         # # x3_1 = self.conv3_1(torch.cat([x3_0A, x3_0B, self.Up4_0(self.cbam4_0b(x4_0B))], 1))
# #         # # x2_2 = self.conv2_2(torch.cat([x2_0A, x2_0B, x2_1, self.Up3_1(self.cbam3_1(x3_1))], 1))
# #         # # x1_3 = self.conv1_3(torch.cat([x1_0A, x1_0B, x1_1, x1_2, self.Up2_2(self.cbam2_2(x2_2))], 1))
# #         # # x0_4 = self.conv0_4(torch.cat([x0_0A, x0_0B, x0_1, x0_2, x0_3, self.Up1_3(self.cbam1_3(x1_3))], 1))
        
# #         # x0_1 = self.conv0_1(torch.cat([x0_0A, x0_0B, self.Up1_0(x1_0B)], 1)) #torch.cat 是将两个张量拼接在一起 拼接
# #         # x0_1 = self.cbam0_1(x0_1)
# #         # x1_1 = self.conv1_1(torch.cat([x1_0A, x1_0B, self.Up2_0(x2_0B)], 1))
# #         # x1_1 = self.cbam1_1(x1_1)
# #         # x0_2 = self.conv0_2(torch.cat([x0_0A, x0_0B, x0_1, self.Up1_1((x1_1))], 1))
# #         # x0_2 = self.cbam0_2(x0_2)
        
# #         # x2_1 = self.conv2_1(torch.cat([x2_0A, x2_0B, self.Up3_0(x3_0B)], 1))
# #         # x2_1 = self.cbam2_1(x2_1)
# #         # x1_2 = self.conv1_2(torch.cat([x1_0A, x1_0B, x1_1, self.Up2_1(self.cbam2_1(x2_1))], 1))
# #         # x1_2 = self.cbam1_2(x1_2)
# #         # x0_3 = self.conv0_3(torch.cat([x0_0A, x0_0B, x0_1, x0_2, self.Up1_2(self.cbam1_2(x1_2))], 1))
# #         # x0_3 = self.cbam0_3(x0_3)

# #         # x3_1 = self.conv3_1(torch.cat([x3_0A, x3_0B, self.Up4_0(x4_0B)], 1))
# #         # x3_1 = self.cbam3_1(x3_1)
# #         # x2_2 = self.conv2_2(torch.cat([x2_0A, x2_0B, x2_1, self.Up3_1(self.cbam3_1(x3_1))], 1))
# #         # x2_2 = self.cbam2_2(x2_2)
# #         # x1_3 = self.conv1_3(torch.cat([x1_0A, x1_0B, x1_1, x1_2, self.Up2_2(self.cbam2_2(x2_2))], 1))
# #         # x1_3 = self.cbam1_3(x1_3)
# #         # x0_4 = self.conv0_4(torch.cat([x0_0A, x0_0B, x0_1, x0_2, x0_3, self.Up1_3(self.cbam1_3(x1_3))], 1))
# #         # x0_4 = self.cbam0_4(x0_4)
# #         # out = torch.cat([x0_1, x0_2, x0_3, x0_4], 1)
        
# #         # intra = torch.sum(torch.stack((x0_1, x0_2, x0_3, x0_4)), dim=0)
# #         # ca1 = self.ca1(intra)
# #         # #后边加SAM
# #         # # ca1 = self.sa(ca1)
# #         # out = self.ca(out) * (out + ca1.repeat(1, 4, 1, 1))
# #         # # print("===========================",out)
# #         # # out = self.sa(self.ca(out)) * (out + ca1.repeat(1, 4, 1, 1))
# #         # out = self.conv_final(out)
# #         # # out = self.final_change(out)
# #         # return (out, )

# #         #############################################################earthquake#############################################
# #         x0_1 = self.conv0_1(torch.cat([x0_0A, x0_0B, self.Up1_0(x1_0B)], 1)) #torch.cat 是将两个张量拼接在一起 拼接
# #         x1_1 = self.conv1_1(torch.cat([x1_0A, x1_0B, self.Up2_0(x2_0B)], 1))
# #         # print(x1_1.shape)
# #         # x1_1 = self.cbam1_1(x1_1)
# #         # print("1")
# #         # print(x1_1.shape)
# #         x0_2 = self.conv0_2(torch.cat([x0_0A, x0_0B, x0_1, self.Up1_1((x1_1))], 1))
# #         # print("2")

        
# #         x2_1 = self.conv2_1(torch.cat([x2_0A, x2_0B, self.Up3_0(x3_0B)], 1))
# #         # x2_1 = self.cbam2_1(x2_1)
# #         # x1_2 = self.conv1_2(torch.cat([x1_0A, x1_0B, x1_1, self.Up2_1(self.cbam2_1(x2_1))], 1))
# #         x1_2 = self.conv1_2(torch.cat([x1_0A, x1_0B, x1_1, self.Up2_1(x2_1)], 1))
# #         # x1_2 = self.cbam1_2(x1_2)
# #         # x0_3 = self.conv0_3(torch.cat([x0_0A, x0_0B, x0_1, x0_2, self.Up1_2(self.cbam1_2(x1_2))], 1))
# #         x0_3 = self.conv0_3(torch.cat([x0_0A, x0_0B, x0_1, x0_2, self.Up1_2(x1_2)], 1))

# #         x3_1 = self.conv3_1(torch.cat([x3_0A, x3_0B, self.Up4_0(x4_0B)], 1))
# #         # x3_1 = self.cbam3_1(x3_1)
# #         # x2_2 = self.conv2_2(torch.cat([x2_0A, x2_0B, x2_1, self.Up3_1(self.cbam3_1(x3_1))], 1))
# #         x2_2 = self.conv2_2(torch.cat([x2_0A, x2_0B, x2_1, self.Up3_1(x3_1)], 1))
# #         # x2_2 = self.cbam2_2(x2_2)
# #         # x1_3 = self.conv1_3(torch.cat([x1_0A, x1_0B, x1_1, x1_2, self.Up2_2(self.cbam2_2(x2_2))], 1))
# #         x1_3 = self.conv1_3(torch.cat([x1_0A, x1_0B, x1_1, x1_2, self.Up2_2(x2_2)], 1))
# #         # x1_3 = self.cbam1_3(x1_3)
# #         # x0_4 = self.conv0_4(torch.cat([x0_0A, x0_0B, x0_1, x0_2, x0_3, self.Up1_3(self.cbam1_3(x1_3))], 1))
# #         x0_4 = self.conv0_4(torch.cat([x0_0A, x0_0B, x0_1, x0_2, x0_3, self.Up1_3(x1_3)], 1))
        
# #         out = torch.cat([x0_1, x0_2, x0_3, x0_4], 1)
        
# #         intra = torch.sum(torch.stack((x0_1, x0_2, x0_3, x0_4)), dim=0)
# #         ca1 = self.ca1(intra)
# #         out = self.ca(out) * (out + ca1.repeat(1, 4, 1, 1))
# #         out = self.conv_final(out)

# #         return (out, )


# class Siam_NestedUNet_Conc(nn.Module):
#     # SNUNet-CD without Attention
#     def __init__(self, in_ch=3, out_ch=2):
#         super(Siam_NestedUNet_Conc, self).__init__()
#         torch.nn.Module.dump_patches = True
#         n1 = 32     # the initial number of channels of feature map
#         filters = [n1, n1 * 2, n1 * 4, n1 * 8, n1 * 16]

#         self.pool = nn.MaxPool2d(kernel_size=2, stride=2)

#         self.conv0_0 = conv_block_nested(in_ch, filters[0], filters[0])
#         self.conv1_0 = conv_block_nested(filters[0], filters[1], filters[1])
#         self.Up1_0 = up(filters[1])
#         self.conv2_0 = conv_block_nested(filters[1], filters[2], filters[2])
#         self.Up2_0 = up(filters[2])
#         self.conv3_0 = conv_block_nested(filters[2], filters[3], filters[3])
#         self.Up3_0 = up(filters[3])
#         self.conv4_0 = conv_block_nested(filters[3], filters[4], filters[4])
#         self.Up4_0 = up(filters[4])

#         self.conv0_1 = conv_block_nested(filters[0] * 2 + filters[1], filters[0], filters[0])
#         self.conv1_1 = conv_block_nested(filters[1] * 2 + filters[2], filters[1], filters[1])
#         self.Up1_1 = up(filters[1])
#         self.conv2_1 = conv_block_nested(filters[2] * 2 + filters[3], filters[2], filters[2])
#         self.Up2_1 = up(filters[2])
#         self.conv3_1 = conv_block_nested(filters[3] * 2 + filters[4], filters[3], filters[3])
#         self.Up3_1 = up(filters[3])

#         self.conv0_2 = conv_block_nested(filters[0] * 3 + filters[1], filters[0], filters[0])
#         self.conv1_2 = conv_block_nested(filters[1] * 3 + filters[2], filters[1], filters[1])
#         self.Up1_2 = up(filters[1])
#         self.conv2_2 = conv_block_nested(filters[2] * 3 + filters[3], filters[2], filters[2])
#         self.Up2_2 = up(filters[2])

#         self.conv0_3 = conv_block_nested(filters[0] * 4 + filters[1], filters[0], filters[0])
#         self.conv1_3 = conv_block_nested(filters[1] * 4 + filters[2], filters[1], filters[1])
#         self.Up1_3 = up(filters[1])

#         self.conv0_4 = conv_block_nested(filters[0] * 5 + filters[1], filters[0], filters[0])

#         self.final1 = nn.Conv2d(filters[0], out_ch, kernel_size=1)
#         self.final2 = nn.Conv2d(filters[0], out_ch, kernel_size=1)
#         self.final3 = nn.Conv2d(filters[0], out_ch, kernel_size=1)
#         self.final4 = nn.Conv2d(filters[0], out_ch, kernel_size=1)
#         self.conv_final = nn.Conv2d(out_ch * 4, out_ch, kernel_size=1)

#         for m in self.modules():
#             if isinstance(m, nn.Conv2d):
#                 nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
#             elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
#                 nn.init.constant_(m.weight, 1)
#                 nn.init.constant_(m.bias, 0)


#     def forward(self, xA, xB):
#         '''xA'''
#         x0_0A = self.conv0_0(xA)
#         x1_0A = self.conv1_0(self.pool(x0_0A))
#         x2_0A = self.conv2_0(self.pool(x1_0A))
#         x3_0A = self.conv3_0(self.pool(x2_0A))
#         # x4_0A = self.conv4_0(self.pool(x3_0A))
#         '''xB'''
#         x0_0B = self.conv0_0(xB)
#         x1_0B = self.conv1_0(self.pool(x0_0B))
#         x2_0B = self.conv2_0(self.pool(x1_0B))
#         x3_0B = self.conv3_0(self.pool(x2_0B))
#         x4_0B = self.conv4_0(self.pool(x3_0B))

#         x0_1 = self.conv0_1(torch.cat([x0_0A, x0_0B, self.Up1_0(x1_0B)], 1))
#         x1_1 = self.conv1_1(torch.cat([x1_0A, x1_0B, self.Up2_0(x2_0B)], 1))
#         x0_2 = self.conv0_2(torch.cat([x0_0A, x0_0B, x0_1, self.Up1_1(x1_1)], 1))


#         x2_1 = self.conv2_1(torch.cat([x2_0A, x2_0B, self.Up3_0(x3_0B)], 1))
#         x1_2 = self.conv1_2(torch.cat([x1_0A, x1_0B, x1_1, self.Up2_1(x2_1)], 1))
#         x0_3 = self.conv0_3(torch.cat([x0_0A, x0_0B, x0_1, x0_2, self.Up1_2(x1_2)], 1))

#         x3_1 = self.conv3_1(torch.cat([x3_0A, x3_0B, self.Up4_0(x4_0B)], 1))
#         x2_2 = self.conv2_2(torch.cat([x2_0A, x2_0B, x2_1, self.Up3_1(x3_1)], 1))
#         x1_3 = self.conv1_3(torch.cat([x1_0A, x1_0B, x1_1, x1_2, self.Up2_2(x2_2)], 1))
#         x0_4 = self.conv0_4(torch.cat([x0_0A, x0_0B, x0_1, x0_2, x0_3, self.Up1_3(x1_3)], 1))


#         output1 = self.final1(x0_1)
#         output2 = self.final2(x0_2)
#         output3 = self.final3(x0_3)
#         output4 = self.final4(x0_4)
#         output = self.conv_final(torch.cat([output1, output2, output3, output4], 1))
#         return (output1, output2, output3, output4, output)


# # class SpatialAttention(nn.Module):
# #     def __init__(self, kernel_size=3):
# #         super(SpatialAttention, self).__init__()

# #         assert kernel_size in (3, 7), 'kernel size must be 3 or 7'
# #         padding = 3 if kernel_size == 7 else 1

# #         self.conv1 = nn.Conv2d(2, 1, kernel_size, padding=padding, bias=False)  # 7,3     3,1
# #         self.sigmoid = nn.Sigmoid()

# #     def forward(self, x):
# #         avg_out = torch.mean(x, dim=1, keepdim=True)
# #         max_out, _ = torch.max(x, dim=1, keepdim=True)
# #         x = torch.cat([avg_out, max_out], dim=1)
# #         x = self.conv1(x)
# #         return self.sigmoid(x)


# # class CBAM(nn.Module):
# #     def __init__(self, in_channels, kernel_size, ratio = 16):
# #         super(CBAM, self).__init__()
# #         self.cam = ChannelAttention(in_channels, ratio=ratio)
# #         self.sam = SpatialAttention(kernel_size)

# #     def forward(self, x):
# #         # x_cam = self.cam(x)
# #         # # x_camx = x_cam *x
# #         # x_sam = self.sam(x_cam)
# #         # # x_cbam = torch.cat([x_cam * x, x_sam * x], 1)
# #         # x_cbam = x * x_sam
# #         # # print(x_cbam.shape)
# #         # # print("--------------------------------------------")
# #         # out = x * self.cam(x)
# #         # x_cbam = out * self.sam(out)
# #         # return x_cbam
# #         x_cam = self.cam(x)
# #         out_cam = x_cam * x
# #         x_sam = self.sam(x)
# #         out_sam = x_sam * x
# #         output = out_cam + out_sam
# #         return output


# #####################################################################
# ####ghost改进多分类########################
# # Kaiyu Li
# # https://github.com/likyoo
# #

import torch.nn as nn
import torch

import math

class GhostModule(nn.Module):
    """
    GhostNet: More Features from Cheap Operations By Kai Han, Yunhe Wang, Qi Tian, Jianyuan Guo, Chunjing Xu, Chang Xu.
    """

    def __init__(self, inp, oup, kernel_size=1, ratio=2, dw_size=3, stride=1, relu=True):
        super(GhostModule, self).__init__()
        self.oup = oup
        init_channels = math.ceil(oup / ratio)
        new_channels = init_channels * (ratio - 1)

        self.primary_conv = nn.Sequential(
            nn.Conv2d(inp, init_channels, kernel_size, stride, kernel_size // 2, bias=False), #1x1卷积
            nn.BatchNorm2d(init_channels),
            nn.ReLU(inplace=True) if relu else nn.Sequential(),
        )

        self.cheap_operation = nn.Sequential(
            nn.Conv2d(init_channels, new_channels, dw_size, 1, dw_size // 2, groups=init_channels, bias=False), #DW卷积
            nn.BatchNorm2d(new_channels),
            nn.ReLU(inplace=True) if relu else nn.Sequential(),
        )
        # self.resnet_attention = ResNetAttention(oup, oup)
        self.eca_attention = eca_block(oup, oup)

    def forward(self, x):
        x1 = self.primary_conv(x)
        x2 = self.cheap_operation(x1)
        out = torch.cat([x1, x2], dim=1)
        # scale = self.resnet_attention(out[:, :self.oup, :, :])
        scale = self.eca_attention(out[:, :self.oup, :, :])
        # return out[:, :self.oup, :, :]
        return scale

class eca_block(nn.Module):
    def __init__(self, channel, b=1, gamma=2):
        super(eca_block, self).__init__()
        kernel_size = int(abs((math.log(channel, 2) + b) / gamma))
        kernel_size = kernel_size if kernel_size % 2 else kernel_size + 1
        
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.conv = nn.Conv1d(1, 1, kernel_size=kernel_size, padding=(kernel_size - 1) // 2, bias=False) 
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        y = self.avg_pool(x)
        y = self.conv(y.squeeze(-1).transpose(-1, -2)).transpose(-1, -2).unsqueeze(-1)
        y = self.sigmoid(y)
        return x * y.expand_as(x)


class conv3x3(nn.Module):
    # 全部使用华为提出的ghostmodel来替换传统卷积
    def __init__(self, in_, out):
        super().__init__()
        self.conv = GhostModule(in_, out)  # 改用了华为提出的ghostnet模块

    def forward(self, x):
        x = self.conv(x)
        return x

# # #添加eca注意力机制
class conv_block_nested(nn.Module):
    def __init__(self, in_ch, mid_ch, out_ch):
        super(conv_block_nested, self).__init__()
        self.activation = nn.ReLU(inplace=True)
        self.conv1 = conv3x3(in_ch, mid_ch)
        self.bn1 = nn.BatchNorm2d(mid_ch)
        self.conv2 = conv3x3(mid_ch, out_ch)
        self.bn2 = nn.BatchNorm2d(out_ch)
        # 更改部分，加入ecaAttention
        self.eca_attention = eca_block(out_ch, out_ch)
        # self.SpatialAttention = SpatialAttention(7)

    def forward(self, x):
        x = self.conv1(x)
        identity = x
        x = self.bn1(x)
        x = self.activation(x)

        x = self.conv2(x)
        x = self.bn2(x)
        scale = self.eca_attention(x)
        # scale = self.SpatialAttention(x)

        # output = self.activation(x + identity)
        output = self.activation(scale + identity)
        return output

class ConvRelu(nn.Module):
    def __init__(self, in_, out):
        super().__init__()
        self.conv = conv3x3(in_, out)
        self.activation = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.conv(x)
        x = self.activation(x)
        return x


# class conv_block_nested(nn.Module):
#     def __init__(self, in_ch, mid_ch, out_ch):
#         super(conv_block_nested, self).__init__()
#         self.activation = nn.ReLU(inplace=True)
#         self.conv1 = nn.Conv2d(in_ch, mid_ch, kernel_size=3, padding=1, bias=True)
#         self.bn1 = nn.BatchNorm2d(mid_ch)
#         self.conv2 = nn.Conv2d(mid_ch, out_ch, kernel_size=3, padding=1, bias=True)
#         self.bn2 = nn.BatchNorm2d(out_ch)

#     def forward(self, x):
#         x = self.conv1(x)
#         identity = x
#         x = self.bn1(x)
#         x = self.activation(x)

#         x = self.conv2(x)
#         x = self.bn2(x)
#         output = self.activation(x + identity)
#         return output

class up(nn.Module):
    def __init__(self, in_ch, bilinear=False):
        super(up, self).__init__()

        if bilinear:
            self.up = nn.Upsample(scale_factor=2,
                                  mode='bilinear',
                                  align_corners=True)
        else:
            self.up = nn.ConvTranspose2d(in_ch, in_ch, 2, stride=2)

    def forward(self, x):

        x = self.up(x)
        return x


class ChannelAttention(nn.Module):
    def __init__(self, in_channels, ratio = 16):
        super(ChannelAttention, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        self.fc1 = nn.Conv2d(in_channels,in_channels//ratio,1,bias=False)
        self.relu1 = nn.ReLU()
        self.fc2 = nn.Conv2d(in_channels//ratio, in_channels,1,bias=False)
        self.sigmod = nn.Sigmoid()
    def forward(self,x):
        avg_out = self.fc2(self.relu1(self.fc1(self.avg_pool(x))))
        max_out = self.fc2(self.relu1(self.fc1(self.max_pool(x))))
        out = avg_out + max_out
        return self.sigmod(out)


class Channel_Attention_Module_Conv(nn.Module):
    def __init__(self, channels, gamma = 2, b = 1):
        super(Channel_Attention_Module_Conv, self).__init__()
        kernel_size = int(abs((math.log(channels, 2) + b) / gamma))
        kernel_size = kernel_size if kernel_size % 2 else kernel_size + 1
        self.avg_pooling = nn.AdaptiveAvgPool2d(1)
        self.max_pooling = nn.AdaptiveMaxPool2d(1)
        self.conv = nn.Conv1d(1, 1, kernel_size = kernel_size, padding = (kernel_size - 1) // 2, bias = False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_x = self.avg_pooling(x)
        max_x = self.max_pooling(x)
        avg_out = self.conv(avg_x.squeeze(-1).transpose(-1, -2)).transpose(-1, -2).unsqueeze(-1)
        max_out = self.conv(max_x.squeeze(-1).transpose(-1, -2)).transpose(-1, -2).unsqueeze(-1)
        v = self.sigmoid(avg_out + max_out)
        return x * v


# # # #源主干网络
class SNUNet_ECAM(nn.Module):
    # SNUNet-CD with ECAM
    def __init__(self, in_ch=3, out_ch=2):
        super(SNUNet_ECAM, self).__init__()
        torch.nn.Module.dump_patches = True
        n1 = 32     # the initial number of channels of feature map
        # filters = [n1, n1 * 2, n1 * 4, n1 * 8, n1 * 16]
        filters = [n1, n1 * 2, n1 * 4, n1 * 8, n1 * 16, n1 * 32]

        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)

        self.conv0_0 = conv_block_nested(in_ch, filters[0], filters[0])
        self.conv1_0 = conv_block_nested(filters[0], filters[1], filters[1])
        self.Up1_0 = up(filters[1])
        self.conv2_0 = conv_block_nested(filters[1], filters[2], filters[2])
        self.Up2_0 = up(filters[2])
        self.conv3_0 = conv_block_nested(filters[2], filters[3], filters[3])
        self.Up3_0 = up(filters[3])
        self.conv4_0 = conv_block_nested(filters[3], filters[4], filters[4])
        self.Up4_0 = up(filters[4])

        #加深一层
        # self.conv5_0 = conv_block_nested(filters[4], filters[5], filters[5])
        # self.Up5_0 = up(filters[5])


        self.conv0_1 = conv_block_nested(filters[0] * 2 + filters[1], filters[0], filters[0])
        self.conv1_1 = conv_block_nested(filters[1] * 2 + filters[2], filters[1], filters[1])
        self.Up1_1 = up(filters[1])
        self.conv2_1 = conv_block_nested(filters[2] * 2 + filters[3], filters[2], filters[2])
        self.Up2_1 = up(filters[2])
        self.conv3_1 = conv_block_nested(filters[3] * 2 + filters[4], filters[3], filters[3])
        self.Up3_1 = up(filters[3])
        # #加深一层
        # self.conv4_1 = conv_block_nested(filters[4] * 2 + filters[5], filters[4], filters[4])
        # self.Up4_1 = up(filters[4])

        self.conv0_2 = conv_block_nested(filters[0] * 3 + filters[1], filters[0], filters[0])
        self.conv1_2 = conv_block_nested(filters[1] * 3 + filters[2], filters[1], filters[1])
        self.Up1_2 = up(filters[1])
        self.conv2_2 = conv_block_nested(filters[2] * 3 + filters[3], filters[2], filters[2])
        self.Up2_2 = up(filters[2])
        #加深一层
        # self.conv3_2 = conv_block_nested(filters[3] * 3 + filters[4], filters[3], filters[3])
        # self.Up3_2 = up(filters[3])

        self.conv0_3 = conv_block_nested(filters[0] * 4 + filters[1], filters[0], filters[0])
        self.conv1_3 = conv_block_nested(filters[1] * 4 + filters[2], filters[1], filters[1])
        self.Up1_3 = up(filters[1])
        #加深一层
        # self.conv2_3 = conv_block_nested(filters[2] * 4 + filters[3], filters[2], filters[2])
        # self.Up2_3 = up(filters[2])

        self.conv0_4 = conv_block_nested(filters[0] * 5 + filters[1], filters[0], filters[0])
        # #加深一层
        # self.conv1_4 = conv_block_nested(filters[1] * 5 + filters[2], filters[1], filters[1])
        # self.Up1_4 = up(filters[1])

        #加深一层
        # self.conv0_5 = conv_block_nested(filters[0] * 6 + filters[1], filters[0], filters[0])


        self.ca = ChannelAttention(filters[0] * 4, ratio=16)
        # self.ca = Channel_Attention_Module_Conv(filters[0] * 4)
        self.sa = SpatialAttention(7)

        self.cbam = CBAM(filters[0] * 4, 7, ratio=16)
        
        self.ca1 = ChannelAttention(filters[0], ratio=16 // 4)
        # self.ca1 = Channel_Attention_Module_Conv(filters[0])
        self.cbam1 = CBAM(filters[0], 7, ratio=16 // 4)
        self.sa1 = SpatialAttention(7)

        # self.conv_final = nn.Conv2d(filters[0] * 5, out_ch, kernel_size=1)
        self.final_change = nn.Sequential(
                ConvRelu(filters[0] * 4, 32),
                nn.Conv2d(32, out_ch, kernel_size=1)
            )

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                # nn.GroupNorm(num_groups=4, num_channels=out_ch)
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)


    def forward(self, xA, xB):
        '''xA'''
        x0_0A = self.conv0_0(xA)
        x1_0A = self.conv1_0(self.pool(x0_0A))
        x2_0A = self.conv2_0(self.pool(x1_0A))
        x3_0A = self.conv3_0(self.pool(x2_0A))
        #加深一层
        # x4_0A = self.conv4_0(self.pool(x3_0A))

        '''xB'''
        x0_0B = self.conv0_0(xB)
        x1_0B = self.conv1_0(self.pool(x0_0B))
        x2_0B = self.conv2_0(self.pool(x1_0B))
        x3_0B = self.conv3_0(self.pool(x2_0B))
        x4_0B = self.conv4_0(self.pool(x3_0B))
        #加深一层
        # x5_0B = self.conv5_0(self.pool(x4_0B))

        x0_1 = self.conv0_1(torch.cat([x0_0A, x0_0B, self.Up1_0(x1_0B)], 1))
        x1_1 = self.conv1_1(torch.cat([x1_0A, x1_0B, self.Up2_0(x2_0B)], 1))
        x0_2 = self.conv0_2(torch.cat([x0_0A, x0_0B, x0_1, self.Up1_1(x1_1)], 1))

        
        x2_1 = self.conv2_1(torch.cat([x2_0A, x2_0B, self.Up3_0(x3_0B)], 1))
        x1_2 = self.conv1_2(torch.cat([x1_0A, x1_0B, x1_1, self.Up2_1(x2_1)], 1))
        x0_3 = self.conv0_3(torch.cat([x0_0A, x0_0B, x0_1, x0_2, self.Up1_2(x1_2)], 1))

        x3_1 = self.conv3_1(torch.cat([x3_0A, x3_0B, self.Up4_0(x4_0B)], 1))
        x2_2 = self.conv2_2(torch.cat([x2_0A, x2_0B, x2_1, self.Up3_1(x3_1)], 1))
        x1_3 = self.conv1_3(torch.cat([x1_0A, x1_0B, x1_1, x1_2, self.Up2_2(x2_2)], 1))
        x0_4 = self.conv0_4(torch.cat([x0_0A, x0_0B, x0_1, x0_2, x0_3, self.Up1_3(x1_3)], 1))

        #加深一层
        # x4_1 = self.conv4_1(torch.cat([x4_0A, x4_0B, self.Up5_0(x5_0B)], 1))
        # x3_2 = self.conv3_2(torch.cat([x3_0A, x3_0B, x3_1, self.Up4_1(x4_1)], 1))
        # x2_3 = self.conv2_3(torch.cat([x2_0A, x2_0B, x2_1, x2_2, self.Up3_2(x3_2)], 1))
        # x1_4 = self.conv1_4(torch.cat([x1_0A, x1_0B, x1_1, x1_2, x1_3, self.Up2_3(x2_3)], 1))
        # x0_5 = self.conv0_5(torch.cat([x0_0A, x0_0B, x0_1, x0_2, x0_3, x0_4, self.Up1_4(x1_4)], 1))

        out = torch.cat([x0_1, x0_2, x0_3, x0_4], 1)
        intra = torch.sum(torch.stack((x0_1, x0_2, x0_3, x0_4)), dim=0)

        # out = torch.cat([x0_1, x0_2, x0_3, x0_4, x0_5], 1)
        # intra = torch.sum(torch.stack((x0_1, x0_2, x0_3, x0_4, x0_5)), dim=0)

        #更换一个cam
        # ca1 = self.ca1(intra)
        # out = self.ca(out) * (out + ca1.repeat(1, 5, 1, 1))
        # sa1 = self.sa1(intra)
        # out = self.sa(out) * (out + sa1.repeat(1, 1, 1, 1))

        #更换一个cbam
        # out = self.cbam(out) * (out + ca1.repeat(1, 4, 1, 1))

        #更换两个cbam
        cbam1 = self.cbam1(intra)
        out = self.cbam(out) * (out + cbam1.repeat(1, 4, 1, 1)) 

        # out = self.conv_final(out)
        out = self.final_change(out)

        return (out, )
        # return out


##############################ghost-samMLANet#################################
# class SNUNet_ECAM(nn.Module):
# #     # SNUNet-CD with ECAM
#     def __init__(self, in_ch=3, out_ch=2):
#         super(SNUNet_ECAM, self).__init__()
#         torch.nn.Module.dump_patches = True
#         n1 = 32     # the initial number of channels of feature map 特征图的初始通道数
#         filters = [n1, n1 * 2, n1 * 4, n1 * 8, n1 * 16] 

#         self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
#         self.conv0_0 = conv_block_nested(in_ch, filters[0], filters[0])  
#         self.conv1_0 = conv_block_nested(filters[0], filters[1], filters[1])  # 融合的过程
#         self.Up1_0 = up(filters[1])
#         self.conv2_0 = conv_block_nested(filters[1], filters[2], filters[2])
#         self.Up2_0 = up(filters[2])
#         self.conv3_0 = conv_block_nested(filters[2], filters[3], filters[3])
#         self.Up3_0 = up(filters[3])
#         self.conv4_0 = conv_block_nested(filters[3], filters[4], filters[4])
#         self.Up4_0 = up(filters[4])

#         self.conv0_1 = conv_block_nested(filters[0] * 2 + filters[1], filters[0], filters[0])
#         self.cbam0_1 = SpatialAttention(3)

#         self.conv1_1 = conv_block_nested(filters[1] * 2 + filters[2], filters[1], filters[1])
#         self.cbam1_1 = SpatialAttention(3)
#         self.Up1_1 = up(filters[1])
#         self.conv2_1 = conv_block_nested(filters[2] * 2 + filters[3], filters[2], filters[2])
#         self.cbam2_1 = SpatialAttention(3)
#         self.Up2_1 = up(filters[2])
#         self.conv3_1 = conv_block_nested(filters[3] * 2 + filters[4], filters[3], filters[3])
#         self.cbam3_1 = SpatialAttention(3)
#         self.Up3_1 = up(filters[3])

#         self.conv0_2 = conv_block_nested(filters[0] * 3 + filters[1], filters[0], filters[0])
#         self.cbam0_2 = SpatialAttention(3)
#         self.conv1_2 = conv_block_nested(filters[1] * 3 + filters[2], filters[1], filters[1])
#         self.cbam1_2 = SpatialAttention(3)
#         self.Up1_2 = up(filters[1])
#         self.conv2_2 = conv_block_nested(filters[2] * 3 + filters[3], filters[2], filters[2])
#         self.cbam2_2 = SpatialAttention(3)
#         self.Up2_2 = up(filters[2])

#         self.conv0_3 = conv_block_nested(filters[0] * 4 + filters[1], filters[0], filters[0])
#         self.cbam0_3 = SpatialAttention(3)
#         self.conv1_3 = conv_block_nested(filters[1] * 4 + filters[2], filters[1], filters[1])
#         self.cbam1_3 = SpatialAttention(3)
#         self.Up1_3 = up(filters[1])

#         self.conv0_4 = conv_block_nested(filters[0] * 5 + filters[1], filters[0], filters[0])
#         self.ca = ChannelAttention(filters[0] * 4, ratio=16)
#         self.sa = SpatialAttention(7)

#         self.cbam = CBAM(filters[0] * 4, 7, ratio=16)
        
#         self.ca1 = ChannelAttention(filters[0], ratio=16 // 4)
#         self.cbam1 = CBAM(filters[0], 7, ratio=16 // 4)
#         self.sa1 = SpatialAttention(7)
#         self.conv_final = nn.Conv2d(filters[0] * 4, out_ch, kernel_size=1)
#         # self.final_change = nn.Sequential(
#         #         ConvRelu(filters[0] * 4, 32),
#         #         nn.Conv2d(32, out_ch, kernel_size=1)
#         #     )

        

#         for m in self.modules():
#             if isinstance(m, nn.Conv2d):
#                 nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
#             elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
#                 nn.init.constant_(m.weight, 1)
#                 nn.init.constant_(m.bias, 0)


#     def forward(self, xA, xB):
#         '''xA'''
#         x0_0A = self.conv0_0(xA)
#         x1_0A = self.conv1_0(self.pool(x0_0A))
#         x2_0A = self.conv2_0(self.pool(x1_0A))
#         x3_0A = self.conv3_0(self.pool(x2_0A))
#         # x4_0A = self.conv4_0(self.pool(x3_0A))
#         '''xB'''
#         x0_0B = self.conv0_0(xB)
#         x1_0B = self.conv1_0(self.pool(x0_0B))
#         x2_0B = self.conv2_0(self.pool(x1_0B))
#         x3_0B = self.conv3_0(self.pool(x2_0B))
#         x4_0B = self.conv4_0(self.pool(x3_0B))

#         #############################################################earthquake#############################################
#         x0_1 = self.conv0_1(torch.cat([x0_0A, x0_0B, self.Up1_0(x1_0B)], 1)) #torch.cat 是将两个张量拼接在一起 拼接
#         x1_1 = self.conv1_1(torch.cat([x1_0A, x1_0B, self.Up2_0(x2_0B)], 1))
#         # print(x1_1.shape)
#         x1_1 = self.cbam1_1(x1_1)
#         # print("1")
#         # print(x1_1.shape)
#         x0_2 = self.conv0_2(torch.cat([x0_0A, x0_0B, x0_1, self.Up1_1((x1_1))], 1))
#         # print("2")

        
#         x2_1 = self.conv2_1(torch.cat([x2_0A, x2_0B, self.Up3_0(x3_0B)], 1))
#         x2_1 = self.cbam2_1(x2_1)
#         # x1_2 = self.conv1_2(torch.cat([x1_0A, x1_0B, x1_1, self.Up2_1(self.cbam2_1(x2_1))], 1))
#         x1_2 = self.conv1_2(torch.cat([x1_0A, x1_0B, x1_1, self.Up2_1(x2_1)], 1))
#         x1_2 = self.cbam1_2(x1_2)
#         # x0_3 = self.conv0_3(torch.cat([x0_0A, x0_0B, x0_1, x0_2, self.Up1_2(self.cbam1_2(x1_2))], 1))
#         x0_3 = self.conv0_3(torch.cat([x0_0A, x0_0B, x0_1, x0_2, self.Up1_2(x1_2)], 1))

#         x3_1 = self.conv3_1(torch.cat([x3_0A, x3_0B, self.Up4_0(x4_0B)], 1))
#         x3_1 = self.cbam3_1(x3_1)
#         # x2_2 = self.conv2_2(torch.cat([x2_0A, x2_0B, x2_1, self.Up3_1(self.cbam3_1(x3_1))], 1))
#         x2_2 = self.conv2_2(torch.cat([x2_0A, x2_0B, x2_1, self.Up3_1(x3_1)], 1))
#         x2_2 = self.cbam2_2(x2_2)
#         # x1_3 = self.conv1_3(torch.cat([x1_0A, x1_0B, x1_1, x1_2, self.Up2_2(self.cbam2_2(x2_2))], 1))
#         x1_3 = self.conv1_3(torch.cat([x1_0A, x1_0B, x1_1, x1_2, self.Up2_2(x2_2)], 1))
#         x1_3 = self.cbam1_3(x1_3)
#         # x0_4 = self.conv0_4(torch.cat([x0_0A, x0_0B, x0_1, x0_2, x0_3, self.Up1_3(self.cbam1_3(x1_3))], 1))
#         x0_4 = self.conv0_4(torch.cat([x0_0A, x0_0B, x0_1, x0_2, x0_3, self.Up1_3(x1_3)], 1))
        
#         out = torch.cat([x0_1, x0_2, x0_3, x0_4], 1)
        
#         intra = torch.sum(torch.stack((x0_1, x0_2, x0_3, x0_4)), dim=0)

#         #更换一个cam
# #         # ca1 = self.ca1(intra)
#         sa1 = self.sa(intra)
# #         # # out = self.ca(out) * (out + ca1.repeat(1, 4, 1, 1))
#         out = self.sa(out) * (out + sa1.repeat(1, 4, 1, 1))

#         # ca1 = self.ca1(intra)
#         # out = self.ca(out) * (out + ca1.repeat(1, 4, 1, 1))
#         out = self.conv_final(out)

#         return (out, )


####################################Siam_NestedUNet_Conc##################################################
class Siam_NestedUNet_Conc(nn.Module):
    # SNUNet-CD without Attention
    def __init__(self, in_ch=3, out_ch=2):
        super(Siam_NestedUNet_Conc, self).__init__()
        torch.nn.Module.dump_patches = True
        n1 = 32     # the initial number of channels of feature map
        filters = [n1, n1 * 2, n1 * 4, n1 * 8, n1 * 16]

        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)

        self.conv0_0 = conv_block_nested(in_ch, filters[0], filters[0])
        self.conv1_0 = conv_block_nested(filters[0], filters[1], filters[1])
        self.Up1_0 = up(filters[1])
        self.conv2_0 = conv_block_nested(filters[1], filters[2], filters[2])
        self.Up2_0 = up(filters[2])
        self.conv3_0 = conv_block_nested(filters[2], filters[3], filters[3])
        self.Up3_0 = up(filters[3])
        self.conv4_0 = conv_block_nested(filters[3], filters[4], filters[4])
        self.Up4_0 = up(filters[4])

        self.conv0_1 = conv_block_nested(filters[0] * 2 + filters[1], filters[0], filters[0])
        self.conv1_1 = conv_block_nested(filters[1] * 2 + filters[2], filters[1], filters[1])
        self.Up1_1 = up(filters[1])
        self.conv2_1 = conv_block_nested(filters[2] * 2 + filters[3], filters[2], filters[2])
        self.Up2_1 = up(filters[2])
        self.conv3_1 = conv_block_nested(filters[3] * 2 + filters[4], filters[3], filters[3])
        self.Up3_1 = up(filters[3])

        self.conv0_2 = conv_block_nested(filters[0] * 3 + filters[1], filters[0], filters[0])
        self.conv1_2 = conv_block_nested(filters[1] * 3 + filters[2], filters[1], filters[1])
        self.Up1_2 = up(filters[1])
        self.conv2_2 = conv_block_nested(filters[2] * 3 + filters[3], filters[2], filters[2])
        self.Up2_2 = up(filters[2])

        self.conv0_3 = conv_block_nested(filters[0] * 4 + filters[1], filters[0], filters[0])
        self.conv1_3 = conv_block_nested(filters[1] * 4 + filters[2], filters[1], filters[1])
        self.Up1_3 = up(filters[1])

        self.conv0_4 = conv_block_nested(filters[0] * 5 + filters[1], filters[0], filters[0])

        self.final1 = nn.Conv2d(filters[0], out_ch, kernel_size=1)
        self.final2 = nn.Conv2d(filters[0], out_ch, kernel_size=1)
        self.final3 = nn.Conv2d(filters[0], out_ch, kernel_size=1)
        self.final4 = nn.Conv2d(filters[0], out_ch, kernel_size=1)
        self.conv_final = nn.Conv2d(out_ch * 4, out_ch, kernel_size=1)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)


    def forward(self, xA, xB):
        '''xA'''
        x0_0A = self.conv0_0(xA)
        x1_0A = self.conv1_0(self.pool(x0_0A))
        x2_0A = self.conv2_0(self.pool(x1_0A))
        x3_0A = self.conv3_0(self.pool(x2_0A))
        # x4_0A = self.conv4_0(self.pool(x3_0A))
        '''xB'''
        x0_0B = self.conv0_0(xB)
        x1_0B = self.conv1_0(self.pool(x0_0B))
        x2_0B = self.conv2_0(self.pool(x1_0B))
        x3_0B = self.conv3_0(self.pool(x2_0B))
        x4_0B = self.conv4_0(self.pool(x3_0B))

        x0_1 = self.conv0_1(torch.cat([x0_0A, x0_0B, self.Up1_0(x1_0B)], 1))
        x1_1 = self.conv1_1(torch.cat([x1_0A, x1_0B, self.Up2_0(x2_0B)], 1))
        x0_2 = self.conv0_2(torch.cat([x0_0A, x0_0B, x0_1, self.Up1_1(x1_1)], 1))


        x2_1 = self.conv2_1(torch.cat([x2_0A, x2_0B, self.Up3_0(x3_0B)], 1))
        x1_2 = self.conv1_2(torch.cat([x1_0A, x1_0B, x1_1, self.Up2_1(x2_1)], 1))
        x0_3 = self.conv0_3(torch.cat([x0_0A, x0_0B, x0_1, x0_2, self.Up1_2(x1_2)], 1))

        x3_1 = self.conv3_1(torch.cat([x3_0A, x3_0B, self.Up4_0(x4_0B)], 1))
        x2_2 = self.conv2_2(torch.cat([x2_0A, x2_0B, x2_1, self.Up3_1(x3_1)], 1))
        x1_3 = self.conv1_3(torch.cat([x1_0A, x1_0B, x1_1, x1_2, self.Up2_2(x2_2)], 1))
        x0_4 = self.conv0_4(torch.cat([x0_0A, x0_0B, x0_1, x0_2, x0_3, self.Up1_3(x1_3)], 1))


        output1 = self.final1(x0_1)
        output2 = self.final2(x0_2)
        output3 = self.final3(x0_3)
        output4 = self.final4(x0_4)
        output = self.conv_final(torch.cat([output1, output2, output3, output4], 1))
        return (output1, output2, output3, output4, output)

class SpatialAttention(nn.Module):
    def __init__(self, kernel_size=7):
        super(SpatialAttention, self).__init__()

        assert kernel_size in (3, 7), 'kernel size must be 3 or 7'
        padding = 3 if kernel_size == 7 else 1

        self.conv1 = nn.Conv2d(2, 1, kernel_size, padding=padding, bias=False)  # 7,3     3,1
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        x = torch.cat([avg_out, max_out], dim=1)
        x = self.conv1(x)
        return self.sigmoid(x)

class CBAM(nn.Module):
    def __init__(self, in_channels, kernel_size, ratio = 16):
        super(CBAM, self).__init__()
        # self.cam = ChannelAttention(in_channels, ratio=ratio)
        self.cam = Channel_Attention_Module_Conv(in_channels)
        self.sam = SpatialAttention(kernel_size)

    def forward(self, x):
        # x_cam = self.cam(x)
        # # x_camx = x_cam *x
        # x_sam = self.sam(x_cam)
        # # x_cbam = torch.cat([x_cam * x, x_sam * x], 1)
        # x_cbam = x * x_sam  
        # # out = x * self.cam(x)
        # # x_cbam = out * self.sam(out)
        # return x_cbam
        x_cam = self.cam(x)
        out_cam = x_cam * x
        x_sam = self.sam(x)
        out_sam = x_sam * x
        output = out_cam + out_sam
        # output = out_cam + out_sam
        return output


