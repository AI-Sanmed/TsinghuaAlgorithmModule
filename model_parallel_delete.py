import torch
import torch.nn as nn
from false_positive_reduction.net.fpred_networks import NetCombineClsSegMod
from nodule_detection.net.nodule_detection_model import UNet3D
from nodule_segmentation.net.precise_seg_net import DenseUNet3DGuideEdgeDistLower2ConvCat
from nodule_classification.net.texture_cls_net import Net32SETextureAux


orig_model_name = '/home/link/data/algorithm/state_dict_model_UNet_28__LUNA_Texture_Classification_Final'
state_dict = torch.load(orig_model_name + '.pt')
from collections import OrderedDict
new_state_dict = OrderedDict()
for k, v in state_dict.items():
    name = k[7:] # remove `module.`
    new_state_dict[name] = v
# load params
model = Net32SETextureAux(1, 1)
model.load_state_dict(new_state_dict)
state_dict = model.state_dict()
torch.save({'state_dict': state_dict}, orig_model_name + '.pth')