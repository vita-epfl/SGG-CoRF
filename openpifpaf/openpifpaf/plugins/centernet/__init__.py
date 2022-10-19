import openpifpaf
from . import headmeta
from .heads import CenterNetHead, CompositeField3_singleDeform_fourier
from .vg import VGModule, VG, OBJ_CATEGORIES, REL_CATEGORIES
#from .gqa import GQAModule
from .losses import CenterNetLoss
from .decoder import CenterNet
from .cifdetraf_cn import CifDetRaf_CN
from .losses_cifdetCN import CifDet_CNLoss

from .transforms.toannotation import ToRafAnnotations

def register():
    openpifpaf.HEADS[headmeta.CenterNet] = CenterNetHead
    openpifpaf.LOSSES[headmeta.CenterNet] = CenterNetLoss

    openpifpaf.LOSSES[headmeta.Raf_GDeform] = CifDet_CNLoss
    openpifpaf.HEADS[headmeta.Raf_GDeform] = CompositeField3_singleDeform_fourier

    openpifpaf.DATAMODULES['vg'] = VGModule

    openpifpaf.DECODERS.add(CenterNet)
    openpifpaf.DECODERS.add(CifDetRaf_CN)
