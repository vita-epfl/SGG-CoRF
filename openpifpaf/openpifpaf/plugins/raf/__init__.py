import openpifpaf

from . import headmeta
#from .cifdetraf import CifDetRaf
#from .painters import RelationPainter
from .cifdetraf_updated import CifDetRaf
from .painters_updated import RelationPainter
from .annotation import AnnotationRaf
from .heads import DeepCompositeField3, DeepSharedCompositeField3
from .refinement_heads import RefinedCompositeField3
from .losses import DCNCompositeLoss

def register():
    openpifpaf.HEADS[headmeta.Raf] = openpifpaf.network.heads.CompositeField3 #DeepCompositeField3 #openpifpaf.network.heads.CompositeField3
    openpifpaf.HEADS[headmeta.Raf_dcn] = RefinedCompositeField3#DeepCompositeField3 #openpifpaf.network.heads.CompositeField3
    openpifpaf.HEADS[headmeta.CifDet_deep] = DeepCompositeField3
    openpifpaf.HEADS[headmeta.CifDet_deepShared] = DeepSharedCompositeField3
    openpifpaf.DECODERS.add(CifDetRaf)
    openpifpaf.PAINTERS['AnnotationRaf'] = RelationPainter
    openpifpaf.PAINTERS['AnnotationRaf_updated'] = RelationPainter

    openpifpaf.LOSSES[headmeta.CifDet_deep] = openpifpaf.network.losses.CompositeLoss
    openpifpaf.LOSSES[headmeta.Raf] = openpifpaf.network.losses.CompositeLoss
    openpifpaf.LOSSES[headmeta.Raf_dcn] = DCNCompositeLoss
    #openpifpaf.LOSSES[headmeta.Raf_dcn] = openpifpaf.network.losses.CompositeLoss
    openpifpaf.LOSSES[headmeta.CifDet_deepShared] = openpifpaf.network.losses.CompositeLoss
