from openpifpaf.transforms.preprocess import Preprocess


class Prior_HM(Preprocess):
    def __init__(self, encoders):
        self.encoder = encoders

    def __call__(self, image, anns, meta):
        inp_hm = self.encoder(image, anns, meta)
        return (image, inp_hm) , anns, meta
