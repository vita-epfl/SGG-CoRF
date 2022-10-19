from .preprocess import Preprocess


class Encoders(Preprocess):
    def __init__(self, encoders, join=False):
        self.encoders = encoders
        self.join = join

    def __call__(self, image, anns, meta):
        anns = [enc(image, anns, meta) for enc in self.encoders]
        meta['head_indices'] = [enc.meta.head_index for enc in self.encoders]
        if self.join:
            temp_tuple = None
            for ann in anns:
                if temp_tuple is None:
                    temp_tuple = ann if isinstance(ann, tuple) else (ann,)
                    continue
                if isinstance(ann, tuple):
                    temp_tuple = temp_tuple + ann
                else:
                    temp_tuple = temp_tuple + (ann,)
            anns = [temp_tuple]
        return image, anns, meta
