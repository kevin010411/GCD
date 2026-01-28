from mmengine import Registry

MODEL = Registry("model")


def build_model(cfg):
    return MODEL.build(cfg)
