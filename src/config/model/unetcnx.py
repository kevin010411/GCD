_base_ = ["../base.py"]

model = dict(type="UNETCNX_A1", out_channels=4, deep_sup=True)  # 模型
ckpt = "checkpoint/60_20_20_fold1/unetcnx.pth"  # 權重檔
default_layer = "decoder4"  # 預設 CAM 層
