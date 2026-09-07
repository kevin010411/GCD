_base_ = ["../base.py"]

model = dict(type="UNETCNX_A1", out_channels=8, deep_sup=True)  # 模型
ckpt = "checkpoint/12_4_4_fold1/unetcnx.pth"  # 權重檔
default_layer = "decoder4"  # 預設 CAM 層
