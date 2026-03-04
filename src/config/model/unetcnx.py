_base_ = ["../base.py"]

model = dict(type="UNETCNX_A1")  # 模型
ckpt = "checkpoint/60_20_20_fold3/unetcnx.pth"  # 權重檔
default_layer = "dec4"  # 預設 CAM 層
