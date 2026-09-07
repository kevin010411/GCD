_base_ = ["../base.py"]

model = dict(
    type="DynUNet",
    spatial_dims=3,
    in_channels=1,
    out_channels=8,
    kernel_size=[[3, 3, 3], [3, 3, 3], [3, 3, 3], [3, 3, 3], [3, 3, 3]],
    strides=[[1, 1, 1], [2, 2, 2], [2, 2, 2], [2, 2, 2], [2, 2, 2]],
    upsample_kernel_size=[[2, 2, 2], [2, 2, 2], [2, 2, 2], [2, 2, 2]],
    filters=[16, 32, 64, 128, 256],
    use_ckpt=True,
)  # 模型
ckpt = "checkpoint/12_4_4_fold1/nnunet.pth"  # 權重檔
default_layer = "decoder0"  # 預設 CAM 層
