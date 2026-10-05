# CAM 處理協定

UI 與 `experiments.benchmark` 讀取相同的 `cam_protocol`；未設定時維持 benchmark 預設。
適用 GradCAM、HiResCAM（XResCAM）、LayerCAM、Score-CAM。

| cfg 欄位 | 目前 benchmark 預設 | 舊 UI 三項設定 |
| --- | --- | --- |
| `target_region` | `full_prediction`：完整 sliding-window logits 的 argmax，切到各窗 | `tile_prediction`：各窗原始 logits 的 argmax |
| `reduction` | `mean`：目標區域 class logits 平均 | `sum`：目標區域 class logits 總和 |
| `relu_stage` | `per_tile`：各窗 CAM 先 ReLU 再融合 | `after_fusion`：帶正負值的 CAM 融合後 ReLU |

完整可用 cfg：目前方式 `config/model/xai_hw_acdc.py`；舊 UI 三項方式
`config/model/xai_hw_acdc_legacy_ui.py`。兩者共用模型、checkpoint、前處理與滑窗；
舊 UI cfg 並非還原歷史版本所有推論／前處理差異。

UNet3D 與 UNetCNX 也有對應的舊 UI 協定 cfg：
`config/model/unet_3d_legacy_ui.py`、`config/model/unetcnx_legacy_ui.py`。
分別繼承原本的 `unet_3d.py`、`unetcnx.py`，保留 checkpoint、前處理、滑窗與預設 layer。

任意模型 cfg 可直接加上以下內容（可獨立修改每一欄）：

```python
cam_protocol = dict(
    target_region="tile_prediction",
    reduction="sum",
    relu_stage="after_fusion",
)
```

`config/cam/benchmark.py` 與 `legacy_ui.py` 是可重用的協定片段。
不要同時繼承已含 `cam_protocol` 的模型與另一片段（MMEngine 會拒絕重複 base key）；
用上面的子 cfg 覆寫方式。class 由 UI 選項或 benchmark 的 `--class-id` 決定，不固定 class 1。

UI 使用 `predicted_target_mask` objective 時，本協定控制目標區域與 reduction；
若手動選其他 objective，使用該 objective 的公式，只有 CAM ReLU 時機仍受此設定控制。
Score-CAM 無梯度；reduction 控制其 masked-forward 分數，目標固定為原始未遮罩的
完整／單窗預測，不會每次遮罩重新選物體。空目標輸出零。
LayerCAM 的 gradient ReLU 與 Score-CAM 的 activation ReLU 是方法公式，不受
`relu_stage` 控制。Score-CAM 的加權 activation 已非負，因此單改此欄通常不改結果。
所有方式都先以相同 window/blend 融合，再做全影像 min-max normalization。

批次以 `--model-config config/model/xai_hw_acdc_legacy_ui.py` 切換；
UI 載入同一 cfg 即可，無須依入口切換演算法。此 cfg 不重新計算已存在的 heatmap；
`evaluate_metrics` 只評分輸入 heatmap，需先用 benchmark 或 UI 重新產生。
