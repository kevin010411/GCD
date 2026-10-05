# GCD 的分割 XAI 基準

`uv run python -m experiments.benchmark` 在 GCD 的 `experiments` 中重用
MMEngine 模型設定，並以模型輸入網格產生 CAM、執行插入／刪除測試。每次執行
必須提供真實標註，因為 GT Dice 和空間指標需要它。可重複傳入 `--case ID 影像 標註`
以取得三病例的平均表。使用 `--dry-run` 先檢查來源及 protocol。

只匯出熱圖時使用 `--heatmap-only --export-nifti`，此時 `--case ID 影像`
不需要標註，並跳過 GT 指標與插入／刪除評估。仍使用同一個 cfg、固定完整
預測 target mask 及共用 tiled CAM 核心；輸出 protocol、summary 與 NIfTI。

## 為什麼舊 GCD 實驗與 xai_hw 不同

| 項目 | 舊 GCD 內部舊 runner（已非對外入口） | ACDC `xai_hw` 基準／新 `benchmark` 相容設定 |
|---|---|---|
| 模型 | GCD config 預設 `INCEPTION_RESBLOCK` 或自己的 UNet，checkpoint 依設定 | MONAI UNet3D、`num_res_units=3`、ACDC `best_model.pth` |
| CT 前處理 | 0.7×0.7×1.0 mm、HU 視窗縮放至 0–1，沒有 RAS | 1.5×1.5×5.0 mm、RAS、非零強度 z-normalization、對稱 pad 至 96×96×32 |
| CAM | 一般方法先將完整 volume 縮成單一 128³ ROI 再放大；ScoreCAM 有另外的 UI tiled 選項 | 在原模型網格逐個滑窗算 raw CAM，以 Gaussian 權重融合，最後做全圖 min-max |
| target | 可能是縮圖上的 prediction | 完整輸入滑窗 logits 的 argmax，切成每個 tile 的固定 class mask |
| 擾動採樣 | 預設等距，且 `steps=20` 表示 21 個點 | dense21：0、0.05%、0.1%、0.25%…100% |
| Dice | GCD 的 empty/empty Dice 是 1；另有 prediction Dice | empty/empty 是 0；GT Dice 除以該病例完整輸入的 GT Dice |
| AUC | 通常 raw probability 或 raw Dice | 0–100% 梯形積分，先算每例，再等權平均 |

即使熱圖公式相同，模型、輸入、目標 mask 和評分網格只要有一項不同，分數就
不應被解讀為 CAM 實作品質的差異。既有同模型、同 tile 的 GCD ScoreCAM 與
`xai_hw` 公式比對曾達到浮點精度一致；那個結論只適用於當時的 checkpoint、
patient0016、`decoder 2`、class 1 和 tile 設定。

## 新實驗入口

在 Windows 11 PowerShell 中，先從 GCD 根目錄執行 `uv sync --locked`，再使用 uv 執行。
下列影像及 checkpoint 路徑請換成實際的 Windows 路徑：

```powershell
uv run python -m experiments.benchmark `
  --model-config config/model/xai_hw_acdc.py `
  --checkpoint "D:/WSL/xai_hw/assignment/colab/checkpoint/best_model.pth" `
  --profile xai_hw_acdc `
  --case patient103 "data/patient103_frame01.nii.gz" "data/patient103_frame01_gt.nii.gz" `
  --case patient121 "data/patient121_frame01.nii.gz" "data/patient121_frame01_gt.nii.gz" `
  --case patient131 "data/patient131_frame01.nii.gz" "data/patient131_frame01_gt.nii.gz" `
  --methods all --class-id 1 --steps 21 --baseline zero `
  --label-policy original --mode both --export-nifti `
  --output output/acdc_cam_benchmark
```

模型、checkpoint、前處理、滑窗和連通元件全部由 `--model-config` 決定；
`--profile` 現在僅是輸出標籤，不會偷偷覆寫 cfg。`xai_hw_acdc.py` 已包含
完整 MONAI UNet 與 checkpoint 路徑，省略 `--checkpoint` 即使用 cfg 權重。GCD 模型可用
`--layers L1 L2 L3` 指定 `xai_layer_targets` 中的別名或完整 module path；
後綴 `:input` 表示擷取該 module 的輸入。ACDC cfg 的 `benchmark.feature_layers` 預設採用
最後三個 `Conv3d` 中排除輸出層的 `[-4:-1]` 規則。GCD 的 UNet 預設用
`decoder 1:input`、`decoder 2`、`decoder 3`，讓 L1 對應最後輸出前的
高維特徵。也可以明確指定 `decoder 1` 的**輸出**作 ScoreCAM 空間遮罩；
這只有 class 數量那麼多個 channel，與 `decoder 1:input` 是不同實驗。
其他 GCD 架構請
明確指定三層。應在結果 `summary.json` 確認實際路徑。
`--roi X Y Z`、`--overlap`、`--sw-batch-size` 可覆寫推論設定。

既有的 內部舊 runner（已非對外入口） 也能在 `XaiMethods` 使用同一個 tiled
heatmap 產生器，例如：

```python
XaiMethods = [dict(
    type="BenchmarkTiledCamMethod",
    id="scorecam_l1_tiled",
    method="scorecam_L1",
    target_class=[1],
    score_batch_size=4,
)]
```

可另外填入 `layers=[L1, L2, L3]` 指定三個**不同**的有效 feature layer；
GCD UNet 省略時採上述 decoder 選層；ACDC cfg 採上述 Conv3d 規則，其他模型須指定。內部舊 runner（已非對外入口）
會照它自己的前處理與
`XaiMetrics` 設定評分，因此需要與 `xai_hw` 比對數值時，應使用本頁的
`benchmark` 入口及相容 profile。

方法包括四種 CAM 的 L1～L3：`gradcam`、`hirescam`、`layercam`、
`scorecam`；三個多層候選：`multilayer_gradcam`、`multilayer_hirescam`、
`multilayer_layercam`；以及 `prediction_logits_minmax`、
`prediction_mask_binary`、`prediction_distance_prior` 控制組。多層 CAM
先平均各層的 raw tile map，再融合及正規化。ScoreCAM 會做每個 channel
的遮罩推論，跑全方法、三病例可能耗時較長。

每個 run 以輸入檔大小／修改時間、設定及實作檔 hash 建立專屬目錄；已完成的
病例／方法會續跑，`--force` 可重算。輸出含 `protocol.json`、
`metrics_by_case.csv`、`metrics_mean.csv`、每方法的 `curve.csv`、
`summary.json`、`heatmap_model_space.npy`；`--export-nifti` 另輸出
`heatmap_original_space.nii.gz`，可在原 CT 幾何空間開啟。熱圖是 0–1
重要性值，不是 CT 強度。`--preserve-answer` 另計算完全保護 GT class voxel
的插入／刪除曲線，與標準曲線分開標記。

每個病例同時保存 `prediction_model_space.npy`；使用 `--export-nifti` 時，
另保存病例目錄中的 `prediction_original_space.nii.gz`。這是全部類別的整數
分割 label map，包含背景，並依 cfg 套用連通元件後處理。以最近鄰插值還原
到原始影像 shape／affine，不做 min-max 正規化，也不使用 CAM 的 class 1
顯示門檻。每病例保存一次，不隨 CAM 方法或層級重複保存。

主要行為指標是原標籤 `standard_insertion_relative_gt_dice_auc`；同一輪
擾動還輸出 raw GT Dice、GT IoU、prediction consistency Dice、固定完整輸入
預測 ROI 的 class probability，以及各自相對值。空間讀數包括 AP、GT 內熱量
比例、GT 大小門檻的 Dice/IoU 和邊界／內部熱量。刪除分數可能很快飽和，
不應單獨用於方法排名。`label_policy=masked` 會改變答案，必須與原標籤
實驗分開分析。

## 已驗證範圍

UI 的預設 CAM 後端現已改為同一協定：完整預測固定 mask、平均 target logit、
每窗 ReLU、raw magnitude 融合後全圖 min-max。原有 XResCAM 保留為 HiResCAM
相容名稱，另提供 HiResCAM／LayerCAM 的正式方法項目。CAM 核心已移到 src，
experiments 的 benchmark_cam 只保留相容匯入。此對齊限定 predicted_target_mask
及相同 layer／feature 範圍／cfg；其他 objective 與一般 resized-ROI 實驗路徑
不應被宣稱與 benchmark 等價。既有 UI 熱圖需重算，不能沿用舊成品。

2026-10-01 在 Main 容器完成 132 個相關測試（含真實案例驗收）。patient103、
原 xai_hw 權重、class 1、L3、96×96×32 ROI、overlap 0.25、Gaussian、12 tiles：
UI／benchmark 的最大 heatmap 絕對差為 GradCAM 1.68e-5、HiResCAM 3.58e-7、
LayerCAM 2.38e-7、ScoreCAM 3.19e-5（CPU／CUDA reduction 及 ScoreCAM batch
方式不同，驗收 atol/rtol 均為 1e-4，不要求逐位元相同）。將 UI 與 benchmark 的
GradCAM 熱圖交給同一推論／評分協定後，dense21 的六種指標、insertion／deletion
共 12 項 AUC 差異皆為 0。尚未跑所有病例、層或其他方法的完整 AUC；也未做 Qt
視窗端到端驗收。可重跑 `tests.test_real_cam_benchmark_parity`，設
`GCD_RUN_REAL_CAM_PARITY=1`；`GCD_PARITY_REPORT` 可指定 JSON 報告路徑。
以上是歷史容器驗證紀錄，不代表 Windows 11／uv 已完成同等驗收。
現在重跑程式與測試時，使用 Windows 11 PowerShell 與 `uv run python`，
並設定 Windows 可存取的影像與 checkpoint 路徑。

在 Docker `Main` 中使用相同 ACDC checkpoint、patient103、class 1 與
原有 `xai_hw` 成品核對：GradCAM L3 模型網格熱圖逐 voxel 相同；
ScoreCAM L1 最大絕對差為 `2.38e-7`。三病例 ScoreCAM L1 的 dense21、
0–100% 標準 insertion 相對 GT Dice AUC 平均為 `0.982613`，
deletion 為 `0.006677`。這是相容 profile 的重現結果，不是對所有方法和
所有 GCD 模型的等價保證。

GCD 自身的 UNet checkpoint 在 patient0016 的 class 1 測試顯示：
誤取編碼器卷積當 L3 時，GradCAM 的 GT spatial AP 為 `0.03449`；
改用 `decoder 3` 後為 `0.80132`。此處的 insertion 僅以兩個端點做
smoke test，AUC `0.5` 沒有辨識力，不能拿來宣稱行為指標改善。

原本會拒絕 `decoder 1` 輸出的 UI Score-CAM 限制也已移除。使用同一
patient0016／GCD checkpoint、32 個 tile，class-logit 遮罩有 91 次有效
masked forward，輸出的 NIfTI 最大值為 1、非零 voxel 數為 18,061,150；
它確實可以產生熱圖，但仍須獨立評估該熱圖的定位與忠實度。
