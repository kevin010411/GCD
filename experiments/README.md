# Experiments：批量 benchmark 與單案例 metrics

CAM 的完整預測／單窗預測目標、logits mean／sum、融合前／後 ReLU 可由共用
`cam_protocol` cfg 切換；預設保持 benchmark。設定範例與限制見
[CAM 協定說明](../config/cam/README.md)。

Experiments 是 GCD 的批次入口。模型、權重、前處理、滑窗推論與評分核心由
`src/` 提供；Qt UI 與 CLI 使用同一套 cfg。實驗排程、結果彙整、CSV／JSON／圖表
才屬於 experiments。不要在此另寫評分公式。

只有這兩個對外 Python 入口：

- `uv run python -m experiments.benchmark`：多案例、多方法／layer 的 CAM benchmark。
- `uv run python -m experiments.evaluate_metrics`：對一組原圖、答案、已有 heatmap 計算 metrics，不重算 CAM。

入口以外的 Python（包含實驗 cfg）已全部移入 `experiments/src/`。
根目錄的 `__init__.py` 僅為套件標記，不是入口。舊 `experiments.predict`、
`experiments.xai_benchmark` 命令與舊模組 import 路徑已停用。

```text
experiments/
├── benchmark.py                # 批量 benchmark 薄入口
├── evaluate_metrics.py         # 單案例 metrics 薄入口
├── __init__.py                 # 套件標記
├── README.md
├── docs/XAI_BENCHMARK.md
└── src/
    ├── benchmark_runner.py
    ├── single_case_runner.py
    ├── args.py / outputs.py / runner.py / metrics.py
    ├── preprocessing.py / xai_design.py / predict.py
    ├── configs/                # 實驗 cfg；_base_ 指向根 config/
    └── xai/                    # 舊適配器與研究診斷工具，非對外 CLI
```

`experiments/src/` 是私有排程／輸出／適配器；不是 GCD 演算法的第二份來源。
共用評分、幾何處理與 CAM 核心仍在 GCD 根目錄 `src/gcd/infrastructure/`。
保留的舊 registry／resized-ROI runner 是內部相容程式，搬檔不代表所有舊路徑已完成核心收斂。

## 共用設定與執行環境

- `config/model/*.py`：模型、checkpoint 與預設 XAI layer。
- `config/preprocessing/gcd.py`：共用前處理、128³ ROI、constant blending。
- `config/preprocessing/acdc.py`：ACDC 有序前處理、96×96×32 ROI、Gaussian blending。
- `preprocessing.spacing`、`inference.roi_size/overlap`、`display.permute` 是正式欄位；
  `base.py`、頂層 size/stride/spacing/permute 與四窗策略已移除。
- 使用 Windows 11 原生環境、PowerShell 與 uv，從 GCD 專案根目錄執行。
- checkpoint 不會自動複製；使用 `--checkpoint` 指定 Windows 可存取的權重路徑。
  下列影像與權重路徑是範例，請換成實際檔案；相對路徑以 GCD 根目錄為基準。

```powershell
Set-Location D:\KevinFu\GCD
uv sync --locked
uv run python -m experiments.evaluate_metrics --help
uv run python -m experiments.benchmark --help
```

## 單案例：已存在的 heatmap

必須提供四項：原始影像、答案 label、heatmap，以及訓練模型的完整 cfg／checkpoint。
沒有模型就無法計算 insertion／deletion，因為每個擾動點都需要重新推論。

```powershell
uv run python -m experiments.evaluate_metrics `
  --image data/patient.nii.gz `
  --label data/patient_gt.nii.gz `
  --heatmap data/patient_heatmap.nii.gz `
  --model-config config/model/xai_hw_acdc.py `
  --checkpoint D:/WSL/xai_hw/assignment/colab/checkpoint/best_model.pth `
  --class-id 1 --steps 21 --preserve-answer `
  --output output/single_metrics
```

亦可直接執行 `uv run python experiments/evaluate_metrics.py ...`。
cfg 的空間／強度步驟必須符合模型訓練協定，不可只根據影像類型任意挑選。

### 輸入空間

預設 `--heatmap-space original`：

- 原圖與答案必須同 shape、同 affine；答案是非負整數 class ID，不可用連續 probability 代替。
- heatmap 必須與原圖同 shape、同 affine。只比較尺寸是不夠的。
- 原圖／答案由共用前處理轉到模型網格；答案使用 nearest，跳過強度轉換。
- heatmap 依 affine 以線性插值重採樣到模型網格，外部補 0，再做全圖 min-max；
  **不套用 CT 的 clipping、強度 normalization，也不當二值 mask。**
- heatmap 必須有限、非負；常數 heatmap 允許評分，但會標記 constant_heatmap。
- `--heatmap-space model`：heatmap 已在前處理後網格，嚴格核對 shape／affine，不再重採樣。
- `--heatmap-space world`：明確允許不同網格，按世界座標對齊；完全不相交會報錯。
  這不是修正錯誤 affine 的功能，使用者仍須確認空間語意正確。
- 輸入只支援 3-D，或末軸為 1 的 4-D NIfTI；多時間點需先明確選取。
- 原空間 export → reload → model-space resampling 會改變排名，
  因此不能保證與原始未匯出的 model-space CAM AUC 完全相同。

### 計算哪些指標

Insertion／Deletion 同時計算：

| 指標 | 意義 |
|---|---|
| gt_dice | 擾動後預測與答案的二值 Dice |
| relative_gt_dice | gt_dice / 完整輸入 Dice |
| gt_iou | 擾動後預測與答案的 IoU |
| prediction_consistency_dice | 擾動後預測與完整輸入預測的 Dice |
| fixed_roi_class_probability | 完整預測目標 ROI 內的 softmax probability 平均 |
| relative_fixed_roi_class_probability | 上述平均 / 完整輸入的平均 |

另提供 AP、答案內 attribution mass、取 GT 大小的 top-k Dice／IoU，
以及距離答案邊界 3 mm 內／外的 attribution 統計。公式在
`src/gcd/infrastructure/xai/benchmark_metrics.py`，benchmark 與單案例共用。

AUC 一律涵蓋 0–100%，梯形積分；steps=21 使用前段較密的 dense21，
其他步數使用等距點（steps=15 保留既有取點表）。分數不裁切到 1：
Relative Dice／AUC 可以超過 1，代表擾動後比完整輸入更好。

二值 Dice 的 empty-empty 定義為 0；若完整 Dice 為 0，
relative Dice 無法定義，輸出 JSON null／CSV 空值，不假裝等於 0。
若完整預測沒有目標 class，固定 ROI probability 也標成 null。
空間指標沒有正例／負例而無法定義時同樣標成 null。

### 擾動協定

- 預設 `--mode both --baseline zero --label-policy original`。
- baseline 是**模型輸入強度空間**中的 zero／minimum／mean，不是原始 HU。
- original：答案固定不變，擾動會刪除答案區域的影像。
- masked：答案也跟著保留／刪除遮罩改變，屬於另一套評分協定。
- `--preserve-answer`：**額外**算一組保護答案的曲線；target GT voxel 永遠保持原強度，
  排名只包含非保護 voxel，proportion 是可擾動 voxel 的比例。
  standard 與 preserve_answer 必須分開比較。
- `--cfg-options inference.overlap=0.5 inference.blend_mode=gaussian` 可覆寫 cfg；
  這會改變模型結果／協定，會記錄於 protocol。
- `--device cpu`、`--no-plots`、`--dry-run` 可控制執行；dry-run 仍檢查輸入、
  cfg、checkpoint 檔案與幾何，但不載入權重做推論。
- 輸出依內容與協定 hash 分資料夾；同一輸出已存在時報錯，
  明確加 `--force` 才覆寫這個協定。

### 輸出位置

```text
<output>/class<class-id>_<protocol-hash>/
├── protocol.json               # 輸入/權重 SHA256、解析 cfg、空間、評分協定、實作 SHA256
├── summary.json
├── summary.csv                 # 各模式/variant 的 AUC + 空間指標
├── curve.csv                   # 每個比例下全部指標
├── heatmap_model_space.nii.gz   # 真正參與排名的熱圖與模型 affine
└── plots/
    ├── gt_dice.png
    ├── relative_gt_dice.png
    ├── gt_iou.png
    ├── prediction_consistency_dice.png
    ├── fixed_roi_class_probability.png
    └── relative_fixed_roi_class_probability.png
```

## 批量 benchmark

```powershell
uv run python -m experiments.benchmark `
  --model-config config/model/xai_hw_acdc.py `
  --checkpoint D:/WSL/xai_hw/assignment/colab/checkpoint/best_model.pth `
  --case patient103 data/patient103.nii.gz data/patient103_gt.nii.gz `
  --methods gradcam_L3 scorecam_L3 multilayer_layercam `
  --class-id 1 --steps 21 --preserve-answer --export-nifti `
  --output output/benchmark
```

重複 `--case ID IMAGE LABEL` 可增加案例；`--methods all` 執行現有方法組合。
不同模型需指定對應 `--layers L1 L2 L3`，不假設任意模型有相同 decoder。
完整 protocol／方法說明及歷史比較見 [XAI_BENCHMARK.md](docs/XAI_BENCHMARK.md)。
批量輸出保留各案例結果與跨案例平均；不能混合不同 cfg／擾動協定的 AUC。

## 驗證

```powershell
uv run python -m unittest tests.test_single_case_metrics tests.test_xai_benchmark tests.test_cam_benchmark_parity
```

單案例測試涵蓋 affine 不符、模型網格、heatmap 強度不套 CT 前處理、
答案幾何／class ID、既有曲線/AUC 共用、無完整預測時 undefined 指標，
以及 CLI 的 JSON／CSV／6 張圖輸出。真實 acceptance 不等同所有模型或 Qt 視窗皆已驗證。

以下為歷史 Docker `Main` 驗證紀錄，尚不代表 Windows 11／uv 驗收。

2026-10-01 Main 搬移後驗證：152 項相關測試通過（包含入口邊界／私有匯入／cfg 繼承，及 patient103 的 UI/benchmark
熱圖與 GradCAM dense21 指標一致性）；既有 batch CLI 的 prediction_logits_minmax
兩端點 smoke test 通過。單案例 CLI 另實際讀取 patient103 原始 CT、最終答案與
既有原空間 ScoreCAM L3 熱圖，完成 dense21 standard/preserve_answer 共 84 個點及
6 張圖。輸出保留在 D:/WSL/gcd_single_metrics_validation/class1_55bb0c65cc72/。
這是入口與 NIfTI 評分驗證，不是重新產生三病例 baseline。
搬移後另以新公開入口執行 batch 兩端點與單案例三點 smoke test；輸出位於
D:/WSL/gcd_experiment_src_cli_validation/。真實 CAM parity 報告：
D:/WSL/gcd_experiment_src_parity_results.json。上述 smoke test 不用於 baseline 排名。
