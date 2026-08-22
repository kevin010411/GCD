# 單一檔案預測實驗

傳入一個目錄，即可對其中所有 `.nii`／`.nii.gz` 影像進行推論。檔名以
`_gt` 結尾的檔案會自動與對應的輸入配對：

```powershell
uv run python -m experiments.predict data/chgh `
  --config experiments/configs/predict.py `
  --output output/chgh
```

每個案例都會寫入各自的子目錄，而 `dataset_summary.json` 會寫入資料集輸出根目錄。

請從儲存庫根目錄執行：

```powershell
uv run python -m experiments.predict data/patient0016.nii.gz `
  --config experiments/configs/predict.py `
  --output output/patient0016
```

加入真實標註（ground truth），即可計算各類別及平均 Dice／IoU：

```powershell
uv run python -m experiments.predict data/patient0016.nii.gz `
  --config experiments/configs/predict.py `
  --ground-truth data/patient0016_gt.nii.gz `
  --output output/patient0016
```

無須編輯設定檔，即可使用 MMEngine 覆寫設定：

```powershell
uv run python -m experiments.predict data/patient0016.nii.gz `
  --config experiments/configs/predict.py `
  --output output/patient0016 `
  --cfg-options inference.device=cpu inference.overlap=0.5
```

`--output` 指定結果目錄。CLI 會顯示包含預估剩餘時間（ETA）的 tqdm 進度列，
並在該目錄中寫入以下分析檔案：

```text
patient0016/
├── patient0016_prediction.nii.gz
├── patient0016_<method>_class_<class>_xai.nii.gz
├── metrics.json
├── summary.csv
├── benchmark_runs.csv
├── class_voxels.csv
├── segmentation_metrics.csv       # 僅在使用 --ground-truth 時產生
├── xai_summary.csv
├── xai_curves.csv
└── plots/
    └── <method>/
        └── class_<class>/
            ├── insertion_0/
            │   ├── insertion_<class>_target_probability.png
            │   ├── insertion_<class>_prediction_dice.png
            │   ├── insertion_<class>_prediction_iou.png
            │   ├── insertion_<class>_ground_truth_dice.png
            │   └── insertion_<class>_ground_truth_iou.png
            └── deletion_0/
                └── deletion_<class>_<metric>.png
```

每項指標都會寫入各自的圖表。設定索引會反映在父目錄名稱中
（`insertion_0`、`insertion_1`，依此類推），避免同一類別的不同設定彼此覆寫。

效能評測指標包含延遲、吞吐量、參數數量、參數／檢查點大小、預測結果大小，
以及各類別的體素數量。只有在提供真實標註影像時，才會包含 Dice／IoU。
`xai` 區段包含目標類別、歸因結果路徑、各自獨立設定的擾動曲線，以及 AUC 值。
梯度歸因會以設定的推論 ROI 解析度進行評估，再重新取樣至原始預處理影像的形狀，
使大型 CT 掃描的 3D 記憶體用量維持在可控範圍內。

XAI 在 `experiments/configs/predict.py` 中設定。`XaiMethods`、`XaiMetrics`
及 `XaiAnswer` 會透過 MMEngine registry 建立為實例。各方法遵循共用的解釋策略；
插入／刪除（insertion／deletion）屬於高階擾動指標，而 Dice／IoU／機率則是可組合的
步驟評分器。最終答案會依據較高的 insertion AUC 與較低的 deletion AUC 對解釋結果
進行排名，並寫入 `xai_answer.csv`。

實驗 CLI 也支援資料集層級的 `organ_occlusion` 方法。此方法會執行或重用快取的
TotalSegmentator 遮罩，逐一遮蔽每個非空器官，並寫出帶正負號的器官重要性 NIfTI，
以及 `organ_occlusion_ranking.csv`。請將 `organ_occlusion` 加入擾動設定的 `methods`
清單，並將它設定為一個 `XaiMethods` 項目：

```python
XaiMethods = [dict(
    type="OrganOcclusionXaiMethod",
    id="organ_occlusion",
    target_class=[1, 2, 3],
    objective=dict(type="PredictionDiceScore"),
    segmenter=dict(type="TotalSegmentatorService", merge_organs=True),
    occluder=dict(
        type="OrganOccluder",
        mode="local_mean",
        feather_mm=2.0,
        preserve_answer=False,
    ),
)]
XaiMetrics = [
    dict(
        type="PerturbationInsertion", target_class=[1], steps=11,
        methods=["organ_occlusion"], baseline=0.0,
        answer_retention=[None],
        scorers=[
            dict(type="TargetProbabilityScore"),
            dict(type="PredictionDiceScore"),
            dict(type="PredictionIoUScore"),
        ],
    ),
    dict(
        type="PerturbationDeletion", target_class=[1], steps=11,
        methods=["organ_occlusion"], baseline=0.0,
        answer_retention=[None],
        scorers=[dict(type="TargetProbabilityScore")],
    ),
]
XaiAnswer = dict(
    type="FaithfulnessAnswerAggregator",
    insertion_weight=0.5,
    deletion_weight=0.5,
)
```

`merge_organs=False` 會保留 TotalSegmentator 的個別 labels（例如左右肺葉），不套用
curated organ 聯集。`preserve_answer=True` 會在 preprocessing 後還原答案 mask 內的原始
input voxels；有 ground truth 時答案 mask 使用 ground truth，否則使用 baseline prediction。
若不設定 `XaiMetrics`，runner 仍會依 `XaiMethods.target_class` 產生並儲存 XAI
NIfTI，但不會計算 insertion/deletion 曲線或 `XaiAnswer`。

器官目標可使用內建評分器，例如 `PredictionDiceScore`、`PredictionIoUScore`、
`TargetProbabilitySumScore` 或 `TargetLogitSumScore`。遮蔽模式包括 `local_mean`、
`gaussian_blur` 及 `fixed_hu`；其中 `fixed_hu` 也接受範圍為 -1000 至 1000 的
`fill_hu`。

插入與刪除是 `XaiMetrics` 中的項目。每個項目都會獨立建立，並擁有自己的
`methods`、`target_class`、`steps`、`baseline`、`answer_retention`，以及巢狀的
`scorers`。移除某項操作的所有項目，即可停用該操作。方法與目標類別的值可以是清單。

`answer_retention=(None, 0.5)` 會同時產生標準曲線，以及一條保留答案的曲線；
在後者中，至少 50% 的目標類別真實標註體素會維持不變。數值型變體需要真實標註；
若未提供，該變體會記錄為已略過，但標準（`None`）變體仍會執行。

若要加快執行速度，可減少擾動步數：

```powershell
uv run python -m experiments.predict data/patient0016.nii.gz `
  --config experiments/configs/predict.py `
  --output output/patient0016 `
  --cfg-options XaiMetrics.0.steps=2 XaiMetrics.1.steps=2
```
