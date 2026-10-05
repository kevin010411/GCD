# 模型選單目錄

共用設定集中於儲存庫根目錄 `config/`：`model/` 放模型／權重／XAI 層，
`preprocessing/` 放有序前處理、推論、後處理與顯示設定。一般模型直接繼承
`preprocessing/gcd.py`；`model/xai_hw_acdc.py` 繼承 `preprocessing/acdc.py`。
UI 與 experiments 均讀取這些設定；experiments 的同名 ACDC cfg 只作相容別名。
所有 `window` padding 跟隨 inference.roi_size；GCD 預設 128×128×128，
ACDC 預設 96×96×32。base.py 與頂層 size/stride/spacing/permute 已移除；
使用 inference.roi_size/overlap、preprocessing.spacing 與 display.permute。
legacy／legacy_four_tile 策略與 legacy_window padding 不再支援，舊 cfg 必須明確遷移。
這會改變原先四窗方法的覆蓋及一般 GCD 的 padding；舊數據不可視為同一協定。
UI 所有 tiled XAI 統一使用 sliding_window；
ACDC cfg 指定所有方法採 sliding_window。模型選單中的 xai_hw 權重預設使用 Main 路徑，
在 Windows 使用時需將 ckpt 改成可存取的本機路徑。

2026-10-01 四窗移除驗證：Main 中 142 個相關測試通過，包含全部模型 cfg 的
繼承／引擎設定載入、矩形 ROI padding 與舊策略拒絕。patient103/class1/L3 的
四種 CAM 與 benchmark 最大熱圖誤差 3.13e-5；GradCAM dense21 的 12 項 AUC 差異皆為 0。
報告：D:/WSL/gcd_sliding_cfg_parity_results.json。未完成其他病例／模型的真實推論驗證。

歷史搬移驗證（移除四窗與更改 padding 前）：Main 中 91 個相關測試通過，全部共用模型 cfg 可載入／切換。
patient103 在 UI 後端與 experiments 的前處理張量／affine 逐值相同。
根目錄 ACDC cfg 的真實 UI GradCAM 推論使用 96×96×32 window、12 個 tile，
得到有限且非零的 0–1 熱圖（display grid 267×32×225）。此驗證不代表 CAM 品質排名。
Main 缺少 libxkbcommon.so.0，完整 Qt 視窗回歸未執行成功；以上為後端與設定載入驗證。

GCD 的模型選單由同目錄下的 `model_catalog.json` 載入，執行時不再掃描
`model/*.py`。JSON 陣列順序就是選單順序；既有的 Python cfg 仍負責模型建構、
checkpoint、前處理和 XAI layer 設定。

## 新增模型

1. 在 `model/` 建立 Python cfg。
2. 在 `model_catalog.json` 的 `models` 陣列新增一筆登錄。
3. 執行 `python -m unittest tests.test_model_catalog tests.test_model_selector`。

範例：

```json
{
  "id": "unet_3d",
  "name": "unet_3d",
  "config": "model/unet_3d.py",
  "family": "UNet 3D",
  "output_classes": 4,
  "tags": [],
  "description": "",
  "enabled": true
}
```

- `id`：穩定且唯一的識別碼。
- `name`：使用者看到的名稱，可以改成更容易辨認的名稱。
- `config`：相對於目錄 JSON 所在目錄的 Python cfg 路徑；不可指向目錄外。
- `family`：模型家族，也是篩選條件。新增家族會自動出現在篩選選單中。
- `output_classes`：模型輸出類別數，包含背景；必須與 cfg 的 `out_channels`
  或 `num_classes` 相符。這個數字不能當成資料集或解剖類別的判定。
- `tags`：搜尋用的字串陣列，可登錄資料集、變體或實驗名稱。
- `description`：選取後顯示的說明，也會納入搜尋。
- `enabled`：省略時為 `true`；設為 `false` 可保留登錄但從選單隱藏。

目錄格式版本為 `1`。啟用的設定檔不存在、ID／設定檔路徑重複，或格式不合法時，
載入器會明確報錯，不會退回目錄掃描。讀取清單不會執行 Python cfg，也不會載入
模型或權重；權重相容性仍由實際模型載入流程檢查。

## 選擇方式

點主工具列的模型選單，開啟搜尋與篩選視窗。搜尋不分大小寫，空白分隔的關鍵字
必須全部符合；可同時套用模型家族與輸出類別數篩選。搜尋涵蓋名稱、ID、設定檔名、
家族、標籤和說明。選取列後按 **Use model** 或雙擊才會切換目前模型；
**Cancel** 或 Escape 會保留原本模型。沒有結果時無法確認。
