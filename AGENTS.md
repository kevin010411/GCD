# GCD 專案指引

## 執行環境

- 本專案使用 Windows 11 原生環境與 PowerShell，依賴及 Python 執行由 `uv` 管理。
- 從 GCD 專案根目錄執行 `uv sync --locked` 安裝依賴；GUI 使用 `uv run main.py`，批次入口與測試使用 `uv run python ...`。
- 不使用 Docker 執行本專案；新增或更新操作說明時，使用 Windows 路徑與 PowerShell 語法，不寫 Docker／Main 容器指令。
- 影像、checkpoint 與輸出路徑必須是 Windows 可存取的路徑；歷史容器驗證紀錄保留原始環境，不視為 Windows 驗收。

## experiments 的用途

`experiments/` 是 GCD 功能的批量執行入口，主要用途是對多個案例、模型、XAI 方法與設定產生可重現的量化數據。它將原本由 Qt UI 操作的 `src/` 功能，透過 CLI、cfg 或批次工作執行。

`src/` 是核心邏輯的唯一來源。Qt UI 與 experiments 是同一套核心功能的兩種操作入口；在相同輸入與設定下，兩者必須得到一致的核心結果。

## 職責與實作規則

- `src/` 負責模型建構與權重載入、影像前處理、sliding-window 推論、XAI 計算、tile 融合、後處理、影像空間轉換及量化指標的核心計算。
- `config/` 是共用 Python cfg 的正式位置，描述模型、權重、前處理步驟、window size、overlap、融合方式與其他執行設定。experiments 的 cfg 可透過 `_base_` 繼承並配置批次任務。
- `experiments/` 負責案例列舉、參數組合、排程、進度、續跑、結果彙整，以及 CSV／JSON／圖表等研究輸出。
- experiments 必須呼叫 `src/` 的共用服務或 API，不得複製或另寫一套前處理、推論、XAI、融合或評分公式。新增量化指標的核心計算也應放在 `src/`，由 experiments 批量呼叫。
- 若現有功能只能從 Qt UI 呼叫，先將核心運算抽成不依賴 Qt widget、事件迴圈或視窗狀態的 `src/` 服務，再讓 UI 與 experiments 共用。批次入口不得靠模擬 UI 操作完成運算。
- 若修改核心邏輯，必須修改共用的 `src/` 實作；不得只在 experiments 修補並留下 UI 與批次兩套不同的行為。
- 若發現現有 experiments 含獨立核心實作，應將它視為需要收斂的架構問題。此指引不代表現有程式已全部符合共用要求。

## UI 與批次結果的一致性

相同案例與 cfg 必須使用相同的模型、權重、前處理順序、強度處理、window size、overlap、padding、融合方式、target class／mask、layer、後處理與影像幾何資訊。方法本身的參數也必須一致。

實驗需要不同設定或評分協定時，透過明確的 cfg／args 與共用服務參數表達，並記錄於結果中。不得根據入口名稱、profile 或 UI／批次模式，隱藏切換核心演算法。

批次與 UI 可以有不同的案例數量、呈現方式、輸出格式及執行排程。任何可能改變數值結果的最佳化，必須先驗證等價性。

## 驗證與結果紀錄

- 修改共用功能或新增批次入口後，使用相同案例與完整設定，比對 UI 後端服務和 experiments 的模型輸入、logits／prediction、heatmap，以及相關指標。
- 應比對張量數值、shape 與 affine；浮點誤差容忍值需明確且有理由。不能只憑視覺相似或兩個入口各自通過測試，就宣稱結果一致。
- 結果應記錄案例識別、模型與權重來源、解析後 cfg、實際執行參數、評分協定及可追溯的實作版本。只在同一協定下彙整或比較數據。
- 驗證只能確認已測試的模型、案例、方法與設定；未測試的路徑及未解決的差異應明確報告。
