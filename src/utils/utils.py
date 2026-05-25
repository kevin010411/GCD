import time
from contextlib import contextmanager


@contextmanager
def timer(
    name: str = None,
    track_gpu: bool = False,
    device: str = "cuda:0",
):
    """
    計算執行時間，並可選擇追蹤最高 GPU 記憶體占用。

    參數：
      name (str): 此區塊名稱，用於輸出或記錄。
      track_gpu (bool): 是否追蹤 GPU 最高記憶體占用。
      device (str): 指定追蹤的 GPU 裝置，例如 "cuda:0"。
    """
    start = time.perf_counter()
    gpu_peak = None
    torch = None

    if track_gpu:
        import torch

    if torch is not None and torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats(device)

    try:
        yield
    finally:
        end = time.perf_counter()
        elapsed = end - start

        if torch is not None and torch.cuda.is_available():
            gpu_peak = torch.cuda.max_memory_allocated(device) / (1024**2)  # 轉換為 MB

        # 格式化輸出
        msg = f"[{name}] " if name else ""
        msg += f"耗時: {elapsed:.4f} 秒"
        if gpu_peak is not None:
            msg += f"，最高 GPU 記憶體: {gpu_peak:.2f} MB"
        print(msg)
