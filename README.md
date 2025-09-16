# 踩地雷自動器 (AutoSweeper)

本專案是一個多模組化的 Python 踩地雷自動器，支援螢幕擷取、模板比對、OCR 辨識與自動推理點格。

## 操作展示

> 若無法直接播放，請下載觀看。

[![Demo](demo.mp4)](demo.mp4)

<video src="demo.mp4" controls width="480"></video>

## 目錄結構

```
minesweeper.py
demo.mp4
temp2/
  unopened.png
  empty.png
  sad.png
  smile.png
```

## 安裝需求

- Python 3.7+
- opencv-python
- numpy
- pyautogui
- pillow
- keyboard
- paddleocr（可選，若需 OCR）

安裝指令：
```sh
pip install opencv-python numpy pyautogui pillow keyboard
pip install paddleocr  
```

## 使用方式

1. 準備模板圖（放在 `temp2/` 資料夾，檔名需對應）。
2. 執行主程式：
   ```sh
   python minesweeper.py
   ```
3. 依指示點選棋盤與臉部區域兩角。
4. 按 Enter 開始自動踩地雷。
5. 遊戲進行中可按 `ESC` 中止。

## 參數調整

可在 [`main()`](minesweeper.py) 裡調整棋盤大小、模板資料夾、DEBUG 模式等。

---

如需更多細節，請參考 [minesweeper.py](minesweeper.py) 內註解。