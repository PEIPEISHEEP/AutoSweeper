# -*- coding: utf-8 -*-
"""
掃雷自動器（多模組版）
- 主要分層：
  1) Config: 全域設定/參數
  2) TemplateManager: 載入/管理模板圖
  3) ScreenHelper: 擷取螢幕、互動選點
  4) FaceDetector: 臉部狀態偵測/點擊重開
  5) BoardRecognizer: 盤面擷取、OCR 與模板/直方圖辨識、棋盤更新
  6) Solver: 基本鄰域邏輯推理，回傳下一步(open/flag)
  7) Visualizer: 調試時在新視窗預覽推理動作
  8) GameController: 主流程與事件迴圈

- 備註：
  * 所有座標、尺寸由使用者用滑鼠點兩角自動計算；支援不同 ROWS/COLS。
  * show_move 僅在 DEBUG_MODE=True 才顯示。
"""

import os
import time
import random
import cv2
import numpy as np
import pyautogui
from PIL import ImageGrab
import keyboard

# ===== 1) Config =============================================================

class Config:
    """全域設定（可依需求調整）"""
    def __init__(
        self,
        rows: int = 9,
        cols: int = 9,
        template_folder: str = "temp2",
        debug_mode: bool = True,
        use_ocr: bool = True,
        ocr_lang: str = "en",
        smile_threshold: float = 0.65,
        recog_size=(120, 120)
    ):
        # 棋盤尺寸
        self.ROWS = rows
        self.COLS = cols
        self.CELL_SIZE = None  # 由使用者框選後自動計算

        # 資源
        self.TEMPLATE_FOLDER = template_folder

        # 行為
        self.DEBUG_MODE = debug_mode
        self.USE_OCR = use_ocr
        self.OCR_LANG = ocr_lang
        self.SMILE_THRESHOLD = smile_threshold
        self.RECOG_SIZE = recog_size


# ===== 2) TemplateManager =====================================================

class TemplateManager:
    """管理模板圖像（未開格/空白格/表情圖）"""
    def __init__(self, cfg: Config):
        self.cfg = cfg
        self.templates = {}  # {-1: unopened_img, 0: empty_img}

    def load_cell_templates(self):
        """依目前 cell size 載入格子模板"""
        assert self.cfg.CELL_SIZE is not None, "CELL_SIZE 尚未設定"
        items = { -1: "unopened.png", 0: "empty.png" }
        self.templates.clear()
        for key, fname in items.items():
            path = os.path.join(self.cfg.TEMPLATE_FOLDER, fname)
            if os.path.exists(path):
                img = cv2.imread(path)
                if img is None:
                    continue
                self.templates[key] = cv2.resize(img, (self.cfg.CELL_SIZE, self.cfg.CELL_SIZE))

    def load_face_templates(self, face_w: int, face_h: int):
        """載入臉部模板"""
        sad_path = os.path.join(self.cfg.TEMPLATE_FOLDER, "sad.png")
        smile_path = os.path.join(self.cfg.TEMPLATE_FOLDER, "smile.png")
        sad_tpl = cv2.imread(sad_path)
        smile_tpl = cv2.imread(smile_path)
        if sad_tpl is None or smile_tpl is None:
            raise RuntimeError("找不到臉部模板（sad.png/smile.png）")
        return cv2.resize(sad_tpl, (face_w, face_h)), cv2.resize(smile_tpl, (face_w, face_h))


# ===== 3) ScreenHelper ========================================================

class ScreenHelper:
    """螢幕擷取、互動選點、通用工具"""
    @staticmethod
    def screenshot_bgr():
        """全螢幕截圖（BGR）"""
        img = ImageGrab.grab()
        return cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)

    @staticmethod
    def grab_region_bgr(x1, y1, x2, y2):
        """指定區域截圖（BGR）"""
        img = ImageGrab.grab(bbox=(x1, y1, x2, y2))
        return cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)

    @staticmethod
    def select_click_point(image_bgr, prompt: str):
        """在圖上點一下，回傳座標"""
        clicked = []
        temp = image_bgr.copy()
        cv2.putText(temp, prompt, (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (255,255,255), 2)

        def on_click(event, x, y, flags, param):
            if event == cv2.EVENT_LBUTTONDOWN:
                clicked.append((x, y))
                cv2.destroyAllWindows()

        cv2.imshow(prompt, temp)
        cv2.setMouseCallback(prompt, on_click)
        cv2.waitKey(0)
        return clicked[0] if clicked else None

    @staticmethod
    def init_rect(image_bgr, tip_left: str, tip_right: str):
        """請使用者點選左上/右下角，回傳 ((x1,y1),(x2,y2))"""
        print(tip_left)
        tl = ScreenHelper.select_click_point(image_bgr, tip_left)
        print(tip_right)
        br = ScreenHelper.select_click_point(image_bgr, tip_right)
        if not tl or not br:
            raise RuntimeError("未正確點擊兩角")
        (x1, y1), (x2, y2) = tl, br
        if x2 <= x1 or y2 <= y1:
            raise RuntimeError("無效的區域範圍（右下需大於左上）")
        return (x1, y1), (x2, y2)


# ===== 4) FaceDetector =========================================================

class FaceDetector:
    """臉部狀態辨識（哭臉/笑臉）與重開點擊"""
    def __init__(self, cfg: Config, tmpl_mgr: TemplateManager, face_rect):
        self.cfg = cfg
        self.tmpl_mgr = tmpl_mgr
        self.face_rect = face_rect  # ((x1,y1),(x2,y2))
        (x1, y1), (x2, y2) = face_rect
        self.sad_tpl, self.smile_tpl = tmpl_mgr.load_face_templates(x2 - x1, y2 - y1)

    def check_face(self):
        """回傳 ("sad"/"smile", max_val)"""
        (x1, y1), (x2, y2) = self.face_rect
        face_bgr = ScreenHelper.grab_region_bgr(x1, y1, x2, y2)
        sad_res = cv2.matchTemplate(face_bgr, self.sad_tpl, cv2.TM_CCOEFF_NORMED)
        smile_res = cv2.matchTemplate(face_bgr, self.smile_tpl, cv2.TM_CCOEFF_NORMED)
        _, sad_max, _, _ = cv2.minMaxLoc(sad_res)
        _, smile_max, _, _ = cv2.minMaxLoc(smile_res)
        print(f"哭臉相似度：{sad_max:.2f}, 笑臉相似度：{smile_max:.2f}")
        return ("sad", sad_max) if sad_max >= smile_max else ("smile", smile_max)

    def click_face_center(self):
        """點擊臉中間（重來）"""
        (x1, y1), (x2, y2) = self.face_rect
        cx, cy = (x1 + x2) // 2, (y1 + y2) // 2
        pyautogui.moveTo(cx, cy, duration=0.1)
        pyautogui.click()


# ===== 5) BoardRecognizer ======================================================

class BoardRecognizer:
    """
    棋盤擷取與格子辨識：
    - 優先 OCR（可選），失敗則用灰階直方圖與模板比較（unopened/empty）
    - 維護 recognized dict 只記錄「已確定」的格子
    """
    def __init__(self, cfg: Config, tmpl_mgr: TemplateManager, board_topleft, board_cols, board_rows):
        self.cfg = cfg
        self.tmpl_mgr = tmpl_mgr
        self.board_tl = board_topleft  # (x1, y1)
        self.COLS = board_cols
        self.ROWS = board_rows
        self.ocr_model = None
        self.history_low = 1.0  # 用於記錄歷史最低相似度，觀察用

        # 嘗試載入 OCR
        if self.cfg.USE_OCR:
            try:
                from paddleocr import PaddleOCR
                self.ocr_model = PaddleOCR(det=False, rec=True, use_angle_cls=False, lang=self.cfg.OCR_LANG)
            except Exception as e:
                print(f"⚠️ 無法載入 PaddleOCR，改用非 OCR 模式：{e}")
                self.ocr_model = None

    # ---- 擷取與前處理 ----
    def capture_board_bgr(self):
        """依目前 ROWS/COLS/CELL_SIZE 擷取棋盤 BGR 圖"""
        x1, y1 = self.board_tl
        x2 = x1 + self.COLS * self.cfg.CELL_SIZE
        y2 = y1 + self.ROWS * self.cfg.CELL_SIZE
        return ScreenHelper.grab_region_bgr(x1, y1, x2, y2)
    
    @staticmethod
    def _resize_for_recog(bgr_img, target_size):
        """★ 新增：將影像縮放到指定辨識大小（target_size=(W,H)）。若為 None 則不縮放。"""
        if target_size is None:
            return bgr_img
        w, h = target_size
        if w is None or h is None:
            return bgr_img
        return cv2.resize(bgr_img, (w, h), interpolation=cv2.INTER_AREA)

    def preprocess_for_ocr(self, cell_bgr):
        """
        OCR 前處理：
        1) 先縮放到 cfg.RECOG_SIZE（若有設定）
        """
        img = self._resize_for_recog(cell_bgr, self.cfg.RECOG_SIZE)
        return img

    # ---- 基礎工具 ----
    @staticmethod
    def calculate_histogram(img):
        # --- 健檢：尺寸與像素分布 ---
        if img is None or img.size == 0:
            return None
        h, w = img.shape[:2]
        if h < 2 or w < 2:
            return None

        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        # 檢查是否幾乎全黑/全白（避免退化直方圖）
        mn, mx, mean, std = np.min(gray), np.max(gray), float(np.mean(gray)), float(np.std(gray))
        # print(f"[DBG] cell h×w={h}x{w}, min={mn}, max={mx}, mean={mean:.1f}, std={std:.1f}")

        if std < 1e-3:
            # 幾乎無對比，當作退化
            return None

        hist = cv2.calcHist([gray], [0], None, [256], [0, 256])
        # 用 L1 正規化比 L2 更不容易被極端值影響
        cv2.normalize(hist, hist, alpha=1.0, norm_type=cv2.NORM_L1)
        return hist

    @staticmethod
    def hist_similarity(h1, h2):
        """越接近 1 越相似"""
        return cv2.compareHist(h1, h2, cv2.HISTCMP_CORREL)

    # ---- 單格辨識 ----
    def recognize_cell(self, cell_bgr):
        """
        回傳：
          -2: 旗子（此處不由影像判斷，交由邏輯流程設定）
          -1: 未開
           0: 空白
         1..8: 數字
        """
        # (A) OCR 嘗試
        if self.ocr_model is not None:
            ocr_input = self.preprocess_for_ocr(cell_bgr)
            try:
                result = self.ocr_model.ocr(ocr_input, cls=False)
                if result and result[0]:
                    text, conf = result[0][0][1]
                    try:
                        val = int(text)
                        if 0 < val <= 8:
                            return val
                    except ValueError:
                        pass
            except Exception as e:
                # OCR 偶發錯誤時容錯，不中斷
                if self.cfg.DEBUG_MODE:
                    print(f"OCR 例外：{e}")

        # (B) 模板/直方圖法（辨識 unopened / empty）
        unopened_tpl = self.tmpl_mgr.templates.get(-1)
        empty_tpl = self.tmpl_mgr.templates.get(0)
        if unopened_tpl is not None and empty_tpl is not None:
            cell_resized = self._resize_for_recog(cell_bgr, self.cfg.RECOG_SIZE)
            un_resized   = self._resize_for_recog(unopened_tpl, self.cfg.RECOG_SIZE)
            em_resized   = self._resize_for_recog(empty_tpl, self.cfg.RECOG_SIZE)

            cell_hist = self.calculate_histogram(cell_resized)
            un_hist   = self.calculate_histogram(un_resized)
            em_hist   = self.calculate_histogram(em_resized)

            if cell_hist is None or un_hist is None or em_hist is None:
                # 這一格資料有問題，先跳過，視為未開，避免把錯誤資訊寫進 recognized
                return -1
            
            s_un = self.hist_similarity(cell_hist, un_hist)
            s_em = self.hist_similarity(cell_hist, em_hist)
            if self.cfg.DEBUG_MODE:
                print(f"直方圖相似度：未開={s_un:.2f}, 空白={s_em:.2f}")
            if s_em>s_un and self.history_low > s_em:
                self.history_low = s_em  # 更新最低相似度
            print(f"⚠️ 觀察到新的歷史最低相似度：{self.history_low:.2f}")
            if s_em < 0.97 or s_un >= s_em:
                return -1
            else:
                return 0

        # (C) 無模板時 fallback
        return -1

    # ---- 整盤更新 ----
    def update_board(self, recognized_dict):
        """
        只對未確定的格子做辨識，減少耗時。
        recognized_dict: {(r,c): val}
        回傳 board: 2D list
        """
        board_bgr = self.capture_board_bgr()
        board = [[-1]*self.COLS for _ in range(self.ROWS)]

        for r in range(self.ROWS):
            for c in range(self.COLS):
                if (r, c) in recognized_dict:
                    board[r][c] = recognized_dict[(r, c)]
                    continue
                y = r * self.cfg.CELL_SIZE
                x = c * self.cfg.CELL_SIZE
                cell = board_bgr[y:y+self.cfg.CELL_SIZE, x:x+self.cfg.CELL_SIZE]
                val = self.recognize_cell(cell)
                board[r][c] = val
                if val != -1:  # 只記錄已確定的內容
                    recognized_dict[(r, c)] = val
        return board

    # ---- 滑鼠行為 ----
    def click_cell_center(self, r, c):
        """移動並左鍵點擊某格中心"""
        x = self.board_tl[0] + c * self.cfg.CELL_SIZE + self.cfg.CELL_SIZE // 2
        y = self.board_tl[1] + r * self.cfg.CELL_SIZE + self.cfg.CELL_SIZE // 2
        pyautogui.moveTo(x, y, duration=0.05)
        pyautogui.click()


# ===== 6) Solver ===============================================================

class Solver:
    """最基本數字邏輯：決定 open/flag；無法推理時挑風險較低的位置猜"""
    def __init__(self, rows, cols):
        self.ROWS = rows
        self.COLS = cols

    def neighbors(self, r, c):
        for dr in [-1,0,1]:
            for dc in [-1,0,1]:
                if dr == 0 and dc == 0:
                    continue
                nr, nc = r+dr, c+dc
                if 0 <= nr < self.ROWS and 0 <= nc < self.COLS:
                    yield (nr, nc)

    def infer(self, board, visited):
        safe_moves = set()
        flag_moves = set()

        for r in range(self.ROWS):
            for c in range(self.COLS):
                val = board[r][c]
                if val <= 0:
                    continue
                nbrs = list(self.neighbors(r, c))
                unopened = [(nr, nc) for (nr, nc) in nbrs if board[nr][nc] == -1]
                flagged = sum(1 for (nr, nc) in nbrs if board[nr][nc] == -2)

                # 地雷數等於未開+旗子 => 未開全部是雷
                if len(unopened) + flagged == val:
                    flag_moves.update(unopened)
                # 已插旗等於數字 => 其餘未開皆安全
                elif flagged == val and unopened:
                    safe_moves.update(unopened)

        # 先 open（安全）
        for (r, c) in safe_moves:
            if (r, c) not in visited:
                return ("open", r, c)

        # 再 flag
        for (r, c) in flag_moves:
            if (r, c) not in visited:
                return ("flag", r, c)

        # 無法推理 → 從未開中挑鄰近數字少的格子猜
        candidates = [(r, c) for r in range(self.ROWS) for c in range(self.COLS)
                      if board[r][c] == -1 and (r, c) not in visited]
        if candidates:
            def danger_level(pos):
                rr, cc = pos
                return sum(1 for (nr, nc) in self.neighbors(rr, cc) if board[nr][nc] > 0)
            candidates.sort(key=danger_level)
            best = candidates[0]
            print(f"⚠️ 無法推理，猜測：{best}")
            return ("open", best[0], best[1])

        return None


# ===== 7) Visualizer ===========================================================

class Visualizer:
    """DEBUG 用：畫出當前棋盤與即將執行的動作"""
    def __init__(self, cfg: Config):
        self.cfg = cfg

    def show_move(self, move, board):
        if not self.cfg.DEBUG_MODE or move is None:
            return
        action, r, c = move
        H = self.cfg.ROWS * self.cfg.CELL_SIZE
        W = self.cfg.COLS * self.cfg.CELL_SIZE
        img = np.zeros((H, W, 3), dtype=np.uint8)

        for row in range(self.cfg.ROWS):
            for col in range(self.cfg.COLS):
                x1, y1 = col * self.cfg.CELL_SIZE, row * self.cfg.CELL_SIZE
                x2, y2 = x1 + self.cfg.CELL_SIZE, y1 + self.cfg.CELL_SIZE
                cv2.rectangle(img, (x1, y1), (x2, y2), (255,255,255), 1)

                v = board[row][col]
                text, color = "", (255,255,255)
                if v >= 0:
                    text, color = str(v), (0,255,0)
                elif v == -2:
                    text, color = "F", (0,0,255)

                if text:
                    fs, th = 0.6, 1
                    (tw, th_text), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, fs, th)
                    tx = x1 + (self.cfg.CELL_SIZE - tw) // 2
                    ty = y1 + (self.cfg.CELL_SIZE + th_text) // 2
                    cv2.putText(img, text, (tx, ty), cv2.FONT_HERSHEY_SIMPLEX, fs, color, th)

        # 標註動作目標
        x1, y1 = c * self.cfg.CELL_SIZE, r * self.cfg.CELL_SIZE
        x2, y2 = x1 + self.cfg.CELL_SIZE, y1 + self.cfg.CELL_SIZE
        color = (0,255,0) if action == "open" else (0,0,255)
        cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)

        cv2.imshow("Move Preview", img)
        cv2.waitKey(1)


# ===== 8) GameController =======================================================

class GameController:
    """主流程：初始化、事件迴圈、整合各模組"""
    def __init__(self, cfg: Config):
        self.cfg = cfg
        self.tmpl_mgr = TemplateManager(cfg)
        self.solver = Solver(cfg.ROWS, cfg.COLS)
        self.visualizer = Visualizer(cfg)

        # 遊戲區與臉部區域
        self.board_tl = None        # (x1, y1)
        self.face_rect = None       # ((x1,y1),(x2,y2))

        # 模組
        self.face = None
        self.board_rec = None

        # 狀態
        self.visited = set()
        self.recognized = {}

    def init_ui_regions(self):
        """讓使用者選棋盤與臉部兩區域，並據此設定 cell size 與模板"""
        print("⌛ 2 秒後開始，請切到遊戲視窗...")
        time.sleep(2)
        img = ScreenHelper.screenshot_bgr()

        # 棋盤區（左上/右下）
        tl, br = ScreenHelper.init_rect(img, "Click Top-left of Board", "Click Bottom-right of Board")
        x1, y1 = tl
        x2, y2 = br

        # 自動估算格子大小
        width = x2 - x1
        height = y2 - y1
        # 以 COLS 為基準計算格寬
        self.cfg.CELL_SIZE = width // self.cfg.COLS
        if self.cfg.CELL_SIZE <= 0:
            raise RuntimeError("CELL_SIZE 計算失敗，請確認框選範圍/列數行數")
        print(f"✅ 棋盤設定完成：左上={tl}, ROWS={self.cfg.ROWS}, COLS={self.cfg.COLS}, CELL_SIZE={self.cfg.CELL_SIZE}")
        self.board_tl = tl

        # 臉部區（左上/右下）
        ftl, fbr = ScreenHelper.init_rect(img, "Click Top-left of Face", "Click Bottom-right of Face")
        self.face_rect = (ftl, fbr)
        print(f"✅ 臉部範圍設定完成：左上={ftl}, 右下={fbr}")

        # 載入模板
        self.tmpl_mgr.load_cell_templates()

        # 建立偵測/辨識模組
        self.face = FaceDetector(self.cfg, self.tmpl_mgr, self.face_rect)
        self.board_rec = BoardRecognizer(self.cfg, self.tmpl_mgr, self.board_tl, self.cfg.COLS, self.cfg.ROWS)

    def loop(self):
        """主事件迴圈"""
        input("✅ 設定完成，按 Enter 開始自動掃雷...")

        while True:
            # 允許 ESC 中止
            if keyboard.is_pressed("esc"):
                print("🛑 使用者中止（ESC）")
                break

            # 臉部狀態
            face_status, max_val = self.face.check_face()
            if face_status == "sad":
                print("❌ 偵測哭臉，自動重來")
                self.face.click_face_center()
                self.visited.clear()
                self.recognized.clear()
                time.sleep(1)
                continue
            elif face_status == "smile" and max_val >= self.cfg.SMILE_THRESHOLD:
                if self.cfg.DEBUG_MODE:
                    print("😄 偵測到笑臉，繼續遊戲")
            else:
                if self.cfg.DEBUG_MODE:
                    print("😐 臉部狀態不明，請檢查")

            # 更新棋盤
            board = self.board_rec.update_board(self.recognized)

            # 同步 visited（所有已知 >=0 的都算已開）
            for r in range(self.cfg.ROWS):
                for c in range(self.cfg.COLS):
                    if board[r][c] >= 0:
                        self.visited.add((r, c))

            # 勝利判定：已開+旗子==總格數
            opened = sum(1 for row in board for v in row if v >= 0)
            flagged = sum(1 for row in board for v in row if v == -2)
            if opened + flagged == self.cfg.ROWS * self.cfg.COLS:
                print("🎉 勝利！")
                self.visualizer.show_move(move, board)
                break

            # 決策
            move = self.solver.infer(board, self.visited)
            if not move:
                # 無推理 → 隨機開一格未開（備援）
                unopened = [(r, c) for r in range(self.cfg.ROWS) for c in range(self.cfg.COLS)
                            if board[r][c] == -1 and (r, c) not in self.visited]
                if not unopened:
                    print("無可行動格子，結束")
                    break
                move = ("open",) + random.choice(unopened)

            # 視覺化（DEBUG）
            self.visualizer.show_move(move, board)

            # 執行動作
            action, rr, cc = move
            self.visited.add((rr, cc))
            if action == "flag":
                # 旗子狀態由邏輯層決定（這裡不做實際右鍵行為，避免誤觸）
                # 若你想要真的插旗，可改成 pyautogui.rightClick()
                self.recognized[(rr, cc)] = -2
            else:
                self.board_rec.click_cell_center(rr, cc)

            time.sleep(0.5)


# ===== 入口點 ==================================================================

def main():
    """主程式進入點（請依需求調整參數）"""
    cfg = Config(
        rows=9,
        cols=9,
        template_folder="temp2",
        debug_mode=True,      # 想更安靜就關掉
        use_ocr=True,         # PaddleOCR 
        ocr_lang="en",
        smile_threshold=0.65,  # 笑臉相似度門檻
        recog_size=(128, 128),  # ★ 例如統一縮放到 128x128 再辨識
    )

    controller = GameController(cfg)
    controller.init_ui_regions()
    controller.loop()
    cv2.waitKey(0)
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
