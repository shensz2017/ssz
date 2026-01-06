import sys
import os
import json
import time
import requests
import base64
import threading
import re
import traceback
import numpy as np 
import cv2 
import math
from datetime import datetime
from concurrent.futures import ThreadPoolExecutor

from PyQt6.QtWidgets import (QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout, 
                             QLabel, QPushButton, QTextEdit, QScrollArea, QFrame, 
                             QGridLayout, QComboBox, QProgressBar, QLineEdit, QMessageBox, 
                             QDialog, QFormLayout, QFileDialog, QGroupBox, QSplitter, QSpinBox, 
                             QMenu, QCheckBox, QTabWidget, QGraphicsDropShadowEffect, QSizePolicy)
from PyQt6.QtCore import (Qt, pyqtSignal, QThread, QMimeData, QObject, pyqtSlot, QRunnable, 
                          QThreadPool, QTimer, QCoreApplication, QRect, QSize)
from PyQt6.QtGui import (QPixmap, QDragEnterEvent, QDropEvent, QDrag, QAction, QCursor, 
                         QTextCursor, QColor, QPalette, QIcon, QFont, QPainter, QAction)

# ==========================================
# 0. 全局配置与工具
# ==========================================
def exception_hook(exctype, value, traceback_obj):
    err_msg = "".join(traceback.format_exception(exctype, value, traceback_obj))
    print(f"CRITICAL ERROR: {err_msg}")

sys.excepthook = exception_hook

if getattr(sys, 'frozen', False):
    BASE_DIR = os.path.dirname(sys.executable)
else:
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))

CONFIG_FILE = os.path.join(BASE_DIR, "config_grsai.json")
MAX_ASYNC_WORKERS = 30
API_TIMEOUT = 300 

API_HOST = "https://api.grsai.com"
URL_CHAT = f"{API_HOST}/v1/chat/completions"       
URL_DRAW_NANO = f"{API_HOST}/v1/draw/nano-banana"  
URL_VIDEO_VEO = f"{API_HOST}/v1/video/veo"         
URL_RESULT = f"{API_HOST}/v1/draw/result"          
URL_IMGBB = "https://api.imgbb.com/1/upload"       

def load_config():
    if os.path.exists(CONFIG_FILE):
        try:
            with open(CONFIG_FILE, 'r') as f: return json.load(f)
        except: pass
    return {"grsai_key": "", "imgbb_key": ""}

def save_config(k1, k2):
    with open(CONFIG_FILE, 'w') as f: json.dump({"grsai_key": k1, "imgbb_key": k2}, f)

def ensure_dir(dir_name):
    full_path = os.path.join(BASE_DIR, dir_name)
    if not os.path.exists(full_path):
        os.makedirs(full_path)
    return os.path.abspath(full_path)

def cv2_save_image_safe(path, img):
    try:
        folder = os.path.dirname(path)
        if not os.path.exists(folder): os.makedirs(folder)
        is_success, buffer = cv2.imencode(".jpg", img)
        if is_success:
            buffer.tofile(path)
            return True, "Success"
        else:
            return False, "Encode Failed"
    except Exception as e:
        return False, str(e)

# ==========================================
# 1. 日志系统
# ==========================================
class Logger(QObject):
    log_signal = pyqtSignal(str, str)
    def log(self, level, msg):
        t = datetime.now().strftime("%H:%M:%S")
        self.log_signal.emit(level, f"[{t}] {msg}")
    def info(self, m): self.log("INFO", m)
    def warn(self, m): self.log("WARN", m)
    def error(self, m): self.log("ERROR", m)
    def step(self, m): self.log("STEP", m)

global_logger = Logger()

# ==========================================
# 2. 网络核心
# ==========================================
class APIClient:
    def __init__(self):
        cfg = load_config()
        self.grsai_key = cfg.get("grsai_key", "")
        self.imgbb_key = cfg.get("imgbb_key", "")
        self.headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self.grsai_key}"
        }

    def _clean_gemini_output(self, raw_text):
        cleaned = re.sub(r'<think>.*?</think>', '', raw_text, flags=re.DOTALL)
        cleaned = re.sub(r'```json', '', cleaned)
        cleaned = re.sub(r'```', '', cleaned)
        return cleaned.strip()

    def upload_imgbb(self, file_path):
        if not self.imgbb_key: raise Exception("缺少 ImgBB Key")
        if str(file_path).startswith("http"): return file_path
        try:
            with open(file_path, "rb") as file:
                payload = {"key": self.imgbb_key}
                files = {"image": file}
                res = requests.post(URL_IMGBB, data=payload, files=files, timeout=60)
                res.raise_for_status()
                return res.json()['data']['url']
        except Exception as e:
            print(f"ImgBB Error: {e}")
            return None

    def download(self, url, save_path):
        try:
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            with requests.get(url, stream=True, timeout=120) as r:
                r.raise_for_status()
                with open(save_path, 'wb') as f:
                    for chunk in r.iter_content(8192): f.write(chunk)
            return True
        except Exception as e:
            global_logger.error(f"Download Error: {e}")
            return False

    def poll_task_with_progress(self, task_id, progress_callback=None):
        payload = {"id": task_id}
        start_t = time.time()
        while True:
            if time.time() - start_t > 600: raise Exception("任务超时")
            try:
                r = requests.post(URL_RESULT, headers=self.headers, json=payload, timeout=30)
                if r.status_code == 200:
                    d = r.json().get('data', {})
                    status = d.get('status')
                    progress = d.get('progress', 0)
                    if progress_callback: progress_callback(progress)
                    if status == 'succeeded': return d
                    elif status == 'failed': 
                        raise Exception(d.get('failure_reason') or d.get('error'))
                time.sleep(2)
            except Exception as e: raise e

    def poll_task(self, task_id):
        return self.poll_task_with_progress(task_id)

    def stream_video_script(self, video_path, model, manual_prompt, ref_urls):
        try:
            with open(video_path, "rb") as video_file:
                video_b64 = base64.b64encode(video_file.read()).decode('utf-8')
        except Exception as e:
            yield f"\n[Error] 读取视频失败: {e}"
            return

        sys_prompt = """
        你是一个专业分镜师。深度分析视频，结合参考图和指令生成分镜脚本。
        输出要求：
        1. 不要表格，不要 <think>。
        2. 按 [00:00 - 00:05] 时间格式分段。
        3. 包含：Theme, Characters, Scene Breakdown (Camera, Action).
        """
        mime_type = "video/mp4"
        if video_path.lower().endswith(".mov"): mime_type = "video/quicktime"
        video_data_uri = f"data:{mime_type};base64,{video_b64}"
        
        user_content = [
            {"type": "text", "text": f"指令: {manual_prompt}. 生成脚本。"},
            {"type": "image_url", "image_url": {"url": video_data_uri}}
        ]
        if ref_urls:
            for url in ref_urls: user_content.append({"type": "image_url", "image_url": {"url": url}})
        
        payload = {"model": model, "stream": True, "messages": [{"role": "system", "content": sys_prompt}, {"role": "user", "content": user_content}], "max_tokens": 4000}

        try:
            response = requests.post(URL_CHAT, headers=self.headers, json=payload, stream=True, timeout=300)
            if response.status_code != 200:
                yield f"\n[API Error] Status: {response.status_code}"
                return

            yield "\n[System] Connecting to Gemini...\n\n"
            for line in response.iter_lines():
                if line:
                    decoded = line.decode('utf-8')
                    if decoded.startswith("data: "):
                        data_str = decoded[6:]
                        if data_str.strip() == "[DONE]": break
                        try:
                            json_data = json.loads(data_str)
                            if 'choices' in json_data:
                                delta = json_data['choices'][0].get('delta', {})
                                content = delta.get('content', "")
                                if content: yield content
                        except: pass
        except Exception as e:
            yield f"\n[System Error] Link broken: {str(e)}"

    def analyze_video_direct(self, video_path, count, model):
        try:
            with open(video_path, "rb") as video_file:
                video_b64 = base64.b64encode(video_file.read()).decode('utf-8')
            sys_prompt = f"Extract {count} highlights. JSON Array: [{{'seconds': 2.5, 'description': '...'}}]"
            mime_type = "video/mp4"
            if video_path.lower().endswith(".mov"): mime_type = "video/quicktime"
            video_data_uri = f"data:{mime_type};base64,{video_b64}"
            user_content = [{"type": "text", "text": f"Extract {count} highlights."}, {"type": "image_url", "image_url": {"url": video_data_uri}}]
            payload = {"model": model, "stream": False, "messages": [{"role": "system", "content": sys_prompt}, {"role": "user", "content": user_content}]}
            r = requests.post(URL_CHAT, headers=self.headers, json=payload, timeout=API_TIMEOUT)
            txt = self._clean_gemini_output(r.json()['choices'][0]['message']['content'])
            match = re.search(r'\[.*\]', txt, re.DOTALL)
            if match: return json.loads(match.group())
            return []
        except: return []

    def agent_multimodal_split(self, script, count, model, image_urls):
        sys_prompt = f"Director. Split into {count} prompts based on script/images. JSON Array."
        user_content = [{"type": "text", "text": f"Script: {script}"}]
        for url in image_urls: user_content.append({"type": "image_url", "image_url": {"url": url}})
        payload = {"model": model, "stream": False, "messages": [{"role": "system", "content": sys_prompt}, {"role": "user", "content": user_content}]}
        try:
            r = requests.post(URL_CHAT, headers=self.headers, json=payload, timeout=API_TIMEOUT)
            content = self._clean_gemini_output(r.json()['choices'][0]['message']['content'])
            j = json.loads(content)
            return j[:count] if isinstance(j, list) else [content]
        except: return []

    def chat_split_script(self, script, count, model):
        sys_prompt = f"Director. Split into {count} prompts. JSON Array."
        payload = {"model": model, "stream": False, "messages": [{"role": "system", "content": sys_prompt}, {"role": "user", "content": script}]}
        try:
            r = requests.post(URL_CHAT, headers=self.headers, json=payload, timeout=API_TIMEOUT)
            content = self._clean_gemini_output(r.json()['choices'][0]['message']['content'])
            j = json.loads(content)
            return j[:count] if isinstance(j, list) else [content]
        except: return []

    def optimize_video_prompt(self, img_url, original_prompt, model):
        sys_prompt = "Optimize for Veo video generation. Short, English."
        user_content = [{"type": "text", "text": f"Intent: {original_prompt}"}, {"type": "image_url", "image_url": {"url": img_url}}]
        payload = {"model": model, "stream": False, "messages": [{"role": "system", "content": sys_prompt}, {"role": "user", "content": user_content}]}
        try:
            r = requests.post(URL_CHAT, headers=self.headers, json=payload, timeout=API_TIMEOUT)
            return self._clean_gemini_output(r.json()['choices'][0]['message']['content'])
        except: return original_prompt

# ==========================================
# 3. 异步任务 Worker
# ==========================================
class WorkerSignals(QObject):
    finished = pyqtSignal(object) 
    error = pyqtSignal(str) 
    progress = pyqtSignal(str)

class ScriptAnalysisThread(QThread):
    chunk_received = pyqtSignal(str); finished_signal = pyqtSignal(); log_signal = pyqtSignal(str, str)
    def __init__(self, client, video_path, model, manual_prompt, local_assets):
        super().__init__(); self.c = client; self.vp = video_path; self.m = model
        self.prompt = manual_prompt; self.assets = local_assets; self.is_running = True
    def run(self):
        self.log_signal.emit("STEP", "🎬 Script Analysis Started...")
        ref_urls = []
        if self.assets:
            for path in self.assets:
                if not self.is_running: return
                u = self.c.upload_imgbb(path)
                if u: ref_urls.append(u)
        for chunk in self.c.stream_video_script(self.vp, self.m, self.prompt, ref_urls):
            if not self.is_running: break
            self.chunk_received.emit(chunk)
        self.finished_signal.emit()
    def stop(self): self.is_running = False

class SmartAnalyzeTask(QRunnable):
    def __init__(self, client, vid_path, model, target_count):
        super().__init__(); self.c = client; self.p = vid_path; self.m = model; self.cnt = target_count
        self.signals = WorkerSignals()
    def run(self):
        try:
            self.signals.progress.emit(f"AI Analyzing...")
            ai_results = self.c.analyze_video_direct(self.p, self.cnt, self.m)
            if not ai_results: raise Exception("No AI results")
            cap = cv2.VideoCapture(self.p)
            ensure_dir("Highlights"); final_res = []; target_timestamps = []
            for item in ai_results:
                if isinstance(item, dict): target_timestamps.append({"sec": float(item.get('seconds', 0)), "desc": item.get('description', "AI Highlight")})
            target_timestamps.sort(key=lambda x: x["sec"]); target_timestamps = target_timestamps[:self.cnt]
            for i, item in enumerate(target_timestamps):
                sec = item["sec"]; cap.set(cv2.CAP_PROP_POS_MSEC, sec * 1000); ret, frame = cap.read()
                if ret:
                    fpath = os.path.abspath(f"Highlights/hl_{int(time.time())}_{i}.jpg")
                    cv2_save_image_safe(fpath, frame)
                    final_res.append({"path": fpath, "prompt": item["desc"]})
            cap.release(); self.signals.finished.emit(final_res)
        except Exception as e: self.signals.error.emit(str(e))

class AgentProcessThread(QThread):
    finished_signal = pyqtSignal(list, list); error_signal = pyqtSignal(str); log_signal = pyqtSignal(str, str) 
    def __init__(self, client, script, count, model, local_assets):
        super().__init__(); self.c=client; self.s=script; self.n=count; self.m=model; self.assets=local_assets
    def run(self):
        try:
            self.log_signal.emit("STEP", f"🚀 Agent: {self.m}")
            ref_urls = []
            if self.assets:
                for path in self.assets:
                    u = self.c.upload_imgbb(path)
                    if u: ref_urls.append(u)
            prompts = self.c.agent_multimodal_split(self.s, self.n, self.m, ref_urls) if ref_urls else self.c.chat_split_script(self.s, self.n, self.m)
            if prompts: self.finished_signal.emit(prompts, ref_urls)
            else: self.error_signal.emit("Empty Result")
        except Exception as e: self.error_signal.emit(str(e))

class ImageGenTask(QRunnable):
    def __init__(self, client, prompt, model, ratio, size, ref_urls, out_dir, target_slot="start"):
        super().__init__()
        self.c=client; self.p=prompt; self.m=model; self.r=ratio; self.s=size
        self.urls=ref_urls; self.d=out_dir; self.target = target_slot
        self.signals=WorkerSignals()
    def run(self):
        try:
            final_urls = []
            if self.urls:
                for path in self.urls:
                    if os.path.exists(path):
                        u = self.c.upload_imgbb(path)
                        if u: final_urls.append(u)
                    elif str(path).startswith("http"): final_urls.append(path)
            payload = {"model": self.m, "prompt": self.p, "aspectRatio": self.r, "imageSize": self.s, "urls": final_urls, "webHook": "-1", "shutProgress": False}
            self.signals.progress.emit(f"🎨 Generating ({self.target})...")
            r = requests.post(URL_DRAW_NANO, headers=self.c.headers, json=payload, timeout=30)
            r.raise_for_status(); tid = r.json()['data']['id']
            data = self.c.poll_task(tid)
            if 'results' in data:
                url = data['results'][0]['url']
                fname = f"gen_{self.target}_{int(time.time())}_{tid[-4:]}.png"
                save = os.path.join(self.d, fname)
                if self.c.download(url, save): 
                    self.signals.finished.emit((save, self.target))
                else: raise Exception("Download Failed")
            else: raise Exception("No Result")
        except Exception as e: self.signals.error.emit(str(e))

class VideoGenTask(QRunnable):
    def __init__(self, client, prompt, img_path, end_img_path, model, ratio, out_dir):
        super().__init__()
        self.c = client; self.p = prompt; self.i = img_path
        self.end_img = end_img_path
        self.m = model; self.r = ratio; self.d = out_dir 
        self.signals = WorkerSignals()
        
    def smart_normalize(self, img_path, target_ratio, reference_size=None):
        # 高清智能处理逻辑 (High-Res Smart Normalizer)
        try:
            img = cv2.imdecode(np.fromfile(img_path, dtype=np.uint8), cv2.IMREAD_COLOR)
            if img is None: return img_path
            
            h, w = img.shape[:2]
            
            # 1. 解析目标比例
            w_ratio, h_ratio = map(int, target_ratio.split(':'))
            
            # 2. 如果提供了参考尺寸 (End Frame)，直接对齐到该尺寸
            if reference_size:
                ref_w, ref_h = reference_size
                if w == ref_w and h == ref_h:
                    return img_path # 尺寸已完美，无需处理
                
                # 使用最高质量插值缩放尾帧
                resized = cv2.resize(img, (ref_w, ref_h), interpolation=cv2.INTER_LANCZOS4)
                
            else:
                # 3. 处理首帧 (Start Frame)：保留最大分辨率，进行中心裁切 (Center Crop)
                # 计算理想宽高
                target_ratio_val = w_ratio / h_ratio
                current_ratio_val = w / h
                
                if current_ratio_val > target_ratio_val:
                    # 图片太宽，裁两边
                    new_w = int(h * target_ratio_val)
                    offset = (w - new_w) // 2
                    resized = img[:, offset:offset+new_w]
                else:
                    # 图片太高，裁上下
                    new_h = int(w / target_ratio_val)
                    offset = (h - new_h) // 2
                    resized = img[offset:offset+new_h, :]
                
                # 4. 确保尺寸是 16 的倍数 (视频编码兼容性)
                final_h, final_w = resized.shape[:2]
                final_w = final_w - (final_w % 16)
                final_h = final_h - (final_h % 16)
                resized = resized[:final_h, :final_w]

            # 保存
            temp_name = f"temp_hd_{int(time.time())}_{'end' if reference_size else 'start'}.jpg"
            temp_path = os.path.abspath(os.path.join(self.d, temp_name))
            
            # 高质量保存
            is_success, buffer = cv2.imencode(".jpg", resized, [cv2.IMWRITE_JPEG_QUALITY, 98])
            if is_success:
                buffer.tofile(temp_path)
                # 如果是首帧，返回 (path, (width, height))，供尾帧参考
                return (temp_path, (resized.shape[1], resized.shape[0])) if not reference_size else temp_path
            return img_path
            
        except Exception as e:
            print(f"Smart Norm Error: {e}")
            return (img_path, None) if not reference_size else img_path

    def run(self):
        try:
            if not self.i or not os.path.exists(self.i):
                raise Exception("Start Image Not Found (Check Path)")

            # 1. 预处理首帧 (保留最大高清分辨率)
            self.signals.progress.emit("🔄 Processing Start Frame (High-Res)...")
            res = self.smart_normalize(self.i, self.r, reference_size=None)
            
            # 解包：可能是 (path, size) 或原 path
            if isinstance(res, tuple):
                norm_start_path, start_size = res
            else:
                norm_start_path, start_size = res, None

            # 2. 预处理尾帧 (强制对齐首帧尺寸)
            norm_end_path = None
            if self.end_img and os.path.exists(self.end_img):
                self.signals.progress.emit("🔄 Aligning End Frame...")
                # 如果首帧获取到了尺寸，强制尾帧对其
                if start_size:
                    norm_end_path = self.smart_normalize(self.end_img, self.r, reference_size=start_size)
                else:
                    # 兜底：如果首帧处理失败，尾帧也独立处理
                    res_end = self.smart_normalize(self.end_img, self.r, reference_size=None)
                    norm_end_path = res_end[0] if isinstance(res_end, tuple) else res_end

            # 3. 上传
            self.signals.progress.emit("📤 Uploading Start Frame...")
            u_start = self.c.upload_imgbb(norm_start_path) 
            if not u_start: raise Exception("Start frame upload failed")
            
            self.signals.progress.emit(f"📝 Prompt: {self.p}")

            payload = {
                "model": self.m, "prompt": self.p, "firstFrameUrl": u_start, 
                "aspectRatio": self.r, "webHook": "-1", "shutProgress": False
            }
            if norm_end_path:
                self.signals.progress.emit("📤 Uploading End Frame...")
                u_end = self.c.upload_imgbb(norm_end_path)
                if u_end:
                    payload["lastFrameUrl"] = u_end
                    self.signals.progress.emit("✅ End frame attached.")

            # 4. 提交
            self.signals.progress.emit(f"🎬 Submitting Task...")
            r = requests.post(URL_VIDEO_VEO, headers=self.c.headers, json=payload, timeout=30)
            r.raise_for_status(); tid = r.json()['data']['id']
            data = self.c.poll_task_with_progress(tid, lambda prog: self.signals.progress.emit(f"⏳ Generating: {prog}%"))
            if 'url' in data:
                fname = f"veo_{int(time.time())}_{tid[-4:]}.mp4"
                save = os.path.join(self.d, fname)
                if self.c.download(data['url'], save): self.signals.finished.emit(save)
                else: raise Exception("Download Failed")
            else: raise Exception("No Video URL")
            
            # 清理
            try:
                if norm_start_path != self.i and os.path.exists(norm_start_path): os.remove(norm_start_path)
                if norm_end_path and norm_end_path != self.end_img and os.path.exists(norm_end_path): os.remove(norm_end_path)
            except: pass
            
        except Exception as e: self.signals.error.emit(str(e))

class PromptOptimizeTask(QRunnable):
    def __init__(self, client, img_path, original_prompt, model):
        super().__init__(); self.c=client; self.i=img_path; self.p=original_prompt; self.m=model; self.signals=WorkerSignals()
    def run(self):
        try:
            self.signals.progress.emit("AI Optimizing...")
            url = self.c.upload_imgbb(self.i)
            if not url: raise Exception("Upload Failed")
            optimized = self.c.optimize_video_prompt(url, self.p, self.m)
            self.signals.finished.emit(optimized)
        except Exception as e: self.signals.error.emit(str(e))

# ==========================================
# 4. 现代 UI 组件
# ==========================================
class LogConsole(QTextEdit):
    def __init__(self):
        super().__init__(); self.setReadOnly(True)
        self.setStyleSheet("""background-color: #09090b; color: #4ade80; font-family: Consolas, monospace; font-size: 12px; border: 1px solid #27272a; border-radius: 8px; padding: 10px;""")
        global_logger.log_signal.connect(self.add_log)
    def add_log(self, level, msg):
        c = "#4ade80" 
        if level=="WARN": c="#facc15" 
        elif level=="ERROR": c="#f87171" 
        elif level=="STEP": c="#60a5fa" 
        self.append(f'<span style="color:{c}">{msg}</span>')
        self.verticalScrollBar().setValue(self.verticalScrollBar().maximum())

class AssetLabel(QLabel):
    delete_signal = pyqtSignal(object) 
    def __init__(self, path):
        super().__init__()
        self.path = path; self.setFixedSize(100, 100)
        self.setStyleSheet("""border: 2px solid #3f3f46; background-color: #27272a; border-radius: 8px;""")
        pix = QPixmap(path)
        if not pix.isNull():
            self.setPixmap(pix.scaled(100, 100, Qt.AspectRatioMode.KeepAspectRatio, Qt.TransformationMode.SmoothTransformation))
        self.setAlignment(Qt.AlignmentFlag.AlignCenter)
    def contextMenuEvent(self, event):
        menu = QMenu(self)
        del_act = menu.addAction("🗑️ Delete")
        if menu.exec(self.mapToGlobal(event.pos())) == del_act: self.delete_signal.emit(self)

class FullScreenViewer(QDialog):
    def __init__(self, image_path, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Full Screen Preview (Press ESC to Close)")
        self.setWindowState(Qt.WindowState.WindowMaximized)
        self.setStyleSheet("background-color: #000;")
        self.pixmap = QPixmap(image_path)
    def paintEvent(self, event):
        if self.pixmap.isNull(): return
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.SmoothPixmapTransform)
        scaled = self.pixmap.scaled(self.size(), Qt.AspectRatioMode.KeepAspectRatio, Qt.TransformationMode.SmoothTransformation)
        x = (self.width() - scaled.width()) // 2
        y = (self.height() - scaled.height()) // 2
        painter.drawPixmap(x, y, scaled)

class ZoomImageLabel(QLabel):
    clear_signal = pyqtSignal()
    def __init__(self, text="Double-click to Zoom", parent=None):
        super().__init__(text, parent)
        self.full_path = None
        self.cached_pixmap = None
        self.setCursor(Qt.CursorShape.PointingHandCursor)
        self.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.setStyleSheet("background-color: #18181b; border-radius: 6px; color: #52525b; font-size: 11px;")

    def set_image(self, path):
        if path is None:
            self.full_path = None
            self.cached_pixmap = None
            self.setText("Empty")
        else:
            self.full_path = path
            self.cached_pixmap = QPixmap(path)
            self.setText("")
        self.update()

    def paintEvent(self, event):
        if self.cached_pixmap and not self.cached_pixmap.isNull():
            painter = QPainter(self)
            painter.setRenderHint(QPainter.RenderHint.SmoothPixmapTransform)
            target_rect = self.rect()
            scaled = self.cached_pixmap.scaled(target_rect.size(), Qt.AspectRatioMode.KeepAspectRatio, Qt.TransformationMode.SmoothTransformation)
            x = (target_rect.width() - scaled.width()) // 2
            y = (target_rect.height() - scaled.height()) // 2
            painter.drawPixmap(x, y, scaled)
        else:
            super().paintEvent(event)

    def mouseDoubleClickEvent(self, event):
        if self.full_path and os.path.exists(self.full_path):
            viewer = FullScreenViewer(self.full_path, self)
            viewer.exec()
            
    def contextMenuEvent(self, event):
        menu = QMenu(self)
        clear_act = menu.addAction("🗑️ Clear Image")
        if menu.exec(self.mapToGlobal(event.pos())) == clear_act:
            self.clear_signal.emit()

# --- 核心卡片 ---
class StoryboardCard(QWidget):
    update_signal = pyqtSignal(object) 
    video_signal = pyqtSignal(object)
    error_signal = pyqtSignal(str)
    progress_signal = pyqtSignal(str)
    delete_signal = pyqtSignal(object)
    
    def __init__(self, index, prompt, parent_pool, client, main_window, ref_urls):
        super().__init__()
        self.img_path = None
        self.end_img_path = None
        self.pool = parent_pool; self.client = client; self.mw = main_window; self.ref_urls = ref_urls
        
        self.setFixedWidth(340) 
        self.setSizePolicy(QSizePolicy.Policy.Fixed, QSizePolicy.Policy.MinimumExpanding)
        
        self.container = QFrame(self)
        self.container.setObjectName("CardFrame")
        self.container.setStyleSheet("""QFrame#CardFrame { background-color: #27272a; border-radius: 12px; border: 1px solid #3f3f46; }""")
        
        self.shadow = QGraphicsDropShadowEffect(self)
        self.shadow.setBlurRadius(15); self.shadow.setYOffset(4); self.shadow.setColor(QColor(0,0,0,80))
        self.container.setGraphicsEffect(self.shadow)

        self.main_layout = QVBoxLayout(self); self.main_layout.setContentsMargins(5,5,5,5)
        self.main_layout.addWidget(self.container)

        l = QVBoxLayout(self.container); l.setContentsMargins(12, 12, 12, 12); l.setSpacing(8)

        # 1. Header
        top_bar = QHBoxLayout()
        self.chk_sel = QCheckBox(f"No.{index}"); self.chk_sel.setChecked(True)
        self.chk_sel.setStyleSheet("color: #a1a1aa; font-weight: bold;")
        btn_del = QPushButton("×"); btn_del.setFixedSize(24, 24)
        btn_del.setStyleSheet("background:transparent; color:#71717a; border:none; font-size:18px;")
        btn_del.clicked.connect(lambda: self.delete_signal.emit(self))
        top_bar.addWidget(self.chk_sel); top_bar.addStretch(); top_bar.addWidget(btn_del)
        l.addLayout(top_bar)

        # --- SECTION A: START FRAME ---
        start_group = QGroupBox("🎬 Start Frame"); start_group.setStyleSheet("color:#60a5fa; border:none; font-weight:bold; margin-top:0;")
        l_start = QVBoxLayout(start_group); l_start.setContentsMargins(0,10,0,0)
        
        self.img = ZoomImageLabel("Double-click to Zoom")
        self.img.setFixedHeight(160)
        self.img.clear_signal.connect(lambda: self.set_image_by_target(None, "start"))
        l_start.addWidget(self.img)

        self.txt_start = QTextEdit(); self.txt_start.setText(str(prompt)); self.txt_start.setFixedHeight(50)
        self.txt_start.setPlaceholderText("Start Frame Prompt...")
        self.txt_start.setStyleSheet("border: 1px solid #3f3f46; background: #18181b; font-size: 11px; color: #d4d4d8; padding: 4px; border-radius: 4px;")
        l_start.addWidget(self.txt_start)

        btns_start = QHBoxLayout()
        self.btn_remake_start = QPushButton("🎨 Remake"); self.btn_remake_start.setObjectName("PrimaryBtn")
        self.btn_remake_start.clicked.connect(lambda: self.on_remake_target("start"))
        self.btn_retry_start = QPushButton("🔄 Retry"); 
        self.btn_retry_start.setStyleSheet("background-color: #3f3f46; color: white;")
        self.btn_retry_start.clicked.connect(lambda: self.on_retry_target("start"))
        
        btn_clear_start = QPushButton("🗑️"); btn_clear_start.setFixedSize(30, 28)
        btn_clear_start.clicked.connect(lambda: self.set_image_by_target(None, "start"))
        
        btns_start.addWidget(self.btn_remake_start); btns_start.addWidget(self.btn_retry_start); btns_start.addWidget(btn_clear_start)
        l_start.addLayout(btns_start)
        l.addWidget(start_group)

        # --- SECTION B: END FRAME ---
        end_group = QGroupBox("🏁 End Frame"); end_group.setStyleSheet("color:#a1a1aa; border:none; font-weight:bold; margin-top:10px;")
        l_end = QVBoxLayout(end_group); l_end.setContentsMargins(0,10,0,0)
        
        self.img_end_preview = ZoomImageLabel("Double-click to Zoom")
        self.img_end_preview.setFixedHeight(160)
        self.img_end_preview.clear_signal.connect(lambda: self.set_image_by_target(None, "end"))
        l_end.addWidget(self.img_end_preview)

        self.txt_end = QTextEdit(); self.txt_end.setFixedHeight(50)
        self.txt_end.setPlaceholderText("End Frame Prompt (For Image Gen only)...")
        self.txt_end.setStyleSheet("border: 1px solid #3f3f46; background: #18181b; font-size: 11px; color: #d4d4d8; padding: 4px; border-radius: 4px;")
        l_end.addWidget(self.txt_end)

        btns_end = QHBoxLayout()
        self.btn_remake_end = QPushButton("🎨 Remake"); self.btn_remake_end.setObjectName("PrimaryBtn")
        self.btn_remake_end.clicked.connect(lambda: self.on_remake_target("end"))
        self.btn_retry_end = QPushButton("🔄 Retry"); 
        self.btn_retry_end.setStyleSheet("background-color: #3f3f46; color: white;")
        self.btn_retry_end.clicked.connect(lambda: self.on_retry_target("end"))
        
        btn_clear_end = QPushButton("🗑️"); btn_clear_end.setFixedSize(30, 28)
        btn_clear_end.clicked.connect(lambda: self.set_image_by_target(None, "end"))

        btns_end.addWidget(self.btn_remake_end); btns_end.addWidget(self.btn_retry_end); btns_end.addWidget(btn_clear_end)
        l_end.addLayout(btns_end)
        
        btn_upload_end = QPushButton("📂 Upload End Frame (Local)"); 
        btn_upload_end.setCursor(Qt.CursorShape.PointingHandCursor)
        btn_upload_end.setStyleSheet("""QPushButton { background-color: #27272a; border: 1px dashed #52525b; color: #a1a1aa; padding: 6px; border-radius: 4px; } QPushButton:hover { border: 1px dashed #60a5fa; color: #60a5fa; }""")
        btn_upload_end.clicked.connect(self.upload_end_frame)
        l_end.addWidget(btn_upload_end)

        l.addWidget(end_group)

        # --- SECTION C: VIDEO ACTION ---
        l.addWidget(QFrame(frameShape=QFrame.Shape.HLine, frameShadow=QFrame.Shadow.Sunken))
        vid_box = QHBoxLayout()
        self.btn_retry_vid = QPushButton("🎬 Generate Video (Veo)"); self.btn_retry_vid.setObjectName("PrimaryBtn")
        self.btn_retry_vid.setFixedHeight(30)
        self.btn_retry_vid.clicked.connect(self.start_video_gen)
        vid_box.addWidget(self.btn_retry_vid)
        l.addLayout(vid_box)
        
        self.lbl_st = QLabel("Ready"); self.lbl_st.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.lbl_st.setStyleSheet("color:#71717a; font-size:10px;")
        l.addWidget(self.lbl_st)
        
        self.update_signal.connect(self.on_image_success)
        self.video_signal.connect(self.on_video_success)
        self.error_signal.connect(self.on_video_error)
        self.progress_signal.connect(self.on_video_progress)

    def contextMenuEvent(self, event):
        menu = QMenu(self)
        del_act = menu.addAction("🗑️ 删除此分镜卡片 (Delete Card)")
        if menu.exec(self.mapToGlobal(event.pos())) == del_act:
            self.delete_signal.emit(self)

    def enterEvent(self, event):
        self.container.setStyleSheet("""QFrame#CardFrame { background-color: #27272a; border-radius: 12px; border: 1px solid #60a5fa; }""")
    def leaveEvent(self, event):
        self.container.setStyleSheet("""QFrame#CardFrame { background-color: #27272a; border-radius: 12px; border: 1px solid #3f3f46; }""")

    def upload_end_frame(self):
        f, _ = QFileDialog.getOpenFileName(self, "End Frame", "", "Images (*.png *.jpg)")
        if f: self.set_image_by_target(f, "end")

    def set_image_by_target(self, path, target):
        if target == "start":
            self.img_path = str(path) if path else None
            self.img.set_image(self.img_path)
        elif target == "end":
            self.end_img_path = str(path) if path else None
            self.img_end_preview.set_image(self.end_img_path)
        
        if path:
            self.lbl_st.setText(f"✅ {target.capitalize()} Done")
            self.lbl_st.setStyleSheet("color:#4ade80;")
        else:
            self.lbl_st.setText(f"🗑️ {target.capitalize()} Cleared")
            self.lbl_st.setStyleSheet("color:#a1a1aa;")

    def on_remake_target(self, target):
        refs = self.mw.get_all_assets().copy()
        current_img = self.img_path if target == "start" else self.end_img_path
        if current_img and os.path.exists(current_img):
            refs.append(current_img)
        self.submit_task(target, refs)

    def on_retry_target(self, target):
        refs = self.mw.get_all_assets().copy()
        self.submit_task(target, refs)

    def submit_task(self, target, ref_list):
        self.lbl_st.setText(f"⏳ Gen {target}...")
        prompt_text = self.txt_start.toPlainText() if target == "start" else self.txt_end.toPlainText()
        task = ImageGenTask(self.client, prompt_text, self.mw.cb_draw.currentText(), self.mw.cb_ratio.currentText(), self.mw.cb_size.currentText(), ref_list, self.mw.out_dir, target_slot=target)
        task.signals.finished.connect(self.update_signal)
        task.signals.error.connect(lambda e: self.lbl_st.setText(f"❌ {e}"))
        task.signals.progress.connect(lambda m: self.lbl_st.setText(m))
        self.pool.start(task)

    def start_gen(self):
        self.on_retry_target("start")

    def start_video_gen(self):
        if not self.img_path: 
            self.lbl_st.setText("❌ Missing Start Frame")
            return
        self.btn_retry_vid.setVisible(False)
        model = self.mw.cb_veo.currentText()
        ratio = self.mw.cb_vr.currentText()
        final_prompt = self.txt_start.toPlainText().strip()
        vid_dir = self.mw.out_dir_video
        
        task = VideoGenTask(self.client, final_prompt, self.img_path, self.end_img_path, model, ratio, vid_dir)
        task.signals.finished.connect(self.video_signal)
        task.signals.error.connect(self.error_signal)
        task.signals.progress.connect(self.progress_signal)
        self.lbl_st.setText("⏳ Video Queue..."); self.pool.start(task)

    def update_prompt(self, new_text): 
        self.txt_start.setText(str(new_text)); self.lbl_st.setText("✨ Optimized")
        
    @pyqtSlot(object)
    def on_image_success(self, result_tuple):
        path, target = result_tuple
        self.set_image_by_target(path, target)

    @pyqtSlot(object)
    def on_video_success(self, path):
        self.lbl_st.setText(f"🎬 Video Done"); self.lbl_st.setStyleSheet("color:#4ade80;")
        self.btn_retry_vid.setVisible(True)
    @pyqtSlot(str)
    def on_video_error(self, err):
        self.lbl_st.setText(f"❌ Video Error"); self.lbl_st.setStyleSheet("color:#f87171;")
        self.btn_retry_vid.setVisible(True)
    @pyqtSlot(str)
    def on_video_progress(self, msg): self.lbl_st.setText(msg)

# ==========================================
# 5. 主程序窗口
# ==========================================
class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("AI Director Pro - V53 (Smart High-Res)")
        self.resize(1600, 950)
        self.pool = QThreadPool(); self.pool.setMaxThreadCount(MAX_ASYNC_WORKERS)
        self.client = APIClient()
        self.out_dir = ensure_dir("Storyboards_Output")
        self.out_dir_video = ensure_dir("Final_Videos") 
        ensure_dir("Highlights")
        self.asset_paths = []; self.storyboard_cards = []
        self.setup_ui(); self.apply_theme()
        if not self.client.grsai_key: global_logger.warn("⚠️ Please Config API Key")
        else: global_logger.info(f"✅ System Ready")
        
        self.rotate_timer = QTimer()
        self.rotate_timer.timeout.connect(self.rotate_icon)
        self.current_angle = 0

    def apply_theme(self):
        dark_qss = """
        QMainWindow, QWidget { background-color: #18181b; color: #e4e4e7; font-family: 'Segoe UI', sans-serif; font-size: 13px; }
        QScrollArea, QWidget#scrollContent { background-color: transparent; border: none; }
        QGroupBox { background-color: #27272a; border: 1px solid #3f3f46; border-radius: 12px; margin-top: 24px; padding-top: 24px; font-weight: bold; color: #60a5fa; }
        QGroupBox::title { subcontrol-origin: margin; subcontrol-position: top left; padding: 0 10px; left: 10px; }
        QLineEdit, QTextEdit, QPlainTextEdit { background-color: #09090b; border: 1px solid #3f3f46; border-radius: 8px; padding: 8px; color: white; selection-background-color: #3b82f6; }
        QLineEdit:focus, QTextEdit:focus { border: 1px solid #60a5fa; background-color: #000000; }
        QPushButton { background-color: #3f3f46; border: none; border-radius: 6px; color: white; padding: 8px 16px; font-weight: 600; }
        QPushButton:hover { background-color: #52525b; }
        QPushButton#PrimaryBtn { background-color: qlineargradient(x1:0, y1:0, x2:1, y2:1, stop:0 #3b82f6, stop:1 #2563eb); color: white; }
        QPushButton#PrimaryBtn:hover { background-color: qlineargradient(x1:0, y1:0, x2:1, y2:1, stop:0 #60a5fa, stop:1 #3b82f6); }
        QPushButton#DangerBtn { background-color: #ef4444; color: white; }
        QComboBox, QSpinBox { background-color: #27272a; border: 1px solid #3f3f46; border-radius: 6px; padding: 6px; }
        QTabWidget::pane { border: 1px solid #3f3f46; border-radius: 8px; background: #18181b; }
        QTabBar::tab { background: #27272a; color: #a1a1aa; padding: 10px 20px; margin-right: 4px; border-top-left-radius: 8px; border-top-right-radius: 8px; }
        QTabBar::tab:selected { background: #3b82f6; color: white; font-weight: bold; }
        """
        self.setStyleSheet(dark_qss)

    def setup_ui(self):
        self.menuBar().addAction("⚙️ Config API Key", self.open_settings)
        sp = QSplitter(Qt.Orientation.Vertical); self.setCentralWidget(sp)
        top = QWidget(); tl = QHBoxLayout(top); tl.setContentsMargins(16,16,16,16); tl.setSpacing(16)
        
        # 1. Assets
        gl = QGroupBox("📂 Assets Library"); ll = QVBoxLayout(gl)
        self.sc_as = QScrollArea(); self.sc_as.setWidgetResizable(True)
        self.wid_as = QWidget(); self.grid_as = QGridLayout(self.wid_as); self.grid_as.setAlignment(Qt.AlignmentFlag.AlignTop)
        self.sc_as.setWidget(self.wid_as); self.sc_as.setAcceptDrops(True)
        self.sc_as.dragEnterEvent = lambda e: e.accept() if e.mimeData().hasUrls() else e.ignore()
        self.sc_as.dropEvent = self.drop_asset
        ll.addWidget(self.sc_as)
        btn_box = QHBoxLayout()
        btn_up = QPushButton("➕ Upload"); btn_up.clicked.connect(self.upload_as)
        btn_cl = QPushButton("🗑️ Clear"); btn_cl.setObjectName("DangerBtn"); btn_cl.clicked.connect(self.clear_assets)
        btn_box.addWidget(btn_up); btn_box.addWidget(btn_cl)
        ll.addLayout(btn_box); tl.addWidget(gl, 1)
        
        # 2. Main Workspace
        gc = QGroupBox("🎬 Director Workspace"); cl = QVBoxLayout(gc)
        hbox = QHBoxLayout(); self.inp_vid = QLineEdit(); self.inp_vid.setPlaceholderText("Video Path..."); self.inp_vid.setAcceptDrops(True)
        self.inp_vid.dragEnterEvent = lambda e: e.accept() if e.mimeData().hasUrls() else e.ignore()
        self.inp_vid.dropEvent = lambda e: self.inp_vid.setText(e.mimeData().urls()[0].toLocalFile())
        hbox.addWidget(QLabel("Highlight Count:")); self.sp_ana_cnt = QSpinBox(); self.sp_ana_cnt.setRange(1,12); self.sp_ana_cnt.setValue(4)
        hbox.addWidget(self.sp_ana_cnt)
        btn_an = QPushButton("🔍 Smart Extract"); btn_an.clicked.connect(self.run_analyze)
        hbox.addWidget(self.inp_vid); hbox.addWidget(btn_an); cl.addLayout(hbox)
        
        self.tab_widget = QTabWidget()
        
        # Tab 1: Storyboard
        tab_sb = QWidget(); l_sb = QVBoxLayout(tab_sb); l_sb.setContentsMargins(10,10,10,10)
        self.sc_sb = QScrollArea(); self.sc_sb.setWidgetResizable(True); self.wid_sb = QWidget(); self.grid_sb = QGridLayout(self.wid_sb)
        self.grid_sb.setAlignment(Qt.AlignmentFlag.AlignTop | Qt.AlignmentFlag.AlignLeft); self.grid_sb.setSpacing(16); self.sc_sb.setWidget(self.wid_sb)
        manage_box = QHBoxLayout()
        btn_imp = QPushButton("📥 Import Image"); btn_imp.clicked.connect(self.import_storyboard_image)
        btn_bat = QPushButton("🎨 Batch Draw"); btn_bat.setObjectName("PrimaryBtn"); btn_bat.clicked.connect(self.batch_remake)
        btn_cls = QPushButton("🗑️ Delete All"); btn_cls.setObjectName("DangerBtn"); btn_cls.clicked.connect(self.clear_all_storyboards)
        manage_box.addWidget(btn_imp); manage_box.addWidget(btn_bat); manage_box.addStretch(); manage_box.addWidget(btn_cls)
        l_sb.addWidget(self.sc_sb); l_sb.addLayout(manage_box)
        self.tab_widget.addTab(tab_sb, "🖼️ Storyboard")
        
        # Tab 2: Script
        tab_sc = QWidget(); l_sc = QVBoxLayout(tab_sc)
        self.txt_script_out = QTextEdit(); self.txt_script_out.setReadOnly(True)
        self.txt_script_out.setStyleSheet("font-size:13px; line-height:1.6; color:#e4e4e7; background:#09090b; border:none; padding:10px;")
        
        sc_box = QVBoxLayout()
        self.txt_manual = QTextEdit(); self.txt_manual.setPlaceholderText("Instructions (e.g. style change)...")
        self.txt_manual.setFixedHeight(60); 
        self.btn_toggle_script = QPushButton("🚀 Start Script Analysis")
        self.btn_toggle_script.setObjectName("PrimaryBtn"); self.btn_toggle_script.setFixedHeight(45)
        self.btn_toggle_script.clicked.connect(self.toggle_script_analysis)
        self.is_script_running = False
        sc_box.addWidget(self.txt_manual); sc_box.addWidget(self.btn_toggle_script)
        l_sc.addWidget(self.txt_script_out); l_sc.addLayout(sc_box)
        self.tab_widget.addTab(tab_sc, "📝 Script Agent")
        
        cl.addWidget(self.tab_widget); tl.addWidget(gc, 3)
        
        # 3. Sidebar
        fr = QFrame(); fr.setFixedWidth(320); rl = QVBoxLayout(fr); rl.setSpacing(16)
        
        g1 = QGroupBox("🧠 Agent"); fl1 = QFormLayout()
        self.txt_sc = QTextEdit(); self.txt_sc.setPlaceholderText("Story Idea..."); self.txt_sc.setFixedHeight(70); fl1.addRow(self.txt_sc)
        
        # Restore Models
        self.cb_agent = QComboBox(); self.cb_agent.setEditable(True); self.cb_agent.addItems(["gemini-3-pro", "gemini-2.5-pro", "gemini-2.5-flash"]); fl1.addRow("Model:", self.cb_agent)
        self.sp_cnt = QSpinBox(); self.sp_cnt.setRange(1,30); self.sp_cnt.setValue(4); fl1.addRow("Count:", self.sp_cnt)
        self.btn_ag = QPushButton("🚀 Split & Gen"); self.btn_ag.setObjectName("PrimaryBtn"); self.btn_ag.clicked.connect(self.run_agent_flow)
        fl1.addRow(self.btn_ag); g1.setLayout(fl1); rl.addWidget(g1)
        
        g2 = QGroupBox("🎨 Image"); fl2 = QFormLayout()
        self.cb_draw = QComboBox(); self.cb_draw.addItems(["nano-banana-pro", "nano-banana-fast", "nano-banana"]); fl2.addRow("Model:", self.cb_draw)
        self.cb_ratio = QComboBox(); self.cb_ratio.addItems(["16:9", "9:16", "1:1", "4:3"]); fl2.addRow("Ratio:", self.cb_ratio)
        self.cb_size = QComboBox(); self.cb_size.addItems(["1K", "2K", "4K"]); fl2.addRow("Res:", self.cb_size)
        g2.setLayout(fl2); rl.addWidget(g2)
        
        g3 = QGroupBox("🎥 Video (Veo)"); fl3 = QFormLayout()
        self.cb_veo = QComboBox(); self.cb_veo.addItems(["veo3.1-pro", "veo3.1-fast"]); fl3.addRow("Model:", self.cb_veo)
        self.cb_vr = QComboBox(); self.cb_vr.addItems(["16:9", "9:16"]); fl3.addRow("Ratio:", self.cb_vr)
        btn_optimize = QPushButton("✨ Auto Prompt"); btn_optimize.clicked.connect(self.run_prompt_optimization); fl3.addRow(btn_optimize)
        btn_vid = QPushButton("🎬 Batch Video"); btn_vid.setObjectName("PrimaryBtn"); btn_vid.clicked.connect(self.run_video_flow); fl3.addRow(btn_vid)
        self.lbl_vid_progress = QLabel("Prog: -/-"); fl3.addRow(self.lbl_vid_progress)
        g3.setLayout(fl3); rl.addWidget(g3); rl.addStretch(); tl.addWidget(fr)
        
        self.log = LogConsole(); self.log.setFixedHeight(180)
        sp.addWidget(top); sp.addWidget(self.log); sp.setSizes([750, 180])

    def toggle_script_analysis(self):
        if self.is_script_running: self.stop_script()
        else: self.start_script()

    def start_script(self):
        vid = self.inp_vid.text()
        if not os.path.exists(vid):
            QMessageBox.warning(self, "Error", "Invalid video path!")
            return
        self.is_script_running = True
        self.txt_script_out.clear()
        self.btn_toggle_script.setText("⏹ Stop"); self.btn_toggle_script.setObjectName("DangerBtn")
        self.rotate_timer.start(100)
        
        assets = self.get_all_assets()
        manual = self.txt_manual.toPlainText()
        self.script_thread = ScriptAnalysisThread(self.client, vid, self.cb_agent.currentText(), manual, assets)
        self.script_thread.chunk_received.connect(self.on_script_chunk)
        self.script_thread.finished_signal.connect(self.stop_script)
        self.script_thread.log_signal.connect(self.log.add_log)
        self.script_thread.start()

    def stop_script(self):
        if hasattr(self, 'script_thread'): self.script_thread.stop()
        self.is_script_running = False
        self.rotate_timer.stop()
        self.btn_toggle_script.setText("🚀 Start Script Analysis"); self.btn_toggle_script.setObjectName("PrimaryBtn")

    def rotate_icon(self):
        self.current_angle = (self.current_angle + 10) % 360
        chars = ['/', '-', '\\', '|']; idx = (self.current_angle // 10) % 4
        self.btn_toggle_script.setText(f"⏹ Stop {chars[idx]}")

    @pyqtSlot(str)
    def on_script_chunk(self, chunk):
        c = self.txt_script_out.textCursor()
        c.movePosition(QTextCursor.MoveOperation.End)
        c.insertText(chunk)
        self.txt_script_out.setTextCursor(c)

    def upload_as(self):
        f, _ = QFileDialog.getOpenFileName(self, "Img", "", "Images (*.png *.jpg *.jpeg)")
        if f: self.add_asset_card(f)
    def drop_asset(self, e):
        for u in e.mimeData().urls(): self.add_asset_card(u.toLocalFile())
    def add_asset_card(self, path):
        if path in self.asset_paths: return
        self.asset_paths.append(path)
        lbl = AssetLabel(path); lbl.delete_signal.connect(self.remove_asset)
        c = self.grid_as.count(); self.grid_as.addWidget(lbl, c//2, c%2)
        global_logger.info(f"➕ Asset: {os.path.basename(path)}")
    def remove_asset(self, widget):
        self.grid_as.removeWidget(widget); widget.deleteLater()
        if widget.path in self.asset_paths: self.asset_paths.remove(widget.path)
    def clear_assets(self):
        for i in reversed(range(self.grid_as.count())): w = self.grid_as.itemAt(i).widget(); 
        if w: w.deleteLater()
        self.asset_paths.clear()
    def get_all_assets(self): return self.asset_paths

    def add_sb_card(self, prompt, ref_urls, img_path=None):
        if isinstance(prompt, dict): prompt = prompt.get('description') or prompt.get('prompt') or str(prompt)
        prompt_str = str(prompt)
        idx = len(self.storyboard_cards) + 1
        card = StoryboardCard(idx, prompt_str, self.pool, self.client, self, ref_urls)
        card.delete_signal.connect(self.remove_sb_card)
        if img_path: card.set_image_by_target(img_path, "start")
        self.storyboard_cards.append(card)
        row = (idx-1) // 3
        col = (idx-1) % 3
        self.grid_sb.addWidget(card, row, col)
        return card

    def remove_sb_card(self, card):
        if card in self.storyboard_cards: self.storyboard_cards.remove(card)
        card.deleteLater(); self.refresh_sb_grid()

    def refresh_sb_grid(self):
        while self.grid_sb.count():
            item = self.grid_sb.takeAt(0)
            if item.widget(): item.widget().setParent(None)
        for i, card in enumerate(self.storyboard_cards):
            card.chk_sel.setText(f"No.{i+1}")
            self.grid_sb.addWidget(card, i//3, i%3)

    def clear_all_storyboards(self):
        for c in self.storyboard_cards: c.deleteLater()
        self.storyboard_cards.clear(); self.refresh_sb_grid()

    def import_storyboard_image(self):
        f, _ = QFileDialog.getOpenFileName(self, "Import", "", "Images (*.png *.jpg)")
        if f: self.add_sb_card("Local Import", [], f)

    def batch_remake(self):
        sel = [c for c in self.storyboard_cards if c.chk_sel.isChecked()]
        for c in sel: c.on_remake_target("start")

    def run_agent_flow(self):
        txt = self.txt_sc.toPlainText()
        if not txt: return
        self.clear_all_storyboards(); self.btn_ag.setEnabled(False)
        self.agent_thread = AgentProcessThread(self.client, txt, self.sp_cnt.value(), self.cb_agent.currentText(), self.get_all_assets())
        self.agent_thread.log_signal.connect(self.log.add_log)
        self.agent_thread.finished_signal.connect(self.on_agent_done)
        self.agent_thread.error_signal.connect(self.on_agent_error)
        self.agent_thread.start()

    def on_agent_done(self, prompts, ref_urls):
        self.btn_ag.setEnabled(True)
        QTimer.singleShot(0, lambda: self._create_cards_safely(prompts, ref_urls))

    def _create_cards_safely(self, prompts, ref_urls):
        global_logger.step(f"🚀 Generating {len(prompts)} cards...")
        for p in prompts:
            card = self.add_sb_card(p, ref_urls)
            QCoreApplication.processEvents()
            card.start_gen()

    def on_agent_error(self, err): self.btn_ag.setEnabled(True); global_logger.error(f"Agent Failed: {err}")

    def run_analyze(self):
        p = self.inp_vid.text(); 
        if not p: return
        self.tab_widget.setCurrentIndex(0)
        self.clear_all_storyboards()
        task = SmartAnalyzeTask(self.client, p, self.cb_agent.currentText(), self.sp_ana_cnt.value())
        task.signals.progress.connect(global_logger.info)
        task.signals.finished.connect(self.on_analyze_done)
        task.signals.error.connect(lambda e: global_logger.error(f"Analyze Failed: {e}"))
        self.pool.start(task)

    def on_analyze_done(self, results):
        global_logger.step(f"✅ Extracted {len(results)} highlights")
        for item in results: self.add_sb_card(item['prompt'], [], item['path'])

    def run_prompt_optimization(self):
        sel = [c for c in self.storyboard_cards if c.chk_sel.isChecked()]
        if not sel: return
        model = self.cb_agent.currentText()
        global_logger.step(f"✨ Optimizing prompts...")
        for c in sel:
            if not c.img_path: continue
            task = PromptOptimizeTask(self.client, c.img_path, c.txt_start.toPlainText(), model)
            task.signals.finished.connect(c.update_prompt)
            task.signals.error.connect(lambda e: global_logger.error(f"Optimize Failed: {e}"))
            self.pool.start(task)

    def run_video_flow(self):
        sel = [c for c in self.storyboard_cards if c.chk_sel.isChecked()]
        if not sel: return
        self._start_video_batch(sel)

    def _start_video_batch(self, cards):
        total = len(cards); completed = 0
        def update_prog():
            nonlocal completed; completed += 1; self.lbl_vid_progress.setText(f"Prog: {completed}/{total}")

        global_logger.step(f"🎬 Starting {total} videos...")
        for card in cards:
            if not card.img_path: continue
            try: card.video_signal.disconnect(update_prog)
            except: pass
            try: card.error_signal.disconnect(update_prog)
            except: pass
            
            card.video_signal.connect(lambda _, c=update_prog: c())
            card.error_signal.connect(lambda _, c=update_prog: c())
            card.start_video_gen()

    def open_settings(self):
        d=QDialog(self); d.setWindowTitle("Settings"); l=QFormLayout(d)
        k1=QLineEdit(load_config().get("grsai_key")); k1.setEchoMode(QLineEdit.EchoMode.Password)
        k2=QLineEdit(load_config().get("imgbb_key")); k2.setEchoMode(QLineEdit.EchoMode.Password)
        l.addRow("GRSAI Key:", k1); l.addRow("ImgBB Key:", k2)
        b=QPushButton("Save"); b.setObjectName("PrimaryBtn")
        b.clicked.connect(lambda: (save_config(k1.text(), k2.text()), self.client.__init__(), d.accept())); l.addRow(b); d.exec()

if __name__ == "__main__":
    app = QApplication(sys.argv)
    app.setStyle("Fusion") 
    w = MainWindow()
    w.show()
    sys.exit(app.exec())