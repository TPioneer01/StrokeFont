"""StrokeFont local WebUI, entirely contained in this file.

Run inside the existing environment:
    conda activate StrokeFont
    python webui.py
Then open http://127.0.0.1:7860 . No additional dependencies are needed.
Optional: python webui.py --port 7861 --open-browser
Configurations, uploaded references and task logs are saved under .webui/.
Existing training, sampling and dataset scripts are run as child processes.
"""
import argparse
import base64
from collections import deque
import csv
from datetime import datetime
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import io
import json
import math
import mimetypes
import os
from pathlib import Path
import re
import secrets
import signal
import subprocess
import sys
import threading
import time
from urllib.parse import parse_qs, urlparse
import webbrowser


ROOT = Path(__file__).resolve().parent
STORE = ROOT / '.webui'
TOKEN = secrets.token_urlsafe(32)
IMAGE_TYPES = {'.png', '.jpg', '.jpeg', '.gif'}
ACTIVE = {'starting', 'running', 'stopping'}


def local_path(value, *, exists=False, directory=False):
    if not isinstance(value, str) or not value.strip():
        raise ValueError('请填写文件或目录路径。')
    path = Path(os.path.expandvars(value.strip().strip('"'))).expanduser()
    if not path.is_absolute():
        path = ROOT / path
    path = path.resolve()
    if exists and (not path.exists() or (path.is_dir() != directory)):
        raise ValueError(f'{"目录" if directory else "文件"}不存在：{path}')
    return path


def script_path(path):
    # blobfile used by the original scripts rejects Windows drive-letter URLs.
    try:
        return os.path.relpath(path, ROOT).replace('\\', '/')
    except ValueError:
        raise ValueError('训练数据、笔画文件、模型和输出目录须与项目位于同一磁盘。')


def number(data, key, default, *, minimum=1, integer=True):
    raw = data.get(key, default)
    value = float(raw)
    if not math.isfinite(value) or value < minimum or (integer and not value.is_integer()):
        raise ValueError(f'{key} 必须为不小于 {minimum} 的{"整数" if integer else "数值"}。')
    return int(value) if integer else value


def characters(path):
    chars = ''.join(path.read_text(encoding='utf-8-sig').split())
    if not chars:
        raise ValueError('字符表为空，请使用 UTF-8 文本。')
    return chars


def read_config(path):
    import yaml
    config = yaml.safe_load(local_path(path, exists=True).read_text(encoding='utf-8-sig'))
    if not isinstance(config, dict):
        raise ValueError('配置文件应为 YAML 字典。')
    return config


def stroke_file(value, count):
    if not str(value or '').strip():
        return None
    path = local_path(value, exists=True)
    lines = path.read_text(encoding='utf-8-sig').splitlines()
    if len(lines) != count:
        raise ValueError(f'笔画文件有 {len(lines)} 行，字符表有 {count} 个字符，二者必须一致。')
    for index, line in enumerate(lines):
        # The original loader expects a trailing space and uses split(" ")[1:-1].
        pieces = line.split(' ')[1:-1]
        if len(pieces) != 32 or any(not p.isdigit() for p in pieces):
            raise ValueError(f'笔画文件第 {index + 1} 行格式无效：需字符、32 个非负整数及末尾空格。')
    return script_path(path)


class App:
    def __init__(self, storage=STORE):
        self.storage = Path(storage)
        self.jobs = {}
        self.processes = {}
        self.lock = threading.RLock()
        self.images = {}
        self.environment = {'python': sys.executable, 'version': sys.version.split()[0], 'gpu': '检测中…'}
        for entry in sorted((self.storage / 'jobs').glob('*/job.json'))[-100:]:
            try:
                job = json.loads(entry.read_text(encoding='utf-8'))
                if job['status'] in ACTIVE:
                    job['status'] = 'interrupted'
                    job['error'] = '界面服务已重启，无法继续追踪原进程；请先确认旧任务已退出再启动新任务。'
                self.jobs[job['id']] = job
            except (OSError, ValueError, KeyError):
                continue

    def probe(self):
        try:
            result = subprocess.run(
                ['nvidia-smi', '--query-gpu=name,memory.total,memory.free', '--format=csv,noheader'],
                capture_output=True, text=True, timeout=10, **hidden_process(),
            )
            self.environment['gpu'] = result.stdout.strip() if result.returncode == 0 else '未检测到 NVIDIA GPU'
        except (OSError, subprocess.TimeoutExpired):
            self.environment['gpu'] = '未检测到 NVIDIA GPU（CPU 仍可做功能检查）'

    def image(self, path):
        path = Path(path).resolve()
        if path.suffix.lower() not in IMAGE_TYPES or not path.is_file():
            raise ValueError('只支持预览 PNG、JPEG、GIF 图片。')
        key = secrets.token_urlsafe(16)
        with self.lock:
            if len(self.images) > 4000:
                self.images.clear()
            self.images[key] = path
        return '/api/image?id=' + key

    def inspect(self, value):
        root = local_path(value, exists=True, directory=True)
        fonts, indices, previews = {}, set(), []
        invalid = []
        for path in root.rglob('*'):
            if path.is_file() and path.suffix.lower() in IMAGE_TYPES:
                font = str(path.parent.relative_to(root))
                fonts[font] = fonts.get(font, 0) + 1
                prefix = path.name.split('_')[0].split('.')[0]
                if prefix.isdigit():
                    indices.add(int(prefix))
                elif len(invalid) < 5:
                    invalid.append(path.name)
                if len(previews) < 12:
                    previews.append({'url': self.image(path), 'label': f'{font}/{path.name}'})
        if not fonts:
            raise ValueError('目录中没有训练图片，请先转换字体。')
        missing = [n for n in range(max(indices, default=-1) + 1) if n not in indices][:20]
        return {'path': str(root), 'fonts': fonts, 'count': sum(fonts.values()),
                'characters': len(indices), 'missing': missing, 'invalid': invalid, 'previews': previews}

    def _persist(self, job):
        folder = Path(job['folder'])
        folder.mkdir(parents=True, exist_ok=True)
        (folder / 'job.json').write_text(json.dumps(job, ensure_ascii=False, indent=2), encoding='utf-8')

    def prepare(self, kind, data, folder):
        import yaml
        output = local_path(data['output']) if data.get('output', '').strip() else folder / 'output'
        if kind == 'dataset':
            fonts = local_path(data.get('fonts'), exists=True, directory=True)
            font_files = list(fonts.iterdir())
            if not font_files or any(p.is_file() and p.suffix.lower() not in {'.ttf', '.otf', '.ttc'} for p in font_files):
                raise ValueError('字体目录应只放字体文件（TTF / OTF / TTC），且至少有一个字体文件。')
            if not any(p.is_file() for p in font_files):
                raise ValueError('所选目录中没有字体文件；原转换脚本不递归读取子目录。')
            chars = characters(local_path(data.get('characters', 'total_chn.txt'), exists=True))
            if len(set(chars)) != len(chars):
                raise ValueError('完整字符表含有重复字符，请先去重并同步笔画编码。')
            size = number(data, 'image_size', 80)
            char_size = number(data, 'chara_size', 60)
            if char_size > size:
                raise ValueError('字体绘制大小不能超过画布大小。')
            char_path = folder / 'characters.txt'
            char_path.write_text(chars, encoding='utf-8')
            command = [sys.executable, '-u', str(ROOT / 'font2img.py'), '--ttf_path', str(fonts),
                       '--chara', str(char_path), '--save_path', str(output),
                       '--img_size', str(size), '--chara_size', str(char_size)]
            target, config = len(chars), None
        else:
            config = read_config(data.get('template', 'cfg/test_cfg.yaml' if kind == 'test' else 'cfg/train_cfg.yaml'))
            extras = yaml.safe_load(data.get('advanced', '') or '{}')
            if not isinstance(extras, dict):
                raise ValueError('高级 YAML 参数应为键值字典。')
            config.update(extras)
            char_path = local_path(data.get('characters', 'total_chn.txt'), exists=True)
            chars = characters(char_path)
            if len(set(chars)) != len(chars):
                raise ValueError('完整字符表含重复字符。')
            config.update(chara_nums=len(chars), image_size=number(data, 'image_size', 80),
                          batch_size=number(data, 'batch_size', 1),
                          stroke_path=stroke_file(data.get('stroke', 'chinese_stroke.txt'), len(chars)))
            config['attention_resolutions'] = str(config.get('attention_resolutions', '40, 20, 10'))
            if config['image_size'] not in (64, 80, 128, 256, 512) and not config.get('channel_mult'):
                raise ValueError('默认模型仅支持 64、80、128、256、512 尺寸，其他尺寸须设置 channel_mult。')
            processes = number(data, 'processes', 1)
            if kind == 'test':
                checkpoint = local_path(data.get('model'), exists=True)
                reference = local_path(data.get('reference'), exists=True)
                from PIL import Image
                with Image.open(reference) as img:
                    img.verify()
                text = ''.join(str(data.get('text', '')).split())
                if not text:
                    raise ValueError('请输入需要生成的字符。')
                unknown = ''.join(dict.fromkeys(ch for ch in text if ch not in set(chars)))
                if unknown:
                    raise ValueError(f'以下字符不在完整字符表中：{unknown[:50]}')
                generation = folder / 'generation.txt'
                generation.write_text(text, encoding='utf-8')
                config.update(model_path=script_path(checkpoint), sty_img_path=script_path(reference),
                              total_txt_file=script_path(char_path), gen_txt_file=script_path(generation),
                              img_save_path=script_path(output), num_samples=len(text),
                              classifier_free=bool(data.get('guidance', True)),
                              cont_scale=number(data, 'cont_scale', 3, minimum=0, integer=False),
                              sk_scale=number(data, 'sk_scale', 3, minimum=0, integer=False),
                              use_ddim=bool(data.get('ddim', True)))
                steps = number(data, 'sample_steps', 25)
                if steps > int(config.get('diffusion_steps', 1000)):
                    raise ValueError('采样步数不能超过扩散总步数。')
                if config['use_ddim']:
                    total = int(config.get('diffusion_steps', 1000))
                    if not any(len(range(0, total, stride)) == steps for stride in range(1, total)):
                        raise ValueError('该 DDIM 步数无法从扩散总步数等间距取样；1000 步扩散建议使用 25、50 或 100。')
                    config['timestep_respacing'] = f'ddim{steps}'
                else:
                    config['timestep_respacing'] = ''
                target, entry = len(text), 'sample.py'
            else:
                dataset = local_path(data.get('dataset'), exists=True, directory=True)
                report = self.inspect(str(dataset))
                if report['invalid'] or report['missing'] or report['characters'] != len(chars):
                    raise ValueError('训练图片编号必须完整覆盖字符表（从 00000 开始）；请先检查数据集和字符表。')
                if report['count'] // processes < config['batch_size']:
                    raise ValueError('每个进程的数据不足一个批次，请减小批大小或进程数。')
                resume_value = str(data.get('resume', '')).strip()
                finetune = kind == 'finetune'
                if finetune and not resume_value:
                    raise ValueError('微调必须选择条件训练的 modelXXXXXX.pt 检查点。')
                resume = local_path(resume_value, exists=True) if resume_value else None
                start_step = 0
                if resume:
                    match = re.fullmatch(r'model(\d+)\.pt', resume.name)
                    if not match:
                        raise ValueError('恢复训练请选择 modelXXXXXX.pt，而不是 EMA 权重，以正确恢复步数。')
                    start_step = int(match.group(1))
                encoder = local_path(data.get('encoder', 'pretrained_models/chinese_styenc.ckpt'), exists=not bool(resume))
                train_steps = number(data, 'train_step', 420000)
                total_steps = number(data, 'total_train_step', 800000)
                target = total_steps if finetune else train_steps
                if target <= start_step:
                    raise ValueError(f'目标累计步数必须大于检查点步数 {start_step}。')
                config.update(data_dir=script_path(dataset), model_save_dir=script_path(output),
                              sty_encoder_path=script_path(encoder), resume_checkpoint=script_path(resume) if resume else '',
                              classifier_free=finetune, train_step=train_steps, total_train_step=total_steps,
                              lr=number(data, 'lr', 0.0001, minimum=1e-12, integer=False),
                              microbatch=number(data, 'microbatch', 1),
                              log_interval=number(data, 'log_interval', 10),
                              save_interval=number(data, 'save_interval', 10000))
                entry = 'train.py'
            config_path = folder / 'config.yaml'
            config_path.write_text(yaml.safe_dump(config, allow_unicode=True, sort_keys=False), encoding='utf-8')
            command = [sys.executable, '-u', str(ROOT / entry), '--cfg_path', script_path(config_path)]
            if processes > 1:
                command = [sys.executable, '-u', str(ROOT / 'launch.py'), '--nproc-per-node', str(processes),
                           entry, '--cfg_path', script_path(config_path)]
        can_resume = kind in ('train', 'finetune') and bool(config.get('resume_checkpoint'))
        if output.exists() and (not output.is_dir() or (any(output.iterdir()) and not can_resume)):
            raise ValueError('输出目录非空，请使用新目录；只有恢复训练可以沿用原模型目录。')
        output.mkdir(parents=True, exist_ok=True)
        return command, output, config, target

    def start(self, kind, data):
        if kind not in ('dataset', 'train', 'finetune', 'test'):
            raise ValueError('未知任务类型。')
        with self.lock:
            if any(job['status'] in ACTIVE for job in self.jobs.values()):
                raise ValueError('已有任务在运行，请先等待完成或停止任务。')
            ident = datetime.now().strftime('%Y%m%d-%H%M%S') + '-' + secrets.token_hex(3)
            folder = self.storage / 'jobs' / ident
            folder.mkdir(parents=True, exist_ok=True)
            command, output, config, target = self.prepare(kind, data, folder)
            job = {'id': ident, 'kind': kind, 'status': 'starting', 'created': time.time(), 'inputs': data,
                   'folder': str(folder), 'output': str(output), 'command': command,
                   'target': target, 'config': config, 'error': '', 'returncode': None}
            self.jobs[ident] = job
            self._persist(job)
            env = os.environ.copy()
            env.update(PYTHONUNBUFFERED='1', PYTHONUTF8='1', PYTHONIOENCODING='utf-8', USE_LIBUV='0')
            for name in ('RANK', 'WORLD_SIZE', 'LOCAL_RANK', 'MASTER_ADDR', 'MASTER_PORT'):
                env.pop(name, None)
            try:
                process = subprocess.Popen(command, cwd=str(ROOT), env=env, stdout=subprocess.PIPE,
                                           stderr=subprocess.STDOUT, text=True, encoding='utf-8', errors='replace',
                                           **hidden_process(group=True))
            except OSError as exc:
                job.update(status='failed', error=str(exc), ended=time.time())
                self._persist(job)
                raise ValueError(f'任务启动失败：{exc}')
            self.processes[ident] = process
            job.update(status='running', pid=process.pid)
            self._persist(job)
            threading.Thread(target=self._watch, args=(ident, process), daemon=True).start()
            return self.view(ident)

    def _watch(self, ident, process):
        job = self.jobs[ident]
        try:
            with (Path(job['folder']) / 'console.log').open('w', encoding='utf-8', buffering=1) as log:
                for line in process.stdout:
                    log.write(line)
            code = process.wait()
            with self.lock:
                job.update(returncode=code, ended=time.time(),
                           status='cancelled' if job['status'] == 'stopping' else ('completed' if code == 0 else 'failed'))
                self._persist(job)
        finally:
            process.stdout.close()

    def stop(self, ident):
        with self.lock:
            job = self.jobs.get(ident)
            process = self.processes.get(ident)
            if not job or not process or process.poll() is not None:
                raise ValueError('该任务没有正在运行的进程。')
            job['status'] = 'stopping'
            self._persist(job)
        if os.name == 'nt':
            try:
                result = subprocess.run(['taskkill', '/PID', str(process.pid), '/T', '/F'],
                                        capture_output=True, timeout=15, **hidden_process())
            except (OSError, subprocess.TimeoutExpired):
                with self.lock:
                    if process.poll() is None:
                        job['status'] = 'running'
                        self._persist(job)
                raise
            if result.returncode and process.poll() is None:
                with self.lock:
                    job['status'] = 'running'
                    self._persist(job)
                raise ValueError('停止进程失败，请在系统任务管理器中检查该任务。')
        else:
            try:
                os.killpg(process.pid, signal.SIGTERM)
            except ProcessLookupError:
                pass
            threading.Thread(target=self._kill_later, args=(process,), daemon=True).start()

    @staticmethod
    def _kill_later(process):
        try:
            process.wait(timeout=8)
        except subprocess.TimeoutExpired:
            try:
                os.killpg(process.pid, signal.SIGKILL)
            except ProcessLookupError:
                pass

    def view(self, ident):
        with self.lock:
            job = dict(self.jobs[ident])
        log = Path(job['folder']) / 'console.log'
        if log.exists():
            with log.open('rb') as stream:
                stream.seek(max(0, log.stat().st_size - 100000))
                job['logs'] = stream.read().decode('utf-8', errors='replace')
        else:
            job['logs'] = ''
        job['metrics'] = []
        progress = Path(job['output']) / 'progress.csv'
        if job['kind'] in ('train', 'finetune') and progress.is_file():
            try:
                with progress.open(encoding='utf-8', errors='replace') as stream:
                    for row in deque(csv.DictReader(stream), maxlen=500):
                        if row.get('step') and row.get('loss'):
                            step, loss = float(row['step']), float(row['loss'])
                            if math.isfinite(step) and math.isfinite(loss):
                                job['metrics'].append({'step': step, 'loss': loss})
            except (OSError, ValueError, csv.Error):
                pass
        return job


def hidden_process(group=False):
    if os.name == 'nt':
        return {'creationflags': subprocess.CREATE_NO_WINDOW | (subprocess.CREATE_NEW_PROCESS_GROUP if group else 0)}
    return {'start_new_session': True} if group else {}


HTML = r'''<!doctype html>
<html lang="zh-CN"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>StrokeFont · 字体实验室</title><style>
:root{--bg:#f5f6f8;--paper:#fff;--ink:#1d2738;--muted:#697587;--line:#e4e8ef;--blue:#395be0;--nav:#162234;--green:#14816f}*{box-sizing:border-box}body{margin:0;background:var(--bg);color:var(--ink);font:14px/1.65 system-ui,-apple-system,"Microsoft YaHei",sans-serif}button,input,select,textarea{font:inherit}button{cursor:pointer}button:disabled{opacity:.5;cursor:wait}button:focus-visible,a:focus-visible{outline:3px solid #90acf9;outline-offset:3px}.shell{display:flex;min-height:100vh}aside{width:240px;background:var(--nav);color:#fff;padding:30px 21px;position:fixed;height:100vh;display:flex;flex-direction:column}.brand{font-size:23px;font-weight:760;letter-spacing:-.8px}.brand em{color:#8da4ff;font-style:normal}.subbrand{color:#9daac0;font-size:12px;margin:4px 0 35px;letter-spacing:1px}.nav{display:grid;gap:7px}.nav button{border:0;text-align:left;background:none;color:#aebcd1;padding:12px 15px;border-radius:9px}.nav button.on{background:#2b3c59;color:#fff}.nav button span{margin-right:11px;font-size:12px;opacity:.65}.sidefoot{margin-top:auto;font-size:12px;color:#9daac0;border-top:1px solid #344055;padding-top:18px}main{margin-left:240px;padding:30px 42px;width:calc(100% - 240px);max-width:1640px}.top{display:flex;justify-content:space-between;align-items:center;color:var(--muted);font-size:12px;gap:15px}.pill{padding:5px 12px;border:1px solid var(--line);border-radius:30px;background:white;color:var(--green)}.eyebrow{font-size:11px;letter-spacing:2px;color:var(--blue);font-weight:750;margin-top:30px}h1{font-size:31px;letter-spacing:-1px;line-height:1.3;margin:10px 0}h2{font-size:17px;margin:0 0 16px}h3{font-size:14px;margin:0 0 9px}.lead{color:var(--muted);margin:0 0 25px;max-width:800px}.stats{display:grid;grid-template-columns:1.2fr 1fr .6fr;gap:16px;margin:25px 0}.card{border:1px solid var(--line);background:var(--paper);border-radius:13px;padding:24px;margin-bottom:20px;box-shadow:0 3px 10px #23375b03}.stats .card{margin:0;padding:18px 21px}.caption{font-size:11px;color:var(--muted);text-transform:uppercase;letter-spacing:1px}.statval{font-size:14px;font-weight:650;margin-top:7px;overflow-wrap:anywhere}.grid{display:grid;grid-template-columns:minmax(0,1.65fr) minmax(260px,1fr);gap:22px;align-items:start}.steps{display:grid;grid-template-columns:repeat(3,1fr);gap:16px}.stepnum{background:#edf1ff;color:var(--blue);width:32px;height:32px;display:grid;place-items:center;border-radius:9px;margin-bottom:16px;font-weight:bold}.muted{color:var(--muted)}.small{font-size:12px}.formgrid{display:grid;grid-template-columns:1fr 1fr;gap:17px}.field.wide{grid-column:1/-1}label{display:block;font-weight:620;font-size:12px;margin-bottom:6px}.help{font-size:11px;color:var(--muted);margin:5px 0 0;line-height:1.55}input,select,textarea{width:100%;background:#fcfdff;border:1px solid #dce2eb;border-radius:7px;color:var(--ink);padding:9px 11px;outline:0;min-width:0}input:focus,select:focus,textarea:focus{border-color:var(--blue);box-shadow:0 0 0 3px #395be010}textarea{resize:vertical;min-height:100px}.pathrow{display:flex;gap:7px}.pathrow button{white-space:nowrap}.button{border:1px solid #dce2eb;border-radius:7px;padding:8px 13px;background:#fff;color:var(--ink)}.primary{background:var(--blue);color:#fff;border-color:var(--blue);font-weight:650}.danger{color:#b34040;border-color:#edcccc}.actions{display:flex;gap:9px;align-items:center;flex-wrap:wrap;margin-top:23px}.note{padding:13px 15px;background:#f0f4ff;border-left:3px solid #7290ec;border-radius:5px;font-size:12px;margin:15px 0;line-height:1.8}.warn{background:#fff8eb;border-color:#ddb45d}.instruction{margin:0;padding-left:20px;color:var(--muted);font-size:13px}.instruction li{margin-bottom:12px}.instruction strong{color:var(--ink)}details{margin:20px 0 0}summary{font-weight:650;cursor:pointer;font-size:12px;margin-bottom:12px}code{font:12px/1.7 Consolas,monospace;background:#f0f2f7;padding:2px 5px;border-radius:3px;overflow-wrap:anywhere}pre{white-space:pre-wrap;overflow-wrap:anywhere}.console{background:#121d2e;color:#cbd5e6;border-radius:9px;padding:18px;min-height:170px;max-height:370px;overflow:auto;font:12px/1.7 Consolas,monospace;white-space:pre-wrap}.page{display:none}.page.on{display:block}.tabs{display:flex;gap:5px;background:#eff2f8;padding:5px;border-radius:9px;margin-bottom:22px}.tabs button{flex:1;border:0;background:none;padding:9px;border-radius:6px;color:var(--muted)}.tabs button.on{background:white;color:var(--blue);box-shadow:0 1px 4px #1d273812}.gallery{display:grid;grid-template-columns:repeat(auto-fill,minmax(100px,1fr));gap:12px;margin-top:15px}.tile{border:1px solid var(--line);border-radius:8px;padding:9px;text-align:center;overflow:hidden}.tile img{max-width:100%;height:88px;object-fit:contain;background:white}.tile div{font-size:11px;color:var(--muted);overflow-wrap:anywhere;margin-top:5px}.tile a{color:var(--blue);font-size:11px}table{border-collapse:collapse;width:100%;font-size:12px}td,th{text-align:left;border-bottom:1px solid var(--line);padding:11px 7px;word-break:break-word}th{color:var(--muted);font-weight:550}.historywrap{overflow:auto}.status{display:inline-block;padding:3px 9px;border-radius:20px;background:#eef1f6;font-size:11px;white-space:nowrap}.status.running{background:#eaf1ff;color:#395be0}.status.completed{background:#e7f5ef;color:#14816f}.status.failed,.status.cancelled,.status.interrupted{background:#fff0ec;color:#b55a3e}.bar{height:5px;background:#eaf0f8;border-radius:8px;margin:13px 0}.bar div{height:100%;background:var(--blue);border-radius:8px}.progresshead{display:flex;justify-content:space-between;gap:12px;align-items:center}.toast{position:fixed;right:28px;bottom:24px;background:#17263d;color:white;border-radius:10px;padding:14px 22px;max-width:520px;z-index:15;display:none;box-shadow:0 8px 25px #0002}.toast.error{background:#a34141}.modal{position:fixed;inset:0;background:#101c3470;z-index:10;display:none;align-items:center;justify-content:center;padding:20px}.modal.on{display:flex}.dialog{background:white;max-width:780px;width:100%;border-radius:14px;padding:25px;max-height:85vh;overflow:auto}.filelist{max-height:370px;overflow:auto;margin-top:12px}.fileline{display:flex;justify-content:space-between;gap:10px;padding:10px;border-bottom:1px solid var(--line)}.fileline button{border:0;background:none;text-align:left;color:var(--blue);word-break:break-all}.upload{padding:10px;border:1px dashed #cbd5e5;border-radius:8px;background:#fafcff;margin-top:10px}.empty{padding:25px;text-align:center;color:var(--muted);font-size:13px}canvas{width:100%;height:190px}.check{display:flex;align-items:center;gap:8px;padding:8px 0}.check input{width:auto}.check label{margin:0}a{color:var(--blue)}@media(max-width:1100px){main{padding:25px}.grid{grid-template-columns:1fr}.stats{grid-template-columns:1fr 1fr}.stats .card:last-child{display:none}.steps{grid-template-columns:1fr}}@media(max-width:720px){aside{position:static;width:100%;height:auto;padding:17px}.shell{display:block}.brand{font-size:20px}.subbrand,.sidefoot{display:none}.nav{display:flex;overflow:auto;margin-top:12px;gap:3px}.nav button{white-space:nowrap;padding:8px}.nav button span{display:none}main{margin:0;width:100%;padding:20px}.stats{grid-template-columns:1fr}.formgrid{grid-template-columns:1fr}.field.wide{grid-column:auto}.top{align-items:flex-start}h1{font-size:26px}.card{padding:19px}}
</style></head><body><div class="shell"><aside><div class="brand">Stroke<em>Font</em></div><div class="subbrand">字 体 生 成 实 验 室</div><nav class="nav" aria-label="主要导航">
<button class="on" data-page="home"><span>00</span>开始使用</button><button data-page="dataset"><span>01</span>数据集准备</button><button data-page="train"><span>02</span>训练与微调</button><button data-page="test"><span>03</span>测试与生成</button><button data-page="jobs"><span>04</span>任务与日志</button></nav><div class="sidefoot">LOCAL WORKSPACE<br>模型在本机运行<br>所有路径相对于项目根目录</div></aside><main><div class="top"><span>WORKSPACE / 字体实验室</span><span class="pill" id="live">● 本地服务已连接</span></div>
<section id="home" class="page on"><div class="eyebrow">FROM GLYPHS TO A TYPEFACE</div><h1>让字体实验，更直观。</h1><p class="lead">从字体文件到新字形，在一个工作台中准备数据、调整训练参数、查看生成结果。</p><div class="stats"><div class="card"><div class="caption">Python 环境</div><div id="python" class="statval">读取中…</div></div><div class="card"><div class="caption">NVIDIA GPU · 总显存 / 空闲显存</div><div id="gpu" class="statval">检测中…</div></div><div class="card"><div class="caption">当前会话</div><div id="jobcount" class="statval">0 个任务</div></div></div><div class="steps"><div class="card"><div class="stepnum">01</div><h2>准备字形数据</h2><p class="muted">将多种字体转换成按字符编号排列的白底黑字图片。</p><button class="button" onclick="navigate('dataset')">准备数据 →</button></div><div class="card"><div class="stepnum">02</div><h2>学习内容与风格</h2><p class="muted">先进行条件训练，再从检查点继续微调以支持引导生成。</p><button class="button" onclick="navigate('train')">配置训练 →</button></div><div class="card"><div class="stepnum">03</div><h2>生成目标字形</h2><p class="muted">选择模型和参考字图，输入文字，预览并下载生成图片。</p><button class="button" onclick="navigate('test')">开始生成 →</button></div></div><div class="grid"><div class="card"><h2>第一次使用？按这条路线开始</h2><ol class="instruction"><li>确认页面顶部的 Python 路径属于 <strong>StrokeFont 虚拟环境</strong>。界面会用同一解释器运行全部任务。</li><li>准备覆盖目标字符集的字体文件。<strong>完整字符表、图片编号、笔画编码必须一一对应。</strong></li><li>先用少量样本检查数据与参数，再启动正式训练。6 GB 显存建议从批大小 <strong>1</strong>、微批大小 <strong>1</strong> 开始。</li><li>训练完成后选择 <code>modelXXXXXX.pt</code> 微调；测试优先选择 <code>ema_0.9999_XXXXXX.pt</code>。</li></ol></div><div class="card"><h2>工作台如何运行</h2><p class="muted small">调用现有 font2img.py、train.py、sample.py。每个任务单独保存配置，不覆盖 cfg 中的原始配置文件。</p><div class="note">一次运行一个任务。关闭浏览器后任务继续；退出 WebUI 服务时会停止正在运行的任务。</div><p class="muted small">运行记录保存在 <code>.webui/jobs</code>。停止训练会中断当前批次，只能从最近已保存的检查点恢复。输出为图片，不能直接安装为 TTF 字体。</p></div></div></section>
<section id="dataset" class="page"><div class="eyebrow">01 / DATASET</div><h1>准备字体数据集</h1><p class="lead">将字体文件渲染成训练图片。相同编号在不同字体目录中表示同一个汉字。</p><div class="grid"><div><form id="dataset-form" class="card"><h2>转换配置</h2><div class="formgrid" id="dataset-fields"></div><div class="actions"><button class="button primary" type="submit">开始转换</button><button class="button" type="button" onclick="inspectDataset()">检查已有数据集</button></div><p class="help">为避免覆盖已有图片，转换输出目录须为空。未填写输出目录时自动分配任务目录。</p></form><div id="dataset-report" class="card" hidden></div></div><div class="card"><h2>准备步骤</h2><ol class="instruction"><li><strong>整理字体：</strong>在同一个目录内放入 TTF、OTF 或 TTC 字体文件，不要放其他文件。脚本不会递归读取子目录。</li><li><strong>选择字符表：</strong>默认 total_chn.txt 包含 6625 字。转换时使用一份去除空白的独立副本。</li><li><strong>调整尺寸：</strong>默认画布 80×80、绘制大小 60，与原模型一致。</li><li><strong>转换后检查：</strong>查看字形是否居中、有无方框占位字、截断或空白。脚本不能识别所有缺字占位符。</li><li><strong>进入训练：</strong>把生成的数据集目录填入训练页；自定义字符集也需要对应的笔画编码。</li></ol><div class="note"><code>data/id_0/00000.png</code><br><code>data/id_1/00000.png</code><br>两个文件应是同一汉字的不同字体。</div></div></div></section>
<section id="train" class="page"><div class="eyebrow">02 / TRAINING</div><h1>训练与微调</h1><p class="lead">设置目标累计步数，监控损失曲线，并从已保存的模型继续实验。</p><div class="grid"><form id="train-form" class="card"><div class="tabs"><button class="on" type="button" id="mode-train" onclick="setMode(false)">条件训练</button><button type="button" id="mode-finetune" onclick="setMode(true)">引导微调</button></div><div class="formgrid" id="train-fields"></div><details><summary>高级模型参数 · YAML</summary><p class="help">与基础配置合并；上方表单字段优先。恢复训练时网络结构必须与检查点一致。</p><textarea id="train-advanced" rows="8" spellcheck="false">num_channels: 128
num_res_blocks: 3
attention_resolutions: '40, 20, 10'
diffusion_steps: 1000
dropout: 0.1</textarea></details><div class="actions"><button class="button primary" type="submit" id="train-start">开始条件训练</button><button class="button" type="button" onclick="importConfig('train')">从基础配置载入</button></div></form><div class="card"><h2 id="train-guide-title">条件训练 · 操作说明</h2><ol class="instruction" id="train-guide"></ol><div class="note warn">训练步数通常较大。建议先把目标步数设为 2、日志间隔设为 1，检查整个流程；这只能验证运行，不代表学到可用字体。</div><details><summary>参数解释</summary><p class="small muted"><strong>批大小：</strong>每个进程一次取多少张图片。<br><strong>微批大小：</strong>拆分批次以降低显存占用，不能消除模型本身的显存需求。<br><strong>进程数：</strong>GPU 训练通常每张卡一个进程。只有一张 GPU 时请保持 1。<br><strong>笔画文件：</strong>可留空禁用笔画条件，训练与测试必须保持一致。<br><strong>检查点：</strong>恢复请选择 modelXXXXXX.pt，其同目录的优化器和 EMA 文件会自动匹配。</p></details></div></div></section>
<section id="test" class="page"><div class="eyebrow">03 / GENERATION</div><h1>用参考字形，生成新的文字。</h1><p class="lead">参考图片提供字体风格，输入字符决定生成内容。网络参数必须与所选模型一致。</p><div class="grid"><form id="test-form" class="card"><h2>生成配置</h2><div class="formgrid" id="test-fields"></div><div class="upload"><label for="reference-upload">也可以上传参考图片</label><input type="file" id="reference-upload" accept="image/png,image/jpeg,image/gif"><p class="help">上传保存在本机 .webui/uploads；建议居中、白底黑字的单字图。</p><img id="reference-preview" alt="参考字图预览" style="max-width:120px;max-height:120px;margin-top:10px" hidden></div><details><summary>高级模型参数 · YAML</summary><textarea id="test-advanced" rows="7" spellcheck="false">num_channels: 128
num_res_blocks: 3
attention_resolutions: '40, 20, 10'
diffusion_steps: 1000</textarea></details><div class="actions"><button type="submit" class="button primary">开始生成</button><button type="button" class="button" onclick="importConfig('test')">从基础配置载入</button></div></form><div class="card"><h2>生成步骤与提示</h2><ol class="instruction"><li><strong>选择完整模型：</strong>使用训练或微调产出的 model / ema 权重。chinese_styenc.ckpt 只是风格编码器，不能单独生成。</li><li><strong>设置一致的模型结构：</strong>字符表、图片尺寸、笔画条件和高级参数应与训练时一致。</li><li><strong>选择参考字图：</strong>图片会缩放为目标尺寸；白底黑字通常更符合训练分布。</li><li><strong>输入文字：</strong>空白会被忽略，每个字符生成一张图片，顺序保持一致。输入必须在完整字符表中。</li><li><strong>预览结果：</strong>完成后在任务页查看、下载图片。默认新建结果目录，保留以前的实验。</li></ol><div class="note">使用条件训练模型时关闭「引导生成」；微调模型可开启，分别调整内容与笔画引导。DDIM 默认 25 步，关闭后使用完整扩散采样，耗时更长。</div></div></div></section>
<section id="jobs" class="page"><div class="eyebrow">04 / EXPERIMENTS</div><h1>任务、曲线与结果</h1><p class="lead">查看当前运行状态，追踪训练损失，并保留每次实验使用的配置。</p><div class="card"><div class="progresshead"><h2 style="margin:0" id="job-title">选择一个任务</h2><button class="button danger" id="stop" disabled onclick="stopJob()">停止任务</button></div><div id="job-summary" class="muted small" style="margin-top:10px">运行任务后，日志将自动显示在这里。</div><div class="bar"><div id="progress-bar" style="width:0%"></div></div><div id="job-progress" class="help"></div><canvas id="loss-chart" role="img" aria-label="训练损失曲线" hidden></canvas><div class="actions"><button class="button" onclick="downloadConfig()">下载本次配置</button><button class="button" id="use-data" hidden onclick="useDataset()">将数据集用于训练 →</button><button class="button" id="use-finetune" hidden onclick="useModel(true)">载入模型继续微调 →</button><button class="button" id="use-test" hidden onclick="useModel(false)">将模型用于测试 →</button><label class="check"><input type="checkbox" id="autoscroll" checked>日志自动滚动</label></div><pre id="logs" class="console">等待任务…</pre><div id="results" class="gallery"></div></div><div class="card"><h2>任务记录</h2><div class="historywrap"><table><thead><tr><th>时间</th><th>任务</th><th>状态</th><th>操作</th></tr></thead><tbody id="history"><tr><td colspan="4" class="empty">还没有任务</td></tr></tbody></table></div></div></section>
</main></div><div id="toast" class="toast" role="status" aria-live="polite"></div><div id="browser" class="modal" role="dialog" aria-modal="true" aria-labelledby="browser-title"><div class="dialog"><div class="progresshead"><h2 id="browser-title">选择本地路径</h2><button class="button" onclick="closeBrowser()">关闭</button></div><div class="pathrow"><input id="browse-path" aria-label="当前目录"><button class="button" onclick="loadDirectory(document.getElementById('browse-path').value)">前往</button><button class="button" onclick="loadDirectory('')">项目根目录</button></div><div id="file-list" class="filelist"></div><div class="actions"><button class="button primary" id="choose-directory" onclick="chooseDirectory()">使用当前目录</button></div></div></div>
<script>
const TOKEN=__TOKEN__, $=id=>document.getElementById(id);let activeJob=null,jobDetail=null,finetune=false,browseTarget=null,currentDir='',pollBusy=false,toastTimer,lastGallery='';
const labels={dataset:'数据集转换',train:'条件训练',finetune:'引导微调',test:'字形生成'},statuses={starting:'准备中',running:'运行中',stopping:'正在停止',completed:'已完成',failed:'失败',cancelled:'已停止',interrupted:'服务已重启'};
const esc=s=>String(s??'').replace(/[&<>"']/g,c=>({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[c]));
function toast(message,error=false){$('toast').textContent=message;$('toast').className='toast'+(error?' error':'');$('toast').style.display='block';clearTimeout(toastTimer);toastTimer=setTimeout(()=>$('toast').style.display='none',error?10000:4500)}
async function api(path,data){const opts={headers:{'X-StrokeFont-Token':TOKEN}};if(data!==undefined){opts.method='POST';opts.headers['Content-Type']='application/json';opts.body=JSON.stringify(data)}const r=await fetch(path,opts);let v;try{v=await r.json()}catch{throw Error('服务返回了无效响应')}if(!r.ok)throw Error(v.error||'请求失败');return v}
function navigate(name){document.querySelectorAll('.page').forEach(e=>e.classList.toggle('on',e.id===name));document.querySelectorAll('.nav button').forEach(e=>e.classList.toggle('on',e.dataset.page===name));window.scrollTo(0,0)}
document.querySelectorAll('.nav button').forEach(e=>e.onclick=()=>navigate(e.dataset.page));
const schemas={
dataset:[['fonts','字体文件目录','path','./ttf_folder','目录中仅放字体文件，不递归读取子目录。',true],['characters','完整字符表','file','./total_chn.txt','UTF-8；字符顺序决定图片编号。',true],['output','数据集输出目录','path','','留空自动新建；也可填写尚未创建的目录。',true],['image_size','图片画布大小','number',80,'默认 80×80。'],['chara_size','字体绘制大小','number',60,'建议小于画布大小，留出边距。']],
train:[['template','基础配置文件','file','./cfg/train_cfg.yaml','未在表单列出的参数从此文件继承。',true],['dataset','训练图片目录','path','./data','选择转换后的数据集根目录。',true],['characters','完整字符表','file','./total_chn.txt','自动读取字符总数，须与图片编号对应。',true],['stroke','笔画编码文件','file','./chinese_stroke.txt','可留空；自定义字符集必须同步修改。',true],['output','模型保存目录','path','./trained_models','新训练需空目录；恢复训练可沿用原目录。',true],['encoder','预训练风格编码器','file','./pretrained_models/chinese_styenc.ckpt','首次条件训练必需；加载后冻结参数。',true],['resume','恢复训练检查点','file','','微调必填；条件训练留空表示从头开始。',true],['batch_size','每进程批大小','number',1,'6 GB GPU 建议先设为 1。'],['microbatch','微批大小','number',1,'将批次拆分，降低显存需求。'],['lr','学习率','decimal',0.0001,'默认 1e-4。'],['processes','进程数','number',1,'一般为参与训练的 GPU 数量。'],['train_step','条件训练累计目标步数','number',420000,'条件训练阶段的停止步数。'],['total_train_step','微调累计目标步数','number',800000,'包括已完成的条件训练步数。'],['log_interval','日志间隔 / 步','number',10,'曲线只在输出日志时刷新。'],['save_interval','保存间隔 / 步','number',10000,'停止后只能从已保存权重恢复。'],['image_size','图片大小','number',80,'须与数据和模型一致。']],
test:[['template','基础配置文件','file','./cfg/test_cfg.yaml','网络结构须与模型训练时一致。',true],['model','完整模型权重','file','','优先选择微调后的 EMA 模型。',true],['reference','参考字形图片','file','','也可以使用下方上传控件。',true],['characters','完整字符表','file','./total_chn.txt','必须与训练字符表的内容、顺序完全一致。',true],['stroke','笔画编码文件','file','./chinese_stroke.txt','训练未使用笔画条件时请留空。',true],['text','需要生成的文字','textarea','你好世界','每个非空白字符生成一张图片。',true],['output','结果输出目录','path','','留空自动分配新目录，避免覆盖旧结果。',true],['batch_size','每进程生成批大小','number',1,'引导生成占用更多显存，建议从 1 开始。'],['image_size','图片大小','number',80,'与模型训练尺寸一致。'],['processes','进程数','number',1,'单 GPU 保持 1。'],['sample_steps','DDIM 采样步数','number',25,'1000 步扩散推荐 25 / 50 / 100。'],['cont_scale','内容引导强度','decimal',3,'开启引导生成后生效。'],['sk_scale','笔画引导强度','decimal',3,'同时启用笔画条件时生效。'],['guidance','引导生成','checkbox',true,'使用经过 classifier-free 微调的模型时开启。'],['ddim','使用 DDIM 快速采样','checkbox',true,'关闭则执行完整扩散采样。']]};
function renderForms(){for(const [name,fields]of Object.entries(schemas)){$(name+'-fields').innerHTML=fields.map(([key,label,type,value,help,wide])=>{const id=name+'-'+key;let el;if(type==='checkbox')el=`<div class="check"><input id="${id}" type="checkbox" ${value?'checked':''}><label for="${id}">${esc(label)}</label></div>`;else{el=`<label for="${id}">${esc(label)}</label>`;if(type==='textarea')el+=`<textarea id="${id}">${esc(value)}</textarea>`;else el+=`<div class="pathrow"><input id="${id}" value="${esc(value)}" type="${['number','decimal'].includes(type)?'number':'text'}" ${type==='decimal'?'step="any" min="0"':type==='number'?'step="1" min="1"':''}>${['path','file'].includes(type)?`<button class="button" type="button" onclick="openBrowser('${id}','${type}')">选择</button>`:''}</div>`}return `<div class="field ${wide?'wide':''}">${el}<p class="help">${esc(help)}</p></div>`}).join('')}}
renderForms();
function values(name){const data={};for(const [key,,type]of schemas[name])data[key]=type==='checkbox'?$(name+'-'+key).checked:$(name+'-'+key).value;if(name!=='dataset')data.advanced=$(name+'-advanced').value;return data}
function setMode(mode){finetune=mode;$('mode-train').classList.toggle('on',!mode);$('mode-finetune').classList.toggle('on',mode);$('train-start').textContent=mode?'开始引导微调':'开始条件训练';$('train-guide-title').textContent=mode?'引导微调 · 操作说明':'条件训练 · 操作说明';$('train-guide').innerHTML=mode?'<li><strong>先完成条件训练</strong>，选择对应的 modelXXXXXX.pt 检查点。</li><li><strong>保留相同结构和字符集</strong>，也应使用相同数据集与笔画设置。</li><li><strong>设置累计目标步数</strong>：从 420000 步到 800000 步，表示再训练约 380000 步。</li><li><strong>启动微调</strong>：训练会随机屏蔽字符及笔画条件，之后可使用引导采样。</li>':'<li><strong>选择数据集根目录</strong>，其中每个子目录对应一种字体。</li><li><strong>核对字符表和笔画文件</strong>，检查每个编号对应的字符。</li><li><strong>选择风格编码器</strong>，从头训练时使用 chinese_styenc.ckpt。</li><li><strong>先用小批次试运行</strong>，检查损失、显存占用和模型保存，再启动正式训练。</li>';}
setMode(false);
for(const name of ['dataset','train','test'])$(name+'-form').onsubmit=async e=>{e.preventDefault();const button=e.submitter;button.disabled=true;try{const kind=name==='train'&&finetune?'finetune':name;const job=await api('/api/start',{kind,data:values(name)});activeJob=job.id;lastGallery='';navigate('jobs');renderJob(job);toast('任务已启动');await poll()}catch(err){toast(err.message,true)}finally{button.disabled=false}};
async function inspectDataset(){try{const path=$('dataset-output').value||$('train-dataset').value;toast('正在扫描数据集…');const report=await api('/api/inspect',{path});$('dataset-report').hidden=false;$('dataset-report').innerHTML=`<h2>数据集检查</h2><p><strong>${report.count}</strong> 张图片 · <strong>${Object.keys(report.fonts).length}</strong> 种字体 · <strong>${report.characters}</strong> 个字符编号</p><p class="small muted">${esc(report.path)}</p>${report.missing.length||report.invalid.length?`<div class="note warn">存在缺失编号或命名异常：${esc([...report.missing,...report.invalid].join(', '))}。训练前请处理。</div>`:''}<details><summary>各字体图片数</summary><pre>${esc(JSON.stringify(report.fonts,null,2))}</pre></details><div class="gallery">${report.previews.map(tile).join('')}</div>`;$('train-dataset').value=report.path}catch(e){toast(e.message,true)}}
function tile(item){return `<div class="tile"><img loading="lazy" src="${esc(item.url)}" alt="${esc(item.label)}"><div>${esc(item.label)}</div><a href="${esc(item.url)}&download=1">下载图片</a></div>`}
async function importConfig(name){try{const result=await api('/api/config',{path:$(name+'-template').value});const map={data_dir:'dataset',model_save_dir:'output',sty_encoder_path:'encoder',resume_checkpoint:'resume',stroke_path:'stroke',model_path:'model',sty_img_path:'reference',total_txt_file:'characters',img_save_path:'output',classifier_free:'guidance',use_ddim:'ddim'};for(const [key,,,]of schemas[name]){const cfgKey=Object.keys(map).find(k=>map[k]===key&&k in result.values)||key;if(cfgKey in result.values){const el=$(name+'-'+key);if(el.type==='checkbox')el.checked=!!result.values[cfgKey];else el.value=result.values[cfgKey]??''}}$(name+'-advanced').value=result.advanced;if(name==='train')setMode(!!result.values.classifier_free);toast('已载入配置；请核对占位路径与批大小。')}catch(e){toast(e.message,true)}}
async function openBrowser(id,type){browseTarget={id,type};$('browser').classList.add('on');$('choose-directory').hidden=type==='file';await loadDirectory($(id).value)}
function closeBrowser(){$('browser').classList.remove('on');if(browseTarget)$(browseTarget.id).focus()}
async function loadDirectory(path){try{const r=await api('/api/browse',{path});currentDir=r.path;$('browse-path').value=r.path;$('file-list').innerHTML='';const add=(name,path,dir)=>{const row=document.createElement('div');row.className='fileline';const b=document.createElement('button');b.textContent=(dir?'▸  ':'·  ')+name;b.onclick=()=>dir?loadDirectory(path):choosePath(path);b.disabled=!dir&&browseTarget.type!=='file';row.append(b);$('file-list').append(row)};if(r.parent!==r.path)add('上一级',r.parent,true);for(const item of r.entries)add(item.name,item.path,item.directory);if(r.truncated)toast('仅显示前 400 项，可直接输入完整路径。')}catch(e){toast(e.message,true)}}
function choosePath(path){$(browseTarget.id).value=path;closeBrowser()}function chooseDirectory(){choosePath(currentDir)}
$('browse-path').onkeydown=e=>{if(e.key==='Enter'){e.preventDefault();loadDirectory(e.target.value)}};document.addEventListener('keydown',e=>{if(e.key==='Escape')closeBrowser()});
$('reference-upload').onchange=async e=>{const file=e.target.files[0];if(!file)return;if(file.size>16*1024*1024){toast('图片不能超过 16 MB。',true);return}try{const encoded=await new Promise((resolve,reject)=>{const r=new FileReader();r.onload=()=>resolve(r.result.split(',')[1]);r.onerror=reject;r.readAsDataURL(file)});const r=await api('/api/upload',{name:file.name,content:encoded});$('test-reference').value=r.path;$('reference-preview').src=r.url;$('reference-preview').hidden=false;toast('参考图片已保存在本机')}catch(err){toast(err.message,true)}};
function renderJob(job){jobDetail=job;$('job-title').textContent=labels[job.kind]+' · '+statuses[job.status];$('job-summary').innerHTML=`<span class="status ${esc(job.status)}">${esc(statuses[job.status])}</span>　${esc(job.id)}<br>输出目录：<code>${esc(job.output)}</code>${job.error?'<br>'+esc(job.error):''}`;$('stop').disabled=!['running','starting'].includes(job.status);const el=$('logs');el.textContent=job.logs||'进程已启动，等待输出…';if($('autoscroll').checked)el.scrollTop=el.scrollHeight;const m=job.metrics||[];const percent=job.status==='completed'?100:m.length?Math.min(100,100*m[m.length-1].step/job.target):0;$('progress-bar').style.width=percent+'%';$('job-progress').textContent=m.length?`累计步数 ${m[m.length-1].step} / ${job.target} · loss ${m[m.length-1].loss.toFixed(5)} · 曲线显示最近 ${m.length} 个记录点`:'状态来自子进程退出码；转换与生成阶段请以日志为准。';$('loss-chart').hidden=!m.length;if(m.length)drawChart(m);if(job.kind==='test'&&job.status==='completed'&&lastGallery!==job.id){lastGallery=job.id;api('/api/results',{id:job.id}).then(r=>$('results').innerHTML=r.images.map(tile).join('')).catch(e=>toast(e.message,true))}else if(job.kind!=='test'||job.status!=='completed')$('results').innerHTML='';}
function drawChart(data){const c=$('loss-chart'),rect=c.getBoundingClientRect(),dpr=window.devicePixelRatio||1;c.width=Math.max(200,rect.width)*dpr;c.height=190*dpr;const x=c.getContext('2d');x.scale(dpr,dpr);const w=c.width/dpr,h=190,l=53,r=17,t=18,b=32;const xs=data.map(v=>v.step),ys=data.map(v=>v.loss),xmin=Math.min(...xs),xmax=Math.max(...xs),ymin=Math.min(...ys),ymax=Math.max(...ys);x.font='11px system-ui';for(let i=0;i<4;i++){const py=t+(h-t-b)*i/3;x.strokeStyle='#e8edf4';x.beginPath();x.moveTo(l,py);x.lineTo(w-r,py);x.stroke();x.fillStyle='#778296';x.fillText((ymax-(ymax-ymin)*i/3).toFixed(3),0,py+4)}x.fillText(String(xmin),l,h-9);x.fillText(String(xmax),w-r-35,h-9);x.strokeStyle='#395be0';x.lineWidth=2;x.beginPath();data.forEach((v,i)=>{const px=l+(v.step-xmin)/(xmax-xmin||1)*(w-l-r),py=t+(ymax-v.loss)/(ymax-ymin||1)*(h-t-b);i?x.lineTo(px,py):x.moveTo(px,py)});x.stroke();if(data.length===1){x.fillStyle='#395be0';x.beginPath();x.arc(l,t,3,0,Math.PI*2);x.fill()}}
const modelKeys=['num_channels','num_res_blocks','num_heads','num_head_channels','num_heads_upsample','attention_resolutions','channel_mult','diffusion_steps','noise_schedule','dropout','use_checkpoint','use_scale_shift_norm','resblock_updown','use_fp16','use_new_attention_order','learn_sigma','predict_xstart','rescale_timesteps','rescale_learned_sigmas','use_kl'];
function useDataset(){if(!jobDetail)return;$('train-dataset').value=jobDetail.output;$('dataset-output').value=jobDetail.output;if(jobDetail.inputs?.characters)$('train-characters').value=jobDetail.inputs.characters;navigate('train');toast('已填入数据集目录，请核对笔画文件和训练参数。')}
async function useModel(forFine){if(!jobDetail)return;try{const j=jobDetail,r=await api('/api/checkpoints',{id:j.id}),name=forFine?'train':'test',path=forFine?r.model:(r.ema||r.model);if(!path)throw Error('该目录中尚未找到可用检查点。');const cfg=j.config,inputs=j.inputs||{};$(name+'-'+(forFine?'resume':'model')).value=path;$(name+'-characters').value=inputs.characters||'./total_chn.txt';$(name+'-stroke').value=cfg.stroke_path||'';$(name+'-image_size').value=cfg.image_size;$(name+'-advanced').value=JSON.stringify(Object.fromEntries(Object.entries(cfg).filter(([k])=>modelKeys.includes(k))),null,2);if(forFine){$('train-dataset').value=cfg.data_dir;$('train-output').value=j.output;$('train-total_train_step').value=cfg.total_train_step;setMode(true)}else $('test-guidance').checked=j.kind==='finetune';navigate(name);toast('已载入模型与结构参数，请核对其余配置。')}catch(e){toast(e.message,true)}}
async function selectJob(id){activeJob=id;lastGallery='';try{renderJob(await api('/api/job?id='+encodeURIComponent(id)));navigate('jobs')}catch(e){toast(e.message,true)}}
async function poll(){if(pollBusy)return;pollBusy=true;try{const s=await api('/api/state');$('python').textContent=s.environment.version+' · '+s.environment.python;$('gpu').textContent=s.environment.gpu;$('jobcount').textContent=s.jobs.length+' 个任务';$('history').innerHTML=s.jobs.length?s.jobs.map(j=>`<tr><td>${esc(new Date(j.created*1000).toLocaleString())}</td><td>${esc(labels[j.kind])}</td><td><span class="status ${esc(j.status)}">${esc(statuses[j.status])}</span></td><td><button class="button" data-job="${esc(j.id)}">查看</button></td></tr>`).join(''):'<tr><td colspan="4" class="empty">还没有任务</td></tr>';$('history').querySelectorAll('[data-job]').forEach(b=>b.onclick=()=>selectJob(b.dataset.job));if(!activeJob&&s.jobs.length)activeJob=s.jobs[0].id;if(activeJob)renderJob(await api('/api/job?id='+encodeURIComponent(activeJob)));$('use-data').hidden=!(jobDetail?.kind==='dataset'&&jobDetail?.status==='completed');const canModel=['train','finetune'].includes(jobDetail?.kind)&&!['starting','running','stopping'].includes(jobDetail?.status);$('use-finetune').hidden=!canModel;$('use-test').hidden=!canModel;$('live').textContent='● 本地服务已连接'}catch(e){$('live').textContent='○ 服务连接中断'}finally{pollBusy=false}}
async function stopJob(){if(!activeJob)return;if(!confirm('停止当前任务及其子进程？训练只能从最近已保存的检查点恢复。'))return;try{await api('/api/stop',{id:activeJob});toast('已发送停止请求');await poll()}catch(e){toast(e.message,true)}}
async function downloadConfig(){if(!jobDetail?.config){toast('该任务没有 YAML 配置。');return}try{const r=await api('/api/job-config?id='+encodeURIComponent(activeJob));const url=URL.createObjectURL(new Blob([r.yaml],{type:'text/yaml;charset=utf-8'})),a=document.createElement('a');a.href=url;a.download=activeJob+'.yaml';a.click();setTimeout(()=>URL.revokeObjectURL(url),1000)}catch(e){toast(e.message,true)}}
window.addEventListener('resize',()=>{if(jobDetail?.metrics?.length)drawChart(jobDetail.metrics)});poll();setInterval(poll,2500);
</script></body></html>'''


class Handler(BaseHTTPRequestHandler):
    server_version = 'StrokeFontWebUI/1.0'

    def log_message(self, *_):
        pass

    @property
    def app(self):
        return self.server.app

    def respond(self, data, code=200, content_type='application/json; charset=utf-8', **headers):
        body = json.dumps(data, ensure_ascii=False, allow_nan=False).encode('utf-8') if content_type.startswith('application/json') else data
        self.send_response(code)
        self.send_header('Content-Type', content_type)
        self.send_header('Content-Length', str(len(body)))
        self.send_header('Cache-Control', 'no-store')
        self.send_header('X-Content-Type-Options', 'nosniff')
        self.send_header('X-Frame-Options', 'DENY')
        self.send_header('Referrer-Policy', 'no-referrer')
        for key, value in headers.items():
            self.send_header(key.replace('_', '-'), value)
        self.end_headers()
        self.wfile.write(body)

    def allowed(self, token=True):
        host = self.headers.get('Host', '')
        permitted = {f'127.0.0.1:{self.server.server_port}', f'localhost:{self.server.server_port}'}
        if host not in permitted or (self.headers.get('Origin') and self.headers['Origin'] not in {'http://' + h for h in permitted}):
            raise PermissionError('只允许本机同源访问。')
        if token and not secrets.compare_digest(self.headers.get('X-StrokeFont-Token', ''), TOKEN):
            raise PermissionError('无效会话，请刷新页面。')

    def do_GET(self):
        try:
            url = urlparse(self.path)
            query = parse_qs(url.query)
            self.allowed(token=url.path not in ('/', '/api/image', '/favicon.ico'))
            if url.path == '/':
                self.respond(HTML.replace('__TOKEN__', json.dumps(TOKEN)).encode('utf-8'), content_type='text/html; charset=utf-8')
            elif url.path == '/api/state':
                with self.app.lock:
                    jobs = [{k: j[k] for k in ('id', 'kind', 'status', 'created')} for j in reversed(list(self.app.jobs.values()))]
                self.respond({'environment': self.app.environment, 'jobs': jobs})
            elif url.path == '/api/job':
                self.respond(self.app.view(query['id'][0]))
            elif url.path == '/api/job-config':
                job = self.app.jobs[query['id'][0]]
                self.respond({'yaml': (Path(job['folder']) / 'config.yaml').read_text(encoding='utf-8')})
            elif url.path == '/api/image':
                path = self.app.images[query['id'][0]]
                headers = {'Content_Disposition': 'attachment; filename="image' + path.suffix.lower() + '"'} if 'download' in query else {}
                self.respond(path.read_bytes(), content_type=mimetypes.guess_type(path.name)[0] or 'image/png', **headers)
            else:
                self.respond({'error': '地址不存在。'}, 404)
        except PermissionError as exc:
            self.respond({'error': str(exc)}, 403)
        except (KeyError, FileNotFoundError):
            self.respond({'error': '记录或文件不存在。'}, 404)
        except (ValueError, OSError) as exc:
            self.respond({'error': str(exc)}, 400)

    def do_POST(self):
        try:
            self.allowed()
            size = int(self.headers.get('Content-Length', '0'))
            if not 0 < size <= 24 * 1024 * 1024:
                raise ValueError('请求大小无效，最大 24 MB。')
            data = json.loads(self.rfile.read(size))
            if not isinstance(data, dict):
                raise ValueError('请求格式无效。')
            route = urlparse(self.path).path
            if route == '/api/start':
                self.respond(self.app.start(data['kind'], data['data']))
            elif route == '/api/stop':
                self.app.stop(data['id'])
                self.respond({'ok': True})
            elif route == '/api/inspect':
                self.respond(self.app.inspect(data['path']))
            elif route == '/api/browse':
                path = local_path(data.get('path') or str(ROOT))
                if not path.is_dir():
                    path = path.parent
                entries = sorted(path.iterdir(), key=lambda p: (not p.is_dir(), p.name.lower()))
                self.respond({'path': str(path), 'parent': str(path.parent), 'truncated': len(entries) > 400,
                              'entries': [{'name': p.name, 'path': str(p), 'directory': p.is_dir()} for p in entries[:400]]})
            elif route == '/api/config':
                import yaml
                config = read_config(data['path'])
                model_keys = {'num_channels', 'num_res_blocks', 'num_heads', 'num_head_channels', 'num_heads_upsample',
                              'attention_resolutions', 'channel_mult', 'diffusion_steps', 'noise_schedule', 'dropout',
                              'use_checkpoint', 'use_scale_shift_norm', 'resblock_updown', 'use_fp16', 'use_new_attention_order',
                              'learn_sigma', 'sigma_small', 'predict_xstart', 'rescale_timesteps', 'rescale_learned_sigmas', 'use_kl'}
                self.respond({'values': config, 'advanced': yaml.safe_dump({k: v for k, v in config.items() if k in model_keys}, sort_keys=False)})
            elif route == '/api/upload':
                content = base64.b64decode(data['content'], validate=True)
                if len(content) > 16 * 1024 * 1024:
                    raise ValueError('图片大小不能超过 16 MB。')
                from PIL import Image
                with Image.open(io.BytesIO(content)) as image:
                    image.verify()
                    suffix = {'PNG': '.png', 'JPEG': '.jpg', 'GIF': '.gif'}.get(image.format)
                if not suffix:
                    raise ValueError('仅支持 PNG、JPEG 或 GIF 参考图。')
                folder = self.app.storage / 'uploads'
                folder.mkdir(parents=True, exist_ok=True)
                path = folder / (secrets.token_hex(12) + suffix)
                path.write_bytes(content)
                self.respond({'path': str(path), 'url': self.app.image(path)})
            elif route == '/api/results':
                job = self.app.jobs[data['id']]
                images = sorted(Path(job['output']).glob('*.png'))[:300]
                chars = characters(Path(job['folder']) / 'generation.txt') if job['kind'] == 'test' else ''
                self.respond({'images': [{'url': self.app.image(p), 'label': (chars[i] + ' · ' if i < len(chars) else '') + p.name}
                                         for i, p in enumerate(images)]})
            elif route == '/api/checkpoints':
                job = self.app.jobs[data['id']]
                folder = Path(job['output'])
                models = sorted((p for p in folder.glob('model*.pt') if re.fullmatch(r'model\d+\.pt', p.name)), key=lambda p: p.stat().st_mtime, reverse=True)
                emas = sorted(folder.glob('ema_*.pt'), key=lambda p: p.stat().st_mtime, reverse=True)
                self.respond({'model': str(models[0]) if models else '', 'ema': str(emas[0]) if emas else ''})
            else:
                self.respond({'error': '地址不存在。'}, 404)
        except PermissionError as exc:
            self.respond({'error': str(exc)}, 403)
        except (KeyError, ValueError, OSError, TypeError) as exc:
            self.respond({'error': str(exc)}, 400)
        except Exception as exc:
            self.respond({'error': f'{type(exc).__name__}: {exc}'}, 400)


def main():
    parser = argparse.ArgumentParser(description='StrokeFont 单文件本地 WebUI，无需额外 Web 框架。')
    parser.add_argument('--port', type=int, default=7860)
    parser.add_argument('--open-browser', action='store_true')
    args = parser.parse_args()
    app = App()
    try:
        server = ThreadingHTTPServer(('127.0.0.1', args.port), Handler)
    except OSError as exc:
        parser.exit(1, f'无法启动服务：{exc}\n可尝试 python webui.py --port 7861\n')
    server.app = app
    threading.Thread(target=app.probe, daemon=True).start()
    url = f'http://127.0.0.1:{server.server_port}'
    print(f'StrokeFont WebUI: {url}\nPython: {sys.executable}\n按 Ctrl+C 退出（会停止正在运行的任务）。', flush=True)
    if args.open_browser:
        webbrowser.open(url)
    try:
        server.serve_forever(poll_interval=0.3)
    except KeyboardInterrupt:
        pass
    finally:
        for job in list(app.jobs.values()):
            if job['status'] in ACTIVE:
                try:
                    app.stop(job['id'])
                    app.processes[job['id']].wait(timeout=12)
                except (OSError, ValueError, subprocess.TimeoutExpired):
                    pass
        server.server_close()


if __name__ == '__main__':
    main()
