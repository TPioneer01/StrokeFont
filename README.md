# StrokeFont

StrokeFont 使用条件扩散模型生成字体图片。输入目标字符、笔画编码和一张字体参考图片，模型生成具有参考字体风格的字形。默认使用 6625 个中文字符，输出为 80×80 的 RGB 图片；不直接生成 TTF 字体文件。

## 1. 环境依赖

建议使用 Python 3.10 和 NVIDIA GPU。当前依赖版本为 PyTorch 2.5.1（CUDA 11.8）、NumPy 1.23.5、Pillow 9.5.0 等，完整列表见 `requirements.txt`。其中 `attrdict3` 提供代码使用的 `attrdict` 模块。

已有环境时直接激活；新机器可先创建：

```powershell
conda create -n StrokeFont python=3.10 -y
conda activate StrokeFont
python -m pip install -r requirements.txt
```

上述依赖文件安装 CUDA 11.8 版 PyTorch，需要兼容的 NVIDIA 驱动。检查安装结果：

```powershell
python -m pip check
python -c "import torch; print(torch.__version__); print(torch.cuda.is_available())"
```

仅使用 CPU 时，将 `requirements.txt` 中的 `torch==2.5.1+cu118` 改为 `torch==2.5.1`，删除额外索引那一行，再安装依赖。CPU 可用于功能检查，完整训练建议使用 GPU。

请在项目根目录运行以下命令。单进程直接使用 `python`，本机多进程使用 `launch.py`。Windows 自动选择 Gloo，支持 NCCL 的 CUDA 环境自动选择 NCCL；各进程通过 `LOCAL_RANK` 选择显卡，不会覆盖用户设置的 `CUDA_VISIBLE_DEVICES`。

## 2. 准备数据集

### 字符表与笔画编码

- `total_chn.txt`：完整字符表，字符顺序决定字符编号。
- `chinese_stroke.txt`：对应字符的 32 维笔画编码，行顺序必须与字符表一致。
- `gen_char.txt`：测试时要生成的字符，字符必须包含在完整字符表中。

使用自定义字符集时，需要同时更新字符表、笔画编码以及配置中的 `chara_nums`。训练图片的编号也必须对应这一字符顺序，不能对不同字体分别重新编号。

### 将字体转换为图片

将用于训练的字体文件放入 `./ttf_folder`，建议使用覆盖目标字符集的 TTF 字体，然后运行：

```powershell
python font2img.py --ttf_path ./ttf_folder --chara ./total_chn.txt --save_path ./data --img_size 80 --chara_size 60
```

`img_size` 为画布大小，`chara_size` 为绘制字体大小。脚本生成白底黑字 RGB 图片，并按字体分别保存：

```text
data/
├── id_0/
│   ├── 00000.png
│   ├── 00001.png
│   └── ...
├── id_1/
│   ├── 00000.png
│   ├── 00001.png
│   └── ...
└── ...
```

同名图片在不同目录中必须代表同一个字符。训练时会从同一字体目录随机抽取参考字图，以学习字体风格。转换后请检查缺字、方框占位字和空白图片，确保目标字符集在数据中完整出现；脚本的空白过滤无法识别所有缺字占位符。

## 3. 开启条件训练

编辑 `cfg/train_cfg.yaml`，主要参数如下：

```yaml
data_dir: './data'
chara_nums: 6625
image_size: 80
batch_size: 16
lr: 0.0001
train_step: 420000
total_train_step: 800000
sty_encoder_path: './pretrained_models/chinese_styenc.ckpt'
model_save_dir: './trained_models'
stroke_path: './chinese_stroke.txt'
classifier_free: false
resume_checkpoint: ''
```

保留配置中其他网络和扩散参数。首次训练会加载预训练风格编码器并冻结其参数。`pretrained_models` 下的风格编码器不能直接作为完整生成模型用于测试。

单 GPU 启动：

```powershell
python train.py --cfg_path ./cfg/train_cfg.yaml
```

多 GPU 启动（以下示例需要两张可用 GPU）：

```powershell
python launch.py --nproc-per-node 2 train.py --cfg_path ./cfg/train_cfg.yaml
```

`launch.py` 使用当前 Python 环境启动子进程，自动设置进程编号和本机通信端口；任一进程失败时会停止其他进程。当前 Windows 版 PyTorch 2.5.1 的 `torchrun` 启动阶段存在 libuv 兼容问题，请使用上述启动脚本。在支持的系统上也可以使用 PyTorch 原生启动器：

```shell
torchrun --standalone --nproc-per-node=2 train.py --cfg_path ./cfg/train_cfg.yaml
```

`batch_size` 是每个进程的批大小，总批大小为它乘以进程数。每个进程分到的图片数量应至少达到一个批次。显存不足时减小 `batch_size`，例如 1 或 2；也可设置 `microbatch: 1`，将一个批次拆成多次前向和反向计算。RTX 2060 6 GB 建议从小批次试运行，默认批大小不保证适用。

模型、优化器、EMA 权重和训练日志保存到 `model_save_dir`：

```text
model420000.pt
ema_0.9999_420000.pt
opt420000.pt
log.txt
progress.csv
```

实际文件编号以保存时的步数为准。中断后可把 `resume_checkpoint` 设为已有的 `modelXXXXXX.pt` 继续训练；同目录中匹配步数的优化器和 EMA 文件也会被恢复。各进程分别记录本地指标，多进程日志文件带 rank 后缀。

## 4. 微调

条件训练完成后，复制训练配置为 `cfg/finetune_cfg.yaml`，保留相同的字符集、网络结构、数据集和笔画设置，修改：

```yaml
classifier_free: true
resume_checkpoint: './trained_models/model420000.pt'
total_train_step: 800000
```

启动微调：

```powershell
python train.py --cfg_path ./cfg/finetune_cfg.yaml
```

多 GPU 时使用上一节的 `launch.py` 命令，并将配置路径换成 `cfg/finetune_cfg.yaml`。

`total_train_step` 是包含条件训练在内的累计目标步数。例如从第 420000 步恢复，到 800000 步结束，约再训练 380000 步。此阶段会随机屏蔽字符和笔画条件，支持测试时的 classifier-free guidance。必须提供条件训练的完整模型检查点，推荐使用 `modelXXXXXX.pt`，以便正确恢复步数和优化器状态。

## 5. 测试与生成

1. 准备一张参考字形图片，建议为清晰、居中的白底黑字，预处理时会缩放到配置中的 `image_size`。
2. 将待生成字符写入 UTF-8 编码的 `gen_char.txt`。按文件中的字符顺序生成，空格和换行会被忽略。
3. 编辑 `cfg/test_cfg.yaml`：

```yaml
chara_nums: 6625
image_size: 80
batch_size: 5
num_samples: 10
stroke_path: './chinese_stroke.txt'
model_path: './trained_models/ema_0.9999_800000.pt'
sty_img_path: './reference.png'
total_txt_file: './total_chn.txt'
gen_txt_file: './gen_char.txt'
img_save_path: './result'
classifier_free: true
cont_scale: 3.0
sk_scale: 3.0
use_ddim: true
timestep_respacing: ddim25
```

保留测试配置中其余网络参数，并确保它们与训练时一致。`model_path` 必须指向实际存在的完整模型权重，推荐使用微调后的 EMA 文件。`num_samples` 表示生成字符数量，必须为正数且不能超过 `gen_char.txt` 的字符数；例如文件只有 3 个字符，就将其设为 3。最后一批不足 `batch_size` 时会自动补齐计算并丢弃多余结果。

`cont_scale` 控制字符内容引导强度，`sk_scale` 控制笔画引导强度。测试条件训练模型时，将 `classifier_free` 设为 `false`，此时这两个引导参数不参与计算。

运行生成：

```powershell
python sample.py --cfg_path ./cfg/test_cfg.yaml
```

多 GPU 生成可用相同的 `launch.py` 启动格式，将入口换为 `sample.py`。多个进程分配不同字符，最终由 0 号进程按输入顺序保存结果。所有进程都需要访问相同的数据和检查点路径。

结果保存在 `img_save_path` 中，文件名为 `00000.png`、`00001.png` 等，对应待生成文件的前 `num_samples` 个字符。结果是字形图片；转换成可安装字体需要另行进行轮廓提取和字体封装。
