import cv2, os, sys, argparse, audio
import subprocess, random, string
from tqdm import tqdm
import torch, face_detection
from models.model_hyperlips import HyperLipsBase,HyperLipsHR
from GFPGAN import *
from face_parsing import init_parser, swap_regions_img
import shutil

#创建ArgumentParser对象，ArgumentParser 用于处理命令行参数
parser = argparse.ArgumentParser(description='Inference code to lip-sync videos in the wild using HyperLipsBase or HyperLipsHR models')
#HyperLipsBase和HyperLipsHR的权重文件路径
parser.add_argument('--checkpoint_path_BASE', type=str,help='Name of saved HyperLipsBase checkpoint to load weights from', default="checkpoints/hyperlipsbase_mead.pth")
parser.add_argument('--checkpoint_path_HR', type=str,help='Name of saved HyperLipsHR checkpoint to load weights from', default="checkpoints/hyperlipshr_mead_128.pth")
#选择哪个模型
parser.add_argument('--modelname', type=str,
                    help='Choosing HyperLipsBase or HyperLipsHR', default="HyperLipsBase")
#测试集的数据
parser.add_argument('--face', type=str,
                    help='Filepath of video/image that contains faces to use', default="test/video/M003-002.mp4")
parser.add_argument('--audio', type=str,
                    help='Filepath of video/audio file to use as raw audio source', default="test/audio/M003-002.mp4")
#输出文件路径
parser.add_argument('--outfile', type=str, help='Video path to save result. See default for an e.g.',
                    default='results/result_voice.mp4')

#处理时的参数设置
parser.add_argument('--pads', nargs='+', type=int, default=[0, 10, 0, 0],
                    help='Padding (top, bottom, left, right). Please adjust to include chin at least')    #nargs='+' 表示这个参数可以接受一个或多个值，这些值会被解析成一个列表
parser.add_argument('--filter_window', default=2, type=int,
                    help='real window is 2*T+1')
parser.add_argument('--face_det_batch_size', type=int,
                    help='Batch size for face detection', default=8)
parser.add_argument('--hyper_batch_size', type=int, help='Batch size for hyperlips model(s)', default=128)

parser.add_argument('--resize_factor', default=1, type=int,
                    help='Reduce the resolution by this factor. Sometimes, best results are obtained at 480p or 720p')

parser.add_argument('--segmentation_path', type=str,
					help='Name of saved checkpoint of segmentation network', default="checkpoints/face_segmentation.pth")
#设置face enhancement
parser.add_argument('--face_enhancement_path', type=str,
					help='Name of saved checkpoint of segmentation network', default="checkpoints/GFPGANv1.3.pth")
parser.add_argument('--no_faceenhance', default=True, action='store_true',
					help='Prevent using face enhancement')
#使用gpu
parser.add_argument('--gpu_id', type=float, help='gpu id (default: 0)',
                    default=0, required=False)
#解析命令行传来的参数并存储到args中
args = parser.parse_args()
#添加新属性
args.img_size = 128

#平滑 Mel 频谱的各个块
def get_smoothened_mels(mel_chunks, T):
    #mel_chunks：Mel 频谱的块数组
    #T：用于平滑的阈值
    for i in range(len(mel_chunks)):
        if i > T-1 and i<len(mel_chunks)-T:   #对于数组中不在开始或结束位置的块（由 T 确定），它会考虑当前块及其前后共 2*T 个块的平均值来平滑当前块
            window = mel_chunks[i-T: i + T]
            mel_chunks[i] = np.mean(window, axis=0)
        else:  #对于位于数组起始和结束位置的块，保持原样
            mel_chunks[i] = mel_chunks[i]

    return mel_chunks #返回经过平滑处理的 Mel 频谱块数组


#检测图像中的人脸
def face_detect(images, detector,pad):
    #images：包含待检测人脸的图像列表
    #detector：用于检测人脸的检测器
    #pad：用于在人脸周围添加的填充值，格式为 [上, 下, 左, 右]
    batch_size = 16
    if len(images) > 1:
        print('error')
        raise RuntimeError('leng(imgaes')
    while 1:
        predictions = []    #矩形坐标：最常见的形式是矩形坐标，表示检测到的人脸区域。这通常是四个值的元组或列表，如 (x1, y1, x2, y2)，其中 x1, y1 是矩形左上角的坐标，x2, y2 是矩形右下角的坐标。
        try:
            for i in range(0, len(images), batch_size):
                predictions.extend(
                    detector.get_detections_for_batch(np.array(images[i:i + batch_size])))  
        except RuntimeError as e:
            print(e)
            #如果捕捉到 `RuntimeError`（通常是因为图像太大导致的内存不足），它首先打印出错误信息。
            #- 如果当前批处理大小已经是1，它会重新抛出一个错误，提示图像太大，无法在 GPU 上运行人脸检测。
            #- 如果批处理大小大于1，它会将批处理大小减半，并输出新的批处理大小，然后继续循环尝试。
            if batch_size == 1:
                raise RuntimeError(
                    'Image too big to run face detection on GPU. Please use the --resize_factor argument')
            batch_size //= 2
            print('Recovering from OOM error; New batch size: {}'.format(batch_size))
            continue
        break

    #这部分代码处理检测到的人脸：
    #主要目的是根据给定的填充值调整人脸的边界框（BoundingBox）
    results = []
    pady1, pady2, padx1, padx2 = pad  # [0, 10, 0, 0]
    for rect, image in zip(predictions, images):
        if rect is None:
            raise ValueError('Face not detected! Ensure the video contains a face in all the frames.')

        y1 = max(0, rect[1] - pady1)
        y2 = min(image.shape[0], rect[3] + pady2)
        x1 = max(0, rect[0] - padx1)
        x2 = min(image.shape[1], rect[2] + padx2)

        results.append([x1, y1, x2, y2])

    boxes = np.array(results)
    results = [[image[y1: y2, x1:x2], (y1, y2, x1, x2)] for image, (x1, y1, x2, y2) in zip(images, boxes)]

    del detector
    return results

def datagen(mels, detector,face_path, resize_factor):
    #这是一个生成器函数，用于准备深度学习模型的输入数据。

    # 参数： Mel 频谱 (mels)、人脸检测器 (detector)、人脸视频路径 (face_path) 、缩放因子 (resize_factor)。
    # 处理流程：
    # 遍历 Mel 频谱的每个部分。
    # 从视频中读取对应的帧。
    # 使用 face_detect 函数对每帧图像进行人脸检测，并调整检测到的人脸大小以符合模型的输入尺寸 (img_size)。
    # 将处理后的人脸图像和相应的 Mel 频谱块存储起来，准备作为模型的输入。
    # 当累积的图像达到一个批处理大小 (hyper_batch_size) 时，将它们转换为 NumPy 数组，并通过 yield 返回。
    img_batch, mel_batch, frame_batch, coords_batch = [], [], [], []
    bbox_face, frame_to_det_list, rects, frame_to_det_batch = [], [], [], []
    img_size = 128
    hyper_batch_size = args.hyper_batch_size
    reader = read_frames(face_path, resize_factor)
    for i, m in enumerate(mels):
        try:
            frame_to_save = next(reader)
        except StopIteration:
            reader = read_frames(face_path, resize_factor)
            frame_to_save = next(reader)
        h, w, _ = frame_to_save.shape

        face, coords = face_detect([frame_to_save], detector,args.pads)[0]
        #调整大小确保了无论输入视频的分辨率或人脸在帧中的大小如何，模型都会接收到标准尺寸的人脸图像。
        face = cv2.resize(face, (img_size, img_size))

        img_batch.append(face)
        mel_batch.append(m)
        frame_batch.append(frame_to_save)
        coords_batch.append(coords)

        #达到预设批次就进行
        if len(img_batch) >= hyper_batch_size:
            img_batch, mel_batch = np.asarray(img_batch), np.asarray(mel_batch)

            #这行代码将每个图像的右半部分像素值设置为 0，从而在图像的右半部分创建一个黑色区域
            img_masked = img_batch.copy()
            img_masked[:, img_size // 2:] = 0

            #将掩码后的图像和原始图像沿着第四维（即颜色通道）拼接起来  / 255用于归一化颜色
            img_batch = np.concatenate((img_masked, img_batch), axis=3) / 255.
            # 将 Mel 频谱数据重塑为适合模型输入的形状
            mel_batch = np.reshape(mel_batch, [len(mel_batch), mel_batch.shape[1], mel_batch.shape[2], 1])
            
            yield img_batch, mel_batch, frame_batch, coords_batch
            #返回处理后的一批图像数据 (img_batch)、Mel 频谱数据 (mel_batch)、原始视频帧 (frame_batch) 和对应的坐标 (coords_batch)
            img_batch, mel_batch, frame_batch, coords_batch = [], [], [], []

    #处理最后一批未完成批处理大小的图像和 Mel 频谱数据
    if len(img_batch) > 0:
        img_batch, mel_batch = np.asarray(img_batch), np.asarray(mel_batch)

        img_masked = img_batch.copy()
        img_masked[:, img_size // 2:] = 0

        img_batch = np.concatenate((img_masked, img_batch), axis=3) / 255.
        mel_batch = np.reshape(mel_batch, [len(mel_batch), mel_batch.shape[1], mel_batch.shape[2], 1])

        yield img_batch, mel_batch, frame_batch, coords_batch

#载模型参数，并将其映射到指定的设备上，然后返回加载的模型参数
def _load(checkpoint_path, device):
    checkpoint = torch.load(checkpoint_path, map_location=device)
    return checkpoint

#这两个函数用于加载 HyperLips 模型
#根据指定的路径和设备，分别加载 HyperLipsHR 和 HyperLipsBase 模型，并将模型放置到指定的 GPU 上
def load_HyperLipsHR(path,path_hr,device):
    model = HyperLipsHR(window_T =args.filter_window ,rescaling=1,base_model_checkpoint=path,HRDecoder_model_checkpoint =path_hr)
    model = model.to(device)
    print("HyperLipsHR model loaded")
    return model.eval()

def load_HyperLipsBase(path, device):
    model = HyperLipsBase()
    checkpoint = _load(path, device)
    s = checkpoint["state_dict"]
    model.load_state_dict(s)
    model = model.to(device)
    print("HyperLipsBase model loaded")
    return model.eval()

def read_frames(face_path, resize_factor):
    #这个函数用于从视频文件中连续读取帧。
    # 使用 OpenCV 打开视频文件，并逐帧读取。
    # 如果指定了 resize_factor，则对帧进行缩放。

    #创建一个逐帧读取视频的对象，VideoCapture是类
    video_stream = cv2.VideoCapture(face_path)

    print('Reading video frames from start...')
    read_frames_index = 0
    while 1:
        still_reading, frame = video_stream.read()  #still_reading布尔值代表是否捕获帧   frame是捕获的帧
        if not still_reading:
            video_stream.release()
            break
        if resize_factor > 1:   #除以resize_factor进行缩放
            frame = cv2.resize(frame, (frame.shape[1] // resize_factor, frame.shape[0] // resize_factor))
        yield frame   #暂停逐帧返回

def main():
    Hyperlips_executor = Hyperlips()  #创建对象
    Hyperlips_executor._HyperlipsLoadModels()
    Hyperlips_executor._HyperlipsInference()

class Hyperlips():
    def __init__(self):
        ## 初始化时，从命令行参数获取模型检查点路径、分割网络路径和批处理大小
        self.checkpoint_path_BASE = args.checkpoint_path_BASE
        self.checkpoint_path_HR = args.checkpoint_path_HR
        self.parser_path = args.segmentation_path
        self.batch_size = args.hyper_batch_size #128
        self.mel_step_size = 16

    # 加载模型并进行初始化
    def _HyperlipsLoadModels(self):
        # 检查GPU可用性和GPU ID
        gpu_id = args.gpu_id
        if not torch.cuda.is_available() or (gpu_id > (torch.cuda.device_count() - 1)):
            raise ValueError('Existing gpu configuration problem.')
        self.device = torch.device(f'cuda:{gpu_id}')  # 设置使用的设备
        print('Using {} for inference.'.format(self.device))

        # 初始化面部增强模型
        self.restorer = GFPGANInit(self.device, args.face_enhancement_path)

        # 根据命令行参数加载相应的模型
        if args.modelname == "HyperLipsBase":
            self.model = load_HyperLipsBase(self.checkpoint_path_BASE, self.device)
        elif args.modelname == "HyperLipsHR":
            self.model = load_HyperLipsHR(self.checkpoint_path_BASE, self.checkpoint_path_HR, self.device)

        # 初始化面部分割网络
        self.seg_net = init_parser(self.parser_path, self.device)
        print('Models initialization successful.')

    def _HyperlipsInference(self):
        # 主推理过程
        face = args.face  # 输入视频路径
        audiopath = args.audio  # 输入音频路径
        print("The input video path is {}, The output audio path is {}".format(face, audiopath))

        outfile = args.outfile
        outfile = os.path.abspath(outfile)
        rest_root_path = os.path.dirname(os.path.realpath(outfile))
        temp_save_path = outfile.rsplit('.', 1)[0]

        # 创建必要的输出目录
        if not os.path.exists(rest_root_path):
            os.mkdir(rest_root_path)
        if not os.path.exists(temp_save_path):
            os.mkdir(temp_save_path)

        # 初始化人脸检测器
        detector = face_detection.FaceAlignment(face_detection.LandmarksType._2D, flip_input=False,
                                                device='cuda:{}'.format(args.gpu_id))

        # 检查输入视频文件是否存在
        if not os.path.isfile(face):
            raise ValueError('--face argument must be a valid path to video/image file')

        # 获取视频的基本信息，如帧率和分辨率
        video_stream = cv2.VideoCapture(face)
        fps = video_stream.get(cv2.CAP_PROP_FPS)
        frame_width = int(video_stream.get(cv2.CAP_PROP_FRAME_WIDTH))
        frame_height = int(video_stream.get(cv2.CAP_PROP_FRAME_HEIGHT))
        video_stream.release()

        # 设置视频写入器，准备输出视频文件
        out = cv2.VideoWriter(os.path.join(temp_save_path, 'result.avi'), cv2.VideoWriter_fourcc(*'DIVX'), fps,
                              (frame_width, frame_height))

        # 处理音频文件，确保是 WAV 格式
        if not audiopath.endswith('.wav'):
            print('Extracting raw audio...')
            command = 'ffmpeg -y -i {} -strict -2 {}'.format(audiopath, os.path.join(temp_save_path, 'temp.wav'))
            subprocess.call(command, shell=True)
            audiopath = os.path.join(temp_save_path, 'temp.wav')

        # 加载音频数据，转换为 Mel 频谱
        wav = audio.load_wav(audiopath, 16000)
        mel = audio.melspectrogram(wav)

        if np.isnan(mel.reshape(-1)).sum() > 0:
            raise ValueError(
                'Mel contains nan! Using a TTS voice? Add a small epsilon noise to the wav file and try again')
            # 将 Mel 频谱切割为小块，准备用于模型输入
            mel_chunks = []
            mel_idx_multiplier = 80. / fps  # 计算 Mel 频谱与视频帧的对应关系
            i = 0
            while 1:
                start_idx = int(i * mel_idx_multiplier)
                if start_idx + self.mel_step_size > len(mel[0]):
                    mel_chunks.append(mel[:, len(mel[0]) - self.mel_step_size:])
                    break
                mel_chunks.append(mel[:, start_idx: start_idx + self.mel_step_size])
                i += 1

            # 平滑 Mel 频谱，如果需要
            if args.filter_window is not None:
                mel_chunks = get_smoothened_mels(mel_chunks, T=args.filter_window)
            print("Length of mel chunks: {}".format(len(mel_chunks)))


        # 转换图像和 Mel 频谱批次为张量，送入 GPU
        gen = datagen(mel_chunks, detector, face, args.resize_factor)
        for i, (img_batch, mel_batch, frames, coords) in enumerate(tqdm(gen,
                                                                        total=int(
                                                                            np.ceil(
                                                                                float(len(mel_chunks))/ self.batch_size)))):

            img_batch = torch.FloatTensor(np.transpose(img_batch, (0, 3, 1, 2))).to(self.device)#([122, 6, 96, 96])
            mel_batch = torch.FloatTensor(np.transpose(mel_batch, (0, 3, 1, 2))).to(self.device)

            # 使用模型进行推理，不计算梯度以节省计算资源
            with torch.no_grad():
                pred = self.model(mel_batch, img_batch)  # 使用模型对图像和 Mel 频谱进行推理

            # 处理模型的输出，生成最终的视频帧
            for p, f, c in zip(pred, frames, coords):
                y1, y2, x1, x2 = c  # 获取人脸区域坐标
                mask_temp = np.zeros_like(f)  # 创建与视频帧大小相同的遮罩
                p = p.cpu().numpy().transpose(1, 2, 0) * 255.  # 将模型输出转换为图像格式

                # 根据配置决定是否进行面部增强
                if not args.no_faceenhance:
                    ori_f = f.copy()
                    p = cv2.resize(p, (x2 - x1, y2 - y1)).astype(np.uint8)
                    f[y1:y2, x1:x2] = p
                    Code_img = GFPGANInfer(f, self.restorer, aligned=False)  # 使用 GFPGAN 进行面部增强
                    p, mask_out = swap_regions_img(ori_f[y1:y2, x1:x2], Code_img[y1:y2, x1:x2],
                                                   self.seg_net)  # 将增强后的人脸和原始视频帧合并
                else:
                    p, mask_out = swap_regions_img(f[y1:y2, x1:x2], p, self.seg_net)  # 不增强时直接合并

                # 应用遮罩和混合技术，使合成看起来更自然

                mask_temp[y1:y2, x1:x2] = mask_out
                kernel = np.ones((5,5),np.uint8)  
                mask_temp = cv2.erode(mask_temp,kernel,iterations = 1)
                mask_temp = cv2.GaussianBlur(mask_temp, (75, 75), 0,0,cv2.BORDER_DEFAULT) 
                f_background = f.copy()
                f[y1:y2, x1:x2] = p
                f = f_background*(1-mask_temp/255.0)+f*(mask_temp/255.0)
                f = f.astype(np.uint8)
                out.write(f)

        out.release()
        outfile_dfl = os.path.join(rest_root_path, args.outfile.split('/')[-1]) 
        command = 'ffmpeg -y -i {} -i {} -strict -2 -q:v 1 {}'.format(
            audiopath, os.path.join(temp_save_path, 'result.avi'), outfile_dfl)
        subprocess.call(command, shell=True)
        if os.path.exists(temp_save_path):
            shutil.rmtree(temp_save_path)
        #释放 GPU 内存
        torch.cuda.empty_cache()

if __name__ == '__main__':
    main()
