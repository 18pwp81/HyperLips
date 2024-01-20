from os.path import dirname, join, basename, isfile
from tqdm import tqdm
from models import SyncNet_color as SyncNet
from models.model_hyperlips import HyperLipsBase, HyperCtrolDiscriminator
import audio

import torch
from torch import nn
from torch.nn import functional as F
from torch import optim
import torch.backends.cudnn as cudnn
from torch.utils import data as data_utils
import numpy as np
from glob import glob
import os, random, cv2, argparse

# 设置了 CUDA 的环境变量，指定使用哪个 GPU
os.environ["CUDA_VISIBLE_DEVICES"] = '0'
from hparams_Base import hparams, get_image_list

# 使用 argparse 解析命令行参数
parser = argparse.ArgumentParser(description='Code to train the Hyperbase model WITH the visual quality discriminator')
parser.add_argument("--data_root", help="Root folder of the preprocessed LRS2 dataset", default='Train_data/imgs')
parser.add_argument('--checkpoint_dir', help='Save checkpoints to this directory', default="checkpoints_hyperlips_base", type=str)
parser.add_argument('--syncnet_checkpoint_path', help='Load the pre-trained audio-visual sync module', default="checkpoints/pretrain_sync_expert.pth", type=str)
parser.add_argument('--checkpoint_path', help='Resume generator from this checkpoint', default=None, type=str)
parser.add_argument('--disc_checkpoint_path', help='Resume quality disc from this checkpoint', default=None, type=str)
args = parser.parse_args()


global_step = 0
global_epoch = 0
use_cuda = torch.cuda.is_available()
print('use_cuda: {}'.format(use_cuda))

syncnet_T = 5
syncnet_mel_step_size = 16

# 一个用于加载和处理数据的数据集类
class Dataset(object):
    def __init__(self, split):
        # 初始化函数，接收一个split参数用于区分数据集的不同部分（如训练集、验证集等）
        self.all_videos = get_image_list(args.data_root, split)  # 从指定的根目录和数据集部分获取所有视频的列表

    def get_frame_id(self, frame):
        # 从文件名中提取帧的ID
        return int(basename(frame).split('.')[0])  # 提取文件名的基础部分，然后分割并转换为整数

    def get_window(self, start_frame):
        # 获取一段时间窗口内的帧的文件路径
        start_id = self.get_frame_id(start_frame)  # 获取起始帧的ID
        vidname = dirname(start_frame)  # 获取视频文件的目录名

        window_fnames = []
        for frame_id in range(start_id, start_id + syncnet_T):
            frame = join(vidname, '{}.jpg'.format(frame_id))  # 拼接路径和文件名
            if not isfile(frame):
                return None  # 如果文件不存在，则返回None
            window_fnames.append(frame)  # 将文件名添加到列表
        return window_fnames  # 返回窗口内的所有文件名

    def read_window(self, window_fnames):
        # 读取和处理get_window函数返回的帧路径列表中的所有图像
        if window_fnames is None: return None
        window = []
        for fname in window_fnames:
            img = cv2.imread(fname)  # 读取图像
            if img is None:
                return None  # 如果图像不存在，则返回None
            try:
                img = cv2.resize(img, (hparams.img_size, hparams.img_size))  # 调整图像大小
            except Exception as e:
                return None

            window.append(img)  # 将图像添加到列表

        return window  # 返回包含所有图像的列表

    def crop_audio_window(self, spec, start_frame):
        # 裁剪单个音频窗口
        if type(start_frame) == int:
            start_frame_num = start_frame
        else:
            start_frame_num = self.get_frame_id(start_frame)  # 获取起始帧的编号
        start_idx = int(80. * (start_frame_num / float(hparams.fps)))  # 计算起始索引

        end_idx = start_idx + syncnet_mel_step_size  # 计算结束索引

        return spec[start_idx: end_idx, :]  # 返回音频频谱的一部分

    def get_segmented_mels(self, spec, start_frame):
        # 获取分段的梅尔频谱：从音频频谱中获取一系列连续的音频窗口，并将它们组合成一个数组
        mels = []
        assert syncnet_T == 5 # 断言，确保syncnet_T等于5
        start_frame_num = self.get_frame_id(start_frame) + 1 # 0-indexing ---> 1-indexing
        if start_frame_num - 2 < 0: return None
        for i in range(start_frame_num, start_frame_num + syncnet_T):
            m = self.crop_audio_window(spec, i - 2)
            if m.shape[0] != syncnet_mel_step_size:
                return None
            mels.append(m.T)

        mels = np.asarray(mels) # 将列表转换为NumPy数组

        return mels # 返回包含所有音频片段的数组

    def prepare_window(self, window):
        # 准备时间窗口，用于模型输入
        x = np.asarray(window) / 255.  # 将图像数据转换为浮点数并归一化
        x = np.transpose(x, (3, 0, 1, 2))  # 调整数组维度以符合模型输入要求

        return x  # 返回准备好的数据

    def __len__(self):
        # 返回数据集中视频的数量
        return len(self.all_videos)

    def __getitem__(self, idx):
        # 获取数据集中的单个项
        while 1:
            idx = random.randint(0, len(self.all_videos) - 1)  # 随机选择一个视频
            vidname = self.all_videos[idx]  # 获取视频名称
            img_names = list(glob(join(vidname, '*.jpg')))  # 获取视频中所有图像的名称
            if len(img_names) <= 3 * syncnet_T:
                continue  # 如果图像数量不足，则继续选择下一个视频

            img_name = random.choice(img_names)  # 随机选择一个正确的图像
            wrong_img_name = random.choice(img_names)  # 随机选择一个错误的图像
            while wrong_img_name == img_name:
                wrong_img_name = random.choice(img_names)  # 确保错误图像和正确图像不同

            window_fnames = self.get_window(img_name)  # 获取正确图像的时间窗口
            wrong_window_fnames = self.get_window(wrong_img_name)  # 获取错误图像的时间窗口
            if window_fnames is None or wrong_window_fnames is None:
                continue  # 如果任一窗口为空，则继续选择

            window = self.read_window(window_fnames) # 读取正确窗口的图像
            if window is None:
                continue # 如果读取失败，则继续选择

            wrong_window = self.read_window(wrong_window_fnames)  # 读取错误窗口的图像
            if wrong_window is None:
                continue  # 如果读取失败，则继续选择

            try:
                wavpath = join(vidname, "audio.wav")  # 构建音频文件的路径
                wav = audio.load_wav(wavpath, hparams.sample_rate)  # 加载音频文件

                orig_mel = audio.melspectrogram(wav).T  # 将音频文件转换为梅尔频谱
            except Exception as e:
                continue  # 如果处理音频失败，则继续选择

            mel = self.crop_audio_window(orig_mel.copy(), img_name)  # 裁剪音频窗口

            if (mel.shape[0] != syncnet_mel_step_size):
                continue  # 如果裁剪后的音频长度不符合预期，则继续选择

            indiv_mels = self.get_segmented_mels(orig_mel.copy(), img_name)  # 获取分段的梅尔频谱
            if indiv_mels is None: continue  # 如果获取失败，则继续选择

            window = self.prepare_window(window)  # 准备正确的时间窗口
            y = window.copy()  # 复制一份时间窗口作为标签
            window[:, :, window.shape[2] // 2:] = 0.  # 修改一半的时间窗口用于输入

            wrong_window = self.prepare_window(wrong_window)  # 准备错误的时间窗口
            x = np.concatenate([window, wrong_window], axis=0)  # 合并正确和错误的时间窗口

            # 模型的输入x：由处理后的正确和错误图像窗口组成，为模型提供了一个同时包含正面和负面例子的综合数据样本
            # 分段的梅尔频谱indiv_mels：原始音频的梅尔频谱中提取的一系列分段，与正确的图像窗口相对应
            # 原始的梅尔频谱mel：与正确图像窗口直接相关的音频窗口的梅尔频谱，没有进行时间上的切割或分段
            # 标签y：正确图像窗口的一个复制
            x = torch.FloatTensor(x)  # 转换为PyTorch张量
            mel = torch.FloatTensor(mel.T).unsqueeze(0)  # 转换为PyTorch张量并调整维度
            indiv_mels = torch.FloatTensor(indiv_mels).unsqueeze(1)  # 转换为PyTorch张量并调整维度
            y = torch.FloatTensor(y)  # 转换为PyTorch张量
            return x, indiv_mels, mel, y  # 返回输入数据、分段的梅尔频谱、原始的梅尔频谱和标签

# 在训练期间定期保存图像以进行检查。这有助于可视化训练进度。
def save_sample_images(x, g, gt, global_step, checkpoint_dir):
    # 定义一个函数来保存样本图像，接收输入图像x，生成图像g，目标图像gt，全局步数和检查点目录作为参数

    # 将输入图像x从PyTorch张量转换为NumPy数组，并进行必要的转置和缩放，以便保存为图像文件
    x = (x.detach().cpu().numpy().transpose(0, 2, 3, 4, 1) * 255.).astype(np.uint8)

    # 将生成图像g进行同样的处理，以便保存
    g = (g.detach().cpu().numpy().transpose(0, 2, 3, 4, 1) * 255.).astype(np.uint8)

    # 将目标图像gt进行同样的处理，以便保存
    gt = (gt.detach().cpu().numpy().transpose(0, 2, 3, 4, 1) * 255.).astype(np.uint8)

    # 分离输入图像x的不同部分，可能是为了区分不同的通道或视角
    refs, inps = x[..., 3:], x[..., :3]

    # 创建一个文件夹名称，包含检查点目录和全局步数，用于存储图像
    folder = join(checkpoint_dir, "samples_step{:09d}".format(global_step))

    # 如果该文件夹不存在，则创建它
    if not os.path.exists(folder): os.mkdir(folder)

    # 将参考图像、输入图像、生成图像和目标图像合并成一个大的数组，以便创建一个拼贴图
    collage = np.concatenate((refs, inps, g, gt), axis=-2)

    # 遍历拼贴图中的每个元素
    for batch_idx, c in enumerate(collage):
        # 遍历元素中的每个时间步长
        for t in range(len(c)):
            # 将图像保存到指定的文件夹中，文件名包含批次索引和时间步长
            cv2.imwrite('{}/{}_{}.jpg'.format(folder, batch_idx, t), c[t])

# 定义一个二元交叉熵损失（Binary Cross-Entropy Loss），用于模型训练中的损失计算
logloss = nn.BCELoss()

# 损失函数,测量预测值和实际值之间的差异
# 定义余弦损失函数，用于计算两个向量之间的余弦相似度损失
def cosine_loss(a, v, y):
    # 计算向量a和v之间的余弦相似度
    d = nn.functional.cosine_similarity(a, v)
    # 使用二元交叉熵损失计算损失值
    loss = logloss(d.unsqueeze(1), y)

    # 返回计算得到的损失值
    return loss

# 设置设备为GPU（如果可用），否则使用CPU
device = torch.device("cuda" if use_cuda else "cpu")

# 初始化SyncNet模型，并将其转移到指定的设备（GPU或CPU）
syncnet = SyncNet().to(device)

# 将SyncNet模型的所有参数的梯度计算设为False，即固定参数，不进行梯度更新
for p in syncnet.parameters():
    p.requires_grad = False

# 初始化L1损失函数，用于测量预测值和实际值之间的差异
recon_loss = nn.L1Loss()

# 定义获取同步损失的函数
def get_sync_loss(mel, g):
    # 对生成的图像g进行裁剪和拼接操作
    g = g[:, :, :, g.size(3)//2:]
    g = torch.cat([g[:, :, i] for i in range(syncnet_T)], dim=1)
    # 将处理后的图像g调整大小以适配SyncNet模型的输入尺寸
    g = torch.nn.functional.interpolate(g, (64, 128), mode='bilinear', align_corners=False)

    # 通过SyncNet模型计算音频特征和视频特征
    a, v = syncnet(mel, g)
    # 创建一个元素全为1的张量，用于计算损失
    y = torch.ones(g.size(0), 1).float().to(device)
    # 调用cosine_loss函数计算并返回同步损失
    return cosine_loss(a, v, y)


# 主要的训练循环。它批量处理数据，计算损失，更新模型参数，保存检查点，并记录进度。
def train(device, model, disc, train_data_loader, test_data_loader, optimizer, disc_optimizer,
          checkpoint_dir=None, checkpoint_interval=None, nepochs=None):
    # 使用全局变量跟踪训练步骤和周期
    global global_step, global_epoch
    resumed_step = global_step  # 记录训练恢复的起始步骤

    # 训练周期循环
    while global_epoch < nepochs:
        # 打印当前周期
        print('Starting Epoch: {}'.format(global_epoch))
        # 初始化各种损失统计
        running_sync_loss, running_l1_loss, disc_loss, running_perceptual_loss = 0., 0., 0., 0.
        running_disc_real_loss, running_disc_fake_loss = 0., 0.
        # 初始化进度条
        prog_bar = tqdm(enumerate(train_data_loader))

        # 遍历训练数据
        for step, (x, indiv_mels, mel, gt) in prog_bar:
            # 设置模型为训练模式
            disc.train()
            model.train()

            # 将数据移动到指定设备
            x = x.to(device)
            mel = mel.to(device)
            indiv_mels = indiv_mels.to(device)
            gt = gt.to(device)

            ### 训练生成器。首先清除之前的梯度。
            optimizer.zero_grad()
            disc_optimizer.zero_grad()

            # 对输入数据进行一次前向传播计算
            g = model(indiv_mels, x)

            # 根据配置计算同步损失和感知损失
            if hparams.syncnet_wt > 0.:
                sync_loss = get_sync_loss(mel, g)
            else:
                sync_loss = 0.

            if hparams.disc_wt > 0.:
                perceptual_loss = disc.perceptual_forward(g)
            else:
                perceptual_loss = 0.

            # 计算重建损失
            l1loss = recon_loss(g, gt)

            # 总损失计算
            loss = hparams.syncnet_wt * sync_loss + hparams.disc_wt * perceptual_loss + \
                                    (1. - hparams.syncnet_wt - hparams.disc_wt) * l1loss

            # 反向传播
            loss.backward()  # 计算损失函数的梯度

            # 优化器步骤
            optimizer.step()  # 更新生成器的参数

            ### Remove all gradients before Training disc
            disc_optimizer.zero_grad()

            # 判别器的真实损失
            pred = disc(gt)#([2, 3, 5, 512, 512])->([10, 1])
            disc_real_loss = F.binary_cross_entropy(pred, torch.ones((len(pred), 1)).to(device))
            disc_real_loss.backward()

            # 判别器的假损失
            pred = disc(g.detach())#([2, 3, 5, 512, 512])->([10, 1])
            disc_fake_loss = F.binary_cross_entropy(pred, torch.zeros((len(pred), 1)).to(device))
            disc_fake_loss.backward()

            # 更新判别器的参数
            disc_optimizer.step()

            # 更新损失统计(判别器)
            running_disc_real_loss += disc_real_loss.item()
            running_disc_fake_loss += disc_fake_loss.item()

            # 每隔一定步数保存样本图像
            if global_step % checkpoint_interval == 0:
                save_sample_images(x, g, gt, global_step, checkpoint_dir)

            # Logs
            global_step += 1
            cur_session_steps = global_step - resumed_step

            # 更新损失统计(生成器相关损失：重建损失，同步损失，感知损失)
            running_l1_loss += l1loss.item()
            if hparams.syncnet_wt > 0.:
                running_sync_loss += sync_loss.item()
            else:
                running_sync_loss += 0.

            if hparams.disc_wt > 0.:
                running_perceptual_loss += perceptual_loss.item()
            else:
                running_perceptual_loss += 0.

            # 每隔一定步数保存检查点
            if global_step == 1 or global_step % checkpoint_interval == 0:
                save_checkpoint(
                    model, optimizer, global_step, checkpoint_dir, global_epoch)
                save_checkpoint(disc, disc_optimizer, global_step, checkpoint_dir, global_epoch, prefix='disc_')

            # eval_model(test_data_loader, global_step, device, model, disc)
            if global_step % hparams.eval_interval == 0:
                with torch.no_grad():
                    # 计算平均同步损失
                    average_sync_loss = eval_model(test_data_loader, global_step, device, model, disc)

                    # 如果同步损失小于阈值，调整权重
                    if average_sync_loss < .75:
                        hparams.set_hparam('syncnet_wt', 0.03)

            # 更新进度条的描述
            prog_bar.set_description('L1: {}, Sync: {}, Percep: {} | Fake: {}, Real: {}'.format(running_l1_loss / (step + 1),
                                                                                        running_sync_loss / (step + 1),
                                                                                        running_perceptual_loss / (step + 1),
                                                                                        running_disc_fake_loss / (step + 1),
                                                                                        running_disc_real_loss / (step + 1)))

        # 完成一个周期后增加周期数
        global_epoch += 1

# 在测试数据集上评估模型：计算和收集多种损失，打印出各种损失，返回平均同步损失
def eval_model(test_data_loader, global_step, device, model, disc):
    # 设置评估步数
    eval_steps = 300
    # 打印评估信息
    print('Evaluating for {} steps'.format(eval_steps))
    # 初始化各种损失统计列表
    running_sync_loss, running_l1_loss, running_disc_real_loss, running_disc_fake_loss, running_perceptual_loss = [], [], [], [], []

    # 无限循环，直到达到评估步数
    while 1:
        # 遍历测试数据
        for step, (x, indiv_mels, mel, gt) in enumerate((test_data_loader)):
            # 设置模型为评估模式
            model.eval()
            disc.eval()

            # 将数据移动到指定设备
            x = x.to(device)
            mel = mel.to(device)
            indiv_mels = indiv_mels.to(device)
            gt = gt.to(device)

            # 使用判别器预测真实图像的损失
            pred = disc(gt)
            disc_real_loss = F.binary_cross_entropy(pred, torch.ones((len(pred), 1)).to(device))

            # 使用模型生成图像，并计算假图像的损失
            g = model(indiv_mels, x)
            pred = disc(g)
            disc_fake_loss = F.binary_cross_entropy(pred, torch.zeros((len(pred), 1)).to(device))

            # 收集真实和假图像的损失
            running_disc_real_loss.append(disc_real_loss.item())
            running_disc_fake_loss.append(disc_fake_loss.item())

            # 计算同步损失
            sync_loss = get_sync_loss(mel, g)

            # 如果有感知损失权重，计算感知损失
            if hparams.disc_wt > 0.:
                perceptual_loss = disc.perceptual_forward(g)
            else:
                perceptual_loss = 0.

            # 计算重建损失
            l1loss = recon_loss(g, gt)

            # 计算总损失
            loss = hparams.syncnet_wt * sync_loss + hparams.disc_wt * perceptual_loss + \
                                    (1. - hparams.syncnet_wt - hparams.disc_wt) * l1loss

            # 收集重建损失和同步损失
            running_l1_loss.append(l1loss.item())
            running_sync_loss.append(sync_loss.item())

            # 收集感知损失
            if hparams.disc_wt > 0.:
                running_perceptual_loss.append(perceptual_loss.item())
            else:
                running_perceptual_loss.append(0.)

            # 如果评估步数达到设定的值，跳出循环
            if step > eval_steps: break

        print('L1: {}, Sync: {}, Percep: {} | Fake: {}, Real: {}'.format(sum(running_l1_loss) / len(running_l1_loss),
                                                            sum(running_sync_loss) / len(running_sync_loss),
                                                            sum(running_perceptual_loss) / len(running_perceptual_loss),
                                                            sum(running_disc_fake_loss) / len(running_disc_fake_loss),
                                                             sum(running_disc_real_loss) / len(running_disc_real_loss)))
        # 返回平均同步损失
        return sum(running_sync_loss) / len(running_sync_loss)


# 保存和加载训练检查点
def save_checkpoint(model, optimizer, step, checkpoint_dir, epoch, prefix=''):
    checkpoint_path = join(
        checkpoint_dir, "{}checkpoint_step{:09d}.pth".format(prefix, global_step))
    optimizer_state = optimizer.state_dict() if hparams.save_optimizer_state else None
    torch.save({
        "state_dict": model.state_dict(),
        "optimizer": optimizer_state,
        "global_step": step,
        "global_epoch": epoch,
    }, checkpoint_path)
    print("Saved checkpoint:", checkpoint_path)

def _load(checkpoint_path):
    if use_cuda:
        checkpoint = torch.load(checkpoint_path)
    else:
        checkpoint = torch.load(checkpoint_path,
                                map_location=lambda storage, loc: storage)
    return checkpoint


def load_checkpoint(path, model, optimizer, reset_optimizer=False, overwrite_global_states=True):
    global global_step
    global global_epoch

    print("Load checkpoint from: {}".format(path))
    checkpoint = _load(path)
    s = checkpoint["state_dict"]

    model.load_state_dict(s)

    if overwrite_global_states:
        global_step = checkpoint["global_step"]
        global_epoch = checkpoint["global_epoch"]

    return model




if __name__ == "__main__":
    # 设置和数据准备阶段

    # 设置检查点目录
    checkpoint_dir = args.checkpoint_dir

    # 数据集和数据加载器的设置
    # 初始化训练集
    train_dataset = Dataset('train')
    # 初始化验证集
    test_dataset = Dataset('val')

    # 创建训练数据加载器
    train_data_loader = data_utils.DataLoader(
        train_dataset, batch_size=hparams.batch_size, shuffle=True,
        num_workers=hparams.num_workers)

    # 创建测试数据加载器
    test_data_loader = data_utils.DataLoader(
        test_dataset, batch_size=hparams.batch_size,
        num_workers=4)

    # 环境和模型初始化阶段

    # 设置设备为GPU（如果可用），否则使用CPU
    device = torch.device("cuda" if use_cuda else "cpu")

    # 初始化模型
    # 创建HyperLipsBase模型实例
    model = HyperLipsBase()
    # 如果有多个GPU，使用数据并行
    if torch.cuda.device_count() > 1:
                print("Let's use", torch.cuda.device_count(), "GPUs!")
                model = nn.DataParallel(model)
    # 将模型移动到指定的设备
    model = model.to(device)

    # 创建HyperCtrolDiscriminator模型实例
    disc = HyperCtrolDiscriminator()
    # 如果有多个GPU，使用数据并行
    if torch.cuda.device_count() > 1:
                print("Let's use", torch.cuda.device_count(), "GPUs!")
                disc = nn.DataParallel(disc)
    # 将模型移动到指定的设备
    disc = disc.to(device)

    # 打印模型和判别器的可训练参数数量
    print('total trainable params {}'.format(sum(p.numel() for p in model.parameters() if p.requires_grad)))
    print('total DISC trainable params {}'.format(sum(p.numel() for p in disc.parameters() if p.requires_grad)))

    # 优化器初始化阶段

    # 初始化优化器
    optimizer = optim.Adam([p for p in model.parameters() if p.requires_grad],
                           lr=hparams.initial_learning_rate, betas=(0.5, 0.999))
    disc_optimizer = optim.Adam([p for p in disc.parameters() if p.requires_grad],
                           lr=hparams.disc_initial_learning_rate, betas=(0.5, 0.999))

    # 检查点加载阶段

    # 如果提供了模型检查点路径，则从检查点加载模型和优化器的状态
    if args.checkpoint_path is not None:
        load_checkpoint(args.checkpoint_path, model, optimizer, reset_optimizer=False)

    # 如果提供了判别器的检查点路径，则从检查点加载判别器和其优化器的状态
    if args.disc_checkpoint_path is not None:
        load_checkpoint(args.disc_checkpoint_path, disc, disc_optimizer,
                                reset_optimizer=False, overwrite_global_states=False)
    # 加载SyncNet模型的检查点
    load_checkpoint(args.syncnet_checkpoint_path, syncnet, None, reset_optimizer=True,
                                overwrite_global_states=False)
    # 如果检查点目录不存在，则创建该目录
    if not os.path.exists(checkpoint_dir):
        os.mkdir(checkpoint_dir)

    # 训练阶段

    # Train!
    train(device, model, disc, train_data_loader, test_data_loader, optimizer, disc_optimizer,
              checkpoint_dir=checkpoint_dir,
              checkpoint_interval=hparams.checkpoint_interval,
              nepochs=hparams.nepochs)
