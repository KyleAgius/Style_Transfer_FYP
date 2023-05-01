import argparse
import os
import torch
import torch.nn as nn
from PIL import Image
from os.path import basename
from os.path import splitext
from torchvision import transforms
from torchvision.utils import save_image
from function import calc_mean_std, normal, coral
from matplotlib import pyplot as plt
import net as net
import numpy as np
import cupy as cp
import cv2
import yaml
import numpy
import segmentation
import time
import statistics


#读取文件函数
def get_files(img_dir):
    files = os.listdir(img_dir)
    paths = []
    for x in files:
        paths.append(os.path.join(img_dir, x))
    # return [os.path.join(img_dir,x) for x in files]
    return paths

#加载图片
def load_images(args):
    assert (args.content or args.content_dir)
    assert (args.style or args.style_dir)
    if not args.content:
        content_paths = get_files(content_dir)
    else:
        content_paths = [args.content]
    if not args.style:
        style_paths = get_files(style_dir)
    else:
        style_paths = [args.style]
    return content_paths, style_paths

#加载模型参数
def load_weights(vgg, decoder, mcc_module):
    vgg.load_state_dict(torch.load(args.vgg_path))
    decoder.load_state_dict(torch.load(args.decoder_path))
    mcc_module.load_state_dict(torch.load(args.transform_path))

#图片预处理
def test_transform(size, crop):
    transform_list = []
    if size != 0: 
        transform_list.append(transforms.Resize(size))
    if crop:
        transform_list.append(transforms.CenterCrop(size))
    transform_list.append(transforms.ToTensor())
    transform = transforms.Compose(transform_list)
    return transform

#风格图预处理
def style_transform(h,w):
    k = (h,w)
    size = int(np.max(k))

    transform_list = []
    transform_list.append(transforms.Resize(size))
    transform_list.append(transforms.ToTensor())
    transform = transforms.Compose(transform_list)
    return transform

#内容图预处理
def content_transform():
    transform_list = []
    transform_list.append(transforms.ToTensor())
    transform = transforms.Compose(transform_list)
    return transform

#风格化函数
def style_transfer(vgg, decoder, sa_module, content, style, alpha=1.0, interpolation_weights=None):
    assert (0.0 <= alpha <= 1.0)

    style_fs, content_f, style_f=feat_extractor(vgg, content, style)
    Fccc = sa_module(content_f,content_f)

    if interpolation_weights:
        _, C, H, W = Fccc.size()
        feat = torch.FloatTensor(1, C, H, W).zero_().to(device)
        base_feat = sa_module(content_f, style_f)
        for i, w in enumerate(interpolation_weights):
            feat = feat + w * base_feat[i:i + 1]
        Fccc=Fccc[0:1]
    else:
        feat = sa_module(content_f, style_f)
    feat = feat * alpha + Fccc * (1 - alpha)
    return decoder(feat)
  
def feat_extractor(vgg, content, style):
  norm = nn.Sequential(*list(vgg.children())[:1])
  enc_1 = nn.Sequential(*list(vgg.children())[:4])  # input -> relu1_1
  enc_2 = nn.Sequential(*list(vgg.children())[4:11])  # relu1_1 -> relu2_1
  enc_3 = nn.Sequential(*list(vgg.children())[11:18])  # relu2_1 -> relu3_1
  enc_4 = nn.Sequential(*list(vgg.children())[18:31])  # relu3_1 -> relu4_1
  enc_5 = nn.Sequential(*list(vgg.children())[31:44])  # relu4_1 -> relu5_1

  norm.to(device)
  enc_1.to(device)
  enc_2.to(device)
  enc_4.to(device)
  enc_5.to(device)
  content3_1 = enc_3(enc_2(enc_1(content)))
  Content4_1 = enc_4(enc_3(enc_2(enc_1(content))))
  Content5_1 = enc_5(Content4_1)
  Style3_1 = enc_3(enc_2(enc_1(style)))
  Style4_1 = enc_4(enc_3(enc_2(enc_1(style))))
  Style5_1 = enc_5(Style4_1)
  

  content_f=[content3_1,Content4_1,Content5_1]
  style_f=[Style3_1,Style4_1,Style5_1]

 
  style_fs = [enc_1(style),enc_2(enc_1(style)),enc_3(enc_2(enc_1(style))),Style4_1, Style5_1]
  
  return style_fs,content_f, style_f

#图片处理，调用style_tansfer
def image_process(content, style):
 

    withAlpha = False
    if(content.shape[2] == 4):
        alphaCopy = content[:, :, 3]
        content = content[:, :, :3]
        withAlpha = True
    
    content_tf1 = content_transform()
    content_frame = content_tf1(content)
    h, w, c = np.shape(content_frame)
        
    content = content_frame.to(device).unsqueeze(0) 
    

    if(args.segment):
        person, content, bin_mask = remove_background(deeplab_model, content_frame)
    
    with torch.no_grad():
        output = style_transfer(vgg, decoder, mcc_module, content, style, alpha)
    output = output.squeeze(0)
    
    
    output = output * 255 + 0.5   
    output = (torch.clamp(output, 0, 255).permute(1, 2, 0)).type(torch.cuda.ByteTensor)

    output = output.cpu()
    output = output.numpy()
    
    if(withAlpha):
        if(output.shape[0] < alphaCopy.shape[0]):
            alphaCopy = alphaCopy[0:output.shape[0],0:output.shape[1]]
        
        if(output.shape[0] > alphaCopy.shape[0]):
            output = output[0:alphaCopy.shape[0],0:alphaCopy.shape[1]]

        full = np.zeros((output.shape[0], output.shape[1], 4), dtype=np.uint8)
        full[:, :, 3] = alphaCopy
        full[:, :, :3] = output
        output = full

    if(args.segment):
        return custom_background(output, person)
    
    return Image.fromarray(output)
    
def custom_background(background, foreground):
  background = Image.fromarray(background)
  final_foreground = Image.fromarray(foreground)

  x = (background.size[0]-final_foreground.size[0])/2
  y = (background.size[1]-final_foreground.size[1])/2

  box = (x, y, final_foreground.size[0] + x, final_foreground.size[1] + y)
  crop = background.crop(box)
  final_image = crop.copy()
  paste_box = (0, final_image.size[1] - final_foreground.size[1], final_image.size[0], final_image.size[1])
  final_image.paste(final_foreground, paste_box, mask=final_foreground)
  return final_image

#加载视频
def load_video(content_path,style_path, outfile):
    video = cv2.VideoCapture(content_path)

    rate = video.get(5)
    width = video.get(cv2.CAP_PROP_FRAME_WIDTH)
    height = video.get(cv2.CAP_PROP_FRAME_HEIGHT)  # 获得帧宽和帧高
    fps = int(rate)

    video_name = outfile + '/{:s}_stylized_{:s}{:s}'.format(
        splitext(basename(content_path))[0], splitext(basename(style_path))[0], '.mp4')

    videoWriter = cv2.VideoWriter(video_name, cv2.VideoWriter_fourcc(*'mp4v'), fps,
                                  (int(width), int(height)))
    return video,videoWriter
#存储视频
def save_frame(output, videoWriter ,j):
    x = cv2.cvtColor(np.array(output), cv2.COLOR_RGB2BGR)
    videoWriter.write(x)  # 写入帧图

#from remove image background deeplabV3
def load_model():
  model = torch.hub.load('pytorch/vision:v0.6.0', 'deeplabv3_resnet101', pretrained=True)
  model.eval()
  return model
  
def make_transparent_foreground(pic, mask):
  pic = pic * 255 + 0.5
  b, g, r = cv2.split( (torch.clamp(pic, 0, 255).permute(1, 2, 0)).type(torch.cuda.ByteTensor).cpu().numpy())
  a = np.ones(mask.shape, dtype='uint8') * 255
  alpha_im = cv2.merge([b, g, r, a], 4)
  bg = np.zeros(alpha_im.shape)
  new_mask = np.stack([mask, mask, mask, mask], axis=2)
  foreground = np.where(new_mask, alpha_im, bg).astype(np.uint8)
  return foreground
  
def remove_background(model, input_tensor):
  input_batch = input_tensor.unsqueeze(0) # create a mini-batch as expected by the model
  # move the input and model to GPU for speed if available
  if torch.cuda.is_available():
      input_batch = input_batch.to('cuda')
      model.to('cuda')

  with torch.no_grad():
      output = model(input_batch)['out'][0]
  output_predictions = output.argmax(0)
  mask = np.array(output_predictions.cpu())

  background = np.zeros(mask.shape)
  bin_mask = np.where(mask, 255, background).astype(cp.uint8)

  # improve mask
  if not args.default_mask:
    kernel = np.ones((15,15),np.uint8)
    bin_mask = cv2.morphologyEx(bin_mask, cv2.MORPH_OPEN, kernel)  
    bin_mask = cv2.morphologyEx(bin_mask, cv2.MORPH_CLOSE, kernel)  
  
  if(args.use_inpaint):
    # border mask
    kernel = np.ones((50,50),np.uint8)
    eroded_mask = cv2.morphologyEx(bin_mask, cv2.MORPH_ERODE, kernel)
    border_mask = (np.where((bin_mask - eroded_mask)==255, 255, 0)).astype(np.uint8)

    # apply inpaint
    input_numpy = tensor_to_numpy(input_tensor)
    dst = cv2.inpaint(input_numpy, border_mask,5,cv2.INPAINT_TELEA)
    dst_tensor = torch.from_numpy(dst)
    dst_tensor = dst_tensor.permute(2, 0, 1)
    dst_tensor = torch.reshape(dst_tensor, (1, 3, dst_tensor.shape[1], dst_tensor.shape[2]))
    dst_tensor = dst_tensor.type(torch.cuda.FloatTensor)
    background_tensor = dst_tensor / 255
  else:
    # without inpaint
    background_tensor = (input_tensor.type(torch.cuda.FloatTensor)).reshape(1,3,input_tensor.shape[1], input_tensor.shape[2])
    
  foreground = make_transparent_foreground(input_tensor ,bin_mask) 
    
  return foreground, background_tensor, bin_mask

#视频风格化
def process_video(content_path, style_path, outfile):
    j = 0
    video, videoWriter = load_video(content_path, style_path, outfile)

    w = video.get(cv2.CAP_PROP_FRAME_WIDTH)
    h = video.get(cv2.CAP_PROP_FRAME_HEIGHT)

    style = Image.open(style_path)
    style_tf1 = style_transform(h, w)
    style = style_tf1(style.convert("RGB"))

    if yaml['preserve_color']:
        style = coral(style, content)
    
    style = style.to(device).unsqueeze(0)
    
    while (video.isOpened()):
        j = j + 1
        ret, frame = video.read()
        if not ret:
            break

        if j % 1 == False:            
            # 对每一帧进行风格化。
            
            frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)   
            
            output = image_process(frame, style)
            # 对风格化后的结果进行额外处理，以存储到视频中

            save_frame(output, videoWriter, j)

#图像风格化
def process_image(content_path, style_path, outfile):
    content_name, extension = os.path.splitext(content_path)

    image_name = outfile + '/{:s}_stylized_{:s}{:s}'.format(
        splitext(basename(content_path))[0], splitext(basename(style_path))[0], extension)
    # 对图像进行风格迁移
    content = Image.open(content_path)
    content = np.array(content)

    w = content.shape[1]
    h = content.shape[0]

    style = Image.open(style_path)
    style_tf1 = style_transform(h, w)
    style = style_tf1(style.convert("RGB"))
    style = style.to(device).unsqueeze(0)

    output = image_process(content, style)
    
    output.save(image_name)

def tensor_to_numpy(tensor):
    output = tensor * 255 + 0.5   
    output = (torch.clamp(output, 0, 255).permute(1, 2, 0)).type(torch.cuda.ByteTensor)

    output = output.cpu()
    return output.numpy()


def test(content_paths, style_paths):
    for content_path in content_paths:
        # process one content and one style
        outfile = output_path + '/' + splitext(basename(content_path))[0] + '/'
        if not os.path.exists(outfile):
            os.makedirs(outfile)

        # 视频风格化
        if 'mp4' in content_path:
            for style_path in style_paths:
                process_video(content_path, style_path, outfile)
        # 图像风格化
        else:
            for style_path in style_paths:
                process_image(content_path, style_path, outfile)

def create_args():
    parser = argparse.ArgumentParser()
    # Basic options
    parser.add_argument('--content', type=str,default="./content/blonde_girl.jpg",
                        help='File path to the content image')
    parser.add_argument('--content_dir', type=str,
                        help='Directory path to a batch of content images')
    parser.add_argument('--style', type=str,default="./input/style/candy.jpg",
                        help='File path to the style image, or multiple style \
                        images separated by commas if you want to do style \
                        interpolation or spatial control')
    parser.add_argument('--style_dir', type=str,
                        help='Directory path to a batch of style images')
    parser.add_argument('--output', type=str, default='output',
                        help='Directory to save the output image(s)')
    parser.add_argument('--decoder_path', type=str, default='./experiments/decoder_iter_160000.pth')
    parser.add_argument('--transform_path', type=str, default='./experiments/mcc_module_iter_160000.pth')
    parser.add_argument('--vgg_path', type=str, default='./experiments/vgg_normalised.pth')
    parser.add_argument('--yaml_path', type=str, default='./yaml/test.yaml')
    parser.add_argument('--a', type=float, default=1.0)
    parser.add_argument('--style_interpolation_weights', type=str, default="")

    parser.add_argument('--segment', action='store_true')
    parser.add_argument('--default_mask', action='store_true')
    parser.add_argument('--use_inpaint', action='store_true')
    parser.set_defaults(segment=False)
    parser.set_defaults(default_mask=False)
    parser.set_defaults(use_inpaint=False)

    args = parser.parse_args()
    if not os.path.exists(args.output):
        os.mkdir(args.output)

    return args

if __name__ == '__main__':
    args = create_args()
    with open(args.yaml_path,'r') as file :
        yaml =yaml.load(file, Loader=yaml.FullLoader)
    alpha = args.a
    output_path = args.output
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    deeplab_model = load_model()
    decoder = net.decoder
    vgg = net.vgg
    network = net.Net(vgg, decoder)
    mcc_module = network.mcc_module
    decoder.eval()
    mcc_module.eval()
    vgg.eval()
    load_weights(vgg, decoder, mcc_module)

    norm = nn.Sequential(*list(vgg.children())[:1])
    enc_1 = nn.Sequential(*list(vgg.children())[:4])  # input -> relu1_1
    enc_2 = nn.Sequential(*list(vgg.children())[4:11])  # relu1_1 -> relu2_1
    enc_3 = nn.Sequential(*list(vgg.children())[11:18])  # relu2_1 -> relu3_1
    enc_4 = nn.Sequential(*list(vgg.children())[18:31])  # relu3_1 -> relu4_1
    enc_5 = nn.Sequential(*list(vgg.children())[31:44])  # relu4_1 -> relu5_1

    norm.to(device)
    enc_1.to(device)
    enc_2.to(device)
    enc_3.to(device)
    enc_4.to(device)
    enc_5.to(device)
    mcc_module.to(device)
    decoder.to(device)

    content_paths, style_paths = load_images(args)
    test(content_paths, style_paths)



