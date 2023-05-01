import os
import torch
import torch.nn as nn
from torchvision import transforms
import net as net

class MCCNet_pytorch(nn.Module):    

    def __init__(self, vgg_path, decoder_path, transform_path):
        super(MCCNet_pytorch, self).__init__()
        self.alpha = 1.0
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        #self.deeplab_model = self.load_deeplab_model()
        self.decoder = net.decoder
        self.vgg = net.vgg
        self.network = net.Net(self.vgg, self.decoder)
        self.mcc_module = self.network.mcc_module

        self.decoder.eval()
        self.mcc_module.eval()
        self.vgg.eval()

        self.load_weights(vgg_path, decoder_path, transform_path)

        self.norm = nn.Sequential(*list(self.vgg.children())[:1])
        self.enc_1 = nn.Sequential(*list(self.vgg.children())[:4])  # input -> relu1_1
        self.enc_2 = nn.Sequential(*list(self.vgg.children())[4:11])  # relu1_1 -> relu2_1
        self.enc_3 = nn.Sequential(*list(self.vgg.children())[11:18])  # relu2_1 -> relu3_1
        self.enc_4 = nn.Sequential(*list(self.vgg.children())[18:31])  # relu3_1 -> relu4_1
        self.enc_5 = nn.Sequential(*list(self.vgg.children())[31:44])  # relu4_1 -> relu5_1

        self.norm.to(self.device)
        self.enc_1.to(self.device)
        self.enc_2.to(self.device)
        self.enc_3.to(self.device)
        self.enc_4.to(self.device)
        self.enc_5.to(self.device)
        self.mcc_module.to(self.device)
        self.decoder.to(self.device)


    #加载模型参数
    def load_weights(self, vgg_path, decoder_path, transform_path):
        self.vgg.load_state_dict(torch.load(vgg_path))
        self.decoder.load_state_dict(torch.load(decoder_path))
        self.mcc_module.load_state_dict(torch.load(transform_path))

    #风格图预处理
    def style_transform(self,size):

        transform_list = []
        transform_list.append(transforms.Resize(size))
        #transform_list.append(transforms.ToTensor())
        transform = transforms.Compose(transform_list)
        return transform


    #风格化函数
    def style_transfer(self, content, style, interpolation_weights=None):
        assert (0.0 <= self.alpha <= 1.0)

        style_fs, content_f, style_f=self.feat_extractor(self.vgg, content, style)
        Fccc = self.mcc_module(content_f,content_f)

        if interpolation_weights:
            _, C, H, W = Fccc.size()
            feat = torch.FloatTensor(1, C, H, W).zero_().to(device)
            base_feat = self.mcc_module(content_f, style_f)
            for i, w in enumerate(interpolation_weights):
                feat = feat + w * base_feat[i:i + 1]
            Fccc=Fccc[0:1]
        else:
            feat = self.mcc_module(content_f, style_f)
        feat = feat * self.alpha + Fccc * (1 - self.alpha)
        return self.decoder(feat)
    
    def feat_extractor(self, vgg, content, style):
        norm = nn.Sequential(*list(vgg.children())[:1])
        enc_1 = nn.Sequential(*list(vgg.children())[:4])  # input -> relu1_1
        enc_2 = nn.Sequential(*list(vgg.children())[4:11])  # relu1_1 -> relu2_1
        enc_3 = nn.Sequential(*list(vgg.children())[11:18])  # relu2_1 -> relu3_1
        enc_4 = nn.Sequential(*list(vgg.children())[18:31])  # relu3_1 -> relu4_1
        enc_5 = nn.Sequential(*list(vgg.children())[31:44])  # relu4_1 -> relu5_1

        norm.to(self.device)
        enc_1.to(self.device)
        enc_2.to(self.device)
        enc_4.to(self.device)
        enc_5.to(self.device)
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
    def image_process(self, content, style):
        
        withAlpha = False
        if(content.shape[1] == 4):
            alphaCopy = content[:, 3]#.permute(1,2,0)
            alphaCopy = alphaCopy.unsqueeze(0) * 255 + 0.5 
            content = content[:, :3]
            withAlpha = True

        with torch.no_grad():
            output = self.style_transfer(content, style)
        output = output * 255 + 0.5   
        output = (torch.clamp(output, 0, 255))#.permute(1, 2, 0))
        
        if(withAlpha):

        #full = torch.zeros(size = (4, output.shape[1], output.shape[2]), dtype = torch.float32)
        
            if(output.shape[2] < alphaCopy.shape[2]):
                alphaCopy = alphaCopy[:,:,0:output.shape[2],0:output.shape[3]]
            
            if(output.shape[2] > alphaCopy.shape[2]):
                output = output[:,:,0:alphaCopy.shape[2],0:alphaCopy.shape[3]]

            output = torch.cat((output, alphaCopy), dim=1)

        #full[3] = torch.squeeze(alphaCopy, 2)
        #print(full[:3].shape)
        #full[:3] = output
        #output = full
        
        return output
    
    #图像风格化
    def process_image(self, content, style):

        w = content.shape[3]
        h = content.shape[2]
        maxSize = max(w,h)
        #print(maxSize)
        #print(maxSize.shape)

        content = content.float()
        content = content.to(self.device)

        style_trans = self.style_transform(maxSize)
        style = style_trans(style)
        style = style.to(self.device)#.unsqueeze(0)
        style = style.float()

        output = self.image_process(content, style)
        #output = output.permute(2,0,1)

        return output

    def forward(self, content, style):
        return self.process_image(content, style)


   


