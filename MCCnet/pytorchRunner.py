import pytorchModel
import torch
import numpy
from PIL import Image
import torchvision.transforms as transforms
from torchsummary import summary
import net as net
from os import listdir
from os.path import isfile, join

device = "cuda" if torch.cuda.is_available() else "cpu"
p1 = pytorchModel.MCCNet_pytorch('./experiments/vgg_normalised.pth','./experiments/decoder_iter_160000.pth','./experiments/mcc_module_iter_160000.pth').to(device)

transform = transforms.Compose([
    transforms.ToTensor()
])

# change path to style image
style = Image.open("input/style/guernica.jpg")
style = transform(style).unsqueeze(0)
  
path = "input/content/DemoVer2"
pathOut = "output/DemVer2"
contentFiles = [f for f in listdir(path) if isfile(join(path, f))]

for fileName in contentFiles:
    print("Stylising " + fileName)
    content = Image.open(path + "/" + fileName)
    content = numpy.asarray(content)

    splits = 1
    failed = True

    while failed:
        try:
            final = Image.new('RGBA', (content.shape[1], content.shape[0]))
            splitWidth = content.shape[0]//splits
            splitHeight = content.shape[1]//splits
            blocks = [content[x:x+splitWidth,y:y+splitHeight] for x in range(0,content.shape[0],splitWidth) for y in range(0,content.shape[1],splitHeight)]
  
            index = 0

            for x in range(splits):
                for y in range(splits):
                    block = blocks[x*splits + y]
                    
                    tensorCon = transform(block).unsqueeze(0)
                    result = p1(tensorCon, style)
                    result = result.squeeze(0).permute(1,2,0)
                    output = result.cpu()
                    result = output.numpy().astype("uint8")

                    img = Image.fromarray(result)

                    final.paste(img, (y * splitHeight, x * splitWidth))          
            

        except Exception as e:
            print(e)
            splits += 1
            print("Failed to stylised. Splitting image into " + str(splits*splits) + " parts.")

        else:
            failed = False
            final.save(pathOut + "/" + fileName)
            print("Finished Stylising " + fileName)
  