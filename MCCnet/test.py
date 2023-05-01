import time
import torch

b = torch.randn(1,3,32,32).cuda()
t1 = time.time()
output = b * 255 + 0.5   
output = (torch.clamp(output, 0, 255)).type(torch.cuda.ByteTensor)

output = output.cpu()
output = output.numpy()
t2 = time.time()
print(t2-t1)