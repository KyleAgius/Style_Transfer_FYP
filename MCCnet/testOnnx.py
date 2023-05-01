import onnxruntime as ort
import numpy as np
import torch
from PIL import Image
import torchvision.transforms as transforms

content = Image.open("input/content/Jog.png")
style = Image.open("input/style/Starry_Night.jpg")


contentMock = torch.abs(torch.randn(1, 4, 1800, 1800).float())
styleMock = torch.abs(torch.randn(1, 3, 950, 1200).float())

print("exp 16")

ort_sess = ort.InferenceSession('mccNet.onnx', providers=['CPUExecutionProvider'])

input_names = ort_sess.get_inputs()
output_name = ort_sess.get_outputs()[0].name

input_data = [
    {input_names[0].name: contentMock},
    {input_names[1].name: styleMock}
]

input_data_2 = {
    input_names[0].name: contentMock,
    input_names[1].name: styleMock
}

print(type(input_data_2))

output = ort_sess.run([output_name], input_data)

# Print Result 
print(output)
