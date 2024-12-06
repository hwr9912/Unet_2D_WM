import torch
from PIL import Image
from dataset import transform
def predict(img_path):
    net = torch.load('model.pkl')
    net = net.to(torch.device(0))
    torch.no_grad()
    img = Image.open(img_path)
    img = transform(img).unsqueeze(0)
    img_ = img.to(torch.device(0))
    outputs = net(img_)
    _, predicted = torch.max(outputs, 1)
    # print(predicted)
    print('this picture maybe :', classes[predicted[0]])