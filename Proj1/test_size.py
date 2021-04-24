from models.convnet import ConvNet

model = ConvNet()
for k in model.parameters():
    print(k.size())