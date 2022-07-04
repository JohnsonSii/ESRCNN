import torch, sys
sys.path.append("..")
from model import cnn
from PIL import Image
from torchvision import transforms

model = cnn.Net()
state_dict = torch.load("./output/params_12.pth")
model.load_state_dict(state_dict)


# img = Image.open(r"E:\Dataset\DIV2K_train_HR\DIV2K_train_HR\0091.png")
# img = img.resize((256, 256))
# img.show()
# img = transforms.ToTensor()(img)
# img = torch.reshape(img, (1, 3, 256, 256))
# out = model(img)
# out = torch.reshape(out, (3, 512, 512))
# out = transforms.ToPILImage()(out)
# out.show()

img = Image.open(r"E:\Dataset\DIV2K_train_HR\DIV2K_train_HR\0093.png")
img = transforms.ToTensor()(img)
img = transforms.RandomCrop(512)(img)
img = transforms.ToPILImage()(img)
img = img.resize((64, 64))
img.show()
img = transforms.ToTensor()(img)

img = torch.reshape(img, (1, 3, 64, 64))
out = model(img)
out = torch.reshape(out, (3, 512, 512))
out = transforms.ToPILImage()(out)
out.show()
