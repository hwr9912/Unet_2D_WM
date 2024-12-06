
from model.unet import UNet
def build_model(model_name, num_classes):

    if model_name == 'unet' :  #
        return UNet(classes=num_classes)

