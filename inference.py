import torch
import argparse
import torchvision.transforms as transforms
from PIL import Image, ImageDraw, ImageFont
from model import Net 

# Classes used in the dataset
classes = ["Airport", "Bridge", "Center", "Desert", "Forest",
    "Industrial", "Mountain", "Pond", "Port", "Stadium"]

# Getting all the transforms used 
def get_transform():

    transform = transforms.Compose([
        transforms.PILToTensor(),
        transforms.ConvertImageDtype(torch.float),
    ])

    return transform

# Parsing the arguments
parser = argparse.ArgumentParser()
parser.add_argument('--model', type=str, required=True, help="Path to model required")
parser.add_argument('--image', type=str, required=True, help="Image path required")

args = parser.parse_args()

# Loading the model and the image
model_path = args.model
img_path = args.image

# Loading the image and transforming it
img = Image.open(img_path)
img_transform = get_transform()
input_img = img_transform(img).unsqueeze(0)

# Loading the model
model = Net()
model.load_state_dict(torch.load(model_path, map_location=torch.device('cpu')))
model.eval()

# Getting the prediction
with torch.no_grad(): 
    output = model(input_img) 
    output_idx = torch.argmax(output, dim=1).item()

    draw = ImageDraw.Draw(img)
    font = ImageFont.truetype("Arial.ttf", 36)
    draw.text((0, 0),f"Predicted class: {classes[output_idx]}",(0,255,255),font=font)
    img.show()