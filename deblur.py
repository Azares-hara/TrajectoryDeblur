import torch
import imageio
import os
from model2 import Model   
from dataset import common 

checkpoint_dir = "../experiment/2026-03-18_16-06-36/models"
checkpoint_epoch = 1000   
device = "cuda:0"         

#trained model 
args = torch.load(os.path.join(checkpoint_dir, f"args-{checkpoint_epoch}.pt"))
model = Model(args).to(device)
model.load(epoch=checkpoint_epoch)  
model.eval()

#Deblur a folder of blurry images 
input_folder = "./my_blurry_images"
output_folder = "./my_deblurred_results"
os.makedirs(output_folder, exist_ok=True)

for fname in os.listdir(input_folder):
    if fname.lower().endswith((".jpg", ".png")):
        blur = imageio.imread(os.path.join(input_folder, fname), pilmode="RGB")
        blur_tensor = common.np2tensor(blur).unsqueeze(0).to(device)

        with torch.no_grad():
            output = model(blur_tensor)["out"].cpu()

        #save deblurred result
        out_img = common.tensor2np(output.squeeze(0))
        imageio.imwrite(os.path.join(output_folder, fname), out_img)

print("Deblurring complete! Results saved to", output_folder)
