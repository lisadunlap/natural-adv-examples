from tkinter import *
from PIL import ImageTk,Image

import torch
import torch.nn as nn
import torchvision.datasets as dset
import torchvision.transforms as trn
import torchvision.transforms.functional as trnF
import torchvision.models as models
import torch.utils.model_zoo as model_zoo
import torch.nn.functional as F
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import ast
from PIL import Image
import csv

from notebooks.utils import get_displ_img


mean = [0.485, 0.456, 0.406]
std = [0.229, 0.224, 0.225]

test_transform = trn.Compose(
    [trn.Resize(256), trn.CenterCrop(224), trn.ToTensor(), trn.Normalize(mean, std)])

errs = pd.read_csv('imagenet-a_images.csv')
registry = {}

def get_new_sample(idx):
    ex_idx = errs.iloc[idx]
    ex_img_file = ex_idx['path'].replace('/home/lisa', '/Users/lisadunlap')
    ex_target_cls = ex_idx['class']
    image = Image.open(ex_img_file).convert(mode='RGB')
    w, h = image.size

    print(ex_img_file, ex_target_cls)

    # Crop the center of the image
    # smaller_image = image.resize((256, 256)).crop((16.0, 16.0, 240.0, 240.0))
    new_img = test_transform(image)
    orig = get_displ_img(new_img)
    smaller_image = Image.fromarray(np.uint8(orig*255))

    return image, smaller_image, ex_target_cls, w, h, ex_img_file

# current image index
curr_idx = 0
results_file = 'results.csv'
image, smaller_image, ex_target_cls, w, h, ex_img_file = get_new_sample(curr_idx)
prev_img, prev_small_img = image, smaller_image


# start of Tkinker
root = Tk()
root.title("Click me!")


def next_image(event):
    """toggle between image2 and image3"""
    global toggle_flag
    global x, y, photo2, photo3
    if toggle_flag == TRUE:
        # display photo2, same center x, y
        canvas1.create_image(x, y, image=photo2)
        toggle_flag = False
    else:
        canvas1.create_image(x, y, image=photo3)
        toggle_flag = True

def update(c, lbl, image, smaller_image, ex_target_cls):
    global x, y, photo1, photo2
    photo1 =ImageTk.PhotoImage(image)
    photo2 = ImageTk.PhotoImage(smaller_image)
    # c.itemconfigure(photo1, image= new_photo1)
    canvas1.create_image(x, y, image=photo1)
    canvas1.create_image(3*x, y, image=photo2)
    # c.itemconfigure(prev_small_img, image= ImageTk.PhotoImage(smaller_image))
    lbl.configure(text=f'Target class: {ex_target_cls}', font=('Courier', 20))


def retrieve_input():
    global results_file
    results_file=textBox.get("1.0","end-1c")
    print(results_file)

def retrieve_idx_input():
    global curr_idx
    curr_idx = int(idx_textBox.get("1.0", "end-1c"))
    idx_lbl["text"] = f'Index: {curr_idx}'
    image, smaller_image, ex_target_cls, w, h, ex_img_file = get_new_sample(curr_idx)
    update(canvas1, lbl, image, smaller_image, ex_target_cls)


toggle_flag = True

# pick three GIF image files you have in your working directory
# image1 is larger than image2 and image3
photo1 = ImageTk.PhotoImage(image)

photo2 = ImageTk.PhotoImage(smaller_image)

# make canvas sufficiently large to fit all image sizes
canvas1 = Canvas(width=1500, height=1000)
canvas1.pack()


x, y = 400, 400
canvas1.create_image(x, y, image=photo1)
canvas1.create_image(3*x - 50, y, image=photo2)


lbl = Label(root, text=f'Target class: {ex_target_cls}', font=('Courier', 20))
lbl.place(x=x+100, y=2*y)

idx_lbl = Label(root, text=f'Index: {curr_idx}', font=('Courier', 20))
idx_lbl.place(x=x - 100, y=2*y)


file_lbl = Label(root, text=f'Results file', font=('Courier', 20))
file_lbl.place(x=20, y=20)

# top buttons
textBox=Text(root, height=1, width=15, font=('Courier', 20))
# textBox.pack()
buttonCommit=Button(root, height=1, width=5, text="Submit",
                    command=lambda: retrieve_input(), font=('Courier', 20))
textBox.place(x = 200, y = 20)
buttonCommit.place(x = 400, y = 20)


idx_skip_lbl = Label(root, text=f'Skip to idx', font=('Courier', 20))
idx_skip_lbl.place(x=700, y=20)

idx_textBox=Text(root, height=1, width=3, font=('Courier', 20))
# textBox.pack()
idx_buttonCommit=Button(root, height=1, width=5, text="Submit",
                    command=lambda: retrieve_idx_input(), font=('Courier', 20))
idx_textBox.place(x = 950, y = 20)
idx_buttonCommit.place(x = 1300, y = 20)

def clicked():
    lbl.configure(text="Button was clicked !!")

def blurr():
    global ex_img_file, results_file, btn2
    btn2["highlightbackground"] = "black"
    idx = int(idx_lbl["text"].replace('Index: ', ''))
    with open(results_file, 'a+', newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow([idx, ex_img_file,"BLURR"])
    # move onto the next image
    # next_clicked()

def invalid():
    global ex_img_file, results_file, btn9
    btn9["highlightbackground"] = "black"
    idx = int(idx_lbl["text"].replace('Index: ', ''))
    with open(results_file, 'a+', newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow([idx, ex_img_file,"INVALID"])
    # move onto the next image
    # next_clicked()

def occlusion():
    global ex_img_file, results_file, btn3
    btn3["highlightbackground"] = "black"
    idx = int(idx_lbl["text"].replace('Index: ', ''))
    with open(results_file, 'a+', newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow([idx, ex_img_file, "OCCLUSION"])
    # move onto the next image
    # next_clicked()

def background():
    global ex_img_file, results_file, btn4
    btn4["highlightbackground"] = "black"
    idx = int(idx_lbl["text"].replace('Index: ', ''))
    with open(results_file, 'a+', newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow([idx, ex_img_file, "SIZE"])
    # move onto the next image
    # next_clicked()

def confusion():
    global ex_img_file, results_file, btn5
    btn5["highlightbackground"] = "black"
    idx = int(idx_lbl["text"].replace('Index: ', ''))
    with open(results_file, 'a+', newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow([idx, ex_img_file, "CAMOUFLAGE"])
    # move onto the next image
    # next_clicked()

def other():
    global ex_img_file, results_file, btn10
    btn10["highlightbackground"] = "black"
    idx = int(idx_lbl["text"].replace('Index: ', ''))
    with open(results_file, 'a+', newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow([idx, ex_img_file, "OTHER"])
    # move onto the next image
    # next_clicked()

def mirror():
    global ex_img_file, results_file, btn6
    btn6["highlightbackground"] = "black"
    idx = int(idx_lbl["text"].replace('Index: ', ''))
    with open(results_file, 'a+', newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow([idx, ex_img_file, "IMAGE/MIRROR"])
    # move onto the next image
    # next_clicked()

def lighting():
    global ex_img_file, results_file, btn7
    btn7["highlightbackground"] = "black"
    idx = int(idx_lbl["text"].replace('Index: ', ''))
    with open(results_file, 'a+', newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow([idx, ex_img_file, "LIGHTING"])
    # move onto the next image
    # next_clicked()

def ood():
    global ex_img_file, results_file, btn8
    btn8["highlightbackground"] = "black"
    idx = int(idx_lbl["text"].replace('Index: ', ''))
    with open(results_file, 'a+', newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow([idx, ex_img_file, "OOD"])
    # move onto the next image
    # next_clicked()

def error():
    global ex_img_file, results_file, btn11
    btn11["highlightbackground"] = "black"
    idx = int(idx_lbl["text"].replace('Index: ', ''))
    with open(results_file, 'a+', newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow([idx, ex_img_file, "ERROR"])
    # move onto the next image
    # next_clicked()

def next_clicked():
    global ex_img_file
    idx = int(idx_lbl["text"].replace('Index: ', ''))
    idx_lbl["text"] = f'Index: {idx+1}'
    image, smaller_image, ex_target_cls, w, h, ex_img_file = get_new_sample(idx+1)
    update(canvas1, lbl, image, smaller_image, ex_target_cls)
    reset_colors()

def prev_clicked():
    global ex_img_file
    idx = int(idx_lbl["text"].replace('Index: ', ''))
    idx_lbl["text"] = f'Index: {idx-1}'
    image, smaller_image, ex_target_cls, w, h, ex_img_file = get_new_sample(idx-1)
    update(canvas1, lbl, image, smaller_image, ex_target_cls)


def reset_colors():
    global btn2, btn3, btn4, btn5, btn6, btn7, btn8, btn9, btn10, btn11
    btn2["highlightbackground"] = "systemWindowBackgroundColor"
    btn3["highlightbackground"] = "systemWindowBackgroundColor"
    btn4["highlightbackground"] = "systemWindowBackgroundColor"
    btn5["highlightbackground"] = "systemWindowBackgroundColor"
    btn6["highlightbackground"] = "systemWindowBackgroundColor"
    btn7["highlightbackground"] = "systemWindowBackgroundColor"
    btn8["highlightbackground"] = "systemWindowBackgroundColor"
    btn9["highlightbackground"] = "systemWindowBackgroundColor"
    btn10["highlightbackground"] = "systemWindowBackgroundColor"
    btn11["highlightbackground"] = "systemWindowBackgroundColor"

btn = Button(root, text="Prev", command=prev_clicked, width=6, height=2, font=('Courier', 20))
btn.place(x=50, y=880)

btn2 = Button(root, text="Blurry", command=blurr, font=('Courier', 20), width=10, height=2, fg="orange")
btn2.place(x=1330, y=100)

btn3 = Button(root, text="Occluded", command=occlusion, font=('Courier', 20), width=10, height=2, fg="blue")
btn3.place(x=1330, y=175)

btn4 = Button(root, text="Size", command=background, font=('Courier', 20), width=10, height=2, fg="purple")
btn4.place(x=1330, y=250)

btn5 = Button(root, text="Camouflage", command=confusion, font=('Courier', 20), width=10, height=2, fg="green2")
btn5.place(x=1330, y=325)

btn6 = Button(root, text="Image/Mirror", command=mirror, font=('Courier', 20), width=10, height=2, fg="turquoise4")
btn6.place(x=1330, y=400)

btn7 = Button(root, text="Lighting", command=lighting, font=('Courier', 20), width=10, height=2, fg="gold")
btn7.place(x=1330, y=475)

btn8 = Button(root, text="OOD", command=ood, font=('Courier', 20), width=10, height=2, fg="SlateBlue1")
btn8.place(x=1330, y=550)

btn9 = Button(root, text="Invalid", command=invalid, font=('Courier', 20), width=10, height=2, fg="red")
btn9.place(x=1330, y=625)

btn10 = Button(root, text="Other", command=other, font=('Courier', 20), width=10, height=2, fg="grey")
btn10.place(x=1330, y=700)

btn11 = Button(root, text="ERROR", command=error, font=('Courier', 20), width=10, height=2, fg="dark slate grey")
btn11.place(x=1330, y=775)

btn12 = Button(root, text="Next", command=next_clicked, width=6, height=2, font=('Courier', 20))
btn12.place(x=1350, y=880)

# canvas1.bind('<Button-1>', next_clicked)  # bind left mouse click

root.mainloop()