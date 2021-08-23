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

errs = pd.read_csv('results/possible_errors_info.csv')
registry = {}

def get_new_sample(idx):
    ex_idx = errs.iloc[idx]
    ex_img_file = ex_idx['image file'].replace('/home/lisa', '/Users/lisadunlap')
    ex_target_cls = ex_idx['target class name']
    ex_pred_cls = ex_idx['pred class name']
    image = Image.open(ex_img_file).convert(mode='RGB')
    w, h = image.size

    print(ex_img_file, ex_target_cls)

    # Crop the center of the image
    # smaller_image = image.resize((256, 256)).crop((16.0, 16.0, 240.0, 240.0))
    new_img = test_transform(image)
    orig = get_displ_img(new_img)
    smaller_image = Image.fromarray(np.uint8(orig*255))

    return image, smaller_image, ex_target_cls, ex_pred_cls, w, h, ex_img_file

# current image index
curr_idx = 0
results_file = 'results.csv'
image, smaller_image, ex_target_cls, ex_pred_cls, w, h, ex_img_file = get_new_sample(curr_idx)
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
    image, smaller_image, ex_target_cls, ex_pred_cls, w, h, ex_img_file = get_new_sample(curr_idx)
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
canvas1.create_image(3*x, y, image=photo2)


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

def ok_clicked():
    global ex_img_file, results_file
    idx = int(idx_lbl["text"].replace('Index: ', ''))
    with open(results_file, 'a+', newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow([idx, ex_img_file,"OK"])
    # move onto the next image
    next_clicked()

def invalid_clicked():
    global ex_img_file, results_file
    idx = int(idx_lbl["text"].replace('Index: ', ''))
    with open(results_file, 'a+', newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow([idx, ex_img_file, "INVALID"])
    # move onto the next image
    next_clicked()

def not_sure_clicked():
    global ex_img_file, results_file
    idx = int(idx_lbl["text"].replace('Index: ', ''))
    with open(results_file, 'a+', newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow([idx, ex_img_file, "NOT SURE"])
    # move onto the next image
    next_clicked()

def next_clicked():
    global ex_img_file
    idx = int(idx_lbl["text"].replace('Index: ', ''))
    idx_lbl["text"] = f'Index: {idx+1}'
    image, smaller_image, ex_target_cls, ex_pred_cls, w, h, ex_img_file = get_new_sample(idx+1)
    update(canvas1, lbl, image, smaller_image, ex_target_cls)

def prev_clicked():
    global ex_img_file
    idx = int(idx_lbl["text"].replace('Index: ', ''))
    idx_lbl["text"] = f'Index: {idx-1}'
    image, smaller_image, ex_target_cls, ex_pred_cls, w, h, ex_img_file = get_new_sample(idx-1)
    update(canvas1, lbl, image, smaller_image, ex_target_cls)

btn = Button(root, text="Prev", command=prev_clicked, width=6, height=2, font=('Courier', 20))
btn.place(x=100, y=880)

btn2 = Button(root, text="OK", command=ok_clicked, font=('Courier', 20), width=6, height=2, bg="green", fg="green")
btn2.place(x=500, y=880)

btn3 = Button(root, text="INVALID", command=invalid_clicked, font=('Courier', 20), width=6, height=2, bg="red", fg="red")
btn3.place(x=700, y=880)

btn3 = Button(root, text="NOT SURE", command=not_sure_clicked, font=('Courier', 20), width=6, height=2, bg="blue", fg="blue")
btn3.place(x=900, y=880)

btn4 = Button(root, text="Next", command=next_clicked, width=6, height=2, font=('Courier', 20))
btn4.place(x=1300, y=880)

# canvas1.bind('<Button-1>', next_image)  # bind left mouse click

root.mainloop()