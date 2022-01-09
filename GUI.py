from tkinter import * 
from tkinter.ttk import *
from tkinter.filedialog import askopenfilename
import HMM
import pandas as pd
import utils



def open_file(widget):
    global train, test
    file_name = askopenfilename(filetypes =[('Text Files', '*.txt')])
    if widget == 'train':
        train = utils.read(file_name)
    elif widget == 'test':
        test = utils.read(file_name)

def tag():
    acc = HMM.compute(train, test)
    entryText.set(str(acc))
    

root = Tk()
root.geometry('500x500') 
root.title("POS Tagger")
tabControl = Notebook(root)
  
tab1 = Frame(tabControl)
tab2 = Frame(tabControl)
  
tabControl.add(tab1, text ='Viterbi')
tabControl.add(tab2, text ='CRF')
tabControl.pack(expand = 1, fill ="both")

label_0 = Label(tab1, text="Viterbi Part-of-speech Tagger",width=30,font=("bold", 20))
label_0.place(x=30,y=53)

train_viter = Button(tab1, text='Train file', command = lambda widget="train": open_file(widget))
train_viter.place(x=90,y=100)
test_viter = Button(tab1, text='Test file', command = lambda widget="test": open_file(widget))
test_viter.place(x=200,y=100)

test_button = Button(tab1, text='Test!', command = tag)
test_button.place(x=310,y=100)

entryText = StringVar(tab1)

output = Entry(tab1, width=30, state=DISABLED, textvariable = entryText)
output.place(x=150,y=230)

root.mainloop()