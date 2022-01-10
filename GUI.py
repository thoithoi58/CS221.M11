from tkinter import * 
from tkinter.ttk import *
from tkinter.filedialog import askopenfilename
import HMM as hm
import pandas as pd
import utils

class HMM:
    def __init__(self, master):
        self.master = master
        self.tab = Frame(self.master)
        self.master.add(self.tab, text ='Viterbi')

        self.label = Label(self.tab, text="Viterbi Part-of-speech Tagger",width=30,font=("bold", 20))
        self.label.place(x=65,y=53)

        self.train_file = Button(self.tab, text='Train file', command = lambda widget="train": self.open_file(widget))
        self.train_file.place(x=90,y=120)

        self.test_file = Button(self.tab, text='Test file', command = lambda widget="test": self.open_file(widget))
        self.test_file.place(x=200,y=120)

        self.test = Button(self.tab, text='Test!', command = self.tag)
        self.test.place(x=310,y=120)

        self.accuracy = StringVar(self.tab)
        self.button = StringVar(self.tab)

        self.output = Entry(self.tab, width=30, state=DISABLED, textvariable = self.accuracy)
        self.output.place(x=150,y=230)

        self.statusbar = Label(self.tab, textvariable=self.button , relief=SUNKEN, anchor=W, state=DISABLED)
        self.statusbar.pack(side=BOTTOM, fill=X)

    def open_file(self, widget):
        global train, test, button
        file_name = askopenfilename(filetypes =[('Text Files', '*.txt')])
        if widget == 'train':
            train = utils.read(file_name)
            self.button.set('Training file loaded')
        elif widget == 'test':
            test = utils.read(file_name)
            self.button.set('Testing file loaded')

    def tag(self):
        acc = hm.compute(train, test)
        self.accuracy.set(str(acc))
    

root = Tk()
root.geometry('500x500') 
root.title("POS Tagger")
tabControl = Notebook(root)
  
hmm = HMM(tabControl)

tabControl.pack(expand = 1, fill ="both")

root.mainloop()