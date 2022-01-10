from tkinter import * 
import tkinter.scrolledtext as scrolledtext
from tkinter.ttk import *
from tkinter.filedialog import askopenfilename
import HMM as hm
import pandas as pd
import re

class HMM:
    def __init__(self, master):
        self.master = master
        self.tab = Frame(self.master)
        self.master.add(self.tab, text ='Viterbi')

        #Load the transition matrix from training file
        self.train_tagged_words, self.tags_df = self.load_model()
        self.status = StringVar(self.tab)
        self.status.set('Model loaded sucessfully!')
        self.statusbar = Label(self.tab, textvariable=self.status , relief=SUNKEN, anchor=W, state=DISABLED)
        self.statusbar.pack(side=BOTTOM, fill=X)

        self.label = Label(self.tab, text="Viterbi Part-of-speech Tagger",width=30,font=("bold", 11))
        self.label.place(x=250,y=10)
   
        self.input = Text(self.tab, height=10, width=80)
        self.input.place(x=30, y=35)

        self.test = Button(self.tab, text='Test', command = self.tag)
        self.test.place(x=300,y=211)

        self.output = Text(self.tab, height=10, width=80)
        self.output.insert(INSERT, 'Waiting for input...')
        self.output.configure(state='disabled')
        self.output.place(x=30, y=250)

        
    def load_model(self):
        train_tagged_words, tags_df = hm.load_train_file('corpus/train.txt')
        return train_tagged_words, tags_df

    def tag(self):
        inputValue=self.input.get("1.0",END).splitlines()
        self.output.configure(state='normal')
        self.output.delete("1.0","end")
        for idx, i in enumerate(inputValue):
            tmp = re.findall(r"'\w+|n't|\w+(?=n't)|\w+|[^\s\w]", i)
            seq = hm.Viterbi(tmp, self.tags_df, self.train_tagged_words)
            self.output.insert(END, seq)
            self.output.insert(END, '\n')



root = Tk()
root.geometry('700x500') 
root.title("POS Tagger")
tabControl = Notebook(root)
  
hmm = HMM(tabControl)

tabControl.pack(expand = 1, fill ="both")

root.mainloop()