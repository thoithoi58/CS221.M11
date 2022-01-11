from tkinter import * 
from tkinter.ttk import *
import HMM as hmm_
import spacy
from joblib import load

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

        self.label = Label(self.tab, text="Hidden Markov Model",width=30,font=("bold", 11))
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
        train_tagged_words, tags_df = hmm_.load_train_file('corpus/train.txt')
        return train_tagged_words, tags_df

    def tag(self):
        inputValue=self.input.get("1.0",END).splitlines()
        self.output.configure(state='normal')
        self.output.delete("1.0","end")
        for sent in inputValue:
            tmp = nlp(sent)
            tmp = [token.text for token in tmp]
            seq = hmm_.Viterbi(tmp, self.tags_df, self.train_tagged_words)
            self.output.insert(END, seq)
            self.output.insert(END, '\n')


class CRF:
    def __init__(self, master):
        self.master = master
        self.tab = Frame(self.master)
        self.master.add(self.tab, text ='CRF')

        #Load the transition matrix from training file
        self.model = self.load_model()
        self.status = StringVar(self.tab)
        self.status.set('Model loaded sucessfully!')
        self.statusbar = Label(self.tab, textvariable=self.status , relief=SUNKEN, anchor=W, state=DISABLED)
        self.statusbar.pack(side=BOTTOM, fill=X)

        self.label = Label(self.tab, text="Conditional Random Field",width=30,font=("bold", 11))
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
        model = load('crf.joblib')
        return model

    def feature_extract(self, sent, idx):
        word = sent[idx]
        if idx == 0:
            first = True
        else:
            first = False
        if idx == len(sent) - 1:
            last = True
        else:
            last = False

        features = {
            'word' : word,
            'word.lower()': word.lower(),
            'number': word.isdigit(),
            'word.istitle()': word.istitle(),
            'word.isupper()': word.isupper(),
            'has_hyphen': '-' in word,
            'is_first': first,
            'is_last': last
        }
        return features

    def tag(self):
        inputValue=self.input.get("1.0",END).splitlines()
        self.output.configure(state='normal')
        self.output.delete("1.0","end")
        for sent in inputValue:
            tmp = nlp(sent)
            tmp = [token.text for token in tmp]
            # print(tmp)
            features = [self.feature_extract(tmp, idx) for idx, i in enumerate(tmp)]
            seq = self.model.predict([features])[0]
            # print(seq)
            out = list(zip(tmp, seq))
            # print(out)
            self.output.insert(END, out)
            self.output.insert(END, '\n')

nlp = spacy.load('en_core_web_sm', disable=['ner', 'parser'])

root = Tk()
root.geometry('700x500') 
root.title("POS Tagger")
tabControl = Notebook(root)
  
hmm = HMM(tabControl)
crf = CRF(tabControl)

tabControl.pack(expand = 1, fill ="both")

root.mainloop()