from tkinter import * 
from tkinter.ttk import *
import pickle
import models.HMM as hmm_
from joblib import load
import spacy
from spacy.tokenizer import Tokenizer
from spacy.util import compile_infix_regex

nlp = spacy.load("en_core_web_sm", disable=['ner', 'parser'])

def custom_tokenizer(nlp):
    inf = list(nlp.Defaults.infixes)               # Default infixes
    inf.remove(r"(?<=[0-9])[+\-\*^](?=[0-9-])")    # Remove the generic op between numbers or between a number and a -
    inf = tuple(inf)                               # Convert inf to tuple
    infixes = inf + tuple([r"(?<=[0-9])[+*^](?=[0-9-])", r"(?<=[0-9])-(?=-)"])  # Add the removed rule after subtracting (?<=[0-9])-(?=[0-9]) pattern
    infixes = [x for x in infixes if '-|–|—|--|---|——|~' not in x] # Remove - between letters rule
    infix_re = compile_infix_regex(infixes)

    return Tokenizer(nlp.vocab, prefix_search=nlp.tokenizer.prefix_search,
                                suffix_search=nlp.tokenizer.suffix_search,
                                infix_finditer=infix_re.finditer,
                                token_match=nlp.tokenizer.token_match,
                                rules=nlp.Defaults.tokenizer_exceptions)

nlp.tokenizer = custom_tokenizer(nlp)

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

        #Load the model
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
        model = load('models/crf.joblib')
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
        # print(self.input.get("1.0",END).splitlines())
        inputValue = self.input.get("1.0",END).splitlines()
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

class Lemma:
    def __init__(self, master):
        self.master = master
        self.tab = Frame(self.master)
        self.master.add(self.tab, text ='Lemmatizer')

        #Load dictionary
        self.lemma_dict, self.noun_exception = self.load_files()
        self.status = StringVar(self.tab)
        self.status.set('Model loaded sucessfully!')
        self.statusbar = Label(self.tab, textvariable=self.status , relief=SUNKEN, anchor=W, state=DISABLED)
        self.statusbar.pack(side=BOTTOM, fill=X)

        self.label = Label(self.tab, text="Dictionary and Rule-based Lemmatizer",width=30,font=("bold", 11))
        self.label.place(x=250,y=10)
   
        self.input_cell = Text(self.tab, height=10, width=80)
        self.input_cell.place(x=30, y=35)

        self.test = Button(self.tab, text='Test', command = self.tag)
        self.test.place(x=300,y=211)

        self.output = Text(self.tab, height=10, width=80)
        self.output.insert(INSERT, 'Waiting for input...')
        self.output.configure(state='disabled')
        self.output.place(x=30, y=250)

        
    def load_files(self):
        with open('models/lemma_dict.pkl', 'rb') as f:
            lemma_dict = pickle.load(f)
        with open('models/noun_exception.pkl', 'rb') as f:
            noun_exception = pickle.load(f)
        return lemma_dict, noun_exception

    def clear_output(self):
        self.output.configure(state='normal')
        self.output.delete("1.0","end")
    
    def inflect_noun_singular(self,word):
        irregular_dict = pickle.load(open('../models/noun_exception.pkl','rb'))
        consonants = "bcdfghjklmnpqrstwxyz"
        vowels = "aeiou"
        word = str(word).lower()
        if len(word) < 2:
            return word
        if word in irregular_dict:
            return irregular_dict[word]
        if word.endswith('s'):
            if len(word) > 3:
                #Leaves, wives, thieves
                if word.endswith('ves'):
                    if len(word[:-3]) > 2:
                        return word.replace('ves','f')
                    else:
                        return word.replace('ves','fe')
                #Parties, stories
                if word.endswith('ies'):
                    return word.replace('ies','y')
                #Tomatoes, echoes
                if word.endswith('es'):
                    if word.endswith('ses') and word[-4] in vowels:
                        return word[:-1]
                    if word.endswith('zzes'):
                        return word.replace('zzes','z')
                    return word[:-2]
                if word.endswith('ys'):
                    return word.replace('ys','y')
                return word[:-1]
        return word

    def lemmatize(self, word, pos):
        if word in self.lemma_dict:
            if pos in self.lemma_dict[word]:
                return (word, self.lemma_dict[word][pos])
        if pos == 'NOUN':
            return (word, self.inflect_noun_singular(word))
        return (word, word)

    def tag(self):
        # print(self.input_cell)
        inputValue = self.input_cell.get("1.0",END).splitlines()
        print(inputValue)
        self.clear_output()
        for sent in inputValue:
            tmp = nlp(sent)
            lemma = [self.lemmatize(token.text, token.pos_) for token in tmp]
            self.output.insert(END, lemma)
            self.output.insert(END, '\n')
            

root = Tk()
root.geometry('700x500') 
root.title("POS Tagger")
tabControl = Notebook(root)
  
hmm = HMM(tabControl)
crf = CRF(tabControl)
lemma = Lemma(tabControl)

tabControl.pack(expand = 1, fill ="both")

root.mainloop()