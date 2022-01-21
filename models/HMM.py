
# Importing libraries
import pandas as pd
import numpy as np


def read(file):
    with open(file, 'r', encoding='utf8') as f:
        file = f.read().splitlines()
    data = [[] for _ in range(len(file))]
    for idx, i in enumerate(file):
        a = i.split()
        for j in a:
            tmp = (j.rsplit('/',1))
            data[idx].append((tmp[0], tmp[1]))
    return data

def preprocess(train_set):
    train_tagged_words = [tup for sent in train_set for tup in sent]
    tags = {tag for _, tag in train_tagged_words}
    return train_tagged_words, tags

def create_vocab(train, test):
    train_tagged_words = [tup for sent in train for tup in sent]
    test_tagged_words = [tup[0] for sent in test for tup in sent]
    tags = {tag for _, tag in train_tagged_words}
    return train_tagged_words, test_tagged_words, tags


def word_given_tag(word,tag,train_bag):
    taglist = [pair for pair in train_bag if pair[1] == tag]
    tag_count = len(taglist) + 16
    w_in_tag = [pair[0] for pair in taglist if pair[0]==word]    
    word_count_given_tag = len(w_in_tag)    
    
    return (word_count_given_tag,tag_count)

def t2_given_t1(t2,t1,train_bag):
    tags = [pair[1] for pair in train_bag]
    t1_tags = [tag for tag in tags if tag==t1]
    count_of_t1 = len(t1_tags) + 16
    t2_given_t1 = [tags[index+1] for index in range(len(tags)-1) if tags[index] == t1 and tags[index+1] == t2]  
    count_t2_given_t1 = len(t2_given_t1) + 1
    return(count_t2_given_t1,count_of_t1)

def transition_matrix(tags, train_tagged_words):
    tags_matrix = np.zeros((len(tags), len(tags)), dtype='float32')
    for i, t1 in enumerate(list(tags)):
        for j, t2 in enumerate(list(tags)): 
            tags_matrix[i, j] = t2_given_t1(t2, t1, train_tagged_words)[0]/t2_given_t1(t2, t1, train_tagged_words)[1]
    tags_df = pd.DataFrame(tags_matrix, columns = list(tags), index=list(tags))
    return tags_df

def Viterbi(words, tags_df, train_bag):
    state = []
    T = list(set([pair[1] for pair in train_bag]))
    for key, word in enumerate(words):
        #initialise list of probability column for a given observation
        p = [] 
        for tag in T:
            if key == 0:
                transition_p = tags_df.loc['PUNCT', tag]
                # continue
            else:
                transition_p = tags_df.loc[state[-1], tag]
            # compute emission and state probabilities
            emission_p = word_given_tag(words[key], tag, train_bag)[0]/word_given_tag(words[key], tag, train_bag)[1]
            state_probability = emission_p * transition_p    
            p.append(state_probability)
            
        pmax = max(p)
        # getting state for which probability is maximum
        state_max = T[p.index(pmax)] 
        state.append(state_max)
    return list(zip(words, state))


def test(test_set, test_tagged_words, train_tagged_words, tags_df):
    # list of tagged words
    test_run_base = [tup for sent in test_set for tup in sent]

    # list of untagged words
    test_tagged_words = [tup[0] for sent in test_set for tup in sent]
    tagged_seq = Viterbi(test_tagged_words, tags_df, train_tagged_words)
    check = [i for i, j in zip(tagged_seq, test_run_base) if i == j]
    vanilla_viterbi_accuracy = len(check)/len(tagged_seq)
    return vanilla_viterbi_accuracy

# print("The accuracy of the Vanilla Viterbi Algorithm is -", vanilla_viterbi_accuracy)

def load_train_file(train_set):
    data = read(train_set)
    train_tagged_words, tags = preprocess(data)
    tags_df = transition_matrix(tags, train_tagged_words)
    return train_tagged_words, tags_df


def compute(train_set, test_set):
    train_tagged_words, test_tagged_words, tags = create_vocab(train_set, test_set)
    tags_df = transition_matrix(tags, train_tagged_words)
    vanilla_viterbi_accuracy = test(test_set, test_tagged_words, train_tagged_words, tags_df)
    return vanilla_viterbi_accuracy