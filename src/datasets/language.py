"""Utils for language models."""

import re
import numpy as np
import json
import torch
import tqdm
import torch.utils.data as Data
# ------------------------
# utils for shakespeare dataset

ALL_LETTERS = "\n !\"&'(),-.0123456789:;>?ABCDEFGHIJKLMNOPQRSTUVWXYZ[]abcdefghijklmnopqrstuvwxyz}"
NUM_LETTERS = len(ALL_LETTERS)


def _one_hot(index, size):
    '''returns one-hot vector with given size and value 1 at given index
    '''
    vec = [0 for _ in range(size)]
    vec[int(index)] = 1
    return vec


def letter_to_index(letter):
    '''returns one-hot representation of given letter
    '''
    return ALL_LETTERS.find(letter)
    


def word_to_indices(word):
    '''returns a list of character indices

    Args:
        word: string
    
    Return:
        indices: int list with length len(word)
    '''
    indices = []
    for c in word:
        indices.append(ALL_LETTERS.find(c))
    return indices


# ------------------------
# utils for sent140 dataset


def split_line(line):
    '''split given line/phrase into list of words

    Args:
        line: string representing phrase to be split
    
    Return:
        list of strings, with each string representing a word
    '''
    return re.findall(r"[\w']+|[.,!?;]", line)


def _word_to_index(word, indd):
    '''returns index of given word based on given lookup dictionary

    returns the length of the lookup dictionary if word not found

    Args:
        word: string
        indd: dictionary with string words as keys and int indices as values
    '''
    if word in indd:
        return indd[word]
    else:
        return len(indd)


def line_to_indices(line, word2id, max_words=25):
    '''converts given phrase into list of word indices
    
    if the phrase has more than max_words words, returns a list containing
    indices of the first max_words words
    if the phrase has less than max_words words, repeatedly appends integer 
    representing unknown index to returned list until the list's length is 
    max_words

    Args:
        line: string representing phrase/sequence of words
        word2id: dictionary with string words as keys and int indices as values
        max_words: maximum number of word indices in returned list

    Return:
        indl: list of word indices, one index for each word in phrase
    '''
    unk_id = len(word2id)
    line_list = split_line(line) # split phrase in words
    indl = [word2id[w] if w in word2id else unk_id for w in line_list[:max_words]]
    indl += [unk_id]*(max_words-len(indl))
    return indl

def idx_to_embedding(idx,emb):

    embs=[emb[line] for line in idx]
    return np.array(embs)# [batch , seq_len , emb_dim]

def bag_of_words(line, vocab):
    '''returns bag of words representation of given phrase using given vocab

    Args:
        line: string representing phrase to be parsed
        vocab: dictionary with words as keys and indices as values

    Return:
        integer list
    '''
    bag = [0]*len(vocab)
    words = split_line(line)
    for w in words:
        if w in vocab:
            bag[vocab[w]] += 1
    return bag


def get_word_emb_arr(path):
    with open(path, 'r') as inf:
        embs = json.load(inf)
    vocab = embs['vocab']
    word_emb_arr = np.array(embs['emba'])
    indd = {}
    # print(len(vocab))
    for i in range(len(vocab)):
        indd[vocab[i]] = i
    vocab = {w: i for i, w in enumerate(embs['vocab'])}
    return word_emb_arr, indd, vocab


def val_to_vec(size, val):
    """Converts target into one-hot.

    Args:
        size: Size of vector.
        val: Integer in range [0, size].
    Returns:
         vec: one-hot vector with a 1 in the val element.
    """
    assert 0 <= val < size
    vec = [0 for _ in range(size)]
    vec[int(val)] = 1
    return vec

def shake_process_x(raw_x_batch):
    x_batch = [word_to_indices(word) for word in raw_x_batch]
    x_batch = np.array(x_batch)
    return x_batch

def shake_process_y(raw_y_batch):
    y_batch = [letter_to_index(c) for c in raw_y_batch]
    return y_batch

def sent_process_x(raw_x_batch,emb_arr,indd,max_words):
    x_batch = [e[4] for e in raw_x_batch]
    x_batch = [line_to_indices(e, indd, max_words) for e in x_batch]
    # x_batch = np.array(x_batch)
    return idx_to_embedding(x_batch,emb_arr)


def shakespeare(data_dir,spc,rs):
    trainx = torch.tensor([], dtype = torch.uint8)
    trainy = torch.tensor([], dtype = torch.uint8)
    testx = torch.tensor([], dtype = torch.uint8)
    testy = torch.tensor([], dtype = torch.uint8)
    try:
        trainx = torch.load(data_dir+'train/xdata.pt')
        trainy = torch.load(data_dir+'train/ydata.pt')
        user_groups = torch.load(data_dir+'train/user_groups.pt')
        testx  = torch.load(data_dir+'test/xdata.pt')
        testy = torch.load(data_dir+'test/ydata.pt')
        
    except: 
        # prepare training set
        user_groups = {}
        start = 0
        with open(data_dir+'train/data.json', 'r') as inf:
            data = json.load(inf)
        for n,u in enumerate(tqdm(data['users'])):
            temp_x = shake_process_x(data['user_data'][u]['x'])
            temp_y = shake_process_y(data['user_data'][u]['y'])
            # print(temp_y[0])
            trainx = torch.cat((trainx, torch.tensor(temp_x, dtype = torch.uint8)))
            trainy = torch.cat((trainy, torch.tensor(temp_y, dtype = torch.uint8)))
            user_groups[n]=np.arange(start,start+len(temp_x))
            start+=len(temp_x)
        # trainy = torch.argmax(trainy,1)
        torch.save(trainx, data_dir+'train/xdata.pt')
        torch.save(trainy, data_dir+'train/ydata.pt')
        torch.save(user_groups,data_dir+'train/user_groups.pt')

        # prepare test set
        with open(data_dir+'test/data.json', 'r') as inf:
            data = json.load(inf)
        for u in tqdm(data['users']):
            temp_x = shake_process_x(data['user_data'][u]['x'])
            temp_y = shake_process_y(data['user_data'][u]['y'])
            testx = torch.cat((testx, torch.tensor(temp_x, dtype = torch.uint8)))
            testy = torch.cat((testy, torch.tensor(temp_y, dtype = torch.uint8)))
        # testy = torch.argmax(testy,1)
        torch.save(testx, data_dir+'test/xdata.pt')
        torch.save(testy, data_dir+'test/ydata.pt')
    
    train_dataset = Data.TensorDataset(trainx.long(),trainy.long()) 
    test_dataset = Data.TensorDataset(testx.long(), testy.long())
    if spc>1:
        new_user_groups = {}
        remain_role = set(range(len(user_groups.keys())))
        i=0
        while len(remain_role)>=spc:
            idxs = []
            s = rs.choice(list(remain_role),spc,replace=False)
            remain_role-=set(s)
            for r in s:
                idxs.append(user_groups[r])
            new_user_groups[i]=np.concatenate(idxs,0)
            i+=1
        user_groups=new_user_groups
    return train_dataset,test_dataset,user_groups

def sent140(data_dir,spc,rs):
    emb_arr,indd,_ = get_word_emb_arr(data_dir+'embs.json')
    trainx = torch.tensor([], dtype = torch.uint8)
    trainy = torch.tensor([], dtype = torch.uint8)
    testx = torch.tensor([], dtype = torch.uint8)
    testy = torch.tensor([], dtype = torch.uint8)
    try:
        trainx = torch.load(data_dir+'train/xdata.pt')
        trainy = torch.load(data_dir+'train/ydata.pt')
        user_groups = torch.load(data_dir+'train/user_groups.pt')
        testx  = torch.load(data_dir+'test/xdata.pt')
        testy = torch.load(data_dir+'test/ydata.pt')
        
    except: 
        # prepare training set
        user_groups = {}
        start = 0
        with open(data_dir+'train/data.json', 'r') as inf:
            data = json.load(inf)
        for n,u in enumerate(tqdm(data['users'])):
            temp_x = sent_process_x(data['user_data'][u]['x'],emb_arr,indd,25)
            temp_y = data['user_data'][u]['y']
            # print(temp_y[0])
            trainx = torch.cat((trainx, torch.tensor(temp_x)))
            trainy = torch.cat((trainy, torch.tensor(temp_y, dtype = torch.uint8)))
            user_groups[n]=np.arange(start,start+len(temp_x))
            start+=len(temp_x)
        # trainy = torch.argmax(trainy,1)
        torch.save(trainx, data_dir+'train/xdata.pt')
        torch.save(trainy, data_dir+'train/ydata.pt')
        torch.save(user_groups,data_dir+'train/user_groups.pt')

        # prepare test set
        with open(data_dir+'test/data.json', 'r') as inf:
            data = json.load(inf)
        for u in tqdm(data['users']):
            temp_x = sent_process_x(data['user_data'][u]['x'],emb_arr,indd,25)
            temp_y = data['user_data'][u]['y']
            testx = torch.cat((testx, torch.tensor(temp_x)))
            testy = torch.cat((testy, torch.tensor(temp_y, dtype = torch.uint8)))
        # testy = torch.argmax(testy,1)
        torch.save(testx, data_dir+'test/xdata.pt')
        torch.save(testy, data_dir+'test/ydata.pt')
    
    train_dataset = Data.TensorDataset(trainx.float(),trainy.long()) 
    test_dataset = Data.TensorDataset(testx.float(), testy.long())
    if spc>1:
        new_user_groups = {}
        remain_role = set(range(len(user_groups.keys())))
        i=0
        while len(remain_role)>=spc:
            idxs = []
            s = rs.choice(list(remain_role),spc,replace=False)
            remain_role-=set(s)
            for r in s:
                idxs.append(user_groups[r])
            new_user_groups[i]=np.concatenate(idxs,0)
            i+=1
        user_groups=new_user_groups
    return train_dataset,test_dataset,user_groups
