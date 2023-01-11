import pandas as pd
import numpy as np
from IPython.display import display, Markdown, Latex
import matplotlib.pyplot as plt
from collections import Counter
from collections import Counter
from pylab import MaxNLocator

def print_tagging(texts, texts_tag):
  '''
  texts: 1d list of words to be tagged
  texts_tag: 1d list of word tags
  '''

  splitted_text = texts.split(" ") 
  splitted_tag = texts_tag.split(" ")
  texts_row = pd.DataFrame([[splitted_text[k] for k in range(len(splitted_text))]])
  tags = [[splitted_tag[k] for k in range(len(splitted_tag))]]
  tag_row = pd.DataFrame(tags)

  frames = [texts_row, tag_row]
  tagged_texts_df = pd.concat(frames)
  with pd.option_context('display.max_rows', None,
                       'display.max_columns', None,
                       'display.precision', 3,
                       ): display(tagged_texts_df)

  return tags[0]
    


def hist_plot(dataset_train, dataset_val, dataset_test, y_lim=None,plt_size=(16,8)):

  train_tags = list(Counter(dataset_train).keys())

  train_tag_counts = [0]*len(train_tags)
  val_tag_counts = [0]*len(train_tags)
  test_tag_counts = [0]*len(train_tags)
  
  for i in range(len(train_tags)):
    for j in dataset_train:
        if train_tags[i] == j:
            train_tag_counts[i] +=1
    for j in dataset_val:
        if train_tags[i] == j:
            val_tag_counts[i] +=1
    for j in dataset_test:
        if train_tags[i] == j:
            test_tag_counts[i] +=1

  x = np.arange(0, len(train_tags)*3, 3)  # the label locations
  width = 1.6  # the width of the bars

  fig, ax = plt.subplots(figsize=plt_size)
  rects1  = plt.bar(x - width/2,train_tag_counts, label="train set",align='center')
  rects2  = plt.bar(x,val_tag_counts, label="val set",align='center')
  rects3 = plt.bar(x + width/2,test_tag_counts, label="test set",align='center')

  ax.set_xticks(x, train_tags)
  ax.autoscale(tight=True)
  ya = ax.get_yaxis()
  ya.set_major_locator(MaxNLocator(integer=True))

  if y_lim:
      plt.ylim(0,y_lim)
  plt.xticks(rotation='vertical')
  plt.legend(loc='upper center')
  plt.show()


    
def obtain_remove_kept_tags(pos_tokenizer, punctuation_tags, predicted_values, gold_labels):
  '''
  This function is useful to compute the right metric(we have to specify the 
  column and the row to drop from the cm) and also the labels associated with 
  each row and column in the cm_plot
  '''
  remove_classes = [0]+[pos_tokenizer.vocab[p] for p in punctuation_tags] # 0,7-8,23-26
  classes = np.union1d(np.unique(np.argmax(gold_labels, axis=-1)),np.unique(np.argmax(predicted_values, axis=-1))) # here we choose the right classes
  #remove_classes = [np.where(classes==el)[0][0] for el in remove_classes if el in classes] # here we choose the right removable classes thanks to the just computed right classes
  classes = [el for el in classes if el not in remove_classes] # and here we remove the padding and the punctuations classes from them
  kept_classes = [pos_tokenizer.tokenizer.index_word[idx] for idx in classes]

  return remove_classes, kept_classes


def count_different_elements(list1, list2):
    count = 0
    indexes = []
    for i in range(len(list1)):
        if list1[i] != list2[i]:
            count +=1
            indexes.append(i)
            
    return count, indexes

def convert_ids_to_tags(dictionary, id_sequence, length = None):
    tags = []
    for s in id_sequence:
        tags.append({i for i in dictionary if dictionary[i]==s}.pop())
    
    if length is not None:
        tags = tags[:length]
    
    return ' '.join(tags)

def plot_sentence_length_dist(texts, tokenizer, quantile, dataset_name):
    sentence_length = [len(seq) for seq in tokenizer.convert_tokens_to_ids(texts)]
    lengths = Counter(sentence_length).keys()
    counts = Counter(sentence_length).values()
    quantile_line = int(np.quantile(sentence_length , quantile))

    lengths, counts = zip(*sorted(zip(lengths, counts)))
    plt.plot(lengths,counts, marker='o',  markersize=2,label="sequence length distribution")
    plt.axvline(x=quantile_line, color='r', label=f"quantile = {quantile}")
    plt.xlabel("sentence length")
    plt.ylabel("n. of samples")
    plt.title("Sentence length in the "+dataset_name)
    plt.legend()
    plt.show()



