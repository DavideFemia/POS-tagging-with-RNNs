import seaborn as sns
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from IPython.display import display, Markdown, Latex
from . import utils

def get_worst_errors(prediction_info, tags, plt_size=(14,18), title=""):
  '''
  prediction_info: dictionary with f1 and accuracy
  tags: pos tags used (excluded punctuation and symbols)
  '''
  f1_scores = prediction_info['f1']
  f1_scores, tags = zip(*sorted(zip(f1_scores, tags)))
  df = pd.DataFrame([f1_scores,tags]).T
  df.columns=['F1','TAG']
  disp = sns.barplot(data= df, x="TAG", y="F1")
  
  for item in disp.get_xticklabels():
    item.set_rotation(45)
  
  ax = plt.gca()
  sns.set(rc={'figure.figsize':(plt_size[0],plt_size[1])})
  disp.plot(ax=ax)
  plt.title(title)
  plt.show()
  return f1_scores


def print_worst_sentences(pos_tokenizer, y_pred, y_test, test_data, n_sentences = 4):
    '''
    In this implementation, we display the sentences containing the preicted tags 
    having the maximum absolute difference with respect to the gold tagging
    
    pos_tokenizer: the pos tags tokenizer
    
    '''

    # obtaining flat worst sentence indexes
    diff = abs(y_test - y_pred)
    diff_shape = diff.shape
    diff = diff.flatten()
    flattened_worst_sentences_indexes = np.argsort(diff)[::-1][:n_sentences]

    worst_sentence_indexes = []

    # de-flattening the worst sentence indexes
    for i in flattened_worst_sentences_indexes:
        computed_index = np.unravel_index(i, diff_shape)
        worst_sentence_indexes.append(computed_index)

    # displayng results
    for i in range(n_sentences):
        print(f"Sentence n.{worst_sentence_indexes[i][0]}, Word n. {worst_sentence_indexes[i][1]}")

        sentence = test_data['text'].values[worst_sentence_indexes[i][0]]
        gold_tags = test_data['tag'].values[worst_sentence_indexes[i][0]]
        predicted_tags = [np.argmax(i) for i in y_pred[worst_sentence_indexes[i][0]]]
        predicted_tags = utils.convert_ids_to_tags(pos_tokenizer, predicted_tags, len(gold_tags.split(' ')))

        display(Markdown('**Correct tagging**'))
        gold_tagging = utils.print_tagging(sentence, gold_tags)
        display(Markdown('**Predicted tagging**'))
        predicted_tagging = utils.print_tagging(sentence, predicted_tags)

        print("_____________________________________________________________ \n \n")
        
    return worst_sentence_indexes