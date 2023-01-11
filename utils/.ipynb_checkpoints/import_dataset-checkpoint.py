#pip install gensim==3.8.3
#pip install pydot
#pip install pydotplus
#pip install graphviz

from IPython.display import display, Markdown, Latex
import os, shutil
import sys 
import pandas as pd
import numpy as np
from urllib import request
from zipfile import ZipFile
import string
import gensim
from tensorflow import keras
from sklearn.metrics import ConfusionMatrixDisplay
from functools import partial
import random


def download_dataset(download_path: str, url: str):
    if not os.path.exists(download_path):
        print("Downloading dataset...")
        request.urlretrieve(url, download_path)
        print("Download complete!")

def extract_dataset_zip(download_path: str, extract_path: str):
  print("Extracting dataset... (it may take a while...)")
  # loading the temp.zip and creating a zip object
  with ZipFile(download_path, 'r') as loaded_zip:
      loaded_zip.extractall(path=extract_path)
  print("Extraction completed!")

def download_treebank_dataset(out_folder, dataset_name = "dependency_treebank"):
  print(f"Current work directory: {os.getcwd()}")
  dataset_folder = os.path.join(os.getcwd(), out_folder)

  if not os.path.exists(dataset_folder):
      os.makedirs(dataset_folder)

  url = 'https://raw.githubusercontent.com/nltk/nltk_data/gh-pages/packages/corpora/dependency_treebank.zip'
  dataset_path = os.path.join(dataset_folder, dataset_name+".zip")
  print(dataset_path)

  download_dataset(dataset_path, url)
  extract_dataset_zip(dataset_path, dataset_folder)

def split_and_pickle_treebank_dataset(validation_start_index: int, test_start_index: int, dataset_name = "dependency_treebank"):
  '''

  In this implementation,
  
  0. the original dataset has to be already downloaded with the 
     'download_treebank_dataset' function
  1. the dataset is splitted in train, validation and test
  2. the dataset is locally stored as pickle file

  params:
  
  validation_start_index: the index of the first element of the validation set 
                          (the index of the last element of the train set + 1) 
  test_start_index: the index of the first element of the test set 
                    (the index of the last element of the validation set + 1) 

  '''
  folder = os.path.join(os.getcwd(), "Datasets", dataset_name)
  dataframe_rows_doc = []
  dataframe_rows_sent = []
  index_to_show = []
  for filename in os.listdir(folder):
    text = []
    text_tag = []
    text_sent = []
    text_sent_tag = []
    file_path = os.path.join(folder,filename)
    file_id = int(filename.split("_")[1].split(".")[0])
    file_split = 'train' if file_id<101 else ('validation' if file_id<151 else 'test')
    sentence_id = 1
    try:
      if os.path.isfile(file_path):
        with open(file_path, mode='r', encoding='utf-8') as text_file:
          lines = text_file.readlines()
          for line in lines:
            if line != '\n':
              split = line.split()
              text.append(split[0])
              text_sent.append(split[0])
              text_tag.append(split[1])
              text_sent_tag.append(split[1])
            else:
              dataframe_rows_sent.append({
                      "file_id": file_id,
                      "sentence_id": sentence_id,
                      "split": file_split,
                      "text": " ".join(text_sent),
                      "tag": " ".join(text_sent_tag)
                  })
              text_sent = []
              text_sent_tag = []
              sentence_id += 1
    except Exception as e:
      print('Failed to process %s. Reason: %s' % (file_path, e))
      sys.exit(0)

    dataframe_rows_doc.append({
                      "file_id": file_id,
                      "split": file_split,
                      "text": " ".join(text),
                      "tag": " ".join(text_tag)
                  })
    if len(text_sent)!=0:
      dataframe_rows_sent.append({
                        "file_id": file_id,
                        "sentence_id": sentence_id,
                        "split": file_split,
                        "text": " ".join(text_sent),
                        "tag": " ".join(text_sent_tag)
                    })

  doc_folder = os.path.join(os.getcwd(), "Datasets", "Dataframes", dataset_name+"_document")
  if not os.path.exists(doc_folder):
      os.makedirs(doc_folder)

  sent_folder = os.path.join(os.getcwd(), "Datasets", "Dataframes", dataset_name+"_sentence")
  if not os.path.exists(sent_folder):
      os.makedirs(sent_folder)

  # transform the list of rows in a proper dataframe
  dataframe_path = os.path.join(doc_folder, dataset_name + ".pkl")
  df = pd.DataFrame(dataframe_rows_doc)
  df = df[["file_id", "split", "text", "tag"]]


  df.sort_values(by=['file_id'], inplace=True)
  df.index = pd.Series(list(range(1,df.shape[0]+1)))

  df.to_pickle(dataframe_path)

  dataframe_path = os.path.join(sent_folder, dataset_name + ".pkl")
  df_sent = pd.DataFrame(dataframe_rows_sent)
  df_sent = df_sent[["file_id", "sentence_id", "split", "text", "tag"]]


  df_sent.sort_values(by=['file_id','sentence_id'], inplace=True)
  df_sent.index = pd.Series(list(range(1,df_sent.shape[0]+1)))

  df_sent.to_pickle(dataframe_path)

def import_local_treebank_df():
  '''
  In this implementation, the dataset (in pkl format) is imported from the 
  'Datasets' folder into the current enviroment.

  returns: The treebank dataset pandas dataframe
  '''
  dataset_name = "dependency_treebank"
  #split = 'document'
  split = 'sentence'
  dataframe_path = os.path.join(os.getcwd(), "Datasets", "Dataframes", "_".join((dataset_name,split)), dataset_name + ".pkl")
  df = pd.read_pickle(dataframe_path)
  return df
