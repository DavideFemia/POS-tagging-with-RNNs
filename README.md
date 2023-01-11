# POS-tagging-with-RNNs :bookmark_tabs:  :arrow_right: :id:

In this assignment we address the task of POS tagging on the well known *Penn Treebank Dataset*. The aim is to implement and compare the performances of classic recurrent models which make use of *GloVe* embeddings to later choose the best two among them and analyze their errors. The obtained results are quite satisfactory, even considering the reduced size of the model (in terms of number of parameters).

____

We've implemented, trained, validated and tested the 4 following models:

-  BiLSTM (baseline): this model has just a bi-directional layer of LSTM cells, in addition to the dense input layer (followed by the fixed embedding layer) and the dense time-distributed output layer <img src="https://raw.githubusercontent.com/DavideFemia/POS-tagging-with-RNNs/main/img/BiLSTM.png" alt="MarineGEO circle logo" style="height: 100px; width:100px;"/>
-  Double BiLSTM: Similar to the baseline, but with an additional bidirectional LSTM layer (Figure \ref{fig:doublebilstm}).
-  BiLSTM + Dense: Similar to the baseline, but with an additional Dense distributed layer before the output layer (Figure \ref{fig:lstm_dense}).
-  BiGRU: this model is similar to the baseline, but with a bi-directional GRU layer instead of a bi-directional LSTM layer (Figure \ref{fig:bigru}).



## Instructions

- Run ```Assignment_1.ipynb``` to visualize and/or reproduce our train-validation-test-error analysis pipeline :green_book:
- Read ```assignment_1.pdf``` to visualize the report :scroll:
