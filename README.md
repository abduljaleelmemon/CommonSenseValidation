# CommonSenseValidation
Introduction
Recently utilizing natural language understanding systems to
common sense has received significant attention in research area.
Nevertheless, this kind of tasks remains pretty complicated to solve
because of the ambiguity which is an one of natural properties of
language and yet to achieve performance compared with human
understanding common sense.
COMMONSENSE VALIDATION AND EXPLANATION (COMVE) is a
challenge in which we have to perform task A. Validation of common
sense by providing two sentences e.g. “Whales are huge”, “All whales
are small”. The trained model should predict the right answer which is
“Whales are huge”. The dataset provided contain Train, Validation and
Test. 10000 dataset for training, 998 for Validation and 1000 for testing.
Due to high demand, there are many proposed approaches by
different researchers e.g. BERT, GPT, XLNet and moreover these models
are upgraded e.g. ALBert, RoBERTa. Which are trained on very large
datasets in GBs.

Approach
After going through research, I found out that BERT is preferred
model for common sense and so I used BERT model through
Transformers libraray. The libraries I have used are torch, transformer,
sklearn, numpy, pandas.
• Data processing
I have used BertTokenizer for turning texts into tokens.
Parameters: total_tk_size = 18
• FineTuning
I have added different layers into my model for better accuracy.
1. nn.linear to Apply a linear transformation to the incoming
data.
y = x*W^T + b, at first I did it for two sentences separately and
after combining both sentences through torch.cat, which
concatenates the given sequence of tensors in the given
dimension.
2. nn.SELU creates an activation layer
• Loss Function
loss(x1, x2, y)=max(0, −y∗(x1−x2) + margin)
Loss function is used to gauge the error between the prediction
output and the provided target value.
• Training and Validation
After the text is converted in required for BERT model e.g. in
tensors(tokens). There inputs are given in batch form and the
inputs are 1st sentence, 2nd sentence and 3rd labels. Which returns
predicted value and loss value as well of certain batch.
After the training of a batch, there is evaluation on validation
data. Following are the parameters used for training.
Hyper parameter:
Learning Rate = 2e-5,
Epochs = 10
Batch Size = 20,
Eps = 1e-8.
Using these parameters, I get an accuracy between 88 and 89.3.
After training the model I saved it in model.pt for the use of testing
• Testing
I load my saved model and pass my testing data into it by only
providing 1st sentence and 2nd sentence. Which outputs the two
tensors x and y.
1. subtracts y from x e.g. z = x – y
2. convert float array into integer by the condition
e.g. (A > 0).astype(int)
Accuracy

I get an accuracy of 89.3 percent on the test data provided following are the
screenshots attached.
1. codalab
2. On codlab

References: https://www.youtube.com/watch?v=mw7ay38--
ak&t=11s&ab_channel=AnalyticsVidhya

https://github.com/dnanhkhoa/pytorch-pretrained-
BERT/blob/master/pytorch_pretrained_bert/modeling.py

https://towardsdatascience.com/bert-explained-state-of-the-art-language-model-
for-nlp-f8b21a9b6270

https://ai.googleblog.com/2018/11/open-sourcing-bert-state-of-art-pre.html
