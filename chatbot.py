import numpy as np
import tensorflow as tf
import re
import time

#importing dataset

lines = open('movie_lines.txt',encoding='utf-8',errors='ignore').read().split('\n')
conversations = open('movie_conversations.txt',encoding='utf-8',errors='ignore').read().split('\n')


#creating a dic to map each ID
id2line = {}

for line in lines:
    _line = line.split(' +++$+++ ')
    if len(_line) == 5:
        #print(_line)
        id2line[_line[0]] = _line[4]


#printing dict 
#print(id2line)

#creating a list of all the conversations
conversations_ids = []

for conversation in conversations:
    _conversation = conversation.split(' +++$+++ ')[-1][1:-1].replace("'","").replace(" ","")
    conversations_ids.append(_conversation.split(','))

#questions and answers in a seperate list
questions = []
answers = []

#seperating 
for conversation in conversations_ids:
    for i in range(len(conversation) - 1):
        questions.append(id2line[conversation[i]])
        answers.append(id2line[conversation[i+1]])


#print(questions)
#print(answers)


#cleaning function
def clean_text(text):
    text = text.lower()
    text = re.sub(r"i'm","i am",text)
    text = re.sub(r"he's","he is",text)
    text = re.sub(r"she's","she is",text)
    text = re.sub(r"that's","that is",text)
    text = re.sub(r"what's","what is",text)
    text = re.sub(r"where's","where is",text)
    text = re.sub(r"\'ll"," will",text)
    text = re.sub(r"\'ve"," have",text)
    text = re.sub(r"\'re"," are",text)
    text = re.sub(r"\'d"," would",text)
    text = re.sub(r"won't","will not",text)
    text = re.sub(r"can't","cannot",text)
    text = re.sub(r"[-()\"\',.?:;+=-@#<>/|]","",text)
    return text


#cleaning the questions
clean_questions = []
for question in questions:
    clean_questions.append(clean_text(question))

#cleaning answers
clean_answers = []
for answer in answers:
    clean_answers.append(clean_text(answer))
#print(clean_questions)


#create a dict which maps each word to its frequency
word2count = {}

#question word counts
for question in clean_questions:
        for word in question.split():
                if word not in word2count:
                        word2count[word] = 1
                else:
                        word2count[word] += 1




#answer word counts
for answer in clean_answers:
        for word in answer:
                if word not in word2count:
                        word2count[word] = 1
                else:
                        word2count[word] += 1

#print(word2count)


#assigning a unique integer value to the words which frequecy is greater than the threshold value
threshold = 20

#question words to int
questionwordstoint = {}
word_number = 0
for word,num in word2count.items():
        if num >= threshold:
                questionwordstoint[word] = word_number
                word_number += 1


#answer word to int

answerwordtoint = {}
word_number = 0
for word,num in word2count.items():
        if num >= threshold:
                answerwordtoint[word] = word_number
                word_number += 1


#print(questionwordstoint)

#builiding a tokens
tokens = ['<PAD>','<EOS>','<OUT>','<SOS>']

for token in tokens:
        questionwordstoint[token] = len(questionwordstoint) + 1


for token in tokens:
        answerwordtoint[token] = len(questionwordstoint) + 1



#creating inverse of answerwordstoint 
answersinttowords = {w_i:w for w,w_i in answerwordtoint.items()}


#print(answersinttowords)

#adding <EOS> to end of each string
for i in range(len(clean_answers)):
        clean_answers[i] += ' <EOS>'


questions_into_int = []

for question in clean_questions:
        ints = []
        for word in question.split():
                if word not in questionwordstoint:
                        ints.append(questionwordstoint['<OUT>'])
                else:
                        ints.append(questionwordstoint[word])
        questions_into_int.append(ints)


#print(questions_to_int)
answers_into_int = []

for answer in clean_answers:
        ints = []
        for word in question.split():
                if word not in questionwordstoint:
                        ints.append(answerwordtoint['<OUT>'])
                else:
                        ints.append(answerwordtoint[word])
        answers_into_int.append(ints)
#print(answers_into_int)



sorted_clean_questions = []
sorted_clean_answers = []


#sorted clean questions
for length in range(1,25+1):
        for i in enumerate(questions_into_int):
                if len(i[1]) == length:
                        sorted_clean_questions.append(questions_into_int[i[0]])
                        sorted_clean_answers.append(answers_into_int[i[0]])
#print(len(questions_into_int))
#print(len(sorted_clean_questions))


#creating a placeholders for inputs and targets
def model_inputs():
        inputs = tf.placeholder(tf.int32,[None,None],'inputs')
        targets = tf.placeholder(tf.int32,[None,None],'targets')
        lr = tf.placeholder(tf.float32,'learning_rate')
        keep_prob = tf.placeholder(tf.float32,'keep_prob')

        return inputs,targets,lr,keep_prob

#processing layer for targets
def preprocess_targets(targets,answerwordstoint,batch_size):
        left_side = tf.fill([batch_size,1],answerwordstoint['<SOS>'])
        right_side = tf.strided_slice(targets,[0,0],[batch_size,-1],[1,1])
        preprocessed_targets = tf.concat([left_side,right_side],1)
        return preprocessed_targets