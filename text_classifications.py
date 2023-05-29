__author__ = "Dilnoza Saidova"
__date__ = "2023/02/15"
import sys
import os
import pandas as pd
import xml.etree.ElementTree as et
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB

# Input and output paths are taken from terminal.
input_path = sys.argv[2]

# Validate input path.
input_path_valid = os.path.exists(input_path)
if not input_path_valid:
    print("Invalid input file or file not found!")
else:
    # Training data path is set up.
    train_path = "/data/training/text/"
    os.chdir(train_path)

    # Create a vocabulary - consists of all the words that
    # occur in any of the texts in training data.
    train_vocab = {"userid": [], "text": []}
    for file in os.listdir():
        if file.endswith(".txt"):
            train_text_file = open(f"{train_path}{file}", "r", encoding='iso-8859-15')
            train_vocab["userid"].append(file[:-4])
            train_vocab["text"].append(train_text_file.read())
            train_text_file.close()

    test_df = pd.read_csv("/data/training/profile/profile.csv")
    temp1 = pd.merge(test_df, pd.DataFrame(train_vocab), on='userid')
    temp2 = pd.DataFrame(temp1.groupby('userid')['text'].apply(list).reset_index(name='text'))
    merged_test_df = pd.merge(test_df, temp2, on=['userid'])

    count_vect = CountVectorizer()
    clf = MultinomialNB()
    x_train = count_vect.fit_transform(merged_test_df['text'])
    y_train = merged_test_df['gender']
    clf.fit(x_train, y_train)

    # Testing data path is set up.
    test_path = input_path + "/text/"
    os.chdir(test_path)

    # Create a vocabulary - consists of all the words that
    # occur in any of the texts in testing data.
    test_vocab = {"userid": [], "text": []}
    for file in os.listdir():
        if file.endswith(".txt"):
            test_text_file = open(f"{test_path}{file}", "r", encoding='iso-8859-15')
            test_vocab["userid"].append(file[:-4])
            test_vocab["text"].append(test_text_file.read())
            test_text_file.close()

    public_df = pd.DataFrame(test_vocab)
    public_csv = pd.read_csv(input_path + "/profile/profile.csv")
    temp1 = pd.DataFrame(pd.merge(public_csv, public_df, on='userid').
                         groupby('userid')['text'].apply(list).reset_index(name='text'))
    merged_public_df = pd.merge(public_csv, temp1, on=['userid'])

    # Combine lists from 2D list of Strings into one to
    # implement CountVectorizer() for model training.
    for i in merged_public_df.index:
        merged_public_df.at[i, 'text'] = ' '.join(merged_public_df.at[i, 'text'])

    x_test = count_vect.transform(merged_public_df['text'])
    y_predicted = clf.predict(x_test)
    merged_public_df['gender'] = y_predicted

    for cnt in merged_public_df.index:
        user = et.Element("user")
        user.set("\nid", merged_public_df['userid'][cnt])
        user.set("\nage_group", "xx-24")
        user.set("\ngender", "female" if merged_public_df['gender'][cnt] == 1 else "male")
        user.set("\nextrovert", "3.49")
        user.set("\nneurotic", "2.73")
        user.set("\nagreeable", "3.58")
        user.set("\nconscientious", "3.45")
        user.set("\nopen", "3.91")

        output_file = open(sys.argv[4] + "/" + merged_public_df['userid'][cnt] + ".xml", "w")
        output_file.write(et.tostring(user).decode())
        output_file.close()
