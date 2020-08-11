# Universal-Sentiment-Analyzer

This project was done with the help of "Streamlit" framework to make simplistic data app based on Python.

Things we have tried to cover:

    The whole idea is divided into 4 major parts : "Data Analysis", "Tokenization", "Visualization" and "Modeling" and an "About" Section.
    There are 3 datasets included within,namely, Movie Review Dataset, Spam-Ham dataset and Amazon Reviews dataset. 
    At any point you can upload your own CSV file of dataset.

1. Data Analysis:

    To make the app run faster, we have subsetted the data to include only 3000 rows (Shuffled).
    To make the full use of it, you can go to github link included in the app and run it locally through instructions from README file.
    This part is included to get a sense of dataset, view data, view shape of data and even plot the distribution of Positive and negative reviews.

2. Tokenization:

    This part is included to tokenize the reviews/sentences and see the changes with the help of just few clicks.
    Tokenization methods included : Alpha only, Lowercase only, Stopwords, Lemmatization.(At this point, entities are not supported in this app.)
    To make this work, you have to select the column containing REVIEWS/TEXT. Selecting a column with boolean value will result in an error.

3. Vectorization :

    This is the most complicated part of this app.
    We have included 2 parts to this: 
        a) Basic Vectors : CV, TFIDF, NMF, LDA and you can perform the following with these:
            Find "Top words" given an Index of Word you are interested in
            Do "Topic Modeling" given an Index and No of Topics you want to see. - Added option to see Cosine Similarity for a particular Topic
        b) Advance vectors : Word2Vec, Glove Vectors with following options:
            Find Closest Embeddings given an Index
            Find Closest Words to a Combination of Words ("king"-"man"+"woman"). For this, you have to select 3 different indexes
            Find N-pairs of words and see how to look like in Array/Table form - Just for getting Insights
            Plot the Word embedding on a "Scatter Plot"

4. Modeling: 

    For Modeling purposes, there are 4 options:
        SV Classifier
        Naive Bayes Classifier
        Random Forest Classifier
        Gradient Boosting Classifier
    These 4 models have further 4 options. You can train, validate and test on different pipeline to see Accuracy differences.
        CV Pipeline
        TFIDF Pipeline
        NMF Pipeline
        LDA Pipeline

Limitations:

    Our dataset requirement is : There must be 2 Columns, first one as Label and Second as Reviews. 
    We are subsetting any dataset that you upload for 3000 Rows.
    There is no option right now to choose to add or remove Entities in Tokenization
    For Glove vectors, we have used Spacy and currently spacy doesn't let us install "en_core_web_sm" module to be installed on a server using a "pip"/PYPI command. To remove this limitation, whenever you choose Glove vectors, we have given an option to check the existence of module and install in case it doesn't exist.

Future Work:

    We will work on including Models like LSTM, RNN etc
    Exploration in the area of different Tokenization methods
    Give user an option to Enter an review in "Text area" and see the score for Sentiment using any of the models that he has used above for training and validating.
