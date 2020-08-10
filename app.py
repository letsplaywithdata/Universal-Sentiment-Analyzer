import streamlit as st
import pandas as pd
import io
import time
#from spacy.cli import download
import sys
import subprocess
import pkg_resources

#required = {'spacy', 'scikit-learn', 'numpy', 'pandas','en_core_web_sm'}
#installed = {pkg.key for pkg in pkg_resources.working_set}
#missing = required - installed

#if missing:
#    python = sys.executable
#    subprocess.check_call([python, '-m', 'pip', 'install', *missing], stdout=subprocess.DEVNULL)

import spacy
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.decomposition import NMF, LatentDirichletAllocation
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import normalize
#import spacy spacy.load('en_core_web_sm')
from spacy.lang.en import English
import numpy as np
from scipy import spatial
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from sklearn.pipeline import Pipeline
from mlxtend.preprocessing import DenseTransformer
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.decomposition import NMF, LatentDirichletAllocation
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.manifold import TSNE
from gensim.models import Word2Vec
from pylab import rcParams
from PIL import Image 
import matplotlib.pyplot as plt
en = English()

#nlp = download('en')
#nlp1 = spacy.load('en')
#nlp = spacy.load('en_core_web_sm')
def explore_data(dataset):
    if dataset == "Movie Reviews":
        df = pd.read_csv('data/movie_reviews.csv')
    elif dataset == "Spam":
        df = pd.read_csv('data/spam_processed.csv')
    elif dataset == "Amazon Reviews":
        df = pd.read_csv('data/amazon_subset.csv')
    else:
        st.set_option('deprecation.showfileUploaderEncoding', False)
        data_file = st.file_uploader("Upload CSV",type=['csv'])
        if data_file is not None:
            df = pd.read_csv(data_file)
    return df

def main():
    """Sentiment Analysis App """

    st.title("Universal Sentiment Analyzer")

    activities = ["Data Analysis","Tokenization","Vectorization", "Modeling", "About"]
    choice = st.sidebar.radio("Select from the Following: ",activities)

    #data = st.selectbox("Select dataset to work on", ["Iris", "Spam", "UploadCSV"])
    #dataset= explore_data(data)
    if choice == 'Data Analysis':
        st.header("Data Analysis")
        data_options = ["Movie Reviews","Spam","Amazon Reviews","UploadCSV"]
        data = st.selectbox("Select dataset to work on", data_options)
        #st.text("We will subset your dataset to include 1000 rows")
        #st.text("To work on full dataset, go to below link:")
        #st.text("https://github.com/letsplaywithdata/Universal-Sentiment-Analyzer")
        dataset = explore_data(data)
        # adding this code lines only for online usage
        st.info("We are subsetting your dataset for maximum 3000 Rows")
        if st.button("Why?"):
            st.write("To make this run faster on this server we have taken this step")
            st.write("To work on full dataset, go to below website and run locally")
            st.write("https://github.com/letsplaywithdata/Universal-Sentiment-Analyzer")
        dataset= subsetcolumns_true(dataset)
        #data = st.selectbox(
        #    "Select dataset to work on", ["Iris", "Spam", "UploadCSV"])
        #dataset = explore_data(data)
        dataanalysisfunc(dataset)

    if choice == 'Tokenization':
        st.header("Let's Tokenize our Data")
        #data = st.selectbox(
        #    "Select dataset to work on", ["Iris", "Spam", "UploadCSV"])
        #dataset = explore_data(data)
        data_options = ["Movie Reviews","Spam","Amazon Reviews","UploadCSV"]
        data = st.selectbox("Select subset dataset to work on", data_options)
        dataset= explore_data(data)
        dataset= subsetcolumns_true(dataset)        
        Column_options = columnselector(dataset)
        if 0 in Column_options.unique():
            st.error("Select Column with Text/Reviews for further process!")
        #if st.button("Show Column"):
        else:
            st.write(Column_options)
        methods = ["Lowercase", "Alpha_Only", "Lemmatize", "Stop_Words"]
        st.warning("Currently we are not supporting Entities to play around with!")
        token_options= st.multiselect("Tokenization methods to perform", methods, default= 
                    ["Lowercase","Alpha_Only"])
        st.write(f"You choose following options :{token_options}")
        if st.button("Let's Tokenize"):
            tokenized = tokenize_full(Column_options, token_options, showtable=True)
            new_df = pd.DataFrame(tokenized,columns=['Modified Sentences'])
            st.dataframe(new_df)
            st.success("Success. Let's explore Vectors. Select an option from the LEFT sidebar.")
        else:
            st.write("Perform any operation!")
        #dataset['clean_msg']=data[Column_options].apply(tokenizer)
        #entity_analyzer(Column_options)
        #c_sentences = [ sent for sent in blob.sentences ]
        #c_sentiment = [sent.sentiment.polarity for sent in blob.sentences]
                
        #new_df = pd.DataFrame(zip(c_sentences,c_sentiment),columns=['Sentence','Sentiment'])
        #st.dataframe(new_df)
        
            
    if choice == 'Vectorization':
        st.header("Vectorization")
        data_options = ["Movie Reviews","Spam","Amazon Reviews","UploadCSV"]
        data = st.selectbox("Select dataset to work on", data_options)
        dataset= explore_data(data)
        dataset= subsetcolumns_true(dataset)
        Column_options = columnselector(dataset)
        if 0 in Column_options.unique():
            st.error("Select Column with Text/Reviews for further process!")
        #if st.button("Show Column"):
        else:
            st.write(Column_options)
        vector_options = ["Count Vectors", "TFIDF Vectors","NMF Vectors","LDA Vectors"]
        #pretrained_vectors = ["Glove Vectors","Word2Vec"]
        #make_vectors = st.selectbox("Select Basic Vectors", vector_options)
        #model, vectors = vectorize(make_vectors,Column_options, showtopwords=  False)
        #st.write(str(vectors))
        #st.title("Please select one of the options from Left Sidebar")
        typeofvecs = ["Basic", "Advance"]
        select_options = st.selectbox("Select between Basic or Advance Vectors",typeofvecs)
        
        if select_options == "Basic":
            #st.title("Please select one of the options from Left Sidebar")
            st_showstatus = st.radio("Perform One of the following:", ("Top Words", "Topic Modelling"))
            if st_showstatus == "Top Words":
                make_vectors = st.radio("Select Basic Vectors", vector_options)
                model, vectors, model2,vectors2,no_of_sentences,tokenized = vectorize2(make_vectors,Column_options, get_sentence=False)
                if st.button("Show Top Words:"):
                    top_words = topwords(make_vectors, model, vectors, model2,no_of_sentences)
                    st.success("Select any other option to explore more.")
                else:
                    st.write("Perform any operation above.")
            if st_showstatus == "Topic Modelling":
                make_vectors = st.radio("Select Basic Vectors", vector_options)
                #options = list(range(1,15))
                #st_ncomponents = st.selectbox("Select No of Components for NMF/LDA",options)
                #n= st_ncomponents
                #make_vectors = st.selectbox("Select Basic Vectors", vector_options)
                model, vectors, model2, vectors2,no_of_sentences,tokenized = vectorize2(make_vectors,Column_options)
                word_level, no_of_sentences = createWordLevelRep(make_vectors,Column_options,model, vectors, model2,vectors2,no_of_sentences,tokenized)
                #if st.button("Perform Topic Modeling!"):
                topic_models =topicmodelling(make_vectors,model,vectors,model2,vectors2,no_of_sentences,word_level,Column_options)
                #else: 
                #    st.write("Perform any operation above.")
                #word_selected = makewords(model, vectors, vector_options)

                #get_cv = 
                #cv = CountVectorizer(tokenizer=lambda doc: doc, lowercase=False, min_df= 0.01, max_df=0.95, ngram_range= (1,1), stop_words= 'english')
                #count_vecs = cv.fit_transform(tokenized)
                #topicmodelling(make_vectors,model,vectors,support,word_selected,no_of_sentences,Column_options) 
            #if st_showstatus == "Cosine Similarity":
            #    st.write()
            #    word1 = st.text_input("Enter Word 1","Type Here...")
            #    word2 = st.text_input("Enter Word 2","Type Here...")
            #    word3 = st.text_input("Enter Word 3","Type Here...")
       
        
        elif select_options == "Advance":
            pretrained_vectors = ["Glove Vectors","Word2Vec"]
            #st.write("Please select one of the options from Left Sidebar")
            select_pretrain = st.selectbox("Select Pre Trained Vectors",pretrained_vectors)
            st_showstatus = st.selectbox("Perform One of the following:", ("Closest Word Embedding", "Similar Words to Combination of words","N-pairs of Words in Vector Form","Scatter Plot of Word Embeddings"))
            #select_pretrain = st.selectbox("Select Pre Trained Vectors",pretrained_vectors)
                #vocab =getVocab(tokenized)
            model, vectors, model2,vectors2,no_of_sentences,tokenized = vectorize2("Count Vectors",Column_options, get_sentence =False)
            vocab = getVocab(tokenized)
            glove_vecs_dict = gloveVecsDict(vocab)
            word2_vecs_dict = word2vecDict(tokenized)

            if st_showstatus == "Closest Word Embedding":
                if select_pretrain == "Glove Vectors":
                    close_words, index = (option_closest(glove_vecs_dict))
                    #st.table(pd.DataFrame(close_words, columns= close_words[index]))
                    if st.button("Find Closest words"):
                        name = close_words[index]
                        st.table(pd.DataFrame(close_words, columns= [f"Words Closest to '{name}'"]))
                    else:
                        st.write("Perform any operation above.")
                if select_pretrain == "Word2Vec":
                    close_words, index = (option_closest(word2_vecs_dict))
                    #st.table(pd.DataFrame(close_words, columns= close_words[index]))
                    if st.button("Find Closest words"):
                        name = close_words[index]
                        st.table(pd.DataFrame(close_words, columns= [f"Words Closest to '{name}'"]))
                    else:
                        st.write("Perform any operation above.")
            if st_showstatus == "Similar Words to Combination of words":
                if select_pretrain == "Glove Vectors":
                    closest_words, index1,index2,index3, No_of_words = options_closest(glove_vecs_dict)
                    close_words1 = closest_index_embeddings(glove_vecs_dict, index = index1)[0:No_of_words]
                    close_words2 = closest_index_embeddings(glove_vecs_dict, index = index2)[0:No_of_words] 
                    close_words3 = closest_index_embeddings(glove_vecs_dict, index = index3)[0:No_of_words]
                    #close_words3, index3 = (option_closest(glove_vecs_dict))
                    #st.table(pd.DataFrame(close_words, columns= close_words[index]))
                    name1 = close_words1[index1]
                    name2 = close_words2[index2]
                    name3 = close_words3[index3]
                    if st.button("Show similar words"):
                        st.table(pd.DataFrame(closest_words, columns= [f"Words Closest to '{name1}' + '{name2}' - '{name3}'"]))
                    else:
                        st.write("Perform any operation above.")
                if select_pretrain == "Word2Vec":
                    closest_words, index1,index2,index3, No_of_words = options_closest(word2_vecs_dict)
                    close_words1 = closest_index_embeddings(word2_vecs_dict, index = index1)[0:No_of_words]
                    close_words2 = closest_index_embeddings(word2_vecs_dict, index = index2)[0:No_of_words] 
                    close_words3 = closest_index_embeddings(word2_vecs_dict, index = index3)[0:No_of_words]
                    #close_words3, index3 = (option_closest(glove_vecs_dict))
                    #st.table(pd.DataFrame(close_words, columns= close_words[index]))
                    name1 = close_words1[index1]
                    name2 = close_words2[index2]
                    name3 = close_words3[index3]
                    if st.button("Show similar words"):
                        st.table(pd.DataFrame(closest_words, columns= [f"Words Closest to '{name1}' + '{name2}' - '{name3}'"]))
                    else:
                        st.write("Perform any operation above.")
            if st_showstatus == "N-pairs of Words in Vector Form":
                    if select_pretrain == "Glove Vectors":
                        n_pairs = st.slider("Select No of pairs you want to see:", 0, 10, 2)
                        pairs = displayNPairsDict(mydict = glove_vecs_dict, n_pairs = n_pairs)
                        options = ["Array", "Table"]
                        box = st.selectbox("Show vectors in Array form or table:",options)
                        if box == "Array":
                            st.markdown(pairs)
                        else:
                            st.table(pairs)
                        #st.write(pairs)

                    if select_pretrain == "Word2Vec":
                        n_pairs = st.slider("Select No of pairs you want to see:", 0, 10, 2)
                        pairs = displayNPairsDict(mydict = word2_vecs_dict, n_pairs = n_pairs)
                        options = ["Array", "Table"]
                        box = st.selectbox("Show vectors in Array form or table:",options)
                        if box == "Array":
                            st.markdown(pairs)
                        else:
                            st.table(pairs)
            
            if st_showstatus == "Scatter Plot of Word Embeddings":
                if select_pretrain == "Glove Vectors":
                    no_words = st.slider("Select no of Words you want to see in Scatter plot", 0,50,20)
                    #%matplotlib inline
                    spacialScatterPlot(glove_vecs_dict, n_words = no_words, model = "Glove")
                
                if select_pretrain == "Word2Vec":
                    no_words = st.slider("Select no of Words you want to see in Scatter plot", 0,50,20)
                    #%matplotlib inline
                    spacialScatterPlot(word2_vecs_dict, n_words = no_words, model = "Glove")
                #pretrained_vectors = ["Glove Vectors","Word2Vec", "NMA Vectors","LDA Vectors"]
                #select_pretrain = st.selectbox("Select Word Embeddings to see Spatial Relationship between words",pretrained_vectors)

                #select_pretrain = st.selectbox("Select Pre Trained Model",pretrained_vectors)
                #displayNPairsDict(mydict = glove_vecs_dict, n_pairs = n_pairs)

        
        
                #c_sentences = [ sent for sent in blob.sentences ]
                #c_sentiment = [sent.sentiment.polarity for sent in blob.sentences]
                
                #new_df = pd.DataFrame(zip(c_sentences,c_sentiment),columns=['Sentence','Sentiment'])
                #st.dataframe(new_df)
    if choice == 'Modeling':
        st.subheader("Modeling using Basic Supervised Machine Learning Models")
        data_options = ["Movie Reviews","Spam","Amazon Reviews","UploadCSV"]
        data = st.selectbox("Select dataset to work on", data_options)
        dataset= explore_data(data)
        #select_rows, dff = subsetcolumn(dataset)
        #st.title("Select Model you want to train, validate and test on:")
        model_options = st.selectbox("Select Model",["SV Classifier", "Naive Bayes Classifier", "Random Forest Classifier", "Gradient Boosting Classifier"])
        text_col = train_col_select(dataset)
        if 0 in text_col.unique():
            st.error("Select Column with Text/Reviews for this part!")
        #if st.button("Show Column"):
        else:
            st.write(text_col)
        target_col = test_col_select(dataset)
        if 0 not in target_col.unique():
            st.error("Select Column with Labels for further process!")
        #if st.button("Show Column"):
        else:
            st.write(target_col)
        pect_train ,pect_val =pct_train_test()
        pipeline_opt = ["CV", "TFIDF","NMF","LDA"]
        pipeline = st.selectbox("Select one of the pipeline for Model Training",pipeline_opt)
        #pipeline_options = st.selectbox("Select Pipeline to be used",["CV", "TFIDF", "NMF", "LDA"])
        text_train,text_val, text_test,target_train,target_val,target_test = trainTestValSplit(dataset,text_col,target_col,pect_train,pect_val)
        list_scores = ["Train","Validation","Test"]
        if model_options == "SV Classifier":
            #list_scores = ["Train","Validation","Test"]
            #list_cv,list_tfidf,list_lda,list_nmf = SupportVectorClassifier(text_train,target_train,text_val,target_val,text_test,target_test)
            #svc_list = list_cv + list_tfidf + list_lda + list_nmf
            if st.button("Load results"):
                #list_cv,list_tfidf,list_lda,list_nmf = SupportVectorClassifier(text_train,target_train,text_val,target_val,text_test,target_test)
                list_pipeline = SupportVectorClassifier(text_train,target_train,text_val,target_val,text_test,target_test,pipeline)
                #st.table(pd.DataFrame(np.array([list_scores,list_cv,list_tfidf,list_lda,list_nmf]).T, columns = ["Type","CV","TFIDF","LDA","NMF"]))
                st.table(pd.DataFrame(np.array([list_scores,list_pipeline]).T, columns = ["Type",f"{pipeline}"]))
                st.success("Successful")
        if model_options == "Naive Bayes Classifier":
            #list_scores = ["Train","Validation","Test"]
            if st.button("Load Results"):
                list_pipeline = NaiveBayes_Classifier(text_train,target_train,text_val,target_val,text_test,target_test,pipeline)
                st.table(pd.DataFrame(np.array([list_scores,list_pipeline]).T, columns = ["Type",f"{pipeline}"]))
                st.success("Successful")
        if model_options == "Random Forest Classifier":
            #list_scores = ["Train","Validation","Test"]
            if st.button("Load results"):
                #list_cv,list_tfidf,list_lda,list_nmf = SupportVectorClassifier(text_train,target_train,text_val,target_val,text_test,target_test)
                list_pipeline = RandomForest_Classifier(text_train,target_train,text_val,target_val,text_test,target_test,pipeline)
                #st.table(pd.DataFrame(np.array([list_scores,list_cv,list_tfidf,list_lda,list_nmf]).T, columns = ["Type","CV","TFIDF","LDA","NMF"]))
                st.table(pd.DataFrame(np.array([list_scores,list_pipeline]).T, columns = ["Type",f"{pipeline}"]))
                st.success("Successful")
        if model_options == "Gradient Boosting Classifier":
            #list_scores = ["Train","Validation","Test"]
            if st.button("Load results"):
                #list_cv,list_tfidf,list_lda,list_nmf = SupportVectorClassifier(text_train,target_train,text_val,target_val,text_test,target_test)
                list_pipeline = BoostingClassifier(text_train,target_train,text_val,target_val,text_test,target_test,pipeline)
                #st.table(pd.DataFrame(np.array([list_scores,list_cv,list_tfidf,list_lda,list_nmf]).T, columns = ["Type","CV","TFIDF","LDA","NMF"]))
                st.table(pd.DataFrame(np.array([list_scores,list_pipeline]).T, columns = ["Type",f"{pipeline}"]))
                st.success("Successful")




    elif choice == 'About':
        st.subheader("Made by Amber Solberg, Sonal Jain and Yogesh Nizzer")
        st.info("Built for Education purposes")
        st.text("Follow us:")
        st.text(''' 1. Yogesh Nizzer
    Student pursuing Master's In Data Science At Northeastern University
    LinkedIn Profile : https://www.linkedin.com/in/yogesh-nizzer/ ''') 
        img = Image.open("data/yogesh.png")
        st.image(img,width=150,caption='Yogesh Nizzer')
        st.text(''' 2. Sonal Jain
    Student pursuing Master's In Data Science At Northeastern University
    LinkedIn Profile : https://www.linkedin.com/in/sjain2212/ ''') 
        img1 = Image.open("data/sonal.png")
        st.image(img1,width=150,caption='Sonal Jain')
        st.text(''' 3. Amber Solberg
    Student pursuing Master's In Data Science At Harvard Extension School''') 
        st.success("Thank you for Exploring this App!")
            
    

def dataanalysisfunc(dataset):
    top = st.selectbox(
            "Select number of rows to show", [5, 10, 25, 50, 100, len(dataset)])
    table_data = (dataset.head(top))
    if st.button("Show Dataset:"):
        time.sleep(1)
        st.table(table_data)
    else:
        st.write("Try Loading the Dataframe by using the Button above")

    with st.spinner("Extracting source data..."):
        dataset["sentiment"] = dataset.iloc[:,0].map(
        {False: "Negative", True: "Positive"}
        )
        st.info(f"{len(dataset)} rows where extracted with **success**!")
    # Show Shape of Data
    if st.checkbox("Show Shape of Dataset"):
        st.text("\nShape:")
        st.write(dataset.shape) 
    #Column_options = columnselector(dataset)
    
    if st.checkbox("Plot of Value Counts"):
        st.text("Value Counts By Target/Class")

        all_columns_names = dataset.columns.tolist()
        primary_col = st.selectbox('Select Primary Column To Group By',all_columns_names)
        selected_column_names = st.multiselect('Select Columns',all_columns_names)
        if st.button("Plot"):
            st.text("Generating Plot for: {} and {}".format(primary_col,selected_column_names))
            if selected_column_names:
                vc_plot = dataset.groupby(primary_col)[selected_column_names].count()		
            else:
                vc_plot = dataset.iloc[:,-1].value_counts()
            st.write(vc_plot.plot(kind='bar'))
            st.pyplot()
            st.success("Successful. Let's move on to Next part, Tokenization!")
    


def columnselector(dataset):
    Column_options = st.selectbox(
        "Select 'TEXT/Reviews' Column:",
        (dataset.columns))
    if Column_options == dataset.columns[1]:
        column_value = dataset.iloc[:,1]
        #st.write(column_value)
    elif Column_options == dataset.columns[0]:
        column_value = dataset.iloc[:,0]
        #st.write(column_value)
    elif Column_options == dataset.columns[2]:
        column_value = dataset.iloc[:,2]
        #st.write(column_value)
    else:
        st.write("Select A Column")
    return column_value

def train_col_select(dataset):
    Column_options = st.selectbox(
        "Select Columns on which you want to train the dataset",
        (dataset.columns))
    if Column_options == dataset.columns[1]:
        column_value = dataset.iloc[:,1]
        #st.write(column_value)
    elif Column_options == dataset.columns[0]:
        column_value = dataset.iloc[:,0]
        #st.write(column_value)
    elif Column_options == dataset.columns[2]:
        column_value = dataset.iloc[:,2]
        #st.write(column_value)
    else:
        st.write("Select A Column")
    return column_value

def test_col_select(dataset):
    Column_options = st.selectbox(
        "Select Label Column for Testing purpose",
        (dataset.columns))
    if Column_options == dataset.columns[1]:
        column_value = dataset.iloc[:,1]
        #st.write(column_value)
    elif Column_options == dataset.columns[0]:
        column_value = dataset.iloc[:,0]
        #st.write(column_value)
    elif Column_options == dataset.columns[2]:
        column_value = dataset.iloc[:,2]
        #st.write(column_value)
    else:
        st.write("Select A Column")
    return column_value


def tokenize_full(docs, token_options, showtable):
    """Full tokenizer with flags for processing steps
    entities: If False, replaces with entity type
    stop_words: If False, removes stop words
    lowercase: If True, lowercases all tokens
    alpha_only: If True, removes all non-alpha characters
    lemma: If True, lemmatizes words
    """

    ##
    #download('en_core_web_sm')
    #nlp = spacy.load('en_core_web_sm')
    #nlp = spacy.load('en')
    #model = nlp
    #modelss= ["en", "en_core_web_sm"]
    #select_model = st.selectbox("Select Model for Tokenisation",modelss)
    #if select_model == "en":
    model = en
    #else:
    #    model = spacy.load('en_core_web_sm')
    tokenized_docs = []
    for d in docs:
        parsed = model(d)
        # token collector
        tokens = []
        # index pointer
        i = 0
        senten = ''
        # entity collector
        ent = ''
        for t in parsed:
            #sentence = ''
            # only need this if we're replacing entities
            if "Entities" in token_options:
                # replace URLs
                if t.like_url:
                    tokens.append('URL')
                    continue
                # if there's entities collected and current token is non-entity
                if (t.ent_iob_=='O')&(ent!=''):
                    tokens.append(ent)
                    ent = ''
                    continue
                elif t.ent_iob_!='O':
                    ent = t.ent_type_
                    continue
            # only include stop words if stop words==True
            if (t.is_stop)&("Stop_Words" in token_options):
                continue
            # only include non-alpha is alpha_only==False
            if (not t.is_alpha)&("Alpha_Only" in token_options):
                continue
            if "Lemmatize" in token_options:
                t = t.lemma_
            else:
                t = t.text
            if "Lowercase" in token_options:
                t = t.lower()
            tokens.append(t)
            if showtable == True:
                senten = " ".join(tokens)
            else:
                senten = tokens
        tokenized_docs.append(senten)
    return(tokenized_docs)

def vectorize2(make_vectors,Column_options, get_sentence = True):
    methods = ["Lowercase", "Alpha_Only", "Lemmatize", "Stop_Words", "Entities"]
    token_options= st.multiselect("Tokenization methods to perform", methods, default= 
                ["Lowercase","Alpha_Only"])
    st.write(f"You choose following options :{token_options}")
    tokenized = tokenize_full(Column_options, token_options, showtable=False)
    if make_vectors == "Count Vectors":
        cv = CountVectorizer(tokenizer=lambda doc: doc, lowercase=False, min_df= 0.01, max_df=0.95, ngram_range= (1,1), stop_words= 'english')
        count_vecs = cv.fit_transform(tokenized)
        if get_sentence == True:
            no_of_sentences = st.slider("Select no of Topics",0,15,10)
        else:
            no_of_sentences = None
        return(cv,count_vecs,None,None,no_of_sentences,tokenized)

        #'''Getting the TFIDF vectors weights of the tokens with min_df of 0.01 and max_df of 0.95, we are using just 1 for n_grams here.'''
        # Creating Tfidf_vectors
    if make_vectors == "TFIDF Vectors":
        tfidf = TfidfVectorizer(tokenizer=lambda doc: doc,lowercase= False, min_df= 0.01, max_df=0.95, ngram_range= (1,1), stop_words= 'english')
        tfidf_vecs = tfidf.fit_transform(tokenized)
        if get_sentence == True:
            no_of_sentences = st.slider("Select no of Topics",0,15,10)
        else:
            no_of_sentences = None
        #no_of_sentences = st.slider("Select no of Topics",0,15,10)
        return(tfidf,tfidf_vecs,None, None, no_of_sentences,tokenized)

    if make_vectors == "NMF Vectors":
        tfidf = TfidfVectorizer(tokenizer=lambda doc: doc,lowercase= False, min_df= 0.01, max_df=0.95, ngram_range= (1,1), stop_words= 'english')
        tfidf_counts = tfidf.fit_transform(tokenized)
        options = list(range(1,15))
        st_ncomponents = st.selectbox("Select No of Components for NMF/LDA",options)
        no_of_sentences= st_ncomponents
        nmf = NMF(n_components= no_of_sentences)
        nmf_vecs = nmf.fit_transform(tfidf_counts)
        return nmf,nmf_vecs,tfidf,tfidf_counts,no_of_sentences,tokenized


        #Function for getting LDA Vectors
    if make_vectors == "LDA Vectors":
        cv = CountVectorizer(tokenizer=lambda doc: doc, lowercase=False, min_df= 0.01, max_df=0.95, ngram_range= (1,1), stop_words= 'english')
        counts = cv.fit_transform(tokenized)
        options = list(range(1,15))
        st_ncomponents = st.selectbox("Select No of Components for NMF/LDA",options)
        no_of_sentences= st_ncomponents
        lda = LatentDirichletAllocation(n_components= no_of_sentences)
        lda_vecs = lda.fit_transform(counts)
        return lda,lda_vecs,cv,counts,no_of_sentences,tokenized

def topwords(make_vectors, model, vectors,model2,no_of_sentences):
    #noofwords = st.slider("Select no of Top words", 0,25,(10))
    top_words = showtopwords(make_vectors,model,vectors,model2,no_of_sentences)
    st.table(pd.DataFrame(top_words, columns = ['Word','Value']))
    return top_words

def showtopwords(make_vectors, model, vectors, model2,no_of_sentences):
    #def topWordsCV(tokenized, n_words):
    if make_vectors == "Count Vectors" or make_vectors == "TFIDF Vectors":
        vectors = vectors.toarray()
        models = model
        n_words = st.slider("Select no of Top words", 0,25,(10))
    elif make_vectors == "NMF Vectors" or make_vectors == "LDA Vectors":
        vectors = vectors
        models = model2
        n_words = no_of_sentences
    word_count = dict(zip(models.get_feature_names(), vectors.sum(axis=0)))
    top_words = sorted(word_count.items(), key=lambda x: x[1], reverse=True)[:n_words]
    return top_words

def topicmodelling(make_vectors,model,vectors,model2,vectors2,no_of_sentences,word_level,Column_options):
    topic_modelling = showtopicmodelling(make_vectors,model,vectors,model2,vectors2,no_of_sentences)
    df_topic_model = pd.DataFrame(topic_modelling, columns = ['Topic'])
    if st.button("Perform Topic Modelling"):
        st.table(df_topic_model)
    else:
        st.write("Peform any of the operation above.")
    temp_cv,temp_count_vecs = vectorizetemp(Column_options)
    st_cosine = st.checkbox("Cosine Similarity")
    if st_cosine:
        cosine_topic = st.multiselect("Select Topic for which you want to check cosine similarirty",df_topic_model.iloc[:,0])
        #cv, count_vecs = vectorize("Count Vectors",Column_options, showtopwords=False)
        #seed_idxs = [model.vocabulary_[w] for w in cosine_topic]
        words = cosine_topic[0].split()
        st.write(words)
        seed_idxs = [temp_cv.vocabulary_[w] for w in words]
        st.write(cosine_similarity(word_level[seed_idxs]))
    return topic_modelling


def showtopicmodelling(make_vectors, model, vectors,model2,vectors2,no_of_sentences):
    topic_models = []
    if make_vectors == "Count Vectors" or make_vectors == "TFIDF Vectors":
        values = vectors.toarray()
        models = model
    elif make_vectors == "NMF Vectors" or make_vectors == "LDA Vectors":
        models = model2
        values = model.components_
        
    #no_of_sentences = st.slider("Select no of Topics",0,15,10)
    top_display = st.slider("Select no of Words in one Topic", 0,8,5)
    for topic_idx, topic in enumerate(values):
        #print("Topic %d:" % (topic_idx))
        if topic_idx != no_of_sentences:
            top_words_idx = topic.argsort()[::-1][:top_display]
            top_words = [models.get_feature_names()[i] for i in top_words_idx]
            #print(" ".join(top_words))
            sentencess = " ".join(top_words)
        #if topic_idx == no_of_sentences:
        #  break
        elif topic_idx == no_of_sentences:
            #topic_models = []
            break
        topic_models.append(sentencess)
    return topic_models
def createWordLevelRep(make_vectors,Column_options,model, vectors, model2,vectors2,no_of_sentences,tokenized):
    """Function that takes in the cv vectors and tfidf vectors,
    both of which can be accessed with the functions above 
    (countVecs and tfidfVecs). 
      
    You can input this using the following 3 lines of code:
    count_vecs = countVecs(tokenized)
    tfidf_vecs = tfidfVecs(tokenized)
    createWordLevelRep(count_vecs, tfidf_vecs)
    """

    if make_vectors == "Count Vectors":
        cv,count_vecs,temp1,no_of_sentences = model, vectors, vectors2,no_of_sentences
        count_words = count_vecs.T
        word_level = count_words
    if make_vectors == "TFIDF Vectors":
        tfidf,tfidf_vecs,temp2,no_of_sentences =model, vectors, vectors2,no_of_sentences
        tfidf_words = tfidf_vecs.T
        word_level = tfidf_words
    if make_vectors == "NMF Vectors":
        nmf, nmf_vecs,temp_tfidf,no_of_sentences =model, vectors, vectors2,no_of_sentences
        nmf_vecs = nmf.fit_transform(temp_tfidf)
        nmf_words = nmf.components_.T
        word_level = nmf_words
    if make_vectors == "LDA Vectors":
        lda,lda_vecs,temp_cv,no_of_sentences =model, vectors, vectors2,no_of_sentences
        lda_vecs = model.fit_transform(temp_cv)
        lda_words = lda.components_.T
        word_level = lda_words

    return (word_level,no_of_sentences)

def vectorizetemp(Column_options):
    #def countVecs(tokenized):
    #temp_methods = ["Lowercase", "Alpha_Only", "Lemmatize", "Stop_Words", "Entities"]
    temp_token_options= ["Lowercase","Alpha_Only"]
    temp_tokenized = tokenize_full(Column_options, temp_token_options, showtable=False)
    #"Count Vectors", "TFIDF Vectors","NMF Vectors","LDA Vectors",
    #                    "Glove Vectors","ELMO Vectors"
    cv = CountVectorizer(tokenizer=lambda doc: doc, lowercase=False, min_df= 0.01, max_df=0.95, ngram_range= (1,1), stop_words= 'english')
    count_vecs = cv.fit_transform(temp_tokenized)
    #no_of_sentences = st.slider("Select no of Topics",0,15,10)
    return(cv,count_vecs)

def getVocab(tokenized):
    cv = CountVectorizer(tokenizer=lambda doc: doc, lowercase=False, min_df= 0.01, max_df=0.95, ngram_range= (1,1), stop_words= 'english')
    cv.fit_transform(tokenized)
    # collect vectors in matrix
    vocab = cv.vocabulary_
    vocab = dict([(v, vocab[v]+2) for v in vocab])
    vocab['_UNK'] = 1
    vocab['_PAD'] = 0
    return (vocab) 

def gloveVecsDict(vocab):
    glove_vecs_dict = {}
    #glove_vecs_dict = np.zeros(shape=(len(vocab), 300))
    nlp = spacy.load('en_core_web_sm')
    #glove_vecs_dict = {}
    for k, v in vocab.items():
      glove_vecs_dict[k] = nlp(k).vector
    return glove_vecs_dict

def word2vecDict(tokenized):
    word2vec_dict = {}
    model = Word2Vec(list(tokenized), min_count=2)
    for idx, key in enumerate(model.wv.vocab):
        word2vec_dict[key] = model.wv[key]
    return word2vec_dict

def closest_index_embeddings(mydict, index):
    return sorted(mydict.keys(), key=lambda word: spatial.distance.euclidean(mydict[word], mydict[list(mydict.keys())[index]]))

def closest_word_embeddings(mydict, embedding):
    return (sorted(mydict.keys(), key=lambda word: spatial.distance.euclidean(mydict[word], embedding)))

def displayNPairsDict(mydict, n_pairs):
    firstNpairs = {k: mydict[k] for k in list(mydict)[:n_pairs]}
    return firstNpairs
    
def option_closest(my_dict):
    st.write("Select Closest Word Embedding to a word of Index 'i'")
    index = st.slider("Select value of Index", 0,20,5)
    No_of_words = st.slider("Select Number of Words closest to index word",0,20,10)
    close_words = closest_index_embeddings(my_dict, index = index)[0:No_of_words]
    return close_words,index

def options_closest(my_dict):
    st.write("Finding words with Combination of Words")
    index1 = st.slider("Select value of Index 1", 0,50,5)
    index2 = st.slider("Select value of Index 2", 0,50,6)
    index3 = st.slider("Select value of Index 3", 0,50,7)
    No_of_words = st.slider("Select Number of Words closest to index word",0,20,10)
    #close_words = closest_index_embeddings(my_dict, index)[0:No_of_words]
    closest_words = (closest_word_embeddings(my_dict, my_dict[list(my_dict.keys())[index1]] + my_dict[list(my_dict.keys())[index2]] -my_dict[list(my_dict.keys())[index3]])[0:No_of_words])
    return closest_words,index1,index2,index3, No_of_words

def spacialScatterPlot(vecs_dict, n_words, model):
    """Function that takes in the selected word embedding dictionary
    (Options are glove, nmf, and lda). All of these dictionaries can
    be created with the gloveVecsDict, nmfVecsDict, and ldaVecsDict
    functions above.

    Input model specification in order to print the name in the
    title of the scatterplot. Using model = "Glove" will post the 
    following:
    
    "Scatter plot of first n words in Glove vector"
    """
    # Using  TSNE to reduce the dimension 
    tsne = TSNE(n_components=2, random_state=0)
    
    # List of words from Glove vectors to be used for finding the Spatial Relationship
    all_words =  list(vecs_dict.keys())             
    
    # Passing own list of words (seed_words)
    vectors_all_words = [vecs_dict[word] for word in all_words]
    
    Y   = tsne.fit_transform(vectors_all_words)
    
    print("Scatter plot of first {} words in {} vector".format(n_words, model))
    plt.scatter(Y[:n_words, 0], Y[:n_words, 1])
    rcParams['figure.figsize'] = 10,10
    
    for label, x, y in zip(all_words, Y[:n_words, 0], Y[:n_words, 1]):
        plt.annotate(label, xy=(x, y), xytext=(0, 0), textcoords="offset points")
    
    #plt.show()
    st.pyplot(plt.show())

def subsetcolumns_true(dataset):
    dataset = dataset.sample(frac=1).reset_index(drop=True)
    dataset = dataset.iloc[:3000,:]
    return dataset

def subsetcolumn(dataset):
    dataset = dataset.sample(frac=1).reset_index(drop=True)
    subsetting = st.checkbox("Do you want to subset the dataset?")
    if subsetting:
        select_rows = st.slider(
        "Select no of rows you want to keep for Subsetting",0,5000,3000)
        dataset = dataset.iloc[:select_rows,:]
        return select_rows,dataset
    else:
        dataset = dataset
        select_rows = len(dataset)
        return select_rows,dataset

def pct_train_test():
    pect_train = st.slider("Select training and testing split",0.0,1.0,float(0.7))
    pect_val = st.slider("Select training and validation split",0.0,1.0,float(0.3))
    return pect_train, pect_val


def trainTestValSplit(dataframe, text_col, target_col, pect_train, pect_val):
       
    # set the seed for numpy
    np.random.seed(seed=42)
    # shuffle, just for safety
    shuffled_idxs = np.random.choice(range(len(dataframe)), size=len(dataframe),replace=False)
    text_col = text_col[shuffled_idxs]
    target_col = target_col[shuffled_idxs]
    # sample random 70% for fitting model (training)
    # we'll also add a validation set, for checking the progress of the model during training
    # 30% will be simulating "new observations" (testing)

    pct_train = pect_train
    train_bool = np.random.random(len(dataframe))<=pct_train
    text_train = text_col[train_bool]
    text_test = text_col[~train_bool]
    target_train = target_col[train_bool]
    target_test = target_col[~train_bool]
    # making a validation set
    pct_val = pect_val
    val_idxs = np.random.random(size=len(text_train))<=pct_val
    target_val = target_train[val_idxs]
    #print("Target validation shape: ", target_val.shape)
    text_val = text_train[val_idxs]
    # reconfigure train so that it doesn't include validation
    text_train = text_train[~val_idxs]
    target_train = target_train[~val_idxs]
    
    # Create dictionary so that we can access results by their label
    return text_train,text_val,text_test,target_train,target_val,target_test

#Count vectorizer pipeline 
def countvec_pipe(model,X_train,y_train,X_val,y_val,X_test,y_test):
    cv_list = []
    cv_pipeline = Pipeline([
            ('count_vectorizer', CountVectorizer(min_df=0.01, max_df=0.99)),
            ('to_dense', DenseTransformer()),
            (model)
            ])

    cv_pipeline.fit(X_train, y_train)
    cv_train_score = cv_pipeline.score(X_train, y_train)
    cv_list.append(cv_train_score)
    cv_val_score = cv_pipeline.score(X_val, y_val)
    cv_list.append(cv_val_score)
    cv_test_score =cv_pipeline.score(X_test, y_test)
    cv_list.append(cv_test_score)
    #print("training accuracy in cv pipeline:", cv_train_score,"val accuracy in cv pipeline:", cv_val_score)
    #print("testing accuracy in cv pipeine:",cv_test_score)
    return cv_list

#TFIDF vectorizer pipeline 
def tfidf_pipe(model,X_train,y_train,X_val,y_val,X_test,y_test):
    tfidf_list = []
    tfidf_pipeline = Pipeline([
            ('tfidf_vectorizer', TfidfVectorizer(min_df=0.01, max_df=0.99)),
            ('to_dense', DenseTransformer()),
            (model)
            ])

    tfidf_pipeline.fit(X_train, y_train)
    tfidf_train_score = tfidf_pipeline.score(X_train, y_train)
    tfidf_list.append(tfidf_train_score)
    tfidf_val_score = tfidf_pipeline.score(X_val, y_val)
    tfidf_list.append(tfidf_val_score)
    tfidf_test_score =tfidf_pipeline.score(X_test, y_test)
    tfidf_list.append(tfidf_test_score)
    #print("training accuracy in tfidf pipeline:", tfidf_train_score,"val accuracy in tfidf pipeline:", tfidf_val_score)
    #print("testing accuracy in tfidf pipeine:",tfidf_test_score)
    return tfidf_list

def LDA_pipe(model,X_train,y_train,X_val,y_val,X_test,y_test):
    lda_list = []
    lda_pipeline = Pipeline([
            ('count_vectorizer', CountVectorizer(min_df=0.01, max_df=0.99)),
            ('lda', LatentDirichletAllocation(n_components=10,)),
            ('to_dense', DenseTransformer()),
            (model)
        ])

    lda_pipeline.fit(X_train, y_train)
    lda_train_score = lda_pipeline.score(X_train, y_train)
    lda_list.append(lda_train_score)
    lda_val_score = lda_pipeline.score(X_val, y_val)
    lda_list.append(lda_val_score)
    lda_test_score =lda_pipeline.score(X_test, y_test)
    lda_list.append(lda_test_score)
    #print("training accuracy in LDA pipeline:", lda_train_score,"val accuracy in LDA pipeline:", lda_val_score)
    #print("testing accuracy in LDA pipeine:",lda_test_score)
    return lda_list
    #print("training accuracy in LDA pipeline:", lda_train_score,"val accuracy in LDA pipeline:", lda_val_score)
    #print("testing accuracy in LDA pipeine:",lda_test_score)

def NMF_pipe(model,X_train,y_train,X_val,y_val,X_test,y_test):
    nmf_list = []
    nmf_pipeline = Pipeline([
            ('tfidf_vectorizer', TfidfVectorizer(min_df=0.01, max_df=0.99)),            
            ('nmf', NMF()),
            ('to_dense', DenseTransformer()),
            (model)
        ])

    nmf_pipeline.fit(X_train, y_train)
    nmf_train_score = nmf_pipeline.score(X_train, y_train)
    nmf_list.append(nmf_train_score*100)
    nmf_val_score = nmf_pipeline.score(X_val, y_val)
    nmf_list.append(nmf_val_score*100)
    nmf_test_score =nmf_pipeline.score(X_test, y_test)
    nmf_list.append(nmf_test_score*100)
    #print("training accuracy in NMF pipeline:", nmf_train_score*100,"val accuracy in NMF pipeline:", nmf_val_score*100)
    #print("testing accuracy in NMF pipeine:",nmf_test_score*100)
    return nmf_list


def SupportVectorClassifier(text_train,target_train,text_val,target_val,text_test,target_test, pipeline):
  #countVectorizer pipeline
    if pipeline == "CV":
        list_pipeline = countvec_pipe(('svc', SVC(random_state=92)),text_train,target_train,text_val,target_val,text_test,target_test)

    #TFIDF vectorizer pipeline 
    if pipeline == "TFIDF":
        list_pipeline = tfidf_pipe(('svc', SVC(random_state=92)), text_train,target_train,text_val,target_val,text_test,target_test)

    #LDA pipeline
    if pipeline == "LDA":
        list_pipeline = LDA_pipe(('svc', SVC(random_state=92)),text_train,target_train,text_val,target_val,text_test,target_test)

    #NMF pipeline
    if pipeline == "NMF":
        list_pipeline = NMF_pipe(('svc', SVC(random_state=92)),text_train,target_train,text_val,target_val,text_test,target_test)

    return list_pipeline


#Function for using Naive Bayes Classifier


def NaiveBayes_Classifier(text_train,target_train,text_val,target_val,text_test,target_test,pipeline):
  #countVectorizer pipeline
    if pipeline == "CV":
        list_pipeline = countvec_pipe(('nb', GaussianNB()),text_train,target_train,text_val,target_val,text_test,target_test)

  #TFIDF vectorizer pipeline 
    if pipeline == "TFIDF":
        list_pipeline = tfidf_pipe(('nb', GaussianNB()),text_train,target_train,text_val,target_val,text_test,target_test)

  #LDA pipeline
    if pipeline == "LDA":
        list_pipeline = LDA_pipe(('nb', GaussianNB()),text_train,target_train,text_val,target_val,text_test,target_test)

  #NMF pipeline
    if pipeline == "NMF":
        list_pipeline = NMF_pipe(('nb', GaussianNB()),text_train,target_train,text_val,target_val,text_test,target_test)

    return list_pipeline


def RandomForest_Classifier(text_train,target_train,text_val,target_val,text_test,target_test,pipeline):
    #countVectorizer pipeline
    if pipeline == "CV":
        list_pipeline = countvec_pipe(('rf', RandomForestClassifier(random_state=92)),text_train,target_train,text_val,target_val,text_test,target_test)

    #TFIDF vectorizer pipeline 
    if pipeline == "TFIDF":
        list_pipeline = tfidf_pipe(('rf', RandomForestClassifier(random_state=92)),text_train,target_train,text_val,target_val,text_test,target_test)

    #LDA pipeline
    if pipeline == "LDA":
        list_pipeline = LDA_pipe(('rf', RandomForestClassifier(random_state=92)),text_train,target_train,text_val,target_val,text_test,target_test)

    #NMF pipeline
    if pipeline == "NMF":
        list_pipeline = NMF_pipe(('rf', RandomForestClassifier(random_state=92)),text_train,target_train,text_val,target_val,text_test,target_test)

    return list_pipeline

def BoostingClassifier(text_train,target_train,text_val,target_val,text_test,target_test,pipeline):
  #countVectorizer pipeline
    #countVectorizer pipeline
    if pipeline == "CV":
        list_pipeline = countvec_pipe(('rf', RandomForestClassifier(random_state=92)),text_train,target_train,text_val,target_val,text_test,target_test)

    #TFIDF vectorizer pipeline 
    if pipeline == "TFIDF":
        list_pipeline = tfidf_pipe(('rf', RandomForestClassifier(random_state=92)),text_train,target_train,text_val,target_val,text_test,target_test)

    #LDA pipeline
    if pipeline == "LDA":
        list_pipeline = LDA_pipe(('rf', RandomForestClassifier(random_state=92)),text_train,target_train,text_val,target_val,text_test,target_test)

    #NMF pipeline
    if pipeline == "NMF":
        list_pipeline = NMF_pipe(('rf', RandomForestClassifier(random_state=92)),text_train,target_train,text_val,target_val,text_test,target_test)

    return list_pipeline

if __name__ == '__main__':
    main()
