# CMPE255-Team12

John Wang
AJ Dela Cruz
Jiayao Li
## Team 12

ACMPE-255 Project Check-in 3

Abstract
In 2021, the recruiting industry was a 136 billion dollar industry1 in the United States alone. However, sorting the most appropriate job for an applicant through a resume is a difficult process for both the applicant as well as the employer. This process is often manual and time consuming. In order to reduce the manual time and effort involved to properly classify an applicant, we propose using a BERT transformer to classify resumes. We plan on feeding cleaned labeled resume data into a BERT transformer in order to train the model to classify resumes. We anticipate that there will be difficulties with the time that it takes to run the model as well as accuracy issues. In order to mitigate these problems, we will try out different BERT models such as: BERT-base, Small BERT, ALBERT, Electra, BERT Experts, and BERT with Talking-Heads Attention. 
In doing so, we hope to not only provide better automated classification of resumes but    recommend potential suggested classifications based on the BERT model. Based on previous BERT attempts with other classification problems such as comment sentiment analysis. We anticipate that we will be able to achieve around 80% accuracy with our resume classification attempts. While this is far from perfect, this method requires zero manual effort and would serve as a great starting point for the resume classification process. We hope this will greatly improve the efficiency of the resume classification process for applicants and employers.

Intro
Recruiting suitable candidates for the job role is time-consuming but an important task. The number of applicants in the job market can be overwhelming. Especially with the different types of job roles existing along with the increasing number of applications from candidates. Resume classification is needed to make the process of selecting the appropriate candidates for the role easier. Resume classification will help recruiters identify suitable candidates based on their skill set according to the job descriptions. Other tools such as applicant tracking systems “help companies in the process of recruiting new professional figures or re-assigning resources [2].” They are extensively used in the recruitment process to find candidates with the required qualifications. However, applicant tracking systems require a manual evaluation that is time-consuming. Recruiters often need to ensure that the resume isn’t manipulated using keywords. This results in inefficiency in the recruitment process of selecting the appropriate candidates for the role. There is a need to categorize resume based on the job descriptions. It would help separate the relevant resumes that the recruiters are looking for from the large amount of resumes that may not have the necessary qualifications.

To overcome the issues and inefficiency of recruitment process, this paper presents BERT to classify resumes. BERT is a transformer based language representation model that uses bidirectional representation and can create models for different processing tasks. We will use BERT for our resume-job classification project as it processes tasks fast compared to other models such as RNN and LSTM. BERT uses a masked language model (MLM) to “combine left and right context in all layers to enable a bidirectional Transformer [3].” To process our model using BERT, we will use the pre-training and fine-tuning process. Fine-tuning BERT is inexpensive compared to pre-training. We will first pre-train our model on labeled data while fine-tuning our model to be initialized with the pre-trained data. This allows us to capture the language modeling of the resume. The span of texts such as skills on a resume is differentiated by their special token and the learned embedding to show where the token belongs. BERT shows significant improvement compared to other systems.


Literature Review
Introduced in 1986 by Rumelhart [4], the recurrent neural network (RNN) is one of the earliest and the most popular architectures in dealing with sequential data inputs. According to Schmidt (2019) [5], as compared to non-recurrent feedforward networks which pass information through the network without cycles, the RNN has cycles and transmits information back into itself in order to detect patterns in sequential data.

However, in the paper “Fundamentals of Recurrent Neural Network (RNN) and Long Short-Term Memory (LSTM) Network” (2020) [6], Sherstinsky demonstrated that RNNs suffer from issues known as “vanishing gradients” and “exploding gradients” during training since the recurrent weights would get amplified as the size of the input grows. This would lead to unsatisfactory accuracy when processing larger corpora, including resumes.

One solution to address the vanishing or exploding gradients problem is a gradient-based method known as Long Short-Term Memory (LSTM) introduced by Hochreiter and Schmidhuber (1997) [7]. According to Hochreiter, LSTMs have a chain-like structure similar to standard RNNs, but the repeating pattern is more complicated, which includes special units such as memory cells, input gates, output gates, and forget gates. The memory cell remembers value in the long term, and the gate units are conditionally activated in order to regulate the flow of information into and out of the cell. This mechanism allows the recurrent gradients to flow unchanged.

Bondielli et. al [2] proposed resume classification using the summarization technique and transformer archterctures to classify resumes. The summarization was used successfully to condense the texts of resumes and remove the redundancy. The authors used BM25-TextRank algorithm to be more efficient in summarization and transformer-based architectures [2]. Hierarchical clustering algorithms was used on resume embeddings to give the best resumes with relevant information. Roy et al. [8] proposed an approach for resume classification by using k-NN to catergorize resume that are the nearest to job descriptions. The authors used Tf-idf for feature extraction after the data was preprocessed. To categorize the resume into their proper categories, the authors used linear support vector classifier as it provided the best accuracy. The resumes that are the closest to the provided job descriptions are identified using k-NN. Overall, this model helps in assisting recruiters of screening resumes that have irrelevant skills necessary for the job application. Barducci et. al [9] proposed an NLP framework to classify resumes by extracting only the relevant information based on skills or work experience. The authors used segment extraction technique to represent resumes based on different information. The authors then used Named Entity Recognition (NER) to extract relevant features from each segment of the resume. Relevant keywords based on skills or work experience are extracted since they contain the most important features of the resume. 


Project Description
Resume classification is a time consuming, labor intensive task that has the potential to be automated away using machine learning techniques. We hope to implement the most cutting edge language models, BERT and GPT-3, and attempt to classify our dataset in order to review the effectiveness of the current state of the art language classification models. Additionally, we will be comparing the latest models against 2 traditional machine learning classification models, logistic regression and random forest, in order to understand the pros and cons of the deep language models compared to the traditional machine learning techniques.
In our project, we will be classifying the 25 different categories of resumes from the kaggle dataset (https://www.kaggle.com/datasets/jillanisofttech/updated-resume-dataset). We would be using the classification models logistic regression, random forest, BERT and GPT-3. Logistic regression and random forest are the two traditional models that we will be using to compare against the two new state of the art language classification models, BERT and GPT-3. We will use TF-IDF as the encoding algorithm for Logistic regression and random forest. BERT and GPT-3 do encoding themselves and do not need an additional encoding algorithm. We will be using RMSE against a holdset test set in order to compare the models to each other. We will be creating a training set, validation set and holdout test set based on stratified sampling of our dataset. This is because we have 25 different classes and if we didn't have stratified sampling we would have an imbalance of training/validation/test data. 
The following is an outline with the associated team member’s roles and responsibilities:
Data exploration - everyone
Create stratified training/validation/test set + Clean up data - John Wang
Model: LSTM - AJ dela Cruz
Model: TF-IDF + Random Forest: - Jiayao Li
Model: Logistic Regression - AJ dela Cruz
Model BERT: John Wang
Model GPT-3 Embedding Babbage-001 + Logistic Regressor: John Wang
Model GPT-3 Embedding Babbage-001 + Random Forest: Jiayao Li
Model evaluation - each team member will be in charge of their respective models.
 
Experimental Methodology
Data preprocessing
Our data source is the kaggle resume dataset (https://www.kaggle.com/datasets/jillanisofttech/updated-resume-dataset) We used pandas and numpy to load our data set from UpdatedResumeDataSet
.csv When then used numpy and pandas to create a stratified split of our data. This is because we have 25 different tagged classes. So we can not just randomly sample as we would undersample certain categories. For BERT, in order to create a keras dataset. We needed to use python to extract the sample resumes into text files for processing. We wrote a data parser for this. 

For cleaning the data, we used regular expressions “re.sub(r"[^a-zA-Z0-9]+", ' ', clean)” to remove all of the special characters in the data. We also had to label encode all of the categories for the resumes for preprocessing. We visualized our data to get a feel for it. 

The least amount of any particular category of resume is 20 and the most is 84. This is a 4x difference, but still reasonable.    
We also visualized it as a histogram:



Data mining models 

LSTM

LSTM model was selected to demonstrate its limitation with long sequences of text for classification. LSTM memorizes and finds the important information of the resume. Although LSTM fixes RNN’s vanishing gradient problem, it still does not perform well when processing longer sequences. This leads to worse accuracy compared to the other models used in this project.



TF-IDF + Random Forest
TF-IDF stands for Term Frequency–Inverse Document Frequency, which is a word embedding technique. It was designed to reflect how important a word is to a corpus, which matches the nature of resume classification. Resumes for different jobs would include different collection of terms, and those terms are usually strong predictors of the category of the resume. For example, the term “Java” is usually found in resumes for Java Developers. With TF-IDF as a suitable embedding, we used Random Forest Classifier.
Diagram of the model architecture:



TF-IDF + Logistic Regression
TF-IDF was also used to extract the relevant information from the resume before Logistic regression classifier is used. Logistic regression is used in finding correlations between variables. It uses a logistic function to do the classification task by using the weighted combination of the input features. The extracted features from TF-IDF is fed into Logistic regression model to predict the category of the resume. 



BERT
BERT was selected as a language classification model because it is state of the art, easy and readily available for import, runs relatively quickly, used in industry, has a lot of documentation, and has great classification accuracy. Due to these criteria we suspect that BERT can be used in a real life scenario for resume classification.
We used BERT as our language model and added 25 neurons for the 25 different categories as the classifier using tensorflow keras. Between BERT and the 25 neuron classifier, we will have a dropout layer in order to prevent model overfitting. Scikit learn was used to compute the RMSE for the models. 
Diagram of the model architecture:

GPT-3 Embedding Babbage-001 + AdaBoost Regressor
We choose GPT-3 Zero Shoot as a classification model for all of the same reasons as BERT (state of the art, easy and readily available for import, runs relatively quickly, used in industry, has a lot of documentation, and has great classification accuracy.) GPT-3 was used via the OpenAI driver via python. Pandas and Numpy were used for data preprocessing. The embedding generation is shared between the two GPT-3 models. 

GPT-3 Embedding Babbage-001 + Random Forest
We also explored GPT-3’s similarity embedding for classification. In addition to GPT-3’s powerful zero-shot text completion endpoint, it also has a variety of embedding models with distinct features. We choose babbage-001 as our embedding model for its ability to capture text features, which is best-suited for resume classification tasks. We then utilized Random Forest as the classifier after getting the embeddings.



Architecture and Work Flow:
Data Preprocessing: We removed all english stopwords and lemmatized the resume corpus. Then we removed several high-frequency words that is uninformative to resume-job classification.
Embedding: We used GPT-3 text-similarity-babbage-001 embedding model for the embedding of all corpus of resumes and the list of jobs. The result was saved as csv for further processing.
Stratified Train Test Split: We split the dataset into 70% training and 30% testing according to the number of data for each category.
Classification and Evaluation: We utilized the Random Forest Classifier for its simplicity and high accuracy. We also created a evaluation report to the model.


  
Results
Future
Discussion
Future
Conclusion
Future

Data source

https://www.kaggle.com/datasets/jillanisofttech/updated-resume-dataset

Google drive folder 

https://drive.google.com/drive/folders/1RnSYLtA1u1u6pcJPuWQJI_Rh6px6Q29q?usp=sharing

Github Repositories 

John Wang: https://github.com/johnwang21/cmpe255

AJ Dela Cruz:  https://github.com/AJ-delaCruz/CMPE255-Team12 

Jiayao Li: https://github.com/JiayaoLi00/cmpe255


Reference
[1] https://www.statista.com/statistics/873648/us-staffing-industry-market-size/

[2] A. Bondielli and F. Marcelloni, “On the use of summarization and transformer architectures for profiling résumés,” Expert Systems with Applications, vol. 184, p. 115521, 2021.

[3] J. Devlin, M.-W. Chang, K. Lee, and K. Toutanova, “Bert: Pre-training of deep bidirectional Transformers for language understanding,” arXiv.org, 24-May-2019. [Online]. Available: https://arxiv.org/abs/1810.04805. [Accessed: 16-Oct-2022].

[4] Rumelhart, D., Hinton, G. & Williams, “R. Learning representations by back-propagating errors”. Nature 323, 533–536 (1986). https://doi.org/10.1038/323533a0

[5] Robin M. Schmidt, “Recurrent Neural Networks (RNNs): A gentle Introduction and Overview,” 23-Nov-2019.

[6] Alex Sherstinsky, “Fundamentals of Recurrent Neural Network (RNN) and Long Short-Term Memory (LSTM) network,” 2020.

[7] Sepp Hochreiter, Jürgen Schmidhuber, “Long Short-Term Memory,” Neural Comput 1997; 9 (8): 1735–1780. doi: https://doi.org/10.1162/neco.1997.9.8.1735

[8] P. K. Roy, S. S. Chowdhary, and R. Bhatia, “A machine learning approach for automation of resume recommendation system,” Procedia Computer Science, vol. 167, pp. 2318–2327, 2020. 

[9] A. Barducci, S. Iannaccone, V. La Gatta, V. Moscato, G. Sperlì, and S. Zavota, “An end-to-end framework for information extraction from Italian resumes,” Expert Systems with Applications, vol. 210, p. 118487, 2022. 



