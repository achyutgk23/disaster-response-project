# **Disaster Response Classification Project**
This project aims to classify real time messages received from places where disaster is happening. These messages are collected in the real time and classified/directed towards relevant categories/departments of Disaster Response Team. The dataset from Figure Eight has been used for this project.
In this project, the data was analyzed, cleaned and a classification model was built to classify the messages to relevant organizations.

# **Packages and Installations**
Clone the repository to run the project in local machine. Install the required dependencies listed below using pip, if not already installed.
The packages required to run the project are
- numpy
- pandas
- scikit-learn
- nltk
- sqlalchemy
- joblib

##### Clone the git repository
`$ git clone`  
`$ cd disaster-response-project`  

##### Open command prompt  
`$ cd disaster-response-project`  
`$ pip install numpy pandas scikit-learn nltk sqlalchemy joblib`  
`$ python data/process_data.py data/disaster_messages.csv data/disaster_categories.csv data/DisasterResponse.db`  
`$ python models/train_classifier.py DisasterResponse.db models/classifier.pkl`  
`$ app/run.py`

After hosting the website use the 'WORKSPACE ID' and 'WORKSPACE DOMAIN' in the url to launch the website as shown below.

https://WORKSPACEID-3001.WORKSPACEDOMAIN  
WORKSPACE ID and WORKSPACE DOMAIN can be found using the below command in cmd

`env | grep WORK`

# **About Dataset**  
The dataset has been provided by Figure Eight.  
The dataset contains two csv files viz. 'disaster_categories.csv' and 'disaster_messages.csv'.   
1. The disaster_messages.csv file contains four variables viz.
- id - Each message is associated with specified id.
- message - Messages in english received, regarding disaster
- original - Messages in their original languages received, regarding disaster  
- genre - The messages were classified into three genres viz. 'direct', 'news' and 'social'  

2. The disaster_categories.csv file contains two variables viz.
- id - common column between two csv files, used to merge two datasets
- categories - A given message was classified into a single or multiple categories  

# **Imbalanced Dataset**  
Imbalanced datasets are those where observations with one class are far more greater in number than other classes.  
For example a medical facility has a dataset of patients regarding a particular disease. If the dataset contains 1000 observations/data points and out of those 1000 observations 950 observations belong to class 'no disease' and 50 observations belong to 'disease', then this dataset is said to have class imbalance.  
##### Problems with Imbalanced dataset
- Since one class is far more greater in number than others, then the classifier may get biased towards the class with majority of observations.

- Let's consider the above example and train a classifier on the dataset. After training, if we evaluate the model's performance on test set then, even if the model predicts all the observations as 'no disease', the model's accuracy will be 95% which is generally considered as very good but choosing accuracy in this case as a metric to evaluate the model will be invalid or irrelevant.
Hence we have to consider other metrics to evaluate these types of datasets.

- There are several ways to tackle data imbalance problem and in this project 'f1 score' and 'macro average' metrics were used to analyze the performance of the classifier.

# **Steps Taken to Clean the Dataset, Model Building and Model Evaluation**

##### Cleaning the dataset
Below are the steps taken to clean the dataset:
- After loading the above mentioned two datasets, these datasets were merged using the common variable 'id' and saved in the new variable 'df'.

- Next, only disaster_categories dataset was considered and the 'categories' column was split into 36 columns. After splitting the columns, they were named accordingly.  
- After splitting and naming the 36 categories, all the values under these columns were converted into 0's and 1's and saved into 'categories_new' variable.
- From the previously merged 'df' dataset, 'categories' column was dropped and 'categories_new' variable was merged to it.
- Then all the duplicates were dropped from 'df' dataset and the cleaned dataset was saved into a sqlite database.

##### Model Buliding
- The data was retrieved from the database file and again stored in 'df' variable

- From the 'df' variable X and y values were extracted along with all the 36 category names which were stored in 'category_names' variable.
- A function named 'tokenize' is written to pre-process the input text data which takes only one argument called 'text'. The text preprocessing includes 'Normalization', 'Tokenization', 'Stop Word Removal' and 'Lemmatization'.
- Next a pipeline was built which includes `CountVectorizer(tokenizer=tokenize)`, `TfidfTransformer()` and `MultiOutputClassifier(RandomForestClassifier())`.
- A parameter grid was created to tune the hyper-parameters of the classifier using `GridSearchcv()` method.

##### Model Evaluation
- The model was then used to make predictions on test set and evaluated using `classification_report` from scikit-learn. The report shows various metrics in which f1-score and macro_avg can be used to measure the performance of the model.

- The the model is then saved in the pickle format.

# **License**

This project is released under [MIT License](https://choosealicense.com/licenses/mit/)
