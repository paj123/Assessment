# BKFS HeavyWater Machine Learning Assessment

### All repository file information: -
* shuffled-full-set-hashed.csv : - BKFS Heavy Water Dataset
* document-classification-test.py: - python file
* document-classification-test.ipynb: - jupyter file
* app.py: - python file to create app using flask
* .pkl files: ML models in pickle form
* requirements.txt: -All packages with versions that need to be installed
* templates: - consists of html files used for front end design
* static: - css files for better look


### Problem Statement

Generally, the goal of this problem is to build a text classification model based on machine learning. Input is given as a csv file, which contains all documents that serve as our dataset to train and test our model. The document entry format is like this:

```
CANCELLATION NOTICE,641356219cbc f95d0bea231b ... [lots more words] ... 52102c70348d b32153b8b30c
```

So each document is composed of a label which indicates document type at its beginning, and a series of obscured OCR(Optical Character Recognition) data seperated by space and each of them maps to a unique word in original document.



### General Steps

Since the label is given, we will do supervised training on our dataset. Our main steps are:
1. Dataset Preparation - Load our dataset and perform basic pre-processing. E.g. figure our our dataset's composition and split dataset into train and test sets.
2. Feature Engineering - Transform raw dataset into flat features that can be used in our machine learning model.
3. Model Training - Train the model on labelled dataset.



#### Step 1: Dataset Preparation

Let's have a glance of our dataset first. There're 62204 documents in total and their distribution is shown below:

![](images/data_plot.jpeg)

Then we need to split our dataset for training and testing. The training part is used to train our model to make correct classification. And the testing part is used to validate the model's correctness. 
The widely used machine learning library **sklearn** provides us with a powerful method-**model_selection.train_test_split**- to do this split. 
The doc for **model_selection.train_test_split** is provided below:
```
https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.train_test_split.html
```



#### Step 2: Feature Engineering

Our raw obscured text dataset cannot work directly with machine learning algorithms, which needs numeric data representation. So in this step we need to convert the text to numbers. 

A simple and effective model for thinking about text documents in machine learning is called the **Bag-of-Words** Model, or **BoW**. The model is simple in that it throws away all of the order information in the words and focuses on the occurrence of words in a document. The **sklearn** library provides 3 different schemes(Word Counts with **CountVectorizer**, Word Frequencies with **TfidfVectorizer** and Hashing with **HashingVectorizer**) that we can use to achieve this goal, they are introduced in the following link:
```
https://machinelearningmastery.com/prepare-text-data-machine-learning-scikit-learn/
```
In our solution we will use **TfidfVectorizer** because it's a refined version of **CountVectorizer** and easier to implement than **HashingVectorizer**. Since the vectorizer requires the raw_documents to be str, unicode or file objects type, we will convert our dataframe into unicode before transforming the data. 

After we vectorize our data, let's print and check the numeric matrix.
```
  (0, 290693)	0.11966728067482815
  (0, 630669)	0.027930338102784877
  (0, 741273)	0.5053247463636006
  (0, 140159)	0.027446260919014562
  (0, 527647)	0.08764998192233651
  (0, 812021)	0.17521859465377745
  (0, 703628)	0.11798909834518305
  (0, 601896)	0.09424065432278207
  :	:
  (49762, 726701)	0.1252014592533339
  (49762, 691873)	0.12747884824803968
  (49762, 857232)	0.11654331817294304
  (49762, 665928)	0.13385960033372477
  (49762, 160794)	0.1302661471113159
  (49762, 769263)	0.1302661471113159
  (49762, 307783)	0.1302661471113159
  (49762, 134960)	0.12747884824803968
```
Now we are good to use this feature set to train our model.



#### Step 3: Model Training

Now it's time to build a model for our training. In this problem we choose the popular classification model - Logistic Regression, which both works on continuous data and discrete data. The document type is predicted by the refined frequency of each word.
```
model = LogisticRegression()
model.fit(tfidf, Y_train)
```
Finally we achieved the accuracy of 85.73%, which is pretty good.

The confusion matrix:

![](images/cm.png)

### Deploy Model

Because of previous Heroku project deployment experience, I plan to deploy on Heroku at first. If there's spare time after the first version completed, I will try to use AWS to deploy.

The general layout of my deployed project:
```
/heavywater
   ├── templates
   │   └── index.html
   │   └── result.html
   ├── static
   │   └── home.css
   │   └── result.css
   ├── venv/
   ├── Procfile
   ├── requirements.txt
   ├── .gitignore
   ├── LRmodel.pkl
   └── app.py
```

#### Preparation Step

**Flask** is a framework for small scale Python web development. In this project it's basically set up for routing and achieving our prediction functionality. In terms of the UI part, Flask supports template to render HTML to the browser. There're two HTML templates in the project, one for input and one for output. Correspondingly, two CSS files are attached for each HTML.

In order to deploy Flask on Heroku, several preparation steps should be done at first.

Since we have several libraries like Flask and sklearn, we need a virtual environment to manage the dependencies. And Python 3 comes bundled with the **venv** module to create virtual environments and that's what we use here.
```
python3 -m venv venv
source venv/bin/activate
```

After virtual environment is set up, we can use **pip** to install libraries just like this:
```
pip install flask
```

Besides, a server is needed to handle real requests and here we will use **Gunicorn**, which is compatible with various web frameworks and fairly speedy. And a **Procfile** is added to the root directory to tell Heroku the entry point for our application and how to start the web server. In Procfile we will write this line:
```
web: gunicorn app:app
```

And importantly, we also need to tell Heroku what libraries we intend to use to run this application successfully. Everything is writen to a **requirements.txt** file. But we can use pip to help us write:
```
pip freeze > requirements.txt
```

After these preparation step, we can safely deploy our app on Heroku.



#### Deploy Step

Basically, we will create an app on Heroku and use git to push the repository on Heroku.

So at first, log in by execute folloiwing command
```
heroku login
```

After enter both account and password, we can create a new app for our project on Heroku
```
heroku create
```
This will generate a URL for the app and another one for remote git repository.

Initialize the local git repository, add whole project and commit before push
```
git init
git add .
git commit -m "initial commit"
```

Finally, push our local repository to remote
```
git push heroku master
```

Now, you are free to open the web and enter the document content you want to classify on
```
https://serene-castle-92796.herokuapp.com
```
