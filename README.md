# BKFS HeavyWater Machine Learning Assessment

### All repository file information: -
* shuffled-full-set-hashed.csv : - BKFS Heavy Water Dataset
* document-classification-test.py: - python file
* document-classification-test.ipynb: - jupyter file
* app.py: - python file to create app using flask
* .pkl files: ML models in pickle form
* requirements.txt: -All packages with versions that need to be installed
* templates: - consists of html files used for front end design

### Problem Statement

The goal of the problem is to build a text classification model based on machine learning. "shuffled-full-set-hashed.csv" is the input file in csv format, which contains all document types in target variable that is required to train and test our model. The document entry format is like this:

```
CANCELLATION NOTICE,641356219cbc f95d0bea231b ............ 52102c70348d b32153b8b30c
```

### General Steps

Since the label is given, we will do supervised training on our dataset. Our main steps are:
1. Data Preprocessing: - Load our dataset and perform pre-processing. 
2. Data  Modeling: - create a model on labelled dataset.
3. Model Evaluation: - look at the performance wheter it's fine or not
4. create app python script using flask
5. upload in github and then deploy in heroku


### Deploy Model

I plan to deploy the model on Heroku at first because I have read about Heroku deployment process earlier. If I get a time, I will try to use AWS to deploy.

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

**Flask** is a framework for Python web development. In UI part, Flask supports template to render HTML to the browser. There're two HTML templates in the project, one for input and one for output. Correspondingly, two CSS files are attached for each HTML.

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
https://bkfs-assessment.herokuapp.com/
```

[Note]:- If any issue comes in heroku deployment, please follow the following steps: -
create an app -> go to the settings -> Add buildpack -> "heroku/python"
