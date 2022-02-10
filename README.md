# Parkinson's Disease Prediction Using Azure ML
## Summary
In this project, we have used the **Parkinson's Disease Dataset** that contains the biomedical voice measurements of various people. Hence, we seek to predict if a person has the disease or not by using two algorithms. We will be comparing the accuracy of **Hyperdrive Run with tuned hyperparameters** and **AutoML Run** on Microsoft Azure. The result will be binary, i.e. "0" for healthy and "1" for those with the disease. After comparing the performances of both algorithms, we deploy the best performing one. The model can be consumed from the generated REST endpoint.

![Diagram](https://user-images.githubusercontent.com/61888364/103487129-08036d00-4e29-11eb-9419-a63e83287971.png)

## Dataset Overview
The dataset was created by Max Little of the University of Oxford, in collaboration with the National Centre for Voice and Speech, Denver, Colorado, who recorded the speech signals. The original study published the feature extraction methods for general voice disorders.

Parkinson's Disease is a brain disorder that targets the nervous system of human body. This results in tremors, stiffness and disturbs or slows the movement. When the nerve cell damage in the brain cause the dopamine levels to go down, it leads to Parkinson's. Immediate medication can help to control symptoms.

It is a multivarite dataset that contains the range of biomedical voice measurements of 31 people, out of which 23 had Parkinson's disease. Each column is itself a voice measure and rows correspond to 195 voice recordings from those individuals.

![parkinson's stages](https://user-images.githubusercontent.com/61888364/103488601-f2e00b80-4e33-11eb-9baa-5f9599c82de0.png)

### Citation: 
Exploiting Nonlinear Recurrence and Fractal Scaling Properties for Voice Disorder Detection', Little MA, McSharry PE, Roberts SJ, Costello DAE, Moroz IM. BioMedical Engineering OnLine 2007, 6:23 (26 June 2007)

### Task:
Since, its a **classification task with binary output**, the column **"status"** will be used to determine if a person is healthy, denoted by "0" or is having Parkinson's disease, denoted by "1". 

### Attributes:
- **Matrix column entries (attributes):**<br>
- **name** - ASCII subject name and recording number <br>
- **MDVP:Fo(Hz)** - Average vocal fundamental frequency<br>
- **MDVP:Fhi(Hz)** - Maximum vocal fundamental frequency<br>
- **MDVP:Flo(Hz)** - Minimum vocal fundamental frequency<br>
- **MDVP:Jitter(%),MDVP:Jitter(Abs),MDVP:RAP,MDVP:PPQ,Jitter:DDP** - Several measures of variation in fundamental frequency<br>
- **MDVP:Shimmer,MDVP:Shimmer(dB),Shimmer:APQ3,Shimmer:APQ5,MDVP:APQ,Shimmer:DDA** - Several measures of variation in amplitude<br>
- **NHR,HNR** - Two measures of ratio of noise to tonal components in the voice<br>
- **status** - Health status of the subject (one) - Parkinson's, (zero) - healthy<br>
- **RPDE,D2** - Two nonlinear dynamical complexity measures<br>
- **DFA** - Signal fractal scaling exponent<br>
- **spread1,spread2,PPE** - Three nonlinear measures of fundamental frequency variation<br>

### Access to the dataset:
Since the dataset was available in ASCII CSV format, therefore it has been provided in this repository itself [here](https://github.com/sg7801/Parkinsons-Disease-Prediction/blob/main/parkinsons.txt).

# Automated ML Run

Firstly, We used the **TabularDatasetFactory** to create a dataset from the provided link. Then we split the train and test sets and upload them to datastore. Then, we define the task as per the below mentioned code.

- **Below are the settings used for AutoML Run** :
```
automl_settings = {
    "n_cross_validations": 5,
    "experiment_timeout_minutes" :20,
    "primary_metric": 'accuracy',
    "max_concurrent_iterations": 4,
}
automl_config = AutoMLConfig(
    task="classification",
    compute_target=compute,
    enable_early_stopping= True,
    max_cores_per_iteration=-1,
    training_data=training_data,
    label_column_name="status",
    **automl_settings
    )
```
- **Below are the defination and reasons why we above settings for our AutoML Run:**

![automlsettings](https://user-images.githubusercontent.com/61888364/103489212-c975ae80-4e38-11eb-8cf8-36dca5859860.jpg)

## Results of AutoML Run

After the submission, we found that **VotingEnsemble Algorithm** resulted with the best model with **accuracy 0.97906**, **precision_score_weighted 0.99740** and **precision_score_micro 0.99580**. Enabling of the automatic featurisation resulted in Data guardrails including Class balancing detection, Missing feature values imputation and High cardinality feature detection that checks over the input data to ensure quality in training the model.
- **Below image shows that the run has been successfully completed in notebook**
![1  Run completed AML](https://user-images.githubusercontent.com/61888364/103493229-0f8d3b00-4e56-11eb-9b4b-95d55418bc72.png)

- **Below is the image showing Best performing model**
![2  AML BEST MODEL](https://user-images.githubusercontent.com/61888364/103493293-7874b300-4e56-11eb-8968-acf18c0b2da5.png)

- **Now, we retrieved and saved the best model**
![3 retrieveing and saving best model](https://user-images.githubusercontent.com/61888364/103503528-92c08800-4e7a-11eb-8526-3652d815b432.png)

- **Below images show the explaination of Voting Ensemble Algorithm**
![4](https://user-images.githubusercontent.com/61888364/103493492-b45c4800-4e57-11eb-99d9-75d50379e658.png)
![5](https://user-images.githubusercontent.com/61888364/103493499-b8886580-4e57-11eb-9c5f-1142a38d885d.png)

# Hyperdrive Run

I started with the Training Script - train.py which used the Scikit-Learn Logistic Regression. It starts with a clean_data function that cleans the missing values from the dataset and one hot encodes data. I passed the required parameters and then imported the data from the specific URL using TabularDatasetFactory. Then, the data was split into the train and test sets. Finally, parameters were passed in the Logistic Regression Algorithm.

## **Hyperparameter tuning, termination policy and estimator in Hyperdrive run**

![6 Hyperdrive](https://user-images.githubusercontent.com/61888364/103493701-d4403b80-4e58-11eb-9b9b-077150466479.png)

Firstly, we create the different parameters that will be used during the training. They are **"--C"** and **"--max_iter"**. On these we have used the **RandomParameterSampling**. Then we use **"uniform"** that specifies the uniform distribution from which the samplers are taken for "--C" and **"choice"** to choose values from the discrete set of values for "--max_iter".

The Parameter Sampler chosen was - RandomParameterSampling. The major edge it has over other Samplers is of choosing random values from the search space with ease. It can choose values for the hyperparameters by exploring wider pool of values than others.

Then, we define our early termination policy with **evaluation_interval=1, slack_factor=0.02, slack_amount=None and delay_evaluation=0** using the **BanditPolicy** class. This is done to terminate the run that are not performing up to the mark. Starting at specified evaluation_interval, any run resulting in smaller value of primary metric gets cancelled automatically.

Then, we create the estimator and the hyperdrive. We have used **train.py** to perform the **Logistic Regression algorithm**. Since the output that we will predict is binary i.e. "0" for healthy or "1" for those with disease , hence we used Logistic Regression.

Now, we define the Hyperdrive Configuration. We give **max_concurrent_runs value of 4**, i.e. the maximum parallel iterations will be four and **max_total_runs** will be 22 since we only have 195 rows to evaluate.

## **Results of the Hyperdrive Run:**
- **Below screenshot shows the completed hyperdrive run:**
![8 run details](https://user-images.githubusercontent.com/61888364/103493978-8a585500-4e5a-11eb-9f27-8f604220e63c.png)

**Best Model:**
The Best Model of Hyperdrive had **Accuracy** of **0.9056603773584906**, **Regularization Strength** of **0.04411012133409599** and **Max iterations** of **200**. This resulted in value of **'--C'** = **0.04411012133409599** and **'--max_iter'** = **200** .

- **Below screenshot shows the best model details:**

![7 Best Model](https://user-images.githubusercontent.com/61888364/103493832-a7d8ef00-4e59-11eb-8910-5bdd1d492ab8.png)

- **Below screenshot shows the hyperdrive run in workspace:**
![9 hyperdrive in workspace](https://user-images.githubusercontent.com/61888364/103494103-1e2a2100-4e5b-11eb-994e-e45feb5fff5f.png)

# Model Deployment
 Since the best performing model came to be **AutoML** run that had **Voting Ensemble Algorithm** with **accuracy** of **0.97906**. Hence now we will deploy it.
- **Firstly, we will register the model, create an inference configuration and then deploy the model as a webservice.**
 
 ![10](https://user-images.githubusercontent.com/61888364/103494330-5f6f0080-4e5c-11eb-9274-882844c8f12c.png)
 
- **Then, we download the conda environment file and define environment. Then we download the scoring file produced by AutoML and set inference configuration.**

![11](https://user-images.githubusercontent.com/61888364/103494393-b1178b00-4e5c-11eb-934b-c78587e8f08d.png)

- **Now, we set the ACI Webservice configuration. Then, we deploy model as webservice.**

![12](https://user-images.githubusercontent.com/61888364/103494490-21bea780-4e5d-11eb-8cff-bf186b0b7f05.png)

- **We get the Service State as "Healthy", Scoring URI and Swagger URI respectively.**

![13](https://user-images.githubusercontent.com/61888364/103494517-3c911c00-4e5d-11eb-9ddb-623cc2a2d895.png)

- **Now, we select any three samples from the dataframe. Then, we conert records to json data file.**

![14](https://user-images.githubusercontent.com/61888364/103494566-795d1300-4e5d-11eb-9c63-a31bd1ad61b5.png)

- **We used 3 random sample points to test our endpoint for actual value and predicted value from dataset.**

![15](https://user-images.githubusercontent.com/61888364/103494707-fab4a580-4e5d-11eb-9159-16be90642d79.png)

![16](https://user-images.githubusercontent.com/61888364/103494714-fe482c80-4e5d-11eb-8a9b-54de970fe878.png)

- **Here, we can see the endpoint in workspace in "Healthy" Deployment State.**

![20](https://user-images.githubusercontent.com/61888364/103494918-dad1b180-4e5e-11eb-8c34-f9a4affc930c.png)

- **Below screenshot shows the REST Endpoint with Authentication keys (both primary key and secondary key) available.**
![21](https://user-images.githubusercontent.com/61888364/103494920-db6a4800-4e5e-11eb-8007-97dae2ac6517.png)

# Screen Recording: 

Link to the screencast is [here](https://drive.google.com/file/d/1i6OwsxOav1sy6aFsuUQYrh8paMg-6PsV/view?usp=sharing).

# Future Work:

- The major areas of future improvement involve the running of the model for much longer time and trying different parameters to get even better accuracy. 
- We can use GPU's instead of CPU's to improve the performance. Since CPU's might reduce the costs but in terms of performance and accuracy GPU's outperform CPU's.
- We can enable Deep Learning as well in the Auto ML Experiment for better results as it will consider different patterns and algorithms. Hence, improving the accuracy.
