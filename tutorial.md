# GSP076 #

# Tutorial: Cloud ML Engine: Qwik Start #

## Cloud ML Engine
This lab will give you hands-on practice with TensorFlow model training, both locally and on Cloud ML Engine. After training, you will learn how to deploy your model to Cloud ML Engine for serving (prediction). You'll train your model to predict income category of a person using the United States Census Income Dataset.

This lab gives you an introductory, end-to-end experience of training and prediction on Cloud Machine Learning Engine. The lab will use a census dataset to:

* Create a TensorFlow training application and validate it locally.
* Run your training job on a single worker instance in the cloud.
* Run your training job as a distributed training job in the cloud.
* Optimize your hyperparameters by using hyperparameter tuning.
* Deploy a model to support prediction.
* Request an online prediction and see the response.
* Request a batch prediction.

### What you will build ###
The sample builds a wide and deep model for predicting income category based on United States Census Income Dataset. The two income categories (also known as labels) are:

>50K — Greater than 50,000 dollars
<=50K — Less than or equal to 50,000 dollars
Wide and deep models use deep neural nets (DNNs) to learn high-level abstractions about complex features or interactions between such features. These models then combine the outputs from the DNN with a linear regression performed on simpler features. This provides a balance between power and speed that is effective on many structured data problems.

The sample defines the model using TensorFlow's prebuilt DNNCombinedLinearClassifier class. The sample defines the data transformations particular to the census dataset, then assigns these (potentially) transformed features to either the DNN or the linear portion of the model.

## Prerequisites ##

 -  You have a Google Cloud Platform account and a Google Project (note the Google Project Id) provided by Gitlab. You can find this on the left side of the QwikLab page.

## Install Prerequisites

1. Verify current account from QwikLab
```bash
gcloud auth list
```

2. Check your current project.
```bash
gcloud config list project
```

## Install TensorFlow ##

3. Run the following command to install TensorFlow:
```bash
pip install --user --upgrade tensorflow
```

4. Verify the installation:
```bash
python -c "import tensorflow as tf; print('TensorFlow version {} is installed.'.format(tf.VERSION))"
```
    1. You can ignore any warnings that the TensorFlow library wasn't compiled to use certain instructions.


5. Clone the example repo
```bash
git clone https://github.com/GoogleCloudPlatform/cloudml-samples.git
```

6. Navigate to the cloudml-samples > census > estimator directory. The commands in this lab must be run from the estimator directory:
```bash
cd cloudml-samples
cd census
cd estimator
```

7. Check your current directory 
```bash
$pwd
```

## Develop and validate your training application locally ##
Before you run your training application in the cloud, get it running locally. Local environments provide an efficient development and validation workflow so that you can iterate quickly. You also won't incur charges for cloud resources when debugging your application locally.

Get your training data
The relevant data files, adult.data and adult.test, are hosted in a public Google Cloud Storage bucket.

You can read the files directly from Cloud Storage or copy them to your local environment. For this lab you will download the samples for local training, and later upload them to your own Cloud Storage bucket for cloud training.

Run the following command to download the data to a local file directory and set variables that point to the downloaded data files:1. Check your current directory 
```bash
mkdir data
gsutil -m cp gs://cloud-samples-data/ml-engine/census/data/* data/
```


8.  Now set the TRAIN_DATA and EVAL_DATA variables to your local file paths by running the following commands:
```bash
export TRAIN_DATA=$(pwd)/data/adult.data.csv
export EVAL_DATA=$(pwd)/data/adult.test.csv
```

9.  To open the adult.data.csv file, run the following command:
`
``bash
head data/adult.data.csv
```

10.  You will see that the data is stored in comma-separated value format that resembles the following:
Output |
-------|
39, State-gov, 77516, Bachelors, 13, Never-married, Adm-clerical, Not-in-family, White, Male, 2174, 0, 40, United-States, <=50K
50, Self-emp-not-inc, 83311, Bachelors, 13, Married-civ-spouse, Exec-managerial, Husband, White, Male, 0, 0, 13, United-States, <=50K
38, Private, 215646, HS-grad, 9, Divorced, Handlers-cleaners, Not-in-family, White, Male, 0, 0, 40, United-States, <=50K
53, Private, 234721, 11th, 7, Married-civ-spouse, Handlers-cleaners, Husband, Black, Male, 0, 0, 40, United-States, <=50K |

Now that you have downloaded and inspected your training data, you will install the necessary dependencies.

11. Install dependencies
Although TensorFlow is installed on Cloud Shell, you must run the sample's requirements.txt file to ensure you are using the same version of TensorFlow required by the sample:
```script
pip install --user -r ../requirements.txt
```
It will take a couple minutes for this command to complete. You will receive a similar output when it does:
Output |
-------|
Successfully installed Keras-2.2.4 future-0.16.0 numexpr-2.6.9 numpy-1.14.5 pandas-0.24.1 python-dateutil-2.8.0 pyyaml-3.13 scipy-1.2.1 setuptools-39.1.0 tensorboard-1.10.0 tensorflow-1.10.0 |


12. Run a local training job
A local training job loads your Python training program and starts a training process in an environment that's similar to that of a live Cloud ML Engine cloud training job.

Specify an output directory and set a MODEL_DIR variable by running the following command: 
```bash
export MODEL_DIR=output
```

13. Run this training job locally by running the following command:
```bash
gcloud ai-platform local train \
    --module-name trainer.task \
    --package-path trainer/ \
    --job-dir $MODEL_DIR \
    -- \
    --train-files $TRAIN_DATA \
    --eval-files $EVAL_DATA \
    --train-steps 1000 \
    --eval-steps 100
```

Note:
When you run the same training job on CMLE later in the lab, you'll see that the command is not much different from the above.

By default, verbose logging is turned off. You can enable it by setting the --verbosity tag to DEBUG. A later example shows you how to enable it.

Inspect the summary logs using Tensorboard
To see the evaluation results, you can use the visualization tool called TensorBoard. With TensorBoard, you can visualize your TensorFlow graph, plot quantitative metrics about the execution of your graph, and show additional data like images that pass through the graph. Tensorboard is available as part of the TensorFlow installation.

Follow the steps below to launch TensorBoard and point it at the summary logs produced during training, both during and after execution.

Launch TensorBoard:
```bash
tensorboard --logdir=$MODEL_DIR --port=8080
```
Click on the Web Preview icon, then Preview on port 8080. A new tab will open with TensorBoard running.
devshell-web-preview-button

Click on Accuracy to see graphical representations of how accuracy changes as your job progresses.

Type CTRL+C in Cloud Shell to shut down TensorBoard.

## Create a TensorFlow training application and validate it locally.

The output/export/census directory holds the model exported as a result of running training locally. List that directory to see the generated timestamp subdirectory:


## Run your training job on a single worker instance in the cloud.
## Run your training job as a distributed training job in the cloud.
## Optimize your hyperparameters by using hyperparameter tuning.
## Deploy a model to support prediction.
## Request an online prediction and see the response.
## Request a batch prediction.


#old lab

## Service Account Creation

First up, let's create an environment variable to store your Project Id. Please use the code snippet below to set the `PROJECT_ID` variable as given below:

```bash
export PROJECT_ID=<your_project_id>
```

Use the following `gcloud` commands to create a service account named `nlpapi-quickstart` as shown below:

```bash
gcloud iam service-accounts create nlpapi-quickstart
```
Next up, we generate the service account JSON key that will get downloaded to your current folder as `key.json` file. 

```bash
gcloud iam service-accounts keys create key.json --iam-account nlpapi-quickstart@$PROJECT_ID.iam.gserviceaccount.com
```
Finally, we use the APPLICATION DEFAULT CREDENTIALS and set the variable as given below:

```bash
export GOOGLE_APPLICATION_CREDENTIALS=key.json
```

## Use npm to install dependencies

Now, let us install the Node.js library for Google Natural Language API via the command given below:

```bash
npm install --save @google-cloud/language
```

## Source Code

Let us go through the JavaScript file to understand the source code. 

Click here: `walkthrough editor-open-file "Google-Cloud-Shell-Tutorial-Creation/index.js" "Open index.js"`

The code is taken from the official Github project present over [here](https://github.com/googleapis/nodejs-language).

## Run the application

We are ready to run our application and see the results. 

Run the `index.js` file via the command below:

```bash
node index.js
```

You should see both a score and magnitude provided for the sentence. You can change the sentence in the code file and try the API again, if you like. 

## Conclusion

`walkthrough conclusion-trophy`

Thanks for completing this tutorial. I hope you enjoyed the power of the Google Natural Language API.

### Next Steps:

 - Check out more information on [Google Cloud Natural Language API](https://cloud.google.com/natural-language/) 
 - [Natural Language API Basics](https://cloud.google.com/natural-language/docs/basics)
