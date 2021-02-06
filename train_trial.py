from sklearn.linear_model import LogisticRegression
import argparse
import os
import numpy as np
from sklearn.metrics import mean_squared_error
import joblib
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
import pandas as pd
from azureml.core.run import Run
from azureml.data.dataset_factory import TabularDatasetFactory

# TODO: Create TabularDataset using TabularDatasetFactory
# Data is located at:
# "https://automlsamplenotebookdata.blob.core.windows.net/automl-sample-notebook-data/bankmarketing_train.csv"

ds = Dataset.Tabular.from_delimited_files(path="https://automlsamplenotebookdata.blob.core.windows.net/automl-sample-notebook-data/bankmarketing_train.csv")

x, y = clean_data(ds)

# TODO: Split data into train and test sets.

### YOUR CODE HERE ###a
x_train, x_test, y_train,y_test = train_test_split(x,y,test_size = 0.3,random_state=123)

run = Run.get_context()

def clean_data(data):
    # Dict for cleaning data
    months = {"jan":1, "feb":2, "mar":3, "apr":4, "may":5, "jun":6, "jul":7, "aug":8, "sep":9, "oct":10, "nov":11, "dec":12}
    weekdays = {"mon":1, "tue":2, "wed":3, "thu":4, "fri":5, "sat":6, "sun":7}

    # Clean and one hot encode data
    x_df = data.to_pandas_dataframe().dropna()
    jobs = pd.get_dummies(x_df.job, prefix="job")
    x_df.drop("job", inplace=True, axis=1)
    x_df = x_df.join(jobs)
    x_df["marital"] = x_df.marital.apply(lambda s: 1 if s == "married" else 0)
    x_df["default"] = x_df.default.apply(lambda s: 1 if s == "yes" else 0)
    x_df["housing"] = x_df.housing.apply(lambda s: 1 if s == "yes" else 0)
    x_df["loan"] = x_df.loan.apply(lambda s: 1 if s == "yes" else 0)
    contact = pd.get_dummies(x_df.contact, prefix="contact")
    x_df.drop("contact", inplace=True, axis=1)
    x_df = x_df.join(contact)
    education = pd.get_dummies(x_df.education, prefix="education")
    x_df.drop("education", inplace=True, axis=1)
    x_df = x_df.join(education)
    x_df["month"] = x_df.month.map(months)
    x_df["day_of_week"] = x_df.day_of_week.map(weekdays)
    x_df["poutcome"] = x_df.poutcome.apply(lambda s: 1 if s == "success" else 0)

    y_df = x_df.pop("y").apply(lambda s: 1 if s == "yes" else 0)
    

def main():
    # Add arguments to script
    parser = argparse.ArgumentParser()

    parser.add_argument('--C', type=float, default=1.0, help="Inverse of regularization strength. Smaller values cause stronger regularization")
    parser.add_argument('--max_iter', type=int, default=100, help="Maximum number of iterations to converge")

    args = parser.parse_args()

    run.log("Regularization Strength:", np.float(args.C))
    run.log("Max iterations:", np.int(args.max_iter))

    model = LogisticRegression(C=args.C, max_iter=args.max_iter).fit(x_train, y_train)

    accuracy = model.score(x_test, y_test)
    run.log("Accuracy", np.float(accuracy))

if __name__ == '__main__':
    main()

#I have added the compute above but below is the code
#IDID ADDED COMPUTE CLUSTERS BUT HAVEN'T PERSONALIZED THEM
from azureml.core.compute import AmlCompute, ComputeTarget
from azureml.exceptions import ComputeTargetException

aml_compute_target = "cpu-cluster"
try:
    aml_compute = AmlCompute(ws, aml_compute_target)
    print("found existing compute target.")
except ComputeTargetException:
    print("creating new compute target")
    
    provisioning_config = AmlCompute.provisioning_configuration(vm_size = "STANDARD_D2_V2",
                                                                min_nodes = 1, 
                                                                max_nodes = 4)    
    aml_compute = ComputeTarget.create(ws, aml_compute_target, provisioning_config)
    aml_compute.wait_for_completion(show_output=True, min_node_count=None, timeout_in_minutes=20)
    
print("Azure Machine Learning Compute attached")

#sampler

ps = RandomParameterSampling(
    {
        '--batch-size': choice(25, 50, 100),
        '--first-layer-neurons': choice(10, 50, 200, 300, 500),
        '--second-layer-neurons': choice(10, 50, 200, 500),
        '--learning-rate': loguniform(-6, -1)
    }
)

#IDI polciies
                
early_termination_policy = BanditPolicy(evaluation_interval=2, slack_factor=0.1)

#estimator
#est = TensorFlow(source_directory=script_folder,                 
#                 compute_target=compute_target,
  #               entry_script='tf_mnist.py', 
    #             use_gpu=True,
      #           framework_version='2.0',
          #       pip_packages=['azureml-dataset-runtime[pandas,fuse]'])

from azureml.train.estimator import Estimator

est = Estimator(source_directory=source_directory, 
                compute_target=aml_compute_target, 
                entry_script='train.py', 
                conda_packages=['scikit-learn'])


#IDID added estimator
#from azureml.core import Dataset
#from azureml.data import OutputFileDatasetConfig

# create dataset to be used as the input to estimator step
#input_data = ds

# OutputFileDatasetConfig by default write output to the default workspaceblobstore
#output = OutputFileDatasetConfig()

#source_directory = 'estimator_train'

#from azureml.train.estimator import Estimator

#est = Estimator(source_directory=source_directory, 
             #   compute_target=aml_compute_target, 
              #  entry_script='train.py', 
               # conda_packages=['scikit-learn'])

#IDID CREATED HYPERDRIVECONFIG
hd_config = HyperDriveConfig(estimator=est, 
                             hyperparameter_sampling=ps,
                             policy=early_termination_policy,
                             primary_metric_name='validation_acc', 
                             primary_metric_goal=PrimaryMetricGoal.MAXIMIZE, 
                             max_total_runs=4,
                             max_concurrent_runs=4)
                             

