# User Guide
Code Snippets are available in notebooks for all the code blocks mentioned below. Snippets can be easily dragged-and-dropped for usage.
## Datasets
TODO: Define dataset here
### Creating datasets
A dataset can be created: 
 1. From scratch:  
    ```python
    dataset = Dataset.create(
		dataset_name='Flight data', # name of dataset
	    dataset_version = '1.0.0', # Optional str
	    dataset_tags = ['travel', 'airplane'],  # Optional list of str
    )  
    ```
    ![Create dataset](images/create_dataset.png)
   
 2. By using another dataset as a base:
		Specify the parent datasets under **parent_datasets**
	```python
	dataset = Dataset.create(
		dataset_name='Flight data', # name of dataset
		dataset_version = '1.0.1', # type: Optional str
		dataset_tags = ['travel', 'airplane'],  # type: Optional list of str
		parent_datasets = ["c6b147a022274092bc3e1b75f5f09d7a"] # specify the ID of the parent dataset
	)
	```
    ![Create dataset](images/create_dataset_from_another.png)

 3. Combining multiple datasets:
	```python
	Dataset.squash(dataset_name="Flight data", dataset_ids=["87033f7004164dd0aa7106d419e3aa1a", "c6b147a022274092bc3e1b75f5f09d7a"])
	```

### Add files
Files can be added to a dataset:
 1.  By uploading local files:
	  **path** refers to the relative path of the files/folder
	  **wildcard** can be used to select specific file, ex. wildcard="*.png" to select all .png files
	  **dataset_path** is the path within the dataset where we want the files to be uploaded
		```python
	 dataset.add_files(path='paths.csv', wildcard=None, dataset_path=None) # local path of files to be uploaded 
	 dataset.upload()
	 ```
     ![Uploaded data](images/uploaded_data.png)

2. By uploading S3 files:
	**source_url** for S3 files is in the format: s3://bucket-name/file-path
	```python
	 dataset.add_external_files(source_url="", wildcard=None, dataset_path=None) # upload files from s3
	dataset.upload()
	 ```
3. Uploading S3 files from the non-default S3 account:
	**source_url** for S3 files is in the format: s3://bucket-name/file-path
	**aws_access_key**, **aws_secret_key**, & **region** are the AWS credentials required to access the S3 files of the account.
	```python
	dataset.add_external_files(source_url="",
                          wildcard=None, 
                          dataset_path=None
                          aws_access_key="",
                          aws_secret_key="",
                          region="")
	dataset.upload()
	```
### Download a dataset
To download a dataset, use:
```python
dataset = Dataset.get(dataset_id="c6b147a022274092bc3e1b75f5f09d7a") # get ID from GUI
dataset_files_path = dataset.useDataset() # dataset_files_path is the local path of the downloaded dataset.
```
### Archive a dataset
Datasets can be archived from the UI, where they can be then viewed under the archived section & can be restored
![Archive datasets](images/archive_datasets.png)
### Delete a dataset
To delete:
```python
Dataset.delete(dataset_id="c6b147a022274092bc3e1b75f5f09d7a") # get ID from GUI
```
### Publish a dataset
Publishing a dataset means making it non-editable
```python
dataset = Dataset.get(dataset_id="c6b147a022274092bc3e1b75f5f09d7a") # get ID from GUI
dataset.publish()
```

## Experiments
An experiment refers to a code execution session. 
An experiment captures the following:
1. Python environment
2. Run information
3. Console output
4. Hyperparameters
5. Artifacts
6. Plots (recorded automatically)

### Create an experiment
```python
experiment = Experiment.init(
    experiment_name='Linear Regression', # experiment name of at least 3 characters
    tags=['Regression','Best fit'], # Add a list of tags (str) to the created Experiment
)
```
![Experiment](images/experiment.png)


### Get an existing experiment
```python
experiment = Experiment.get_experiment(experiment_id='c6b147a022274092bc3e1b75f5f09d7a') # get experiment ID from GUI
```


### Monitor an experiment's progress
```python
experiment = Experiment.get_experiment(experiment_id='c6b147a022274092bc3e1b75f5f09d7a') # get experiment ID from GUI
experiment.set_progress(0)
# experiment doing stuff
experiment.set_progress(50)
print(experiment.get_progress())
```
An experiment's progress can be seen in the experiments list panel as a loader & in the INFO tab. In the image below, the progress is 50%.

![Progress](images/progress.png)

### Hyperparameters
Hyperparameters are a script's configuration. Many command line parameters are automatically logged by CaiML including:
- click
- argparse 
- Python Fire 
- LightningCLI

TensorFlow Definitions are also automatically logged: TensorFlow MNIST & TensorBoard PR Curve

Here's an example of automatic logging of tensorflow hyperparameters

![Hyperparameters](images/hyperparameters.png)

#### Connect hyperparameters to experiment
Connect a set of hyperparameters to an already existing experiment
```python
# Connect Hyperparameters to Experiment.
# experiments
from cai.automl import Experiment

# Get an instance of an experiment
experiment = Experiment.get_experiment(experiment_id='') # Get experiment_id from GUI or console.

# Define hyperparameters to connect to experiment
args = {
        '': val,  # Fill in '' with name of hyperparameter, replace val with the hyperparameter's default value. Add more hyperparameter-value pairs as required.
       }
# Example:
# args = {
#        'batch_size': 64,
#        'layer_1_units': 128,
#        }
# Connect Hyperparameters to experiment
experiment.connect(args)
```

#### Get hyperparameters
Get hyperparameters connected to an already existing experiment
```python
# Get an experiment's hyperparameters
# experiments
from cai.automl import Experiment

# get an instance of an exeriment
experiment = Experiment.get_experiment(experiment_id='') # get experiment_id from GUI

# Returns all hyperparameters of the experiment
experiment.get_parameters()
```

#### Create optimizer experiment
Create hyperparameter optimizer experiment from an already existing experiment
```python
# Create an optimizer experiment.
# experiments
from cai.automl import Experiment

#Initialize an optimizer experiment
experiment = Experiment.init(project_name='',
                 experiment_name='',
                 task_type=Experiment.ExperimentTypes.optimizer,
                 reuse_last_task_id=False) # Fill the project and experiment names.

# Experiment arguments containing the id of an already created experiment whose hyperparameters we want to optimize.
args = {
    'template_experiment_id': None, # Provide the template experiment id we want to optimize, if available.
}
args = experiment.connect(args)

# Get the template experiment id that we want to optimize, using the experiment name instead.
if not args['template_experiment_id']:
    args['template_experiment_id'] = Experiment.get_experiment(
        experiment_name='').id # Fill in the experiment name.
```

#### Define optimizer
Define the optimizer class
```python
# Define the optimizer.
# experiments
!pip install --quiet optuna
!pip install --quiet hpbandster
from cai.automl import Experiment
from cai.automl.automation import (
    DiscreteParameterRange, HyperParameterOptimizer, GridSearch,
    RandomSearch, UniformIntegerParameterRange, UniformParameterRange, ParameterSet
    )
from cai.automl.automation.optuna import OptimizerOptuna
from cai.automl.automation.hpbandster import OptimizerBOHB

# Get an instance of an experiment
experiment = Experiment.get_experiment(experiment_id='') # Get optimizer experiment_id from GUI or console.

optimizer = HyperParameterOptimizer(
    # This is the experiment we want to optimize
    base_task_id='', # Fill id of the base experiment we want to optimize.
    # We define the hyper-parameters to optimize here.
    # The parameter name should match with UI: <section_name>/<parameter>
    # For example, if in base experiment we have 'General' section, and a parameter 'batch_size' in it, naming should be 'General/batch_size'
    # The hyperparameters are found in the 'General' section by default.
    # Example use case for ANN -
    # hyper_parameters=[
    #    UniformIntegerParameterRange('General/layer_1', min_value=128, max_value=512, step_size=1),
    #    UniformIntegerParameterRange('General/layer_2', min_value=128, max_value=512, step_size=1),
    #    UniformParameterRange('General/drop_rate', min_value=0, max_value=1, step_size=0.01),
    #    UniformIntegerParameterRange('General/batch_size',  min_value=96, max_value=160, step_size=10),
    #    UniformIntegerParameterRange('General/epochs', min_value=20, max_value=40, step_size=1),
    # ],
    hyper_parameters=[
        UniformIntegerParameterRange('', min_value=val, max_value=val, step_size=val), # Sample parameter as integer uniformly from min_value to max_value with difference of step_size. Fill name in '' and values in val.
        UniformParameterRange('', min_value=val, max_value=val, step_size=val), # Sample parameter as float uniformly from min_value to max_value with difference of step_size. Fill name in '' and values in val.
        DiscreteParameterRange('', values=[val, val]), # Sample parameter discretely from the given values. Fill name in '' and values in val.
        ParameterSet(parameter_combinations=[{'': val, '': val},
                                             {'': val, '': val}]), # Sample parameter discretely according to the combinations provided as sets. Fill name in '' and values in val.
    ],
    
    # The objective metric we want to maximize/minimize
    objective_metric_title='', # Fill title name for objective metric.
    objective_metric_series='', # Fill series name for objective metric.
    
    # Define whether to maximimize or minimize objective.
    objective_metric_sign='', # Fill 'max' for maximize, 'min' for minimize.
    
    # Define number of concurrent tasks to run.
    max_number_of_concurrent_tasks=5, # Fill number of concurrent tasks.
    
    # Define the search strategy to be used.
    optimizer_class=OptimizerOptuna, # Fill in the optimizer class. GridSearch, RandomSearch, OptimizerOptuna and OptimizerBOHB are currently available.

    # Provide the sampler class, if using OptunaOptimizer. The classes are available in the module optuna.samplers.
    # If None, TPESampler class will be used by default.
    sampler_class=None, # Fill in the sampler class.

    # Provide the pruner class, if using OptunaOptimizer. The classes are available in the module optuna.pruners.
    # If None, MedianPruner class will be used by default.
    pruner_class=None, # Fill in the pruner class.

    # Set number of maximum jobs to launch for optimization. None is unlimited.
    total_max_jobs=10, # Fill in maximum total jobs.

    # Set maximum number of iterations for experiment to execute.
    max_iteration_per_job=30, # Fill in maximum no. of iterations per job.
)
```

#### Run optimizer
Run the hyperparameter optimization experiment
```python
# Run the optimizer.
# experiments
from cai.automl import Experiment

# Get an instance of an experiment
experiment = Experiment.get_experiment(experiment_id='') # Get optimizer experiment_id from GUI or console.

# Set reporting period in minutes - the interval after which optimization report is generated in console. 
optimizer.set_report_period(0.5) # Fill in reporting period.

# Start optimization process locally.
optimizer.start_locally()

# Set time period for the optimization prcoess in minutes.
optimizer.set_time_limit(in_minutes=120) # Fill in time period.

# Wait until process is done.
optimizer.wait()

# Print the top performing k experiments' ids.
top_exp = an_optimizer.get_top_experiments(top_k=5) # Fill in value of k.
print([t.id for t in top_exp])

# Stop background optimization.
optimizer.stop()

print('Hyperparameter Optimization is complete.')
```

#### Manually log hyperparameters
Hyperparameters can be manually logged
```python
experiment = Experiment.get_experiment(experiment_id='c6b147a022274092bc3e1b75f5f09d7a') # get experiment ID from GUI
experiment.set_parameters_as_dict({'epochs': 20, 'max_value':100}) # pass hyperparameters as string:number dictionary
```

#### Log python objects
Python objects such as variables, classes, numpy objects can be logged
```python 
experiment = Experiment.get_experiment(experiment_id='c6b147a022274092bc3e1b75f5f09d7a') # get experiment ID from GUI
params_dict = {'epochs': 20, 'max_value':100}
experiment.connect(params_dict)
```

#### Log complex objects
Log objects more complex than a dictionary
```python 
# Logging blob objects
# experiments

from cai.automl import Experiment

# get an instance of an exeriment
experiment = Experiment.get_experiment(experiment_id='') # get experiment_id from GUI

# To log objects more complicated than a dictionary
# configuration: variable name of the data being passed
experiment.connect_configuration(
  name='', configuration=None
)
# Example:
# model_config_dict = {
#    'value': 13.37,  'dict': {'sub_value': 'string'},  'list_of_ints': [1, 2, 3, 4],
# }
# experiment.connect_configuration(
#   name='dictionary', configuration=model_config_dict
# )
```

#### Log user properties
Log user metadata that does not impact code execution
```python 
# Set User Properties
# experiments
from cai.automl import Experiment

# get an instance of an exeriment
experiment = Experiment.get_experiment(experiment_id='') # get experiment_id from GUI

# User Properties do not impact code execution
# Can be used to log metadata as a dictionary
# A user property can contain the fields - name, value, description and type
experiment.set_user_properties(
  {"": "", "": ""}
)
# Example:
# experiment.set_user_properties(
# {"name": "my_name", "description": "my_desc", "value": "my_val"}
# )
```

### Artifacts
CaiML allows easy storage of experiments' output products as artifacts that can later be accessed easily and used. Some examples of artifacts are:
-   Numpy objects
-   Pandas DataFrames
-   PIL
-   Files and folders
-   Python objects

![Artifacts](images/artifacts.png)


#### Upload python objects as artifacts

Python objects can be uploaded as artifacts:
```python
experiment = Experiment.get_experiment(experiment_id='c6b147a022274092bc3e1b75f5f09d7a') # get experiment ID from GUI
params_dict = {'epochs': 20, 'max_value':100}
experiment.upload_artifact(name='Parameters dictionary', artifact_object=params_dict)
```
#### Upload files as objects
To upload files/folders as artifacts:
**artifact_object** refers to the relative path of the file/folder
```python
experiment = Experiment.get_experiment(experiment_id='c6b147a022274092bc3e1b75f5f09d7a') # get experiment ID from GUI
experiment.upload_artifact(name='Sunflower image', artifact_object='images/sunflower.jpg')
```

### Finalise an experiment
To make an experiment non-editable:
```python
experiment = Experiment.get_experiment(experiment_id='') # get experiment ID from GUI
experiment.close()
```

### Plots
Matplotlib plots are automatically recorded by CaiMl & can be viewed under the PLOTS tab.

## Models
CaiML provides automatic recording of standard python models such as:
- Tensorflow
- Keras
- Pytorch
- scikit-learn (only using joblib)
- XGBoost (only using joblib)
- FastAI
- MegEngine
- CatBoost

To record a model, simply initiate  a CaiML experiment in your notebook code:
```python
experiment = Experiment.init(
    experiment_name='Tensorflow Model', # experiment name of at least 3 characters
)
```

![Models](images/models.png)

To manually log a model:

### Manually log a model
```python
experiment = Experiment.init(experiment_name="Model experiment")
output_model = OutputModel(experiment=experiment, 
                            name="Sample model", # Optional
                            framework=None, # Optional, framework to be used like PyTorch
                            config_dict=None, # Optional, configuration as a dictionary
                            label_enumeration=None, # Optional, add labels to your model as a dictionary
                            tags=None) # Optional, add tags to your model as list of strings 

# For the model to be recorded, add a weights file
output_model.update_weights(weights_filename='', # Either add a local weights file here
                            registered_uri='https://github.com/ultralytics/yolov5/releases/download/v6.2/yolov5x6.pt') # or add a valid URL of the weights file here
```

### Get instance of a registered model
Get instance of a model registered with CaiML
```python
from caiml import InputModel
input_model = InputModel(model_id="c6b147a022274092bc3e1b75f5f09d7a") # get model_id from the GUI
```

### Get instance of an external model
```python
from caiml import InputModel
input_model = InputModel.import_model(weights_url="https://github.com/ultralytics/yolov5/releases/download/v6.2/yolov5x6.pt", # A valid URL for the weights file. For a local file user file://
                            name="Sample Model", # Optional
                            framework=None, # Optional, framework to be used like PyTorch
                            config_dict=None, # Optional, configuration as a dictionary
                            label_enumeration=None, # Optional, add labels to your model as a dictionary
                            tags=None) # Optional, add tags to your model as list of strings
```

### Update a model
A model's weight file can be updated
```python
from caiml import OutputModel
output_model.update_weights(weights_filename='', # Either add a local weights file here
                            registered_uri='https://github.com/ultralytics/yolov5/releases/download/v6.2/yolov5x6.pt') # or add a valid URL of the weights file here
```

### Add metadata to model
```python
from caiml import InputModel, OutputModel
model.set_metadata("epochs", 20) # model can be an instance of either of OutputModel/InputModel class
```

### Finalize a model
Make a model non-editable
```python
from caiml import InputModel, OutputModel
model.publish() 
```

### Get Model URL
Similarly other properties can be gotten from a model. An exhaustive list can be found in the code snippets in notebook.
```python
from caiml import InputModel, OutputModel
url = model.url # model can be an instance of either of OutputModel/InputModel class
```
