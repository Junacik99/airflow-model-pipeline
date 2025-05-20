# Automated Training Pipeline
An automated model training pipeline for mostly image classification tasks.

## Airflow
Apache Airflow is a powerful open-source platform for programmatically orchestrating complex workflows, making it an excellent choice for managing machine learning (ML) model training pipelines. With its Python-based DSL, Airflow allows data scientists and engineers to define, schedule, and monitor workflows as Directed Acyclic Graphs (**DAGs**), ensuring tasks like data preprocessing, model training, and evaluation run in the correct order with proper dependencies.

One of Airflow’s key strengths is its built-in monitoring and retry mechanism, which helps handle failures gracefully—critical for long-running model training jobs. Features like dynamic pipeline generation, backfilling, and rich logging make it easy to debug and maintain ML workflows. By decoupling pipeline logic from execution, Airflow ensures that model training processes are robust, maintainable, and adaptable to changing data or requirements, making it a preferred tool for production-grade ML operations.

To **install** Airflow, please follow [this](https://airflow.apache.org/docs/apache-airflow/stable/installation/index.html) guide.

## Run
After successful installation of airflow, clone this **whole** repository to the *dags* folder. 

Run airflow with command `airflow standalone`, open `http://localhost:8080/` in browser and look for a DAG called **model_pipeline**. Trigger it with corresponding parameters:
- dataset_path: path to the dataset
- delete_synthetic: flag, to delete previously generated synthetic data
- generate_synthetic: flag, to generate synthetic data
- synthetic_per_image: how many synthetic data should be generated per image

- split_dataset: flag, split dataset into train, test and validation
- train_ratio: training split ratio
- val_ratio: validation split ratio
- test_ratio: testing split ratio
- copy_images: flag, move or copy data into split folders

- model_name: name of the created model
- model_extension: file extension for the model (should be *keras*, it will be automatically converted into *tflite* later)
- batch_size
- learning_rate
- num_epochs
- img_width: input image width
- img_height: input image height

## Customize
To create a custom model that should be trained, create a class in [models](./src/models.py) module. In [train_model](./src/train_model.py#35) module, change it to use newly created class instead of the *CardModel*.

By default, *BinaryCardModel* is used, when the number of classes specified is 2, otherwise *CardModel* is used.