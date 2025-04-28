from airflow import DAG
from airflow.operators.empty import EmptyOperator
from airflow.operators.bash import BashOperator
from airflow.operators.python import ShortCircuitOperator, PythonOperator
from airflow.utils.task_group import TaskGroup
from datetime import datetime
import yaml
import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.load_dataset import check_dataset
from src.generate_synthetic_data import generate
from src.split_dataset import split_dataset
from src.train_model import train_model

HOME_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

config_path = os.path.join(HOME_DIR, 'configs', 'train_model_config.yaml')
with open(config_path, 'r') as f:
    config = yaml.safe_load(f)

default_args = {
    'owner': 'airflow',
    'start_date': datetime(2023, 10, 10)
}

with DAG(
    dag_id='model_dag',
    default_args=default_args,
    schedule=None,
    catchup=False,
    params=config['params'],
    tags=config['tags'],
) as dag:
    
    # 1. Load dataset
    # 2. Generate synthetic data
    # 3. Split dataset
    # 4. Train model
    # 5. TODO Evaluate model on test set
    # 6. TODO Deploy for android


    # Load dataset
    dataset_task = PythonOperator(
        task_id = 'load_dataset',
        python_callable = check_dataset,
        op_kwargs = {
            'data_path': os.path.join(HOME_DIR,'{{ params.dataset_path }}')
        }
    )

    # Synthetic data
    with TaskGroup("synthetic_data") as synthetic_data_task:
        with TaskGroup("delete") as delete_synthetic_data:
            check_delete_synthetic = ShortCircuitOperator(
                task_id='check_delete_synthetic',
                python_callable=lambda **kwargs: kwargs['params'].get('delete_synthetic', False),
                ignore_downstream_trigger_rules=False
            )

            # Delete synthetic data
            delete_data = BashOperator(
                task_id='delete_synthetic_data',
                bash_command = f"find {os.path.join(HOME_DIR, '{{ params.dataset_path }}')} -type f -name '*_synthetic_*' -delete",
            )
            check_delete_synthetic >> delete_data

        with TaskGroup("generate") as generate_synthetic_data:
            # Check if synthetic data generation should be skipped
            check_generate_synthetic = ShortCircuitOperator(
                task_id='check_generate_synthetic',
                python_callable=lambda **kwargs: kwargs['params'].get('generate_synthetic', False),
                ignore_downstream_trigger_rules=False
            )

            # Generate synthetic data
            generate_data = PythonOperator(
                task_id = 'generate_synthetic_data',
                python_callable = generate,
                op_kwargs = {
                    'data_path': os.path.join(HOME_DIR,'{{ params.dataset_path }}'),
                    'num_images': '{{ params.synthetic_per_image }}'
                }
            )
            check_generate_synthetic >> generate_data

        delete_synthetic_data >> generate_synthetic_data
    

    # Split dataset
    with TaskGroup("split") as split_dataset_task:
        # Check if dataset splitting should be skipped
        check_split_dataset = ShortCircuitOperator(
            task_id='check_split_dataset',
            python_callable=lambda **kwargs: kwargs['params'].get('split_dataset', False),
            ignore_downstream_trigger_rules=False
        )

        # Split dataset
        split_task = PythonOperator(
            task_id = 'split_dataset',
            python_callable = split_dataset,
            op_kwargs = {
                'data_path': os.path.join(HOME_DIR,'{{ params.dataset_path }}'),
                'train_ratio': '{{ params.train_ratio }}',
                'val_ratio': '{{ params.val_ratio }}',
                'test_ratio': '{{ params.test_ratio }}',
                'copy': '{{ params.copy_images }}'
            },
            trigger_rule='all_done'
        )
        check_split_dataset >> split_task


    # Train model
    train_model_task = PythonOperator(
        task_id='train_model',
        python_callable=train_model,
        op_kwargs={
            'train_dir': os.path.join(HOME_DIR, '{{ params.dataset_path }}', 'train'),
            'valid_dir': os.path.join(HOME_DIR, '{{ params.dataset_path }}', 'valid'),
            'test_dir': os.path.join(HOME_DIR, '{{ params.dataset_path }}', 'test'),
            'model_path': os.path.join(HOME_DIR, 'output', 'models', '{{ params.model_name }}.{{ params.model_extension }}'),
            'img_width': '{{ params.img_width }}',
            'img_height': '{{ params.img_height }}',
            'learning_rate': '{{ params.learning_rate }}',
            'num_classes': "{{ ti.xcom_pull(task_ids='load_dataset') }}",
            'batch_size': '{{ params.batch_size }}',
            'epochs': '{{ params.num_epochs }}',
            'fig_path': os.path.join(HOME_DIR, 'output', 'plots', '{{ params.model_name }}_history.png')
        }
    )

    dataset_task >> synthetic_data_task >> split_dataset_task >> train_model_task