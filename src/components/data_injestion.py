# ============================================================
# IMPORTING REQUIRED LIBRARIES
# ============================================================

# os module is used for:
# - creating folders
# - joining file paths
# - interacting with operating system
import os

# sys module is used for:
# - handling system-level operations
# - getting detailed exception information
import sys

# Importing our custom exception class
# This helps in showing detailed custom error messages
from src.exception import CustomException

# Importing logging functionality v
# Logging helps us track execution step-by-step
from src.logger import logging

# Pandas library is used for:
# - reading CSV files
# - handling tabular data
# - performing data analysis
import pandas as pd

# train_test_split is used to divide dataset into:
# - training data
# - testing data
from sklearn.model_selection import train_test_split
# Importing dataclass decorator
from dataclasses import dataclass


# @dataclass automatically creates constructor (__init__)
# so we do not need to write it manually.
#
# It is mainly used for storing variables/configurations neatly.
#
# Example:
#
# Without dataclass:
#
# class Student:
#     def __init__(self,name,age):
#         self.name=name
#         self.age=age
#
#
# With dataclass:
#
# @dataclass
# class Student:
#     name:str
#     age:int
#
# Python automatically creates the constructor internally.
#
# This makes code:
# - shorter
# - cleaner
# - easier to read
#
# Commonly used in ML projects for:
# - file paths
# - configurations
# - model parameters

# Importing DataTransformation class
# This class handles preprocessing and feature engineering
"""means:

Go inside src folder
then inside components folder
then open data_transformation.py file
and import the class named DataTransformation"""
from src.components.data_transformation import DataTransformation

# Importing configuration class for DataTransformation
from src.components.data_transformation import DataTransformationConfig

# Importing configuration class for ModelTrainer
from src.components.model_trainer import ModelTrainerConfig

# Importing ModelTrainer class
# This class trains machine learning models
from src.components.model_trainer import ModelTrainer


# ============================================================
# CONFIGURATION CLASS
# ============================================================

# @dataclass automatically creates constructor methods
# Used mainly for storing configuration variables
@dataclass
class DataIngestionConfig:

    # Path where training dataset will be saved
    # os.path.join() creates proper file path
    train_data_path: str = os.path.join('artifacts', "train.csv")

    # Path where testing dataset will be saved
    test_data_path: str = os.path.join('artifacts', "test.csv")

    # Path where complete raw dataset will be saved
    raw_data_path: str = os.path.join('artifacts', "data.csv")


# ============================================================
# DATA INGESTION CLASS
# ============================================================

# Data Ingestion means:
# Loading and preparing data for the ML pipeline
class DataIngestion:

    # Constructor method
    # Automatically runs when object is created
    def __init__(self):

        # Creating object of configuration class
        # So we can access all file paths easily
        self.ingestion_config = DataIngestionConfig()


    # Main function for data ingestion
    def initiate_data_ingestion(self):

        # Logging message
        # Helps us track execution in log files
        logging.info("Entered the data ingestion method or component")

        try:

            # Reading CSV dataset using pandas
            # Data is loaded into dataframe called df
            df = pd.read_csv('notebook\\data\\stud.csv')

            # Logging successful dataset reading
            logging.info('Read the dataset as dataframe')


            # Creating artifacts folder if it does not exist
            # dirname extracts directory path
            os.makedirs(
                os.path.dirname(self.ingestion_config.train_data_path),
                exist_ok=True
            )


            # Saving raw/original dataset into artifacts folder
            # index=False prevents extra index column creation
            # header=True keeps column names
            df.to_csv(
                self.ingestion_config.raw_data_path,
                index=False,
                header=True
            )

            """Simple Meaning

Whenever your ML pipeline runs, it creates many files like:

train.csv
test.csv
model.pkl
preprocessor.pkl
logs
transformed data

These generated files are stored inside:

artifacts/"""


            # Logging before train-test split
            logging.info("Train test split initiated")


            # Splitting dataset into:
            # - 80% training data
            # - 20% testing data
            #
            # random_state=42 ensures same random split every run
            train_set, test_set = train_test_split(
                df,
                test_size=0.2,
                random_state=42
            )


            # Saving training dataset into train.csv
            train_set.to_csv(
                self.ingestion_config.train_data_path,
                index=False,
                header=True
            )


            # Saving testing dataset into test.csv
            test_set.to_csv(
                self.ingestion_config.test_data_path,
                index=False,
                header=True
            )


            # Logging successful completion of data ingestion
            logging.info("Ingestion of the data is completed")


            # Returning paths of train and test files
            # These paths will be used in next pipeline stages
            return (
                self.ingestion_config.train_data_path,
                self.ingestion_config.test_data_path
            )


        # Exception handling block
        except Exception as e:

            # Raising custom exception
            # sys gives detailed traceback information
            raise CustomException(e, sys)


# ============================================================
# MAIN EXECUTION STARTS HERE
# ============================================================

# This condition checks:
# If this file is being run directly,
# then execute below code
#

if __name__ == "__main__":


    # Creating object of DataIngestion class
    obj = DataIngestion()


    # Calling data ingestion function
    # Returns:
    # - train data path
    # - test data path
    train_data, test_data = obj.initiate_data_ingestion()


    # Creating object of DataTransformation class
    data_transformation = DataTransformation()


    # Calling data transformation method
    #
    # This step performs:
    # - missing value handling
    # - encoding categorical features
    # - scaling numerical features
    # - preprocessing
    #
    # Returns:
    # - transformed training array
    # - transformed testing array
    # - preprocessing object path
    train_arr, test_arr, _ = data_transformation.initiate_data_transformation(
        train_data,
        test_data
    )


    # Creating object of ModelTrainer class
    modeltrainer = ModelTrainer()


    # Calling model training method
    #
    # This step:
    # - trains ML models
    # - evaluates performance
    # - selects best model
    #
    # Printing final model score/result
    print(
        modeltrainer.initiate_model_trainer(
            train_arr,
            test_arr
        )
    )