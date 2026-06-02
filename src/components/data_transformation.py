# ============================================================
# IMPORTING REQUIRED LIBRARIES
# ============================================================

import sys
import os
from dataclasses import dataclass

import numpy as np
import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from src.exception import CustomException
from src.logger import logging
from src.utils import save_object


# ============================================================
# CONFIGURATION CLASS
# ============================================================

@dataclass
class DataTransformationConfig:

    preprocessor_obj_file_path = os.path.join(
        'artifacts',
        "preprocessor.pkl"
    )


# ============================================================
# DATA TRANSFORMATION CLASS
# ============================================================

class DataTransformation:

    def __init__(self):

        self.data_transformation_config = DataTransformationConfig()


    # ========================================================
    # FUNCTION TO CREATE PREPROCESSOR OBJECT
    # ========================================================

    def get_data_transformer_object(self):

        try:

            logging.info("Entered get_data_transformer_object method")

            # Numerical columns
            numerical_columns = [
                "writing_score",
                "reading_score"
            ]

            # Categorical columns
            categorical_columns = [
                "gender",
                "race_ethnicity",
                "parental_level_of_education",
                "lunch",
                "test_preparation_course",
            ]


            # ====================================================
            # NUMERICAL PIPELINE
            # ====================================================

            num_pipeline = Pipeline(

                steps=[

                    ("imputer", SimpleImputer(strategy="median")),

                    ("scaler", StandardScaler())

                ]

            )


            # ====================================================
            # CATEGORICAL PIPELINE
            # ====================================================

            cat_pipeline = Pipeline(

                steps=[

                    ("imputer", SimpleImputer(strategy="most_frequent")),

                    ("one_hot_encoder", OneHotEncoder(handle_unknown='ignore')),

                    ("scaler", StandardScaler(with_mean=False))

                ]

            )


            logging.info(f"Categorical columns: {categorical_columns}")

            logging.info(f"Numerical columns: {numerical_columns}")


            # ====================================================
            # COLUMN TRANSFORMER
            # ====================================================

            preprocessor = ColumnTransformer(

                [

                    ("num_pipeline", num_pipeline, numerical_columns),

                    ("cat_pipeline", cat_pipeline, categorical_columns)

                ]

            )


            logging.info("Preprocessor object created successfully")


            return preprocessor


        except Exception as e:

            raise CustomException(e, sys)


    # ========================================================
    # MAIN DATA TRANSFORMATION FUNCTION
    # ========================================================

    def initiate_data_transformation(self, train_path, test_path):

        try:

            logging.info("Started data transformation")

            # Reading train dataset
            train_df = pd.read_csv(train_path)

            # Reading test dataset
            test_df = pd.read_csv(test_path)


            logging.info("Read train and test data completed")


            # Getting preprocessing object
            preprocessing_obj = self.get_data_transformer_object()


            # Target column
            target_column_name = "math_score"


            # ====================================================
            # TRAIN DATA
            # ====================================================

            input_feature_train_df = train_df.drop(
                columns=[target_column_name],
                axis=1
            )

            target_feature_train_df = train_df[target_column_name]


            # ====================================================
            # TEST DATA
            # ====================================================

            input_feature_test_df = test_df.drop(
                columns=[target_column_name],
                axis=1
            )

            target_feature_test_df = test_df[target_column_name]


            logging.info(
                "Applying preprocessing object on training and testing dataframes"
            )


            # ====================================================
            # APPLYING TRANSFORMATION
            # ====================================================

            input_feature_train_arr = preprocessing_obj.fit_transform(
                input_feature_train_df
            )

            input_feature_test_arr = preprocessing_obj.transform(
                input_feature_test_df
            )


            # ====================================================
            # COMBINING INPUT + TARGET
            # ====================================================

            train_arr = np.c_[

                input_feature_train_arr,

                np.array(target_feature_train_df)

            ]


            test_arr = np.c_[

                input_feature_test_arr,

                np.array(target_feature_test_df)

            ]


            # ====================================================
            # SAVING PREPROCESSOR OBJECT
            # ====================================================

            save_object(

                file_path=self.data_transformation_config.preprocessor_obj_file_path,

                obj=preprocessing_obj

            )


            logging.info("Preprocessor pickle file saved successfully")


            return (

                train_arr,

                test_arr,

                self.data_transformation_config.preprocessor_obj_file_path

            )


        except Exception as e:

            raise CustomException(e, sys)


# ============================================================
# MAIN FUNCTION
# ============================================================

if __name__ == "__main__":

    obj = DataTransformation()

    train_path = "artifacts/train.csv"

    test_path = "artifacts/test.csv"

    obj.initiate_data_transformation(train_path, test_path)

    print("Data Transformation Completed")