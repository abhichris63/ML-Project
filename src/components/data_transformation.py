import os
import sys
from dataclasses import dataclass

import pandas as pd
import numpy as np
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from src.exception import CustomException
from src.logger import logging
from src.utils import save_object


@dataclass
class DataTransformationConfig:
    """
    Configuration class for setting up file paths related to data transformation.

    Attributes:
        preprocessor_obj_file_path (str): Path where the preprocessor object (pipeline) will be saved.
    """
    # Define the path where the preprocessor object will be saved
    preprocessor_obj_file_path = os.path.join("artifacts","preprocessor.pkl")


class DataTransformation:
    """
    A class to handle the preprocessing of input data, including numerical and categorical feature transformation.

    Methods:
        __init__(): Initializes the configuration for storing the preprocessor file path.
        get_data_transformer_object(): Prepares and returns the preprocessing pipeline.
    """
    def __init__(self):
        self.data_transformation_config = DataTransformationConfig()

    def get_data_transformer_object(self):
        """
        Prepares and returns a preprocessing pipeline for both numerical and categorical features.

        The preprocessing steps include:
        - Handling missing values in numerical features using the median strategy.
        - Scaling numerical features using StandardScaler.
        - Handling missing values in categorical features using the most frequent strategy.
        - Encoding categorical features using OneHotEncoder.
        - Scaling categorical features after encoding.

        Returns:
            preprocessor (ColumnTransformer): A ColumnTransformer object that applies the
            preprocessing pipelines to the specified numerical and categorical columns.

        Raises:
            CustomException: If an error occurs during the creation of the preprocessor object.
        """
        try:
            # List of numerical columns to be processed
            numerical_columns = ['writing_score','reading_score']
            
            # List of categorical columns to be processed
            categorical_columns = [
                'gender',
                'race_ethnicity',
                'parental_level_of_education',
                'lunch',
                'test_preparation_course'
                ]
            
            # Define the numerical pipeline
            num_pipeline = Pipeline(
                steps= [
                    ("imputer",SimpleImputer(strategy='median')), # Replace missing values with median of the column
                    ("scaler",StandardScaler()) # Scaling numerical features to standard normal distribution
                ]
            )

            # Define the categorical pipeline
            cat_pipeline = Pipeline(
                steps= [
                    ("imputer",SimpleImputer(strategy='most_frequent')),  # Replace missing values with the most frequent value
                    ("one_hot_encoder",OneHotEncoder(sparse_output=False)),  # Convert categorical variables to one-hot encoded vectors
                    ("scaler",StandardScaler())  # Scale the categorical features (after encoding)
                ]
            )

            # Logging information for successful completion of scaling and encoding
            logging.info(f"Categorical columns scaling completed --- Categorical Columns: {categorical_columns}")
            logging.info(f"Numerical columns encoding completed --- Numerical Columns: {numerical_columns}")

            # Combining numerical and categorical pipelines into a single preprocessor
            preprocessor = ColumnTransformer(
                [
                    ("num_pipeline", num_pipeline, numerical_columns),
                    ("cat_pipeline", cat_pipeline, categorical_columns)
                ]
            )
            
            # Return the complete preprocessing object
            return preprocessor

        except Exception as e:
            raise CustomException(e,sys)
    
    
    def initiate_data_transformation(self,train_path, test_path):
        try:
            train_df = pd.read_csv(train_path)
            test_df = pd.read_csv(test_path)
            logging.info("Reading train and test data completed...")

            logging.info("Obtaining preprocessing object")

            preprocessing_obj = self.get_data_transformer_object()
            
            target_column_name = "math_score"
            numerical_columns = ["writing_score","reading_score"]

            input_feature_train_df = train_df.drop(columns=[target_column_name],axis=1)
            target_feature_train_df = train_df[target_column_name]

            input_feature_test_df = test_df.drop(columns=[target_column_name],axis=1)
            target_feature_test_df = test_df[target_column_name]

            logging.info(f"Appling preprocessing object on Training Dataframe and Testing Dataframe.")

            input_feature_train_arr = preprocessing_obj.fit_transform(input_feature_train_df)
            input_feature_test_arr = preprocessing_obj.transform(input_feature_test_df)

            train_arr = np.c_[
                input_feature_train_arr, np.array(target_feature_train_df)
            ]

            test_arr = np.c_[input_feature_test_arr, np.array(target_feature_test_df)]

            logging.info(f"Saved Preprocessing Object...")

            save_object(
                file_path = self.data_transformation_config.preprocessor_obj_file_path,
                obj = preprocessing_obj
            )

            return(
                train_arr,
                test_arr,
                self.data_transformation_config.preprocessor_obj_file_path
            )
        except Exception as e:
            raise CustomException(e,sys)
