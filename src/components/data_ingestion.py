import os
import sys
from src.exception import CustomException
from src.logger import logging
import pandas as pd

from sklearn.model_selection import train_test_split
from dataclasses import dataclass

from src.components.data_transformation import DataTransformation
from src.components.data_transformation import DataTransformationConfig
from src.components.model_trainer import ModelTrainerConfig
from src.components.model_trainer import ModelTrainer

@dataclass
class DataIngestionConfig:
    """
    Configuration class for data ingestion process.

    Attributes:
        train_data_path (str): The file path where the training dataset will be saved.
        test_data_path (str): The file path where the testing dataset will be saved.
        raw_data_path (str): The file path where the raw dataset will be saved.
    """

    train_data_path: str = os.path.join('artifacts','train.csv')
    test_data_path: str = os.path.join('artifacts','test.csv')
    raw_data_path: str = os.path.join('artifacts','data.csv')

class DataIngestion:
    """
    A class responsible for ingesting data, performing train-test split, 
    and saving the resulting datasets to specified file paths.

    Methods:
        __init__(): Initializes the configuration for data ingestion paths.
        initiate_data_ingestion(): Reads the dataset, splits it into training and testing sets, 
                                   and saves them to the respective file paths.
    """


    def __init__(self):
        self.ingestion_config = DataIngestionConfig()

    def initiate_data_ingestion(self):
        """
        Reads the dataset, performs train-test split, and saves the data to specified locations.

        Steps:
        - Reads data from a CSV file.
        - Logs the progress of the process.
        - Splits the data into training and testing sets (80% train, 20% test).
        - Saves the datasets to CSV files.
        
        Returns:
            tuple: Paths of the training and testing data files.
        
        Raises:
            CustomException: If any error occurs during the data ingestion process.
        """


        logging.info("Entered the data ingestion component")
        try:
            # Read the dataset into a pandas DataFrame
            df = pd.read_csv('notebook\data\stud.csv')
            logging.info("Read the Dataset as DataFrame")

            # Create directory if it does not exist
            os.makedirs(os.path.dirname(self.ingestion_config.train_data_path),exist_ok=True)

            # Save raw data to the specified path
            df.to_csv(self.ingestion_config.raw_data_path,index=False,header=True)

            logging.info("Train Test Split is Initiated...")

            # Split the data into training and testing sets (80% train, 20% test)
            train_set,test_set = train_test_split(df,test_size=0.2,random_state=42)

            # Save the training and testing sets to their respective paths
            train_set.to_csv(self.ingestion_config.train_data_path, index=False,header=True)
            test_set.to_csv(self.ingestion_config.test_data_path, index=False, header=True)
            logging.info("Ingestion of the data is Completed...")

            # Return paths of the train and test datasets
            return(
                self.ingestion_config.train_data_path,
                self.ingestion_config.test_data_path
            )
        except Exception as e:
            # Raise a custom exception if any error occurs during the data ingestion process
            raise CustomException(e,sys)
        
if __name__ == "__main__":

    # Create a DataIngestion object and start the data ingestion process
    obj = DataIngestion()
    train_data, test_data = obj.initiate_data_ingestion()

    data_transformation = DataTransformation()
    train_arr, test_arr,_ = data_transformation.initiate_data_transformation(train_data, test_data)
    
    modeltrainer = ModelTrainer()
    print(modeltrainer.initiate_model_trainer(train_arr,test_arr))

    