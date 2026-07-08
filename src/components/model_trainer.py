import sys
from typing import Tuple

import numpy as np
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score

from src.exception import MyException
from src.logger import logging
from src.utils.main_utils import load_numpy_array_data, load_object, save_object
from src.entity.config_entity import ModelTrainerConfig
from src.entity.artifact_entity import DataTransformationArtifact, ModelTrainerArtifact, ClassificationMetricArtifact
from src.entity.estimator import MyModel

class ModelTrainer:
    def __init__(self, data_transformation_artifact: DataTransformationArtifact,
                 model_trainer_config: ModelTrainerConfig):
        """
        :param data_transformation_artifact: Output reference of data transformation artifact stage
        :param model_trainer_config: Configuration for model training
        """
        self.data_transformation_artifact = data_transformation_artifact
        self.model_trainer_config = model_trainer_config

    def get_model_object_and_report(self, train: np.array, test: np.array) -> Tuple[object, object]:
        """
        Method Name :   get_model_object_and_report
        Description :   This function trains a XGBClassifier with specified parameters
        
        Output      :   Returns metric artifact object and trained model object
        On Failure  :   Write an exception log and then raise an exception
        """
        try:
            logging.info("Training XGBClassifier with specified parameters")

            # Splitting the train and test data into features and target variables
            x_train, y_train, x_test, y_test = train[:, :-1], train[:, -1], test[:, :-1], test[:, -1]
            logging.info("train-test split done.")
            
            x_train = x_train.astype(np.float32)
            x_test = x_test.astype(np.float32)

            y_train = y_train.astype(np.int64)
            y_test = y_test.astype(np.int64)

          # Initialize XGBoost Classifier with specified parameters
            model = XGBClassifier(
                n_estimators=self.model_trainer_config.n_estimators,
                max_depth=self.model_trainer_config.max_depth,
                learning_rate=self.model_trainer_config.learning_rate,
                subsample=self.model_trainer_config.subsample,
                min_child_weight=self.model_trainer_config.min_child_weight,
                gamma=self.model_trainer_config.gamma,
                colsample_bytree=self.model_trainer_config.colsample_bytree,
                random_state=42,
                objective="binary:logistic",
                eval_metric="logloss",
                use_label_encoder=False
            )

            # Fit the model
            logging.info("Model training going on...")
            model.fit(x_train, y_train)
            logging.info("Model training done.")


            # Predictions and evaluation metrics
            y_pred = model.predict(x_test)
            
            logging.info(f"Train dtype: {train.dtype}")
            logging.info(f"Test dtype: {test.dtype}")

            logging.info(f"y_test dtype: {y_test.dtype}")
            logging.info(f"y_pred dtype: {y_pred.dtype}")

            logging.info(f"Unique y_test: {np.unique(y_test)}")
            logging.info(f"Unique y_pred: {np.unique(y_pred)}")

            logging.info(f"First 10 y_test: {y_test[:10]}")
            logging.info(f"First 10 y_pred: {y_pred[:10]}")
            accuracy = accuracy_score(y_test, y_pred)
            f1 = f1_score(y_test, y_pred)
            precision = precision_score(y_test, y_pred)
            recall = recall_score(y_test, y_pred)

            # Creating metric artifact
            metric_artifact = ClassificationMetricArtifact(accuracy=accuracy, f1_score=f1, precision_score=precision, recall_score=recall)
            return model, metric_artifact
        
        except Exception as e:
            raise MyException(e, sys) from e

    def initiate_model_trainer(self) -> ModelTrainerArtifact:
        logging.info("Entered initiate_model_trainer method of ModelTrainer class")
        """
        Method Name :   initiate_model_trainer
        Description :   This function initiates the model training steps
        
        Output      :   Returns model trainer artifact
        On Failure  :   Write an exception log and then raise an exception
        """
        try:
            logging.info("------------------------------------------------------------------------------------------------")
            logging.info("Starting Model Trainer Component")
            # Load transformed train and test data
            train_arr = load_numpy_array_data(file_path=self.data_transformation_artifact.transformed_train_file_path)
            test_arr = load_numpy_array_data(file_path=self.data_transformation_artifact.transformed_test_file_path)
            logging.info("train-test data loaded")
            
            # Train model and get metrics
            trained_model, metric_artifact = self.get_model_object_and_report(train=train_arr, test=test_arr)
            logging.info("Model object and artifact loaded.")
            
            # Load preprocessing object
            preprocessing_obj = load_object(file_path=self.data_transformation_artifact.transformed_object_file_path)
            logging.info("Preprocessing obj loaded.")
            
            # Calculate training accuracy
            train_accuracy = accuracy_score(
                train_arr[:, -1],
                trained_model.predict(train_arr[:, :-1])
            )

            logging.info(f"Training Accuracy: {train_accuracy:.4f}")
            
            # Check if the training accuracy meets the expected accuracy threshold
            if train_accuracy < self.model_trainer_config.expected_accuracy:
                raise Exception("No model found with score above the base score")

            # Save the final model object that includes both preprocessing and the trained model
            logging.info("Saving new model as performace is better than previous one.")
            my_model = MyModel(preprocessing_object=preprocessing_obj, trained_model_object=trained_model)
            save_object(self.model_trainer_config.trained_model_file_path, my_model)
            logging.info("Saved final model object that includes both preprocessing and the trained model")

            # Create and return the ModelTrainerArtifact
            model_trainer_artifact = ModelTrainerArtifact(
                trained_model_file_path=self.model_trainer_config.trained_model_file_path,
                metric_artifact=metric_artifact,
            )
            logging.info(f"Model trainer artifact: {model_trainer_artifact}")
            return model_trainer_artifact
        
        except Exception as e:
            raise MyException(e, sys) from e