from src.entity.config_entity import ModelEvaluationConfig
from src.entity.artifact_entity import ModelTrainerArtifact, DataIngestionArtifact, ModelEvaluationArtifact
from sklearn.metrics import f1_score
from src.exception import MyException
from src.constants import TARGET_COLUMN
from src.logger import logging
from src.utils.main_utils import load_object
import sys
import pandas as pd
from typing import Optional
from src.entity.s3_estimator import Proj1Estimator
from dataclasses import dataclass
import numpy as np
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer
from src.utils.main_utils import read_yaml_file
from src.constants import SCHEMA_FILE_PATH

@dataclass
class EvaluateModelResponse:
    trained_model_f1_score: float
    best_model_f1_score: float
    is_model_accepted: bool
    difference: float


class ModelEvaluation:

    def __init__(self, model_eval_config: ModelEvaluationConfig, data_ingestion_artifact: DataIngestionArtifact,
                 model_trainer_artifact: ModelTrainerArtifact):
        try:
            self.model_eval_config = model_eval_config
            self.data_ingestion_artifact = data_ingestion_artifact
            self.model_trainer_artifact = model_trainer_artifact
            self._schema_config = read_yaml_file(file_path=SCHEMA_FILE_PATH)
        except Exception as e:
            raise MyException(e, sys) from e

    def get_best_model(self) -> Optional[Proj1Estimator]:
        """
        Method Name :   get_best_model
        Description :   This function is used to get model from production stage.
        
        Output      :   Returns model object if available in s3 storage
        On Failure  :   Write an exception log and then raise an exception
        """
        try:
            bucket_name = self.model_eval_config.bucket_name
            model_path=self.model_eval_config.s3_model_key_path
            proj1_estimator = Proj1Estimator(bucket_name=bucket_name,
                                               model_path=model_path)

            if proj1_estimator.is_model_present(model_path=model_path):
                return proj1_estimator
            return None
        except Exception as e:
            raise  MyException(e,sys)
        
    def change_dtype(self, df):
        """changing dtype of TotalCharges column to float"""
        logging.info("Changing dtype of TotaCharges col to float")
        df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")
        return df
    
    def droping_null(self, df):
        """droping null values if any"""
        logging.info("Dropping null values")

        df.dropna(inplace=True)
        return df

    def _map_churn_column(self, df):
        """Map Churn column to 0 for no and 1 for yes."""
        logging.info("Mapping 'Churn' column to binary values")
        df["Churn"] = df["Churn"].map({"Yes": 1, "No": 0})
        return df

    
    def evaluate_model(self) -> EvaluateModelResponse:
     try:

        # =====================================================
        # Load Test Data
        # =====================================================
        test_df = pd.read_csv(
            self.data_ingestion_artifact.test_file_path
        )

        logging.info("Test data loaded.")

        # =====================================================
        # Same Cleaning as Training
        # =====================================================
        test_df = self.change_dtype(test_df)

        test_df = self._map_churn_column(test_df)

        test_df = self.droping_null(test_df)

        X = test_df.drop(columns=[TARGET_COLUMN])

        y = test_df[TARGET_COLUMN]

        # =====================================================
        # Drop ID Columns
        # =====================================================
        drop_cols = self._schema_config["drop_columns"]

        X = X.drop(columns=drop_cols)

        # =====================================================
        # Load Saved Preprocessor
        # =====================================================
        preprocessor = load_object(
            self.model_trainer_artifact.trained_model_file_path
        )

        y_pred = preprocessor.predict(X)

        # =====================================================
        # Convert to DataFrame
        # =====================================================

        # =====================================================
        # Load Current Model
        # =====================================================
        trained_model_f1_score = (
            self.model_trainer_artifact.metric_artifact.f1_score
        )

        logging.info(
            f"Current Model F1 Score : {trained_model_f1_score}"
        )

        # =====================================================
        # Compare with Production Model
        # =====================================================
        best_model = self.get_best_model()

        best_model_f1_score = None

        if best_model is not None:

            logging.info("Production model found.")

            y_pred = best_model.predict(X)

            best_model_f1_score = f1_score(
                y,
                y_pred
            )

            logging.info(
                f"Production Model F1 Score : {best_model_f1_score}"
            )

        best_score = (
            0
            if best_model_f1_score is None
            else best_model_f1_score
        )

        return EvaluateModelResponse(
            trained_model_f1_score=trained_model_f1_score,
            best_model_f1_score=best_model_f1_score,
            is_model_accepted=trained_model_f1_score > best_score,
            difference=trained_model_f1_score - best_score,
        )

     except Exception as e:
        raise MyException(e, sys) from e

    def initiate_model_evaluation(self) -> ModelEvaluationArtifact:
        """
        Method Name :   initiate_model_evaluation
        Description :   This function is used to initiate all steps of the model evaluation
        
        Output      :   Returns model evaluation artifact
        On Failure  :   Write an exception log and then raise an exception
        """  
        try:
            print("------------------------------------------------------------------------------------------------")
            logging.info("Initialized Model Evaluation Component.")
            evaluate_model_response = self.evaluate_model()
            s3_model_path = self.model_eval_config.s3_model_key_path

            model_evaluation_artifact = ModelEvaluationArtifact(
                is_model_accepted=evaluate_model_response.is_model_accepted,
                s3_model_path=s3_model_path,
                trained_model_path=self.model_trainer_artifact.trained_model_file_path,
                changed_accuracy=evaluate_model_response.difference)

            logging.info(f"Model evaluation artifact: {model_evaluation_artifact}")
            return model_evaluation_artifact
        except Exception as e:
            raise MyException(e, sys) from e