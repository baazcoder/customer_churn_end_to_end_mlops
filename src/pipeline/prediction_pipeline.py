import sys
from pandas import DataFrame

from src.entity.config_entity import ChurnPredictorConfig
from src.entity.s3_estimator import Proj1Estimator
from src.exception import MyException
from src.logger import logging


class ChurnData:
    """
    Customer data class used for prediction.
    """

    def __init__(
        self,
        gender,
        SeniorCitizen,
        Partner,
        Dependents,
        tenure,
        PhoneService,
        MultipleLines,
        InternetService,
        OnlineSecurity,
        OnlineBackup,
        DeviceProtection,
        TechSupport,
        StreamingTV,
        StreamingMovies,
        Contract,
        PaperlessBilling,
        PaymentMethod,
        MonthlyCharges,
        TotalCharges,
    ):

        try:
            self.gender = gender
            self.SeniorCitizen = SeniorCitizen
            self.Partner = Partner
            self.Dependents = Dependents
            self.tenure = tenure
            self.PhoneService = PhoneService
            self.MultipleLines = MultipleLines
            self.InternetService = InternetService
            self.OnlineSecurity = OnlineSecurity
            self.OnlineBackup = OnlineBackup
            self.DeviceProtection = DeviceProtection
            self.TechSupport = TechSupport
            self.StreamingTV = StreamingTV
            self.StreamingMovies = StreamingMovies
            self.Contract = Contract
            self.PaperlessBilling = PaperlessBilling
            self.PaymentMethod = PaymentMethod
            self.MonthlyCharges = MonthlyCharges
            self.TotalCharges = TotalCharges

        except Exception as e:
            raise MyException(e, sys) from e

    def get_churn_data_as_dict(self):
        """
        Returns customer data as dictionary.
        """
        try:

            input_data = {
                "gender": [self.gender],
                "SeniorCitizen": [self.SeniorCitizen],
                "Partner": [self.Partner],
                "Dependents": [self.Dependents],
                "tenure": [self.tenure],
                "PhoneService": [self.PhoneService],
                "MultipleLines": [self.MultipleLines],
                "InternetService": [self.InternetService],
                "OnlineSecurity": [self.OnlineSecurity],
                "OnlineBackup": [self.OnlineBackup],
                "DeviceProtection": [self.DeviceProtection],
                "TechSupport": [self.TechSupport],
                "StreamingTV": [self.StreamingTV],
                "StreamingMovies": [self.StreamingMovies],
                "Contract": [self.Contract],
                "PaperlessBilling": [self.PaperlessBilling],
                "PaymentMethod": [self.PaymentMethod],
                "MonthlyCharges": [self.MonthlyCharges],
                "TotalCharges": [self.TotalCharges],
            }

            logging.info("Customer input dictionary created.")

            return input_data

        except Exception as e:
            raise MyException(e, sys) from e

    def get_churn_input_data_frame(self) -> DataFrame:
        """
        Returns customer input as pandas DataFrame.
        """
        try:
            return DataFrame(self.get_churn_data_as_dict())

        except Exception as e:
            raise MyException(e, sys) from e


class ChurnDataClassifier:
    """
    Loads the trained model from S3 and performs prediction.
    """

    def __init__(
        self,
        prediction_pipeline_config: ChurnPredictorConfig = ChurnPredictorConfig(),
    ):

        try:
            self.prediction_pipeline_config = prediction_pipeline_config

        except Exception as e:
            raise MyException(e, sys) from e

    def predict(self, dataframe):

        try:

            logging.info("Loading trained model from S3.")

            model = Proj1Estimator(
                bucket_name=self.prediction_pipeline_config.model_bucket_name,
                model_path=self.prediction_pipeline_config.model_file_path,
            )

            prediction = model.predict(dataframe)

            return prediction

        except Exception as e:
            raise MyException(e, sys) from e