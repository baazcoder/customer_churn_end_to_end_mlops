import sys
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from imblearn.combine import SMOTEENN
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer, OneHotEncoder


from src.constants import TARGET_COLUMN, SCHEMA_FILE_PATH
from src.entity.config_entity import DataTransformationConfig
from src.entity.artifact_entity import DataTransformationArtifact, DataIngestionArtifact, DataValidationArtifact
from src.exception import MyException
from src.logger import logging
from src.utils.main_utils import save_object, save_numpy_array_data, read_yaml_file


class DataTransformation:
    def __init__(self, data_ingestion_artifact: DataIngestionArtifact,
                 data_transformation_config: DataTransformationConfig,
                 data_validation_artifact: DataValidationArtifact):
        try:
            self.data_ingestion_artifact = data_ingestion_artifact
            self.data_transformation_config = data_transformation_config
            self.data_validation_artifact = data_validation_artifact
            self._schema_config = read_yaml_file(file_path=SCHEMA_FILE_PATH)
        except Exception as e:
            raise MyException(e, sys)

    @staticmethod
    def read_data(file_path) -> pd.DataFrame:
        try:
            return pd.read_csv(file_path)
        except Exception as e:
            raise MyException(e, sys)

    
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

    def get_preprocessor(self, X_train):

        categorical_columns = X_train.select_dtypes(
            include=["object"]
        ).columns.tolist()

        numeric_columns = X_train.select_dtypes(
            exclude=["object"]
        ).columns.tolist()

        logging.info(f"Categorical Columns : {categorical_columns}")
        logging.info(f"Numerical Columns : {numeric_columns}")

        preprocessor = ColumnTransformer(
            transformers=[
                (
                    "cat",
                    OneHotEncoder(
                        drop="first",
                        handle_unknown="ignore"
                    ),
                    categorical_columns,
                ),
            ],
            remainder="passthrough",
        )

        return preprocessor



    def initiate_data_transformation(self) -> DataTransformationArtifact:
        """
        Performs data transformation:
        1. Read train & test data
        2. Clean data
        3. Split features and target
        4. Apply preprocessing pipeline
        5. Handle class imbalance
        6. Save preprocessing object and transformed datasets
        """

        try:
            logging.info("=" * 60)
            logging.info("Data Transformation Started")
            logging.info("=" * 60)

            if not self.data_validation_artifact.validation_status:
                raise Exception(self.data_validation_artifact.message)

            # ==========================================================
            # Read Data
            # ==========================================================
            train_df = self.read_data(
                self.data_ingestion_artifact.trained_file_path
            )

            test_df = self.read_data(
                self.data_ingestion_artifact.test_file_path
            )

            logging.info("Train and Test datasets loaded successfully.")

            # ==========================================================
            # Data Cleaning
            # ==========================================================
            train_df = self.change_dtype(train_df)
            test_df = self.change_dtype(test_df)

            train_df = self._map_churn_column(train_df)
            test_df = self._map_churn_column(test_df)

            train_df = self.droping_null(train_df)
            test_df = self.droping_null(test_df)

            logging.info("Data cleaning completed.")

            # ==========================================================
            # Split Features & Target
            # ==========================================================
            X_train = train_df.drop(columns=[TARGET_COLUMN])
            y_train = train_df[TARGET_COLUMN]

            X_test = test_df.drop(columns=[TARGET_COLUMN])
            y_test = test_df[TARGET_COLUMN]

            logging.info("Target column separated.")

            # ==========================================================
            # Drop Unnecessary Columns
            # ==========================================================
            drop_columns = self._schema_config["drop_columns"]

            X_train = X_train.drop(columns=drop_columns)
            X_test = X_test.drop(columns=drop_columns)

            logging.info(f"Dropped Columns : {drop_columns}")

            # ==========================================================
            # Preprocessing
            # ==========================================================
            preprocessor = self.get_preprocessor(X_train)

            X_train = preprocessor.fit_transform(X_train)
            X_test = preprocessor.transform(X_test)

            feature_names = preprocessor.get_feature_names_out()

            X_train = pd.DataFrame(
                X_train,
                columns=feature_names
            )

            X_test = pd.DataFrame(
                X_test,
                columns=feature_names
            )

            logging.info(
                f"Training Features Shape : {X_train.shape}"
            )

            logging.info(
                f"Testing Features Shape : {X_test.shape}"
            )

            # ==========================================================
            # Handle Imbalanced Dataset
            # ==========================================================
            logging.info("Applying SMOTEENN...")

            smoteenn = SMOTEENN(random_state=42)

            X_train_resampled, y_train_resampled = smoteenn.fit_resample(
                X_train,
                y_train
            )

            logging.info(
                f"Resampled Training Shape : {X_train_resampled.shape}"
            )

            # ==========================================================
            # Convert to Numpy Arrays
            # ==========================================================
            train_arr = np.c_[
                X_train_resampled.to_numpy(dtype=np.float32),
                y_train_resampled.to_numpy(dtype=np.int64)
            ]

            test_arr = np.c_[
                X_test.to_numpy(dtype=np.float32),
                y_test.to_numpy(dtype=np.int64)
            ]

            # ==========================================================
            # Save Artifacts
            # ==========================================================
            save_object(
                file_path=self.data_transformation_config.transformed_object_file_path,
                obj=preprocessor,
            )

            save_numpy_array_data(
                file_path=self.data_transformation_config.transformed_train_file_path,
                array=train_arr,
            )

            save_numpy_array_data(
                file_path=self.data_transformation_config.transformed_test_file_path,
                array=test_arr,
            )

            logging.info("Transformation artifacts saved successfully.")
            logging.info("=" * 60)
            logging.info("Data Transformation Completed")
            logging.info("=" * 60)

            return DataTransformationArtifact(
                transformed_object_file_path=self.data_transformation_config.transformed_object_file_path,
                transformed_train_file_path=self.data_transformation_config.transformed_train_file_path,
                transformed_test_file_path=self.data_transformation_config.transformed_test_file_path,
            )

        except Exception as e:
            raise MyException(e, sys) from e