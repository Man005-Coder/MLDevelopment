import os
import sys

import numpy as np
import pandas as pd
from dataclasses import dataclass
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OrdinalEncoder

from src.exception import CustomException
from src.logger import logging
from src.utils import save_object


@dataclass
class DataTransformationConfig:
    preprocessor_obj_file_path: str = os.path.join('artifacts', 'preprocessor.pkl')


class DataTransformation:
    def __init__(self):
        self.transformation_config = DataTransformationConfig()

    def get_preprocessor(self, input_df):
        try:
            logging.info("Data Transformation initiated")

            categorical_cols = [
                col for col in input_df.columns
                if not pd.api.types.is_numeric_dtype(input_df[col])
            ]
            numeric_cols = [col for col in input_df.columns if col not in categorical_cols]

            transformers = []
            if categorical_cols:
                transformers.append(
                    (
                        'categorical',
                        OrdinalEncoder(handle_unknown='use_encoded_value', unknown_value=-1),
                        categorical_cols,
                    )
                )
            if numeric_cols:
                transformers.append(('numeric', 'passthrough', numeric_cols))

            preprocessor_obj = ColumnTransformer(transformers=transformers, remainder='drop')
            preprocessor_obj.fit(input_df)

            logging.info("Data Transformation completed")
            return preprocessor_obj
        except Exception as e:
            raise CustomException(e, sys)

    def initiate_data_transformation(self, train_path, test_path):
        try:
            train_df = pd.read_csv(train_path)
            test_df = pd.read_csv(test_path)
            logging.info("Read train and test data completed")
            logging.info("Obtaining preprocessing object")

            input_feature_train_df = train_df.drop(columns=['Salary'])
            target_feature_train_df = train_df['Salary']

            input_feature_test_df = test_df.drop(columns=['Salary'])
            target_feature_test_df = test_df['Salary']

            preprocessor_obj = self.get_preprocessor(input_feature_train_df)

            logging.info("Applying preprocessing object on training and testing datasets.")
            input_feature_train_arr = preprocessor_obj.transform(input_feature_train_df)
            input_feature_test_arr = preprocessor_obj.transform(input_feature_test_df)

            train_arr = np.c_[input_feature_train_arr, np.array(target_feature_train_df)]
            test_arr = np.c_[input_feature_test_arr, np.array(target_feature_test_df)]
            logging.info("Saved preprocessing object.")

            save_object(
                file_path=self.transformation_config.preprocessor_obj_file_path,
                obj=preprocessor_obj
            )

            return (
                train_arr,
                test_arr,
                self.transformation_config.preprocessor_obj_file_path,
            )
        except Exception as e:
            raise CustomException(e, sys)