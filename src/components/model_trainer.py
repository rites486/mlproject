import os
import sys
from dataclasses import dataclass

# ============================================================
# PATH SETUP - Fix for ModuleNotFoundError
# ============================================================
import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

# ============================================================
# IMPORTING MACHINE LEARNING MODELS
# ============================================================

from catboost import CatBoostRegressor

from sklearn.ensemble import (
    AdaBoostRegressor,
    GradientBoostingRegressor,
    RandomForestRegressor,
)

from sklearn.linear_model import LinearRegression

from sklearn.metrics import r2_score

from sklearn.neighbors import KNeighborsRegressor

from sklearn.tree import DecisionTreeRegressor

from xgboost import XGBRegressor


# ============================================================
# IMPORTING CUSTOM MODULES
# ============================================================

from src.exception import CustomException

from src.logger import logging

from src.utils import save_object, evaluate_models


# ============================================================
# CONFIGURATION CLASS
# ============================================================

@dataclass
class ModelTrainerConfig:

    # Path where trained model will be saved
    trained_model_file_path = os.path.join(
        "artifacts",
        "model.pkl"
    )


# ============================================================
# MODEL TRAINER CLASS
# ============================================================

class ModelTrainer:

    # ========================================================
    # CONSTRUCTOR
    # ========================================================

    def __init__(self):

        self.model_trainer_config = ModelTrainerConfig()


    # ========================================================
    # MAIN MODEL TRAINING FUNCTION
    # ========================================================

    def initiate_model_trainer(self, train_array, test_array):

        try:

            logging.info("Started model training")


            # ====================================================
            # SPLITTING TRAIN AND TEST DATA
            # ====================================================

            logging.info("Splitting dependent and independent variables")

            X_train, y_train, X_test, y_test = (

                train_array[:, :-1],

                train_array[:, -1],

                test_array[:, :-1],

                test_array[:, -1]

            )


            # ====================================================
            # MACHINE LEARNING MODELS
            # ====================================================

            models = {

                "Random Forest": RandomForestRegressor(),

                "Decision Tree": DecisionTreeRegressor(),

                "Gradient Boosting": GradientBoostingRegressor(),

                "Linear Regression": LinearRegression(),

                "KNeighbors Regressor": KNeighborsRegressor(),

                "XGBRegressor": XGBRegressor(),

                "CatBoosting Regressor": CatBoostRegressor(
                    verbose=False
                ),

                "AdaBoost Regressor": AdaBoostRegressor(),

            }


            # ====================================================
            # HYPERPARAMETERS
            # ====================================================

            params = {

                "Decision Tree": {

                    'criterion': [

                        'squared_error',

                        'friedman_mse',

                        'absolute_error',

                        'poisson'

                    ]

                },


                "Random Forest": {

                    'n_estimators': [

                        8, 16, 32, 64, 128, 256

                    ]

                },


                "Gradient Boosting": {

                    'learning_rate': [

                        .1, .01, .05, .001

                    ],

                    'subsample': [

                        0.6, 0.7, 0.75, 0.8, 0.85, 0.9

                    ],

                    'n_estimators': [

                        8, 16, 32, 64, 128, 256

                    ]

                },


                "Linear Regression": {

                },


                "KNeighbors Regressor": {

                    'n_neighbors': [

                        3, 5, 7, 9

                    ]

                },


                "XGBRegressor": {

                    'learning_rate': [

                        .1, .01, .05, .001

                    ],

                    'n_estimators': [

                        8, 16, 32, 64, 128, 256

                    ]

                },


                "CatBoosting Regressor": {

                    'depth': [

                        6, 8, 10

                    ],

                    'learning_rate': [

                        0.01, 0.05, 0.1

                    ],

                    'iterations': [

                        30, 50, 100

                    ]

                },


                "AdaBoost Regressor": {

                    'learning_rate': [

                        .1, .01, 0.5, .001

                    ],

                    'n_estimators': [

                        8, 16, 32, 64, 128, 256

                    ]

                }

            }


            # ====================================================
            # MODEL EVALUATION
            # ====================================================

            logging.info("Model evaluation started")

            model_report: dict = evaluate_models(

                X_train=X_train,

                y_train=y_train,

                X_test=X_test,

                y_test=y_test,

                models=models,

                param=params

            )


            logging.info(f"Model Report: {model_report}")


            # ====================================================
            # BEST MODEL SELECTION
            # ====================================================

            best_model_score = max(model_report.values())


            best_model_name = list(model_report.keys())[

                list(model_report.values()).index(
                    best_model_score
                )

            ]


            best_model = models[best_model_name]


            logging.info(f"Best Model Found: {best_model_name}")

            logging.info(f"Best Model Score: {best_model_score}")


            # ====================================================
            # CHECKING MODEL PERFORMANCE
            # ====================================================

            if best_model_score < 0.6:

                raise CustomException(
                    "No best model found",
                    sys
                )


            # ====================================================
            # TRAINING BEST MODEL
            # ====================================================

            best_model.fit(X_train, y_train)


            logging.info(
                "Best model trained successfully"
            )


            # ====================================================
            # SAVING MODEL
            # ====================================================

            save_object(

                file_path=self.model_trainer_config.trained_model_file_path,

                obj=best_model

            )


            logging.info(
                "Trained model saved successfully"
            )


            # ====================================================
            # PREDICTION
            # ====================================================

            predicted = best_model.predict(X_test)


            # ====================================================
            # CALCULATING R2 SCORE
            # ====================================================

            r2_square = r2_score(y_test, predicted)


            logging.info(
                f"Model R2 Score: {r2_square}"
            )


            return r2_square


        # ========================================================
        # EXCEPTION HANDLING
        # ========================================================

        except Exception as e:

            raise CustomException(e, sys)


# ============================================================
# MAIN FUNCTION - UPDATED TO RUN SEPARATELY
# ============================================================
if __name__ == "__main__":

    logging.info("=" * 60)
    logging.info("MODEL TRAINER - Running Separately")
    logging.info("=" * 60)

    try:
        # Step 1: Get the data from data transformation
        logging.info("Step 1: Getting transformed data")
        
        from src.components.data_transformation import DataTransformation
        
        data_transformation = DataTransformation()
        train_arr, test_arr, _ = data_transformation.initiate_data_transformation(
            "artifacts/train.csv",
            "artifacts/test.csv"
        )
        
        logging.info(f"Train array shape: {train_arr.shape}")
        logging.info(f"Test array shape: {test_arr.shape}")
        
        # Step 2: Train models
        logging.info("Step 2: Starting model training")
        
        modeltrainer = ModelTrainer()
        r2_score = modeltrainer.initiate_model_trainer(train_arr, test_arr)
        
        # Step 3: Display results
        print("\n" + "=" * 60)
        print("MODEL TRAINING COMPLETED!")
        print("=" * 60)
        print(f"✅ Best Model R2 Score: {r2_score:.4f}")
        print(f"✅ Model saved at: artifacts/model.pkl")
        print(f"✅ Logs saved in: logs/ folder")
        print("=" * 60)
        
        logging.info(f"Model training completed successfully with R2 Score: {r2_score}")
        
    except FileNotFoundError as e:
        logging.error(f"File not found: {str(e)}")
        print("\n" + "=" * 60)
        print("❌ ERROR: Required files not found!")
        print("=" * 60)
        print("Please run data ingestion first:")
        print("  python src/components/data_ingestion.py")
        print("=" * 60)
        
    except Exception as e:
        logging.error(f"Training failed: {str(e)}")
        print(f"\n❌ Error: {str(e)}")