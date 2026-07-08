from src.logger import logging
from src.exception import MyException
from src.pipeline.training_pipeline import TrainPipeline

logging.info("This is debug message")
logging.debug("lun t vjooo")

# import sys

# try:
#     a = 1+'Z'
# except Exception as e:
#     logging.info(e)
#     raise MyException(e, sys) from e

TrainPipeline().run_pipeline()
