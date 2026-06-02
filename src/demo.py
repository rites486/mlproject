import sys
from exception import CustomException
from logger import logging

try:
    
    a = 10
    b = 0

    c = a / b

except Exception as e:


    logging.info(CustomException(e, sys))

    print(CustomException(e, sys))