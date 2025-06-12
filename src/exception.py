import sys 
import logging
# from src.logger import logging

def error_message_details(error, error_detail: sys):
    _, _, exc_tb = error_detail.exc_info()
    filename = exc_tb.tb_frame.f_code.co_filename
    line_no = exc_tb.tb_lineno
    message = f"Error: {error} | File: {filename} | Line: {line_no}"
    return message



class CustomException(Exception):
    def __init__(self, error_message, error_detail: sys):
        super().__init__(error_message)
        self.error_message = error_message_details(error_message,error_detail)

    def __str__(self):
        return self.error_message
    

# if __name__=="__main__":
#     try:
#         a = 10/0
#     except Exception as e:
#         ce= CustomException(e,sys) 
#         logging.error(ce.error_message)