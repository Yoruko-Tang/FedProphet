from scheduler.base_scheduler import base_AT_scheduler
import numpy as np

class kd_scheduler(base_AT_scheduler):

    def __init__(self, args, model_profiles):
        super().__init__(args)
        