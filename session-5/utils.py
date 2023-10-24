from enum import auto, Enum

class DatasetType(Enum):
    TRAIN = auto()
    VALIDATION = auto()
    
class LoggerType(Enum):
    TENSORBOARD = auto()
    WANDB = auto()
    
class TaskType(Enum):
    CLASSIFICATION = auto()
    RECONSTRUCTION = auto()
        