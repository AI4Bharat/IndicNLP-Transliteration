import enum

class RunMode(enum.Enum):
    train = "training"
    infer = "inference"
    test  = "testing"