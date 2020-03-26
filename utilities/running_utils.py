import enum

class RunMode(enum.Enum):
    train = "training"
    infer = "inference"
    test  = "testing"

def load_pretrained(model, weight_path):
    if not weight_path:
        return model

    pretrain_dict = torch.load(weight_path)
    model_dict = model.state_dict()
    pretrain_dict = {k: v for k, v in pretrain_dict.items() if k in model_dict}
    print("Pretrained layers Loaded:", pretrain_dict.keys())
    model_dict.update(pretrain_dict)
    model.load_state_dict(model_dict)

    return model


def count_train_param(model):
    train_params_count = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print('The model has {} trainable parameters'.format(train_params_count))
    return train_params_count