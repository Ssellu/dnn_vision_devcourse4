from model.lenet5 import Lenet5

def get_model(model_name):
    if model_name == 'lenet5':
        return Lenet5
    else:
        print("Unknow Model Name")