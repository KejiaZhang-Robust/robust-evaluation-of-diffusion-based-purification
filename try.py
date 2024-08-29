import yaml
import torch
import torchvision.models as models
from robustbench.utils import load_model as load_clf

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

clf = load_clf(model_name='Standard', dataset='cifar10').to(device).eval()

print(clf)