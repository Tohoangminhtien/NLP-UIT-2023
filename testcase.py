from builders.vocab_builder import build_vocab
from configs import *
from data_utils import *
from models import *
from utils import *
from builders.model_builder import build_model

from tqdm import tqdm


# Dùng để test từng hàm một, hiểu rõ hơn về phương pháp thực hiện bài toán
config = get_config('configs/GLAICHEVE.yaml')
device = torch.device(config.MODEL.DEVICE)

vocab = build_vocab(config.DATASET.VOCAB)