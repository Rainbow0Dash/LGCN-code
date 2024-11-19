from scipy import io
import time
import warnings

import numpy as np

import loader
from model import *
from tqdm import trange
from torch.utils.tensorboard import SummaryWriter

os.environ['CUDA_LAUNCH_BLOCKING'] = '1'

# path_hha = r"E:\Eevee\model\hha+img\nyud-hha.pth"
# model_hha = torch.load(path_hha)
path_img = r'E:\Eevee\newnet\model\model-10\2024-11-19-08-12-07.pth'
model_img = torch.load(path_img)
root = r'E:\Eevee\newnet\model\model-10\test'
count = 0
threshold = 0
fps = 0


def warmup(model):
    data = torch.rand([1, 3, 512, 512])
    data = data.cuda()
    # data = data.half()
    model(data)
    print('warm up endding')


# warmup(model_hha)
warmup(model_img)

# data = loader.NYUD_Loader(split='test')
data = loader.BSDS_Loader(split='test')
test_data = DataLoader(dataset=data, batch_size=1,
                       num_workers=0, shuffle=True)

print(len(data))
print(threshold)

with torch.no_grad():
    total_time = 0
    for idx, (img, name) in enumerate(test_data):
        img = img.cuda()
        shape = img.shape
        h = shape[-2]
        w = shape[-1]
        t1 = time.time()
        result = model_img(img)
        torch.cuda.synchronize()
        t2 = time.time()
        total_time = total_time + (t2 - t1)
        print(t2 - t1, total_time)
        answer = (result[-1]).clone()


        answer0 = answer.clone()
        torchvision.utils.save_image(answer,
                                     root + (r'\result\%s.png' % name))
        answer0 = torch.reshape(answer0, [answer0.shape[-2], answer0.shape[-1]])
        answer0 = np.array(answer0.cpu())
        io.savemat(root + (r'\mat\mat\%s.mat' % name), {'result': answer0})
        print("[%s/%s] fps:%s" % (count + 1, len(data), (1 / (total_time / (count + 1)))))
        count += 1

    print("end")
