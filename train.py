import gc
import time
import warnings

import loader
from model import *
from tqdm import trange
from torch.utils.tensorboard import SummaryWriter

os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
warnings.filterwarnings('ignore', category=FutureWarning)
# model
# model = model(out_channel=10)
# model = torch.load(r'')

model = torch.load(r'E:\Eevee\newnet\model\model-2\2024-11-13-17-36-57.pth')

model = model.cuda()

# dataset = nyud.NYUD_Loader(setting=['hha'])
# dataset = loader.BSDS_Loader()
dataset = loader.NYUD_Loader(split='test')
train_data = DataLoader(dataset=dataset, batch_size=1,
                        num_workers=0, shuffle=True)
print(len(dataset))

# opti
conv_weights, bn_weights, relu_weights = model.get_weight()
param_groups = [{
    'params': conv_weights,
    'weight_decay': 1e-4,
    'lr': 0.05}, {
    'params': bn_weights,
    'weight_decay': 0.1 * 1e-4,
    'lr': 0.05}, {
    'params': relu_weights,
    'weight_decay': 0.0,
    'lr': 0.05
}]
opt = torch.optim.Adam(params=param_groups, betas=(0.9, 0.99))

##############################
lr = 0.05

writer = SummaryWriter('runs/experiment')
for epoch in range(120):
    start = time.time()
    lr = adjust_learning_rate(opt, epoch, lr)
    print('-------第%s训练-------' % (epoch + 1))
    opt.zero_grad()
    counter = 0
    iter_size = 8
    loss_value = 0
    total_loss = 0
    for progress_bar, (img, lb) in zip(trange(len(dataset)), train_data):
        # print(lb.size())
        img = img.cuda()
        lb = lb.cuda()
        outputs = model(img)
        for o in outputs:
            loss = cross_entropy_loss_RCF(o, lb, 1.1)
        counter = counter + 1
        loss_value += loss.item()
        loss = loss / iter_size
        loss.backward()

        if counter == iter_size:
            opt.step()
            opt.zero_grad()
            counter = 0
            total_loss += loss_value
            loss_value = 0
    writer.add_scalar('Loss/train', total_loss, epoch)
    print('loss:%s,epoch:%s' % (total_loss, epoch))
    total_loss = 0
    torch.save(model, r'E:\Eevee\newnet\model\model-5\%s.pth' % (time.strftime('%Y-%m-%d-%H-%M-%S', time.localtime())))
    gc.collect()
    torch.cuda.empty_cache()
writer.close()
