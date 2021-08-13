import torch
import numpy as np
import sys
import time
from os.path import join
import os
from warnings import simplefilter
import utils
import model
import sys
import Procedure
# args = config.parse_args()
# world_config = world.world_config()
# config = world.config()
if not os.path.exists(utils.FILE_PATH):
    os.makedirs(utils.FILE_PATH, exist_ok=True)
# ==============================
utils.set_seed(utils.seed)
print(">>SEED:", utils.seed)
# ==============================

dataset = utils.construct_dataset()

print('---recModel---')
recModel = model.CF_MO(dataset)
print('--recModel finished---')

recModel = recModel.to(utils.device)

loss = utils.MMoELoss(recModel)

print('load and save to {}'.format(utils.weight_file))

if utils.LOAD:#world_config['LOAD']:
    try:
        # 导入现有模型
        recModel.load_state_dict(torch.load(utils.weight_file, map_location=torch.device('cpu')))
        utils.cprint(f"loaded model weights from {utils.weight_file}")
    except FileNotFoundError:
        print(f"{utils.weight_file} not exists, start from beginning")
Neg_k = 1

if utils.tensorboard:
# if world_config['tensorboard']:
    summary_path = utils.BOARD_PATH + '/' + (
            time.strftime("%m-%d-%Hh%Mm%Ss-") + utils.dataset_name + '-' + utils.comment)
    from tensorboardX import SummaryWriter

    w: SummaryWriter = SummaryWriter(str(summary_path))
# join(world.BOARD_PATH, time.strftime("%m-%d-%Hh%Mm%Ss-") + "-" + world.comment)
else:
    w = None
    utils.cprint("not enable tensorflowboard")


for epoch in range(utils.TRAIN_epochs):
        start = time.time()
        if epoch % 10 == 0:
            utils.cprint('[TEST]')
            Procedure.Test(dataset, recModel, epoch, w, utils.multicore)
        # recModel.train_attn_weight()
        Procedure.MMoE_train_original(dataset, recModel, loss, epoch, Neg_k, w)
Procedure.Test(dataset, recModel, utils.TRAIN_epochs, w, utils.multicore)
torch.save(recModel.state_dict(), utils.weight_file)

if utils.tensorboard:
    w.close()

