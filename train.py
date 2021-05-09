
"""
    @Author: Junjie Jin
    @Code: Junjie Jin
    @Description: train our model (Relying on our loader framework in https://github.com/sfwyly/loader)

    UnSupervise mask image -> image

"""


from tqdm import tqdm
from utils import *
from trainer import *
from loader import *
from configs.config import *

configs = get_config("configs/unit_edges2handbags_folder.yaml")

def train(epochs = 100, val_per_epochs = 10):

    unit_trainer = UNIT_Trainer(configs)

    for i in range(epochs):

        all_gen_loss,all_dis_loss = trainer(unit_trainer)

        print(i," / ",epochs," gen_loss, dis_loss: ",all_gen_loss , all_dis_loss)
        if ((i + 1) % val_per_epochs == 0):
            val_loss = validate()
            print(i, " / ", epochs, " val_loss: ", val_loss)
        log_save() # save log

def log_save():

    pass

def save(i,generator):
    generator.save_weights(configs['save_path'] +str(i)+".h5")

def trainer(unit_trainer):

    train_dataloader = DataLoader(Dataset(root_path=configs['train_path']), batch_size=configs['batch_size'],
                                  image_size=(configs['image_size'], configs['image_size']), shuffle=True)
    # val_dataloader = DataLoader(Dataset(root_path=configs['val_path']), batch_size=configs['batch_size'],
    #                               image_size=(configs['image_size'], configs['image_size']), shuffle=True)

    if(configs['generated_mask']):

        train_mask_dataloader = DataLoader(Dataset(root_path = configs['train_mask_path']), batch_size=configs['batch_size'],
                                  image_size=(configs['image_size'], configs['image_size']), shuffle=True, is_mask=True)
        # val_mask_dataloader = DataLoader(Dataset(root_path = configs['val_mask_path']), batch_size=configs['batch_size'],
        #                         image_size=(configs['image_size'], configs['image_size']), shuffle=True)

    train_length = len(train_dataloader)
    all_gen_loss = []
    all_dis_loss = []
    for i,(X_trains,_) in enumerate(tqdm(train_dataloader)):

        if(not configs['generated_mask']):
            mask_list = getHoles((configs['image_size'],configs['image_size']),configs['batch_size'])
        else:
            length = len(train_mask_dataloader)
            mask_list = 1. - train_mask_dataloader[np.random.randint(length)][0][..., np.newaxis]
        print(X_trains.shape, mask_list.shape)
        loss_gen_total, loss_dis_total = trainer_step(X_trains * mask_list, X_trains, mask_list, unit_trainer)
        all_gen_loss.append(loss_gen_total)
        all_dis_loss.append(loss_dis_total)
    return np.mean(all_gen_loss), np.mean(all_dis_loss)

def validate():

    return 0

if(__name__=="__main__"):

    train(epochs=100, val_per_epochs= 10)