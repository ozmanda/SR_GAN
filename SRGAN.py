import PhIREGANs
import numpy as np
import os
import gan_utils
import utils


class SRGAN(PhIREGANs.PhIREGANs):
    '''
    Inherits from the PhIREGANs class developed by Stengel et al. (2021) and extends it for training and 
    inference on QRF-produced temperature maps. 
    '''

    def __init__(self, data_type: str) -> None:
        '''
        
        '''
        self.datatype = data_type
        self.times = {'pretraintime': None, 'traintime': None, 'testtime': None, 'inferencetime': None}

        
        self.pretrain_tfrecord: str = ''
        self.pretrained_model_dir: str = ''
        
        self.train_tfrecord: str = ''
        self.test_tfrecord: str = ''
        self.hr_test: np.ndarray = None
        self.trained_model_dir: str = ''
        
        self.inference_tfrecord: str = ''
        
    # PRETRAINING ------------------------------------------------------------------------------------------------------
    def set_pretrained_model(self, pretrained_model_path):
        assert os.path.isdir(pretrained_model_path), f'Pretrained model path {pretrained_model_path} does not exist'
        self.pretrained_model_dir = pretrained_model_path


    def set_pretrain_data(self, pretrain_path, sf):
        '''
        Sets the pretraining data based on the scaling factor and the data path. The correct functions 
        are called automatically based on the file extension.
        '''
        gan_utils.check_file(pretrain_path)
        self.scaling_factor = sf
        if os.path.splitext(pretrain_path)[1] != '.tfrecord':
            self.pretrain_tfrecord = self.generate_pretrain_dataset(pretrain_path, sf)
        else:
            self.pretrain_tfrecord = pretrain_path


    def generate_pretrain_dataset(self, pretrain_path):
        #! image generation isn't transferred to this class yet
        imgdir = os.path.join(os.getcwd(), f'Images/{filename}_{self.scaling_factor}xSR')

        filename, ext = os.path.splitext(os.path.basename(pretrain_path))
        tfrecordpath = os.path.join(os.path.dirname(pretrain_path), f'{filename}_pretrain.tfrecord')
        if not os.path.isfile(tfrecordpath):
            print(f'\nGenerating Pretraining dataset from {pretrain_path}{ext}\n')
            imgarray_HR, imgarray_LR = gan_utils.generate_LRHR(pretrain_path, self.scaling_factor)
            utils.generate_TFRecords(tfrecordpath, data_HR=imgarray_HR, data_LR=imgarray_LR, mode='train')
            print(f'utils: {tfrecordpath}')
        return tfrecordpath
    

    def run_pretraining(self, epochs_pretrain, savepath, batchsize, pretrainedmodel):
        print(f'    Initialising Pretraining')
        phiregans = PhIREGANs(datatype=self.datatype, r=[self.scaling_factor], N_epochs_pretrain=epochs_pretrain)
        gan_utils.start_timer()
        self.pretrain_model_dir = self.pretrain(self.scaling_factor, save_path=savepath, datapath=self.pretrain_tfrecord, 
                                       batch_size=batchsize, pretrainedmodel_path=pretrainedmodel)


    # TRAINING ---------------------------------------------------------------------------------------------------------
    def set_trained_model(self, trained_model_path):
        assert os.path.isdir(trained_model_path), f'Trained model path {trained_model_path} does not exist'
        self.trained_model_dir = trained_model_path
    
    def set_train_data(self, train_path):
        if len(train_path) == 1:
            # must be .json or .nc --> requries train_test_split
            gan_utils.check_file(train_path)
            self.train_tfrecord, self.test_tfrecord = self.train_test_tfrecord(train_path)

        elif len(train_path) == 2:
            exts = [os.path.splitext(path)[-1] for path in train_path]
            
            # if both paths are .tfrecords
            if set(['.tfrecord']) == set(exts):
                for path in train_path:
                    gan_utils.check_files(path)
                self.train_tfrecord = path[0]
                self.test_tfrecord = path[1]

            else:
                for path in train_path:
                    gan_utils.check_file(path)
                self.train_test_dataset(train_path, split=False)

        else:
            raise ValueError('Invalid number of paths given for training data')


    def train_test_dataset(self, path):
        ''' For the case where two .nc or .json files are given, the data is directly loaded into tfrecords '''
        self.train_tfrecord = self.training_tfrecord(path[0], 'train')
        self.test_tfrecord, self.hr_test = self.training_tfrecord(path[1], 'test')

    
    def training_tfrecord(self, path, mode):
        
        filename, _ = os.path.splitext(os.path.basename(path))
        tfrecordpath = os.path.join(os.path.dirname(path), f'{filename}_{mode}.tfrecord')
        hr, lr = gan_utils.generate_LRHR(path, self.scaling_factor)
        if mode == 'train':
            utils.generate_TFRecords(tfrecordpath, data_HR=hr, data_LR=lr, mode=mode)
            return tfrecordpath
        elif mode == 'test':
            utils.generate_TFRecords(tfrecordpath, data_LR=lr, mode=mode)
            np.save(os.path.join(os.path.dirname(path), f'{filename}_test_HR.npy'), hr)
            return tfrecordpath, hr
        
    
    def train_test_tfrecord(self, path):
        '''
        For the training case where one .nc or .json file is given, the data is split into training and testing sets
        '''
        filename, _ = os.path.splitext(os.path.basename(path))
        self.train_tfrecord = os.path.join(os.path.dirname(path), f'{filename}_train.tfrecord')
        self.test_tfrecord = os.path.join(os.path.dirname(path), f'{filename}_test.tfrecord')
        hr, lr = gan_utils.generate_LRHR(path, self.scaling_factor)
        hr_train, hr_test, lr_train, lr_test = gan_utils.train_test_split(hr, lr)
        utils.generate_TFRecords(self.train_tfrecord, data_HR=hr_train, data_LR=lr_train, mode='train')
        utils.generate_TFRecords(self.test_tfrecord, data_LR=lr_test, mode='test')
        np.save(os.path.join(os.path.dirname(path), f'{filename}_test_HR.npy'), hr_test)

        
