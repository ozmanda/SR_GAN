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

    def __init__(self, data_type: str, sf: int) -> None:
        '''
        
        '''
        self.datatype = data_type
        self.scaling_factor = sf
        self.times = {'pretraintime': None, 'traintime': None, 'testtime': None, 'inferencetime': None}

        
        self.pretrain_tfrecord: str = ''
        self.pretrain_lr: float = 0
        self.pretrain_batchsize: int = 0
        self.pretrain_epochs: int = 0
        self.pretrained_model_dir: str = ''
        self.pretrain_savepath: str = ''
        
        self.train_tfrecord: str = ''
        self.train_lr: float = 0
        self.train_batchsize: int = 0
        self.train_epochs: int = 0
        self.trained_model_dir: str = ''
        self.train_savepath: str = ''

        self.test_tfrecord: str = ''
        self.test_batchsize: int = 0
        self.hr_test: np.ndarray = None
        
        self.inference_tfrecord: str = ''
        self.inference_batchsize: int = 0
        self.inference_hr: np.ndarray = None
        
    # PRETRAINING ------------------------------------------------------------------------------------------------------
    def set_pretrained_model(self, pretrained_model_path):
        assert os.path.isdir(pretrained_model_path), f'Pretrained model path {pretrained_model_path} does not exist'
        self.pretrained_model_dir = pretrained_model_path

    def configure_pretraining(self, datapath, epochs, batchsize, learningrate, pretrainedmodel, savepath):
        self.pretrain_epochs = epochs
        self.pretrain_batchsize = batchsize
        self.pretrain_lr = learningrate
        self.pretrain_savepath = savepath
        self.set_pretrain_data(datapath)
        self.set_pretrained_model(pretrainedmodel)

    def set_pretrain_data(self, pretrain_path):
        '''
        Sets the pretraining data based on the scaling factor and the data path. The correct functions 
        are called automatically based on the file extension.
        '''
        gan_utils.check_file(pretrain_path)
        if os.path.splitext(pretrain_path)[1] != '.tfrecord':
            self.pretrain_tfrecord = self.generate_pretrain_dataset(pretrain_path, self.scaling_factor)
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
    

    def run_pretraining(self):
        print(f'    Initialising Pretraining')
        phiregans = PhIREGANs(datatype=self.datatype, r=[self.scaling_factor], N_epochs_pretrain=self.pretrain_epochs)
        gan_utils.start_timer()
        self.pretrain_model_dir = phiregans.pretrain(self.scaling_factor, save_path=self.pretrain_savepath, datapath=self.pretrain_tfrecord, 
                                                     batch_size=self.pretrain_batchsize, pretrainedmodel_path=self.pretrained_model_dir)
        self.times['pretraintime'] = gan_utils.end_timer()


    # TRAINING ---------------------------------------------------------------------------------------------------------
    def set_trained_model(self, trained_model_path):
        assert os.path.isdir(trained_model_path), f'Trained model path {trained_model_path} does not exist'
        self.trained_model_dir = trained_model_path

    def configure_training(self, datapath, epochs, batchsize_train, batchsize_test, learningrate, trainedmodel, savepath):
        if trainedmodel:
            self.set_trained_model(trainedmodel)
        self.set_train_data(datapath)
        self.train_epochs = epochs
        self.train_batchsize = batchsize_train
        self.test_batchsize = batchsize_test
        self.training_lr = learningrate
        self.train_savepath = savepath        
    
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
        hr_train, self.hr_test, lr_train, lr_test = gan_utils.train_test_split(hr, lr)
        utils.generate_TFRecords(self.train_tfrecord, data_HR=hr_train, data_LR=lr_train, mode='train')
        utils.generate_TFRecords(self.test_tfrecord, data_LR=lr_test, mode='test')
        np.save(os.path.join(os.path.dirname(path), f'{filename}_test_HR.npy'), self.hr_test)


    def run_training(self):
        if self.trained_model_dir:
            model_dir = self.trained_model_dir
            if self.pretrain_model_dir:
                Warning('Both a pretrained and trained model have been set. The trained model will be used for training')
        elif self.pretrain_model_dir:
            model_dir = self.pretrain_model_dir
        else:
            model_dir = None

        phiregans = PhIREGANs(data_type='temperature', N_epochs_train=self.train_epochs, learning_rate=self.training_lr)

        gan_utils.start_timer()
        model_dir = phiregans.train(r=[self.scaling_factor],
                                    data_path=self.train_tfrecord,
                                    model_path=self.train_savepath,
                                    trainedmodelpath=model_dir,
                                    batch_size=self.train_batchsize)
        self.times['traintime'] = gan_utils.nd_timer()

        gan_utils.start_timer()
        data_out, data_out_path = phiregans.test(r=[self.scaling_factor],
                                                 data_path=self.test_tfrecord,
                                                 model_path=model_dir,
                                                 batch_size=self.test_batchsize)
        self.times['testtime'] = gan_utils.end_timer()

        mse = round((1 / len(self.hr_test)) * np.sum((self.hr_test - data_out) ** 2), 4)
        self.train_mse = mse
        return data_out_path

        

    # INFERENCE --------------------------------------------------------------------------------------------------------
    def configure_inference(self, inference_path, batchsize):
        self.set_inference_data(inference_path, batchsize)
        self.inference_batchsize = batchsize


    def set_inference_data(self, inference_path):
        gan_utils.check_file(inference_path)
        self.load_inference_data(inference_path)


    def load_inference_data(self, inference_path):
        filename, _ = os.path.splitext(os.path.basename(inference_path))
        self.inference_tfrecord = os.path.join(os.path.dirname(inference_path), f'{filename}_inference.tfrecord')
        if not os.path.isfile(self.inference_tfrecord):
            self.inference_hr = self.generate_inference_dataset(inference_path, filename)
        else:
            self.inference_hr = np.load(os.path.join(os.path.dirname(inference_path), f'{filename}_inference_HR.npy'))


    def generate_inference_dataset(self, inference_path, filename):
        hr, lr = gan_utils.generate_LRHR(inference_path, self.scaling_factor)
        utils.generate_TFRecords(self.inference_tfrecord, data_LR=lr, mode='inference')
        np.save(os.path.join(os.path.dirname(inference_path), f'{filename}_inference_HR.npy'), hr)
        return hr


    def run_inference(self):
        assert self.trained_model_dir, 'A trained model must be given'
        phiregans = PhIREGANs(data_type=self.data_type)
        gan_utils.start_timer()
        data_out, data_out_path = phiregans.test(r=[self.scaling_factor],
                                                 data_path=self.inference_tfrecord,
                                                 model_path=self.trained_model_dir,
                                                 batch_size=self.inference_batchsize)
        self.times['inferencetime'] = gan_utils.end_timer()
        mse = round((1 / len(self.inference_hr)) * np.sum((self.inference_hr - data_out) ** 2), 4)
        self.inference_mse = mse
        return data_out_path


    # GENERAL FUNCTIONS ------------------------------------------------------------------------------------------------
    def write_run_info(self, modes, path, batchsize, epochs, mse):
        infofile = open(os.path.join(os.path.dirname(path), f'model_information.txt'), 'w')
        infofile.writelines([f'{self.model_name} MODEL INFORMATION\n',
                             f'Scaling factor: {self.scaling_factor}\n',
                             f'Training data: {self.train_tfrecord}\n',
                             f'Batch size: {batchsize}\n',
                             f'Epochs: {epochs} training',
                             f'Times: {self.times["pretraintime"]} pretraining, {self.times["traintime"]} training, '
                             f'{self.times["testtime"]} testing'],
                             f'Mean squared error: {np.round(mse, 2)}')
        infofile.close()