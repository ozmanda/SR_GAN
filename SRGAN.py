import PhIREGANs
import numpy as np
import os
import gan_utils
import utils


class SRGAN(PhIREGANs.PhIREGANs):
    # TODO: image generation isn't transferred to this class yet
    '''
    Inherits from the PhIREGANs class developed by Stengel et al. (2021) and extends it for training and 
    inference on QRF-produced temperature maps. 
    '''

    def __init__(self, data_type: str, sf: int) -> None:
        self.datatype = data_type
        self.scaling_factor = sf
        self.times = {'pretraintime': None, 'traintime': None, 'testtime': None, 'inferencetime': None}

        
        self.pretrain_tfrecord: str = ''
        self.pretrain_lr: float = 0
        self.pretrain_batchsize: int = 0
        self.pretrain_epochs: int = 0
        self.pretrained_model_dir: str = None
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
        if pretrainedmodel:
            self.set_pretrained_model(pretrainedmodel)

    def set_pretrain_data(self, pretrain_path):
        '''
        Sets the pretraining data based on the scaling factor and the data path. The correct functions 
        are called automatically based on the file extension.
        '''
        gan_utils.check_file(pretrain_path)
        if os.path.splitext(pretrain_path)[1] != '.tfrecord':
            self.pretrain_tfrecord = self.generate_pretrain_dataset(pretrain_path)
        else:
            self.pretrain_tfrecord = pretrain_path


    def generate_pretrain_dataset(self, pretrain_path):
        tfrecordpath = gan_utils.tfrecord_filename(pretrain_path, 'pretrain')
        if not os.path.isfile(tfrecordpath):
            print(f'\nGenerating Pretraining dataset from {pretrain_path}\n')
            imgarray_HR, imgarray_LR = gan_utils.generate_LRHR(pretrain_path, self.scaling_factor)
            utils.generate_TFRecords(tfrecordpath, data_HR=imgarray_HR, data_LR=imgarray_LR, mode='train')
            print(f'utils: {tfrecordpath}')
        return tfrecordpath
    

    def run_pretraining(self):
        print(f'    Initialising Pretraining')
        phiregans = PhIREGANs.PhIREGANs(data_type=self.datatype, N_epochs_pretrain=self.pretrain_epochs)
        gan_utils.start_timer()
        self.pretrain_model_dir = phiregans.pretrain(r = [self.scaling_factor], 
                                                     save_path = self.pretrain_savepath, 
                                                     data_path = self.pretrain_tfrecord, 
                                                     batch_size = self.pretrain_batchsize, 
                                                     pretrainedmodel_path = self.pretrained_model_dir)
        self.times['pretraintime'] = gan_utils.end_timer()


    # TRAINING ---------------------------------------------------------------------------------------------------------
    def set_trained_model(self, trained_model_path):
        assert os.path.isdir(os.path.dirname(trained_model_path)), f'Trained model path {trained_model_path} does not exist'
        self.trained_model_dir = trained_model_path

    def configure_training(self, datapath, epochs, batchsize_train, batchsize_test, learningrate, trainedmodel, savepath, split):
        if trainedmodel:
            self.set_trained_model(trainedmodel)
        elif self.pretrain_model_dir:
            self.set_trained_model(self.pretrain_model_dir)

        self.set_train_data(datapath, split)
        self.train_epochs = epochs
        self.train_batchsize = batchsize_train
        self.test_batchsize = batchsize_test
        self.training_lr = learningrate
        self.train_savepath = savepath        
    
    def set_train_data(self, train_path, split):
        '''
        Checks multiple training data options: 
        1. If one file is given, it must be .nc or .json data --> split into training and test sets, then converted to .tfrecord
        2. If two files are given:
                2.1. If both are .tfrecords, they are directly loaded
                2.2. If both are .nc or .json:
                        2.2.1. If split is True, the first file is the HR data and the second is the LR file, requiring train_test_split
                        2.2.2. If split is False, the first file is the training data and the second is the testing data, requiring direct conversion to .tfrecord
        '''
        if len(train_path) == 1:
            # must be .json or .nc --> requries train_test_split
            train_path = train_path[0]
            gan_utils.check_file(train_path)
            self.train_test_tfrecord(train_path, split)

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
                if split:
                    gan_utils.load_training_data(train_path, self.scaling_factor)
                else:
                    self.train_test_dataset(train_path)

        else:
            raise ValueError('Invalid number of paths given for training data')


    def train_hrlr_dataset(self, path):
        ''' For the case where HR and LR images come from separate files and require splitting '''
        imgarrayHR_train, imgarrayHR_test, imgarrayLR_train, imgarrayLR_test = gan_utils.load_training_data(path, self.scaling_factor)
        self.train_tfrecord = gan_utils.tfrecord_filename(path[0], 'train')
        gan_utils.generate_TFRecords(self.train_tfrecord, data_HR=imgarrayHR_train, data_LR=imgarrayLR_train, mode='train')
        gan_utils.generate_TFRecords(self.test_tfrecord, data_HR=imgarrayHR_test, data_LR=imgarrayLR_test, mode='test')

    
    def train_test_dataset(self, path):
        ''' For the case where two .nc or .json files are given, the data is directly loaded into tfrecords '''
        self.train_tfrecord = self.training_tfrecord(path[0], 'train')
        self.test_tfrecord, self.hr_test = self.training_tfrecord(path[1], 'test')

    
    def training_tfrecord(self, path, mode):
        '''
        Performs the TFRecord generation for either training or testing data
        '''
        tfrecordpath = gan_utils.tfrecord_filename(path, mode)
        hr, lr = gan_utils.generate_LRHR(path, self.scaling_factor)
        if mode == 'train':
            utils.generate_TFRecords(tfrecordpath, data_HR=hr, data_LR=lr, mode=mode)
            return tfrecordpath
        elif mode == 'test':
            utils.generate_TFRecords(tfrecordpath, data_LR=lr, mode=mode)
            np.save(os.path.join(os.path.dirname(path), f'{os.path.splitext(os.path.basename(path))[0]}_test_HR.npy'), hr)
            return tfrecordpath, hr
        
    
    def train_test_tfrecord(self, path, split):
        '''
        For the training case where one .nc or .json file is given, the data is split into training and testing sets
        '''
        filename, _ = os.path.splitext(os.path.basename(path))
        self.train_tfrecord = gan_utils.tfrecord_filename(path, 'train')
        self.test_tfrecord = gan_utils.tfrecord_filename(path, 'test')
        hr, lr = gan_utils.generate_LRHR(path, self.scaling_factor)
        hr_train, self.hr_test, lr_train, lr_test = gan_utils.train_test_split(hr, lr, test_size=split)
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

        phiregans = PhIREGANs.PhIREGANs(data_type='temperature', N_epochs_train=self.train_epochs, learning_rate=self.training_lr)

        gan_utils.start_timer()
        model_dir = phiregans.train(r=[self.scaling_factor],
                                    data_path=self.train_tfrecord,
                                    model_path=self.train_savepath,
                                    trainedmodelpath=model_dir,
                                    batch_size=self.train_batchsize)
        self.times['traintime'] = gan_utils.end_timer()

        gan_utils.start_timer()
        data_out, data_out_path = phiregans.test(r=[self.scaling_factor],
                                                 data_path=self.test_tfrecord,
                                                 model_path=model_dir,
                                                 batch_size=self.test_batchsize)
        self.times['testtime'] = gan_utils.end_timer()

        rmse = round(np.sqrt((1 / len(self.hr_test)) * np.sum((self.hr_test - data_out) ** 2)), 4)
        self.train_rmse = rmse
        self.write_training_info(data_out_path, phiregans.run_id)
        return data_out_path
    

    def write_training_info(self, data_out_path, run_id):
        infofile = open(os.path.join(os.path.dirname(data_out_path), f'training_information.txt'), 'w')
        infofile.writelines([f'{run_id} MODEL INFORMATION\n',
                             f'Scaling factor: {self.scaling_factor}\n',
                             f'Training data: {self.train_tfrecord}\n',
                             f'Batch size: {self.train_batchsize}\n',
                             f'Epochs: {self.train_epochs} training',
                             f'Times: {self.times["traintime"]} training, '
                             f'{self.times["testtime"]} testing'],
                             f'Mean squared error: {self.train_rmse}')
        infofile.close()
        

    # INFERENCE --------------------------------------------------------------------------------------------------------
    def configure_inference(self, inference_path, batchsize):
        self.set_inference_data(inference_path)
        self.inference_batchsize = batchsize


    def set_inference_data(self, inference_path):
        gan_utils.check_file(inference_path, mode='inference')
        if inference_path.endswith('.tfrecord'):
            self.inference_tfrecord = inference_path
        else:
            self.load_inference_data(inference_path)


    def load_inference_data(self, inference_path):
        filename, _ = os.path.splitext(os.path.basename(inference_path))
        self.inference_tfrecord = gan_utils.tfrecord_filename(inference_path, 'inference')
        if not os.path.isfile(self.inference_tfrecord):
            self.inference_hr = self.generate_inference_dataset(inference_path, filename)
        else:
            self.inference_hr = np.load(os.path.join(os.path.dirname(inference_path), f'{filename}_inference_HR.npy'))


    def generate_inference_dataset(self, inference_path, filename):
        hr, lr = gan_utils.generate_LRHR(inference_path, self.scaling_factor)
        utils.generate_TFRecords(self.inference_tfrecord, data_LR=lr, mode='test')
        np.save(os.path.join(os.path.dirname(inference_path), f'{filename}_inference_HR.npy'), hr)
        return hr


    def run_inference(self):
        assert self.trained_model_dir, 'A trained model must be given'
        phiregans = PhIREGANs.PhIREGANs(data_type=self.datatype)
        #! removed mu_sig calculation
        phiregans.mu_sig = utils.calculate_mu_sig(self.inference_hr)
        # phiregans.set_mu_sig(self.inference_tfrecord, 100)
        gan_utils.start_timer()
        data_out, data_out_path = phiregans.test(r=[self.scaling_factor],
                                                 data_path=self.inference_tfrecord,
                                                 model_path=self.trained_model_dir,
                                                 batch_size=self.inference_batchsize)
        self.times['inferencetime'] = gan_utils.end_timer()
        rmse = round(np.sqrt((1 / len(self.inference_hr)) * np.sum((self.inference_hr - data_out) ** 2)), 4)
        self.inference_rmse = rmse
        self.write_inference_info(data_out_path, phiregans.run_id)
        return data_out_path
    

    def write_inference_info(self, data_out_path, run_id):
        infofile = open(os.path.join(data_out_path, f'inference_information.txt'), 'w')
        infofile.writelines([f'{run_id} MODEL INFORMATION\n',
                             f'Trained model path: {self.trained_model_dir}\n',
                             f'Scaling factor: {self.scaling_factor}\n',
                             f'Inference data: {self.inference_tfrecord}\n',
                             f'Batch size: {self.inference_batchsize}\n',
                             f'Times: {self.times["inferencetime"]} inference\n',
                             f'Residual Mean Squared Error: {self.inference_rmse}'])
        infofile.close()
