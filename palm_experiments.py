import os
import argparse
import numpy as np
import netCDF4 as nc
import pandas as pd
import gan_utils
import insitu_maps
import SRGAN
from PhIREGANs import PhIREGANs

TRAININGFILES = ['mb4', 'mb5', 'mb6', 'mb7']
TESTFILES = ['mb8']
PALMPATHS =  {'mb4': 'mb_4_multi_stations_xy_N02.00m.nc',
              'mb5': 'mb_5_multi_stations_LCZ_xy_N02.00m.nc',
              'mb6': 'mb_6_multi_stations_LCZ_xy_N02.00m.nc',
              'mb7': 'mb_7_multi_stations_LCZ_xy_N02.00m.nc',
              'mb8': 'mb_8_multi_stations_LCZ_xy_N02.00m.nc'}
SPARSITIES = [0.8, 0.6, 0.4, 0.2, 0.1]
FAULTY_STATIONS = ['C059A2225266', 'D63DFE9B164B', 'D07769DF208C', 'DF15D23E4B15', 'E2A0DF1A4941', 'E437CB2AF225', 'F5C16A4B6340',
                   'F033A8C6BB79', 'F4683D808CFB', 'D3FE8EEF188C', 'D883D89E6A24', 'EC032D8260EB', 'C3FD36A6C1BC', 'D083B9FD07FB', 
                   'EB90524D4F3E', 'FCBBD3B1DB2C']

TRAINING_HYPERPARAMETERS = {'learning_rate': 1e-4, 'batchsize': 100, 'epochs': 10, 
                            'save_every': 10, 'print_every': 2}


class SparsePALM():
    def __init__(self, palmpath='Data/PALM', measurementpath='Data/Messdaten', scalingfactor=50, modelpath='models') -> None:
        self.SF: int = scalingfactor
        self.palmpath: str = palmpath
        self.measurementpath: str = measurementpath
        self.standardHRpath: dict = {}
        self.modelpath = modelpath

        # palmfile paths
        self.palmfilepaths = {}
        for palmnumber in TRAININGFILES:
            assert os.path.isfile(os.path.join(self.palmpath, PALMPATHS[palmnumber])), f'PALM file {PALMPATHS[palmnumber]} not found.'
            self.palmfilepaths[palmnumber] = os.path.join(self.palmpath, PALMPATHS[palmnumber])
        for palmnumber in TESTFILES:
            assert os.path.isfile(os.path.join(self.palmpath, PALMPATHS[palmnumber])), f'PALM file {PALMPATHS[palmnumber]} not found.'
            self.palmfilepaths[palmnumber] = os.path.join(self.palmpath, PALMPATHS[palmnumber])

        # palmfile subpaths
        self.palmsubpaths = {}
        for palmnumber in TRAININGFILES:
            self.palmsubpaths[palmnumber] = os.path.join(self.palmpath, palmnumber)
            if not os.path.isdir(self.palmsubpaths[palmnumber]):
                os.mkdir(self.palmsubpaths[palmnumber])
        for palmnumber in TESTFILES:
            self.palmsubpaths[palmnumber] = os.path.join(self.palmpath, palmnumber)
            if not os.path.isdir(self.palmsubpaths[palmnumber]):
                os.mkdir(self.palmsubpaths[palmnumber])


    def run_estimation(self, baseline=True, sparse=True, insitu=True) -> None:
        if baseline:
            self.baseline_run()
        if sparse:
            for sparsity in SPARSITIES:
                self.sparse_run(sparsity)
        if insitu:
            self.insitu_run()

    
    def baseline_run(self) -> None:
        modelsavepath = os.path.join(self.modelpath, 'baseline')
        trainingfile_name = 'baseline_train.tfrecord'
        print('Training baseline model...')
        print('    ', end='')
        trainedmodelpath, trainedmodel = self.training_pass(trainingfile_name, modelsavepath)
        self.test('baseline_test.tfrecord', trainedmodelpath, trainedmodel)

    
    def sparse_run(self, sparsity: float) -> None:
        for sparsity in SPARSITIES:
            print(f'Training sparse model with sparsity {sparsity}...')
            print('    ', end='')
            modelsavepath = os.path.join(self.modelpath, f'sparse_{sparsity}')
            trainingfile_name = f'sparse_{sparsity}_train.tfrecord'
            trainedmodelpath, trainedmodel = self.training_pass(trainingfile_name, modelsavepath)
            self.test(f'sparse_{sparsity}_test.tfrecord', trainedmodelpath, trainedmodel)

    def insitu_run(self) -> None:
        modelsavepath = os.path.join(self.modelpath, 'insitu')
        trainingfile_name = 'insitu_train.tfrecord'
        print('Training insitu model...')
        print('    ', end='')
        trainedmodelpath, trainedmodel = self.training_pass(trainingfile_name, modelsavepath)
        self.test('insitu_test.tfrecord', trainedmodelpath, trainedmodel)
        pass


    def training_pass(self, trainingfilename: str, modelpath: str) -> str:
        assert len(TRAININGFILES) >= 2, 'At least two PALM files are required for pretraining and training.'
        phiregans = PhIREGANs(data_type='temperature', N_epochs_train=TRAINING_HYPERPARAMETERS['epochs'], learning_rate=TRAINING_HYPERPARAMETERS['learning_rate'])
        model_dir = phiregans.pretrain(r=[self.SF],
                                       data_path=os.path.join(self.palmsubpaths[TRAININGFILES[0]], trainingfilename),
                                       save_path=modelpath)
        
        for palmnumber in TRAININGFILES[1:]:
            model_dir = phiregans.train(r=[self.SF],
                                        data_path=os.path.join(self.palmsubpaths[palmnumber], trainingfilename),
                                        model_path=modelpath,
                                        trainedmodelpath=model_dir)
            print(f', {palmnumber}', end='')
        print('    done.\n')

        return model_dir, phiregans


    def test(self, testfilename: str, trainedmodelpath: str, trainedmodel: PhIREGANs) -> None:
        for palmnumber in TESTFILES:
            data_out, data_out_path = trainedmodel.test(r=[self.SF],
                                                    data_path=os.path.join(self.palmsubpaths[palmnumber], testfilename),
                                                    model_path=trainedmodelpath)
            ground_truth = np.load(os.path.join(self.palmsubpaths[palmnumber], 'HR.npy'))
            print(f'       Test output of model {os.path.basename(trainedmodelpath)} for {palmnumber} saved at {data_out_path}')
            self.evalute_test(data_out, ground_truth, palmnumber)            


    def evalute_test(self, data_out: np.ndarray, ground_truth: np.ndarray, palmnumber: str) -> None:
        errors = ground_truth - data_out
        max_error = np.nanmax(errors)
        min_error = np.nanmin(errors)
        average_error = round(np.nanmean(errors), 4)
        mse = round((1 / len(ground_truth)) * np.nansum(errors ** 2), 4)
        rmse = round(np.sqrt(mse), 4)
        print(f'           RMSE: {rmse} | MSE: {mse} | Average Error: {average_error} | Max Error: {max_error} | Min Error: {min_error}')
        infofile = open(os.path.join(self.palmsubpaths[palmnumber], 'info.txt'), 'w')
        infofile.writelines([f'RMSE: {rmse}\n', f'MSE: {mse}\n', f'Average Error: {average_error}\n', f'Max Error: {max_error}\n', f'Min Error: {min_error}\n'])


    # -------------------------------------------- DATASET GENERATION --------------------------------------------
    def dataset_generation(self, stationinfo) -> None:
        '''
        Datasets to be generated: 
            1. standard HR & LR arrays (PALM)
            2. sparse LR arrays
            3. in-situ LR array (measurements)
        '''
        for palmnumber in TRAININGFILES:
            self.datagen_pass(palmnumber, stationinfo)
        for palmnumber in TESTFILES:
            self.datagen_pass(palmnumber, stationinfo, mode='test')

    def datagen_pass(self, palmnumber, stationinfo, mode='train') -> None:
        print(f'Generating testing datasets for {palmnumber}...')
        surfacetemps, times = self.extract_palm_data(palmnumber, self.palmfilepaths[palmnumber])
        np.save(os.path.join(self.palmsubpaths[palmnumber], 'times.npy'), times)
        hr_array = self.generate_baseline_arrays(surfacetemps, self.palmsubpaths[palmnumber], mode=mode)
        self.generate_sparse_arrays(hr_array, self.palmsubpaths[palmnumber], mode=mode)
        self.generate_insitu_array(hr_array, palmnumber, self.palmsubpaths[palmnumber], stationinfo, times, mode=mode)


    def extract_palm_data(self, palmnumber, palmfilepath: str):
        palmfile = nc.Dataset(palmfilepath, 'r', format='NETCDF4')
        times = gan_utils.palm_times(palmfile)
        surfacetemps = self.generate_surfacetemps(self.palmsubpaths[palmnumber], palmfile)
        return surfacetemps, times

            
    def generate_surfacetemps(self, palmsubpath: str, palmfile: nc.Dataset) -> np.ndarray:
        print('  surfacetemps...')
        if not os.path.isfile(os.path.join(palmsubpath, 'surfacetemps.npy')):
            surfacetemps: np.ndarray = gan_utils.extract_surfacetemps(palmfile)
            np.save(os.path.join(palmsubpath, 'surfacetemps.npy'), surfacetemps)
        else:
            surfacetemps = np.load(os.path.join(palmsubpath, 'surfacetemps.npy'))
        print('  done.')
        return surfacetemps
    

    def generate_baseline_arrays(self, surfacetemps: np.ndarray, palmsubpath: str, mode='train') -> np.ndarray:
        print('  baseline...')
        tfrecordpath: str = os.path.join(palmsubpath, f'baseline_{mode}.tfrecord')
        if not os.path.isfile(tfrecordpath):
            if not os.path.isfile(os.path.join(palmsubpath, 'HR.npy')):
                hr_array, lr_array = gan_utils.generate_LRHR_from_array(surfacetemps, scalingfactor=self.SF)
                np.save(os.path.join(palmsubpath, 'HR.npy'), hr_array)
                np.save(os.path.join(palmsubpath, 'LR.npy'), lr_array)
            else: 
                hr_array = np.load(os.path.join(palmsubpath, 'HR.npy'))
                lr_array = np.load(os.path.join(palmsubpath, 'LR.npy'))
            gan_utils.generate_TFRecords(tfrecordpath, data_HR=hr_array, data_LR=lr_array, mode=mode)
        else: 
            hr_array = np.load(os.path.join(palmsubpath, 'HR.npy'))
        print('  done.')
        return hr_array


    def generate_sparse_arrays(self, hr_array, palmsubpath, mode='train') -> None:
        print('  sparse...')
        for sparsity in SPARSITIES:
            print(f'    {sparsity}')
            tfrecordpath = os.path.join(palmsubpath, f'sparse_{sparsity}_{mode}.tfrecord')
            if not os.path.isfile(tfrecordpath):
                if not os.path.isfile(os.path.join(palmsubpath, f'LR_{sparsity}.npy')):
                    sparse_hr = self.sparse_array(sparsity, hr_array)
                    _, sparse_lr = gan_utils.generate_LRHR_from_array(sparse_hr, scalingfactor=self.SF)
                    np.save(os.path.join(palmsubpath, f'LR_{sparsity}.npy'), sparse_lr)
                else: 
                    sparse_lr = np.load(os.path.join(palmsubpath, f'LR_{sparsity}.npy'))
                gan_utils.generate_TFRecords(tfrecordpath, data_HR=hr_array, data_LR=sparse_lr, mode=mode)
            
        print('  done.')


    def sparse_array(self, sparsity: float, array: np.ndarray) -> np.array:
        unraveled: np.ndarray = array.reshape(-1)
        indices: np.ndarray = np.random.choice(unraveled.shape[0], int(sparsity * unraveled.shape[0]), replace=False)
        unraveled[indices] = np.nan
        unraveled = unraveled.reshape(array.shape)
        return unraveled


    def generate_insitu_array(self, hr_array, palmnumber, palmsubpath, stationinfo, times, mode='train') -> None:
        print('  in-situ...') 
        tfrecordpath = os.path.join(palmsubpath, f'insitu_{mode}.tfrecord')
        if not os.path.isfile(tfrecordpath):
            if not os.path.isfile(os.path.join(palmsubpath, 'insitu_LR.npy')):
                insitu_hr_array = insitu_maps.highres_maps(os.path.join(self.palmpath, PALMPATHS[palmnumber]), self.measurementpath, stationinfo, times, hr_array.shape)
                _ , insitu_lr_array = gan_utils.generate_LRHR_from_array(insitu_hr_array, scalingfactor=self.SF)
                np.save(os.path.join(palmsubpath, 'insitu_LR.npy'), insitu_lr_array)
            else:
                insitu_lr_array = np.load(os.path.join(palmsubpath, 'insitu_LR.npy'))
            gan_utils.generate_TFRecords(tfrecordpath, data_HR=hr_array, data_LR=insitu_lr_array, mode=mode)
        print('  done.')


if __name__ == '__main__': 
    parser = argparse.ArgumentParser()
    parser.add_argument('--palmfolder', type=str, help='Folder to store PALM data', default='Data/PALM')
    parser.add_argument('--measurementpath', help='Folder to measurememtn data', default='Data/Messdaten')
    parser.add_argument('--scalingfactor', type=int, help='Scaling factor for super resolution', default=50)
    parser.add_argument('--modelpath', type=str, help='Folder to store model data', default='models')
    parser.add_argument('--generate_dataset', type=bool, help='Generate datasets', default=False)
    parser.add_argument('--baseline', type=bool, help='Run baseline estimation', default=False)
    parser.add_argument('--sparse', type=bool, help='Run sparse estimator', default=False)
    parser.add_argument('--insitu', type=str, help='Run in-situ estimator', default=False)
    parser.add_argument('--stationinfo', type=str, help='Station information file', default='Data/stations.csv')
    parser.add_argument('--palmnumber', type=str, help='PALM number', default=None, nargs='*')
    args = parser.parse_args()
    if args.palmnumber:
        TRAININGFILES = args.palmnumber

    print(args.generate_dataset)
    sparse_estimator = SparsePALM()
    if args.generate_dataset:
        sparse_estimator.dataset_generation(stationinfo=args.stationinfo)
    sparse_estimator.run_estimation(baseline=args.baseline, sparse=args.sparse, insitu=args.insitu)