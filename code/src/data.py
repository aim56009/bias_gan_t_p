from dataclasses import dataclass
import cftime
import numpy as np
import pytorch_lightning as pl
import torch
import xarray as xr
from torch.utils.data import DataLoader
from bias_gan_t_p.code.src.utils import (log_transform, norm_minus1_to_plus1_transform, norm_transform)



# +
from dataclasses import dataclass
import cftime
from torch.utils.data import DataLoader


@dataclass
class TestData():
    
    era5: xr.DataArray
    gan: xr.DataArray
    climate_model: xr.DataArray = None
    uuid: str = None
    model = None
    

    def model_name_definition(self, key):
        dict = {
            'era5': 'ERA5',
            'gan': 'GAN (unconstrained)',
            'climate_model': 'Climate model',
        }
        return dict[key]


    def colors(self, key):
        dict = {
            'era5': 'k',
            'gan': 'brown',
            'climate_model': 'r',
        }
        return dict[key]

        
    def convert_units(self):
        # from mm/s to mm/d
        climate_model_pr = self.climate_model.precipitation*3600*24
        era5_pr = self.era5.precipitation*3600*24
        gan_pr = self.gan.precipitation*3600*24
        
        
        self.era5['precipitation'] = era5_pr
        self.climate_model['precipitation'] = climate_model_pr
        self.gan['precipitation'] = gan_pr
        
        
    def crop_test_period(self):
        
        print('')
        print(f'Test set period: {self.gan.tas.time[0].values} - {self.gan.tas.time[-1].values}')
        

        """
        print("self.era5['precipitation'] before:",self.era5['precipitation'])
        climate_model_pr = self.climate_model.precipitation.sel(time=slice(self.gan.tas.time[0], self.gan.tas.time[-1]))
        climate_model_t = self.climate_model.tas.sel(time=slice(self.gan.tas.time[0], self.gan.tas.time[-1]))
        era5_pr = self.era5.precipitation.sel(time=slice(self.gan.tas.time[0], self.gan.tas.time[-1]))
        era5_t = self.era5.tas.sel(time=slice(self.gan.tas.time[0], self.gan.tas.time[-1]))
        
        self.era5['precipitation'] = era5_pr
        self.climate_model['precipitation'] = climate_model_pr
        
        self.era5['tas'] = era5_t
        self.climate_model['tas'] = climate_model_t
        
        print("self.era5['precipitation'] after:",self.era5['precipitation'])
        """
        #print("self.era5['precipitation'] before:",self.era5['precipitation'])      
        climate_model_pr = self.climate_model.precipitation.sel(time=slice(self.gan.tas.time[0], self.gan.tas.time[-1]))
        climate_model_t = self.climate_model.tas.sel(time=slice(self.gan.tas.time[0], self.gan.tas.time[-1]))
        era5_pr = self.era5.precipitation.sel(time=slice(self.gan.tas.time[0], self.gan.tas.time[-1]))
        era5_t = self.era5.tas.sel(time=slice(self.gan.tas.time[0], self.gan.tas.time[-1]))
                
        time_slice = slice(self.gan.tas.time[0], self.gan.tas.time[-1])
        self.era5 = self.era5.sel(time=time_slice)
        self.era5['precipitation'] = era5_pr
        self.era5['tas'] = era5_t

        time_slice = slice(self.gan.tas.time[0], self.gan.tas.time[-1])
        self.climate_model = self.climate_model.sel(time=time_slice)
        self.climate_model['precipitation'] = climate_model_pr
        self.climate_model['tas'] = climate_model_t
        
        #print("self.era5['precipitation'] after:",self.era5['precipitation'])
        


        
    def show_mean(self):
        print('')
        print(f'Mean precipitation [mm/d]:')
        print(f'ERA5: {self.era5.precipitation.mean().values:2.3f}')
        print(f'Climate Model: {self.climate_model.precipitation.mean().values:2.3f}')
        print(f'GAN:  {self.gan.precipitation.mean().values:2.3f}')
        
        print('')
        print(f'Mean temperature [K]:')
        print(f'ERA5: {self.era5.tas.mean().values:2.3f}')
        print(f'Climate Model: {self.climate_model.tas.mean().values:2.3f}')
        print(f'GAN:  {self.gan.tas.mean().values:2.3f}')



class CycleDataset(torch.utils.data.Dataset):
    
    def __init__(self, stage, config, epsilon=0.0001): 
        # stage: train, valid, test
        
        self.transforms = config.transforms
        self.epsilon = config.epsilon
        self.config = config

        if config.lazy:
            self.cache = False
            self.chunks = {'time': 1}
        else:#this
            self.cache = True
            self.chunks = None
        
        self.splits = {
                "train": [str(config.train_start), str(config.train_end)],
                "valid": [str(config.valid_start), str(config.valid_end)],
                "test":  [str(config.test_start), str(config.test_end)],
        }

        self.stage = stage
        
        self.climate_model = self.load_climate_model_data()
        climate_model_reference = self.load_climate_model_data(is_reference=True)
        
        self.era5 = self.load_era5_data()
        era5_reference = self.load_era5_data(is_reference=True)

        self.num_samples = len(self.era5.tas.time.values)
        
        self.era5 = self.apply_transforms(self.era5, era5_reference)
        self.climate_model = self.apply_transforms(self.climate_model, climate_model_reference)



    def load_climate_model_data(self, is_reference=False):
        # Y-domain samples 

        climate_model_pr = xr.open_dataset(self.config.model_pr_path, cache=self.cache, chunks=self.chunks)
        climate_model_pr =  climate_model_pr.precipitation
                
        climate_model_t = xr.open_dataset(self.config.model_t_path, cache=self.cache, chunks=self.chunks)
        climate_model_t =  climate_model_t.tas
                

        if not self.config.lazy:
            climate_model_pr = climate_model_pr.load()
            climate_model_t = climate_model_t.load()

        if is_reference:
            climate_model_pr = climate_model_pr.sel(time=slice(self.splits['train'][0], self.splits['train'][1]))
            climate_model_t = climate_model_t.sel(time=slice(self.splits['train'][0], self.splits['train'][1]))
        else:
            climate_model_pr = climate_model_pr.sel(time=slice(self.splits[self.stage][0],self.splits[self.stage][1]))
            climate_model_t = climate_model_t.sel(time=slice(self.splits[self.stage][0],self.splits[self.stage][1]))
        
        
        climate_model = xr.Dataset({'precipitation': climate_model_pr, 'tas': climate_model_t})

        return climate_model


    def load_era5_data(self, is_reference=False):
        #X-domain samples 

        era5_pr = xr.open_dataset(self.config.era5_pr_path,cache=self.cache, chunks=self.chunks).era5_precipitation
        era5_t = xr.open_dataset(self.config.era5_t_path,cache=self.cache, chunks=self.chunks).tas

        if not self.config.lazy:
            era5_pr = era5_pr.load()
            era5_t = era5_t.load()

        if is_reference:
            era5_pr = era5_pr.sel(time=slice(self.splits['train'][0],self.splits['train'][1]))
            era5_t = era5_t.sel(time=slice(self.splits['train'][0],self.splits['train'][1]))

        else:
            era5_pr = era5_pr.sel(time=slice(self.splits[self.stage][0], self.splits[self.stage][1]))
            era5_t = era5_t.sel(time=slice(self.splits[self.stage][0], self.splits[self.stage][1]))    
        
        era5 = xr.Dataset({'precipitation': era5_pr, 'tas': era5_t})

        return era5
        

    def apply_transforms(self, data, data_ref):
        
        data_prec = data.precipitation
        data_tas = data.tas
                
        data_prec_ref = data_ref.precipitation
        data_tas_ref = data_ref.tas
        
                
        if 'log' in self.transforms:
            data_prec = log_transform(data_prec, self.epsilon)
            data_prec_ref = log_transform(data_prec_ref, self.epsilon)

        if 'normalize' in self.transforms:
            data = norm_transform(data, data_ref)

        if 'normalize_minus1_to_plus1' in self.transforms:
            data_prec = norm_minus1_to_plus1_transform(data_prec, data_prec_ref)
            data_tas = norm_minus1_to_plus1_transform(data_tas, data_tas_ref)
        
        data_ref['precipitation'] = data_prec_ref
        
        data['precipitation'] = data_prec
        data['tas'] = data_tas
        
                
        return data


    def __getitem__(self, index):
        
        x_p = torch.from_numpy(self.era5.precipitation.isel(time=index).values).float().unsqueeze(0)
        x_t = torch.from_numpy(self.era5.tas.isel(time=index).values).float().unsqueeze(0)
        x = torch.cat((x_p, x_t), dim=0)
        
        y_p = torch.from_numpy(self.climate_model.precipitation.isel(time=index).values).float().unsqueeze(0)
        y_t = torch.from_numpy(self.climate_model.tas.isel(time=index).values).float().unsqueeze(0)
        y = torch.cat((y_p, y_t), dim=0)
        
        
        sample = {'A': x, 'B': y}
        
        return sample

    def __len__(self):
        return self.num_samples

# -


class ProjectionDataset(torch.utils.data.Dataset):

    def __init__(self, config, epsilon=0.0001):


        self.transforms = config.transforms
        self.epsilon = config.epsilon
        self.config = config
        self.climate_model = self.load_climate_model_data()
        climate_model_reference = self.load_climate_model_reference_data()
        self.num_samples = len(self.climate_model.time.values)
        self.climate_model = self.apply_transforms(self.climate_model, climate_model_reference)


    def load_climate_model_reference_data(self):
        
        climate_model_pr = xr.open_dataset(self.config.model_pr_path, cache=self.cache, chunks=self.chunks)
        climate_model_pr =  climate_model_pr.precipitation
        
        climate_model_t = xr.open_dataset(self.config.model_t_path, cache=self.cache, chunks=self.chunks)
        climate_model_t =  climate_model_t.tas
        

        
        climate_model_pr = climate_model_pr.load()
        climate_model_t = climate_model_t.load()

        climate_model_pr = climate_model_pr.sel(time=slice(str(self.config.train_start), str(self.config.train_end)))
        climate_model_t = climate_model_t.sel(time=slice(str(self.config.train_start), str(self.config.train_end)))
        
        
        climate_model = xr.Dataset({'precipitation': climate_model_pr, 'tas': climate_model_t})

        return climate_model

    
    def load_climate_model_data(self):

        climate_model_pr = xr.open_dataset(self.config.model_pr_path, cache=self.cache, chunks=self.chunks)
        climate_model_pr =  climate_model_pr.precipitation
        
        climate_model_t = xr.open_dataset(self.config.model_t_path, cache=self.cache, chunks=self.chunks)
        climate_model_t =  climate_model_t.tas
        
        climate_model_pr = climate_model_pr.load()
        climate_model_t = climate_model_t.load()

        climate_model = xr.Dataset({'precipitation': climate_model_pr, 'tas': climate_model_t})
        
        return climate_model


    def apply_transforms(self, data, data_ref):
        
        data_prec = data.precipitation
        data_tas = data.tas
                
        data_prec_ref = data_ref.precipitation
        data_tas_ref = data_ref.tas
        
                
        if 'log' in self.transforms:
            data_prec = log_transform(data_prec, self.epsilon)
            data_prec_ref = log_transform(data_prec_ref, self.epsilon)

        if 'normalize' in self.transforms:
            data = norm_transform(data, data_ref)

        if 'normalize_minus1_to_plus1' in self.transforms:
            data_prec = norm_minus1_to_plus1_transform(data_prec, data_prec_ref)
            data_tas = norm_minus1_to_plus1_transform(data_tas, data_tas_ref)
        
        data_ref['precipitation'] = data_prec_ref
        
        data['precipitation'] = data_prec
        data['tas'] = data_tas


        return data


    def __getitem__(self, index):
        
        y_p = torch.from_numpy(self.climate_model.precipitation.isel(time=index).values).float().unsqueeze(0)
        y_t = torch.from_numpy(self.climate_model.tas.isel(time=index).values).float().unsqueeze(0)
        y = torch.cat((y_p, y_t), dim=0)

        #y = torch.from_numpy(self.climate_model.isel(time=index).values).float().unsqueeze(0)

        return {'B': y}


    def __len__(self):
        return self.num_samples
