import h5py
import numpy as np
from enum import IntEnum

class Timestamp(IntEnum):
    n48hrs = 0
    n36hrs = 1
    n24hrs = 2
    n12hrs = 3
    n0hrs  = 4
    p12hrs = 5
    p24hrs = 6

class Landscape(IntEnum):
    ASP = 0     # Aspect
    CBD = 1     # Canopy Bult Density
    CBH = 2     # Canopy Base Height
    CCV = 3     # Canopy Cover
    CHT = 4     # Canopy Height
    ELV = 5     # Elevelation
    SLP = 16    # Slope

class Vegetation(IntEnum):
    NOD = 6      # No Data
    SPR = 7      # Sparse
    TRE = 8      # Tree
    SRB = 9      # Shrub
    HRB = 10     # Hurb
    WTR = 11     # Water
    BRN = 12     # Barren
    DEV = 13     # Developed
    SNI = 14     # Snow-Ice
    AGC = 15     # Agriculture

class Meteorology(IntEnum):
    TMP  = 0     # Temperature @ 2m (Kelvin)
    HMD  = 1     # Relative Humidity @ 2m (%)
    UWND = 2     # U Wind Component @ 10 m (m/s)
    VWND = 3     # V Wind Component @ 10 m (m/s)
    PERT = 4     # Precipitation Rate (kg/s*m^2)

class DataTransformer:

    @staticmethod
    def rolling_window(img: np.ndarray, shape: np.ndarray) -> np.ndarray:
        s = (img.shape[0] - shape[0] + 1,) + (img.shape[1] - shape[1] + 1,) + shape
        strides = img.strides + img.strides
        return np.lib.stride_tricks.as_strided(img, shape=s, strides=strides)
    
    @staticmethod
    def get_neighbors(img: np.ndarray, shape=(3, 3), flatten=True) -> np.ndarray:
        ##   
        ##  Get neighbors at given location
        ##      [1,2,3,4,5]
        ##      [6,7,8,9,1]                                  
        ##   x= [2,3,4,5,6], 
        ##      [7,8,9,1,2]                                  
        ##      [3,4,5,6,7]
        ##
        ##                                   [7,8,9]
        ##   get_neighbors(x,(3,3))[2][2] => [3,4,5] 
        ##                                   [8,9,1]

        r_extra = np.floor(shape[0] / 2).astype(int)
        c_extra = np.floor(shape[1] / 2).astype(int)
        out = np.empty((img.shape[0] + 2 * r_extra, img.shape[1] + 2 * c_extra))
        out[:] = 0
        out[r_extra:-r_extra, c_extra:-c_extra] = img
        view = DataTransformer.rolling_window(out, shape)
        x,y,z = view.shape[0],view.shape[1],view.shape[2]*view.shape[3]
        return view.reshape((x,y,z)) if flatten else view
    
    @staticmethod
    def get_delta_mask(VIIRS_img: np.ndarray) -> np.ndarray:
        ## An empty 30x30 zero matrix
        zeros = np.zeros(VIIRS_img[0].shape)

        ## Delta in Active Fire between -48 and -36 hrs
        delta_48_36 = np.bitwise_and(VIIRS_img[0], VIIRS_img[1])

        ## Delta in Active Fire between -36 and -24 hrs
        delta_36_24 = np.bitwise_and(VIIRS_img[1], VIIRS_img[2])

        ## Delta in Active Fire between -24 and -12 hrs
        delta_24_12 = np.bitwise_and(VIIRS_img[2], VIIRS_img[3])

        ## Delta in Active Fire between -12 and 0 hrs
        delta_12_00 = np.bitwise_and(VIIRS_img[3], VIIRS_img[4])

        delta_t0 = np.zeros((30,30,4))
        delta_t1 = np.stack((delta_48_36, zeros, zeros, zeros),axis=2)
        delta_t2 = np.stack((delta_48_36, delta_36_24, zeros, zeros),axis=2)
        delta_t3 = np.stack((delta_48_36, delta_36_24, delta_24_12, zeros),axis=2)
        delta_t4 = np.stack((delta_48_36, delta_36_24, delta_24_12, delta_12_00),axis=2)

        return delta_t0, delta_t1, delta_t2, delta_t3, delta_t4
    
    @staticmethod
    def get_wind_direction(uwind: np.ndarray, vwind: np.ndarray) -> np.ndarray:
        return np.arctan2(uwind, vwind)

    @staticmethod
    def get_wind_speed(uwind: np.ndarray, vwind: np.ndarray) -> np.ndarray:
        return (uwind ** 2 + vwind ** 2) ** 0.5

    @staticmethod
    def active_fire_ensemble(VIIRS_img: np.ndarray):
        # Active Fire over timestamps from -46 hrs to 0 hrs
        active_fire = np.stack(map(DataTransformer.get_neighbors, VIIRS_img), axis=0)

        return active_fire

    @staticmethod
    def delta_fire_ensemble(VIIRS_img: np.ndarray):
        # Delta Fires
        delta_fire = np.stack(DataTransformer.get_delta_mask(VIIRS_img), axis=0)

        return delta_fire

    @staticmethod
    def land_scape_ensemble(LDSCP_img: np.ndarray):
        aspect = LDSCP_img[Landscape.ASP]
        cbd    = LDSCP_img[Landscape.CBD]
        cbh    = LDSCP_img[Landscape.CBH]
        ccv    = LDSCP_img[Landscape.CCV]
        cht    = LDSCP_img[Landscape.CHT]
        elv    = LDSCP_img[Landscape.ELV]
        slp    = LDSCP_img[Landscape.SLP]

        return np.stack((aspect, cbd, cbh, ccv, cht, elv, slp), axis=2)
    
    @staticmethod
    def vegetation_ensemble(LDSCP_img: np.ndarray):
        nod = LDSCP_img[Vegetation.NOD]
        spr = LDSCP_img[Vegetation.SPR]
        tre = LDSCP_img[Vegetation.TRE]
        srb = LDSCP_img[Vegetation.SRB]
        hrb = LDSCP_img[Vegetation.HRB]
        wtr = LDSCP_img[Vegetation.WTR]
        brn = LDSCP_img[Vegetation.BRN]
        dev = LDSCP_img[Vegetation.DEV]
        sni = LDSCP_img[Vegetation.SNI]
        agc = LDSCP_img[Vegetation.AGC]

        return np.stack((nod, spr, tre, srb, hrb, wtr, brn, dev, sni, agc), axis=2)

    @staticmethod
    def weather_ensemble(METEO_img: np.ndarray):
        # Temperature @ 2m (Kelvin)
        tmp_t0  = METEO_img[0][Meteorology.TMP]
        tmp_t1  = METEO_img[1][Meteorology.TMP]
        # Relative Humidity @ 2m (%)
        hmd_t0  = METEO_img[0][Meteorology.HMD]
        hmd_t1  = METEO_img[1][Meteorology.HMD]
        # U Wind Component @ 10 m (m/s)
        uwnd_t0 = METEO_img[0][Meteorology.UWND]
        uwnd_t1 = METEO_img[1][Meteorology.UWND]
        # V Wind Component @ 10 m (m/s)
        vwnd_t0 = METEO_img[0][Meteorology.VWND]
        vwnd_t1 = METEO_img[1][Meteorology.VWND]
        # Wind Speed
        wnd_spd_t0 = DataTransformer.get_wind_speed(uwnd_t0, vwnd_t0)
        wnd_spd_t1 = DataTransformer.get_wind_speed(uwnd_t1, vwnd_t1)
        # Wind Direction
        wnd_dic_t0 = DataTransformer.get_wind_direction(uwnd_t0, vwnd_t0)
        wnd_dic_t1 = DataTransformer.get_wind_direction(uwnd_t1, vwnd_t1)
        # Precipitation Rate (kg/s*m^2)
        pert_t0 = METEO_img[0][Meteorology.PERT]
        pert_t1 = METEO_img[1][Meteorology.PERT]

        weather_t0 = np.stack((tmp_t0,hmd_t0,uwnd_t0,vwnd_t0,wnd_spd_t0,wnd_dic_t0,pert_t0),axis=2)
        weather_t1 = np.stack((tmp_t1,hmd_t1,uwnd_t1,vwnd_t1,wnd_spd_t1,wnd_dic_t1,pert_t1),axis=2)

        return np.stack((weather_t0, weather_t1),axis=0)
        
    def __init__(self, path):
        self.VIIRS_data = None       # Shape = (5,n,30,30,9)
        self.DELTA_data = None       # Shape = (5,n,30,30,4)
        self.LDSCP_data = None       # Shape = (n,30,30,7)
        self.VEGTN_data = None       # Shape = (n,30,30,10)
        self.METEO_data = None       # Shape = (2,n,30,30,7)

        self.DATASET_PATH = path
        with h5py.File(path, 'r') as f:
            self.train_data = {}
            for k in list(f):
                self.train_data[k] = f[k][:]

        # Shape = (5,n,30,30)
        self.OBSVD_data = np.stack(tuple(self.train_data['observed']),axis=1)     
        # Shape = (2,n,30,30)  
        self.TARGT_data = np.stack(tuple(self.train_data['target']),axis=1) 

    def filter_fire_idx(self, active_fire_cnts: int):
        return np.where(
            (np.sum(self.train_data['observed'][:,0],axis=(1,2)) > active_fire_cnts) & 
            (np.sum(self.train_data['observed'][:,1],axis=(1,2)) > active_fire_cnts) & 
            (np.sum(self.train_data['observed'][:,2],axis=(1,2)) > active_fire_cnts) & 
            (np.sum(self.train_data['observed'][:,3],axis=(1,2)) > active_fire_cnts) & 
            (np.sum(self.train_data['observed'][:,4],axis=(1,2)) > active_fire_cnts) & 
            (np.sum(self.train_data['target'][:,0],axis=(1,2)) > active_fire_cnts) 
        )[0]

    def filter_data_counts(self, active_fire_cnts: int, epoch=0):
        return np.where(
            np.sum(self.train_data['observed'][:,epoch], axis=(1,2)) == active_fire_cnts
        )[0].shape[0]

    def data_ensemble(self, indices=None):
        if indices is None:
            VIIRS_imgs = self.train_data['observed']
            LDSCP_imgs = self.train_data['land_cover']
            METEO_imgs = self.train_data['meteorology']
        else:
            VIIRS_imgs = np.take(self.train_data['observed'], indices)
            LDSCP_imgs = np.take(self.train_data['land_cover'], indices)
            METEO_imgs = np.take(self.train_data['meteorology'], indices)
        
        print(f"VIIRS_imgs: {VIIRS_imgs.shape}")
        print(f"LDSCP_imgs: {LDSCP_imgs.shape}")
        print(f"METEO_imgs: {METEO_imgs.shape}")

        # n = self.train_data['observed'].shape[0]
        # t_f = self.train_data['observed'].shape[1]  # timestamps of active fires
        # x = self.train_data['observed'].shape[2]
        # y = self.train_data['observed'].shape[3]
        # g = 3 * 3   # grid size for rolling window
        # l = len(Landscape)
        # v = len(Vegetation)
        # t_w = self.train_data['meteorology'][1]     # timestamps of weather
        # w = self.train_data['meteorology'][2] + 2   # 5 var + wind speed + wind direction

        self.VIIRS_data = np.stack(map(DataTransformer.active_fire_ensemble, VIIRS_imgs), axis=1)
        self.DELTA_data = np.stack(map(DataTransformer.delta_fire_ensemble, VIIRS_imgs), axis=1)
        self.LDSCP_data = np.stack(map(DataTransformer.land_scape_ensemble, LDSCP_imgs), axis=0)
        self.VEGTN_data = np.stack(map(DataTransformer.vegetation_ensemble, LDSCP_imgs), axis=0)
        self.METEO_data = np.stack(map(DataTransformer.weather_ensemble, METEO_imgs), axis=1)

        return 

    def get_obs_input_data(self, timestamp: int):
        return np.concatenate(
            (
                self.VIIRS_data[timestamp], self.DELTA_data[timestamp], self.LDSCP_data, 
                self.VEGTN_data, self.METEO_data[0]
            ),
            axis = 3
        )

    def get_target_data(self, timestamp: int):
        if timestamp < Timestamp.n0hrs:
            return self.OBSVD_data[timestamp+1]
        elif timestamp == Timestamp.n0hrs:
            return self.TARGT_data[Timestamp.p12hrs-Timestamp.p12hrs]
        elif timestamp == Timestamp.p12hrs:
            return self.TARGT_data[Timestamp.p24hrs-Timestamp.p12hrs]
        else:
            return None
        

if __name__ == "__main__":
    path = './uci_ml_hackathon_fire_dataset_2012-05-09_2013-01-01_10k_train.hdf5'
    dataset = DataTransformer(path)
    dataset.data_ensemble()
    t1 = dataset.get_obs_input_data(Timestamp.n0hrs)
    t2 = dataset.get_target_data(Timestamp.n0hrs)
    print(t1.shape)
    print(t2.shape)
