import xarray as xr
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import brier_score_loss, roc_curve, roc_auc_score, accuracy_score, f1_score, precision_score, recall_score, average_precision_score, precision_recall_curve
import matplotlib.pyplot as plt
import pandas as pd
from typing import Union, Tuple, List, Optional, Any
from joblib import dump, load
from deap import base, creator, tools, algorithms
import random
from sklearn.model_selection import cross_val_score
import os
import json
import src.Utils as utils
class BaseLineModel:
    """
    BaseLineModel is designed to serve as a foundational framework for flood prediction
    modeling. It incorporates both dynamic and static features to train a model
    and predict flood occurrence over specific geographical areas and time periods.
    It supports custom configurations for model training and evaluation.
    """
    
    def __init__(self, 
            labels_root_path: str = None,  # path to the labels xarray dataset
            dynamic_features_path: str = None,  # path to the predictions score of the first model
            static_features_root_path: str = None,  # path to the static features xarray dataset
            labels_FR_path: str = None,  # path to the labels xarray dataset
            dynamic_features_FR_path: str = None,  # path to the predictions score of the first model
            inf_dynamic_features_FR_path: str = None,  # path to the predictions score of the first model
            static_features_FR_path: str = None,  # path to the static features xarray dataset
            labels_ERA5_path: str = None,  # path to the static features xarray dataset
            nan_value: int = 0,  # value to replace nan values in labels
            replace_nan: bool = False,  # replace nan values in labels
            reduce_train: bool = False,  # reduce the train to take only band with flood 
            train_start: str = "2002-08-03",  # date where to split train test
            train_end: str = "2003-01-01",  # date where to split train test
            val_start: str = "2003-01-10",  # date where to split train test
            val_end: str = "2003-03-17",  # date where to split train test
            test_start: str = "2003-01-01",  # date where to split train test
            test_end: str = "2003-03-17",  # date where to split train test
            nb_train_minicube: int = 200,  # number of minicube to train the model
            nb_test_minicube: int = 100,  # number of minicube to train the model
            nb_val_minicube: int = 10,  # number of minicube to train the model
            name: str = "Baseline_Model_2_full_test",  # name of the model
            min_score_model1: float = .1,  # min score of the first model
            compute_full_test_set: bool = False,
            seed: int = 42,
            Cross_Val = True,
            score_mode = "roc"):  # seed for random number generation
        
        """Initialize the BaselineModel with specified parameters.

        Parameters
        ----------
        labels_root_path : str, optional
            Path to the labels xarray dataset.
        dynamic_features_path : str, optional
            Path to the predictions score of the first model.
        static_features_root_path : str, optional
            Path to the static features xarray dataset.
        labels_FR_path : str, optional
            Path to the labels xarray dataset at full resolution.
        dynamic_features_FR_path : str, optional
            Path to the dynamic features xarray dataset at full resolution.
        inf_dynamic_features_FR_path : str, optional
            Path to the dynamic features xarray dataset at full resolution for inferance only 
            i.e. dynamic features not associated to a label (to compute score for evaluation of the model)
        static_features_FR_path : str, optional
            Path to the static features xarray dataset at full resolution.
        nan_value : float, optional
            Value to replace NaN values in labels.
        replace_nan : bool, optional
            Whether to replace NaN values in labels.
        reduce_train : bool, optional
            Whether to reduce the train to only bands with flood.
        train_start, train_end, val_start, val_end, test_start, test_end : str, optional
            Dates for splitting the data into training, validation, and testing sets.
        nb_train_minicube, nb_test_minicube, nb_val_minicube : int, optional
            Number of mini cubes for training, testing, and validation.
        name : str, optional
            Name of the model.
        min_score_model1 : float, optional
            Minimum score of the first model to consider for dynamic features.
        compute_full_test_set : bool, optional
            Whether to compute the full test set.
        seed : int, optional
            Seed for random number generation.

        Raises
        ------
        ValueError
            If paths to either labels or dynamic/static features are not provided.
        """

        if seed == False:
            seed = np.random.randint(0,1000)
            print("Randomly generated seed :",seed)

        self.score_mode = score_mode
        self.Cross_Val = Cross_Val
        self.seed = seed
        np.random.seed(self.seed)
        self.labels_FR_path = labels_FR_path
        self.dynamic_features_FR_path = dynamic_features_FR_path
        self.inf_dynamic_features_FR_path = inf_dynamic_features_FR_path
        self.static_features_FR_path = static_features_FR_path
        self.labels_ERA5_path = labels_ERA5_path

        self.compute_full_test_set = compute_full_test_set
        if (labels_root_path is None) | (
            (dynamic_features_path is None) & (static_features_root_path is None)
        ):
            raise ValueError("You must provide a path to the labels and either the dynamic or static features")

        self.nb_train_minicube = nb_train_minicube
        self.nb_test_minicube = nb_test_minicube
        self.nb_val_minicube = nb_val_minicube
        # split train test mode
        self.reduce_train=reduce_train     

        self.labels_root_path = labels_root_path
        self.dynamic_features_path = dynamic_features_path
        self.static_features_root_path = static_features_root_path
        self.replace_nan = replace_nan
        self.nan_value = nan_value
        self.min_score_model1 = min_score_model1
        self.model = None

        # train test split date
        self.dataset_limits = {
            'train': {'start':  pd.to_datetime(train_start), 'end':  pd.to_datetime(train_end)},
            'val': {'start':  pd.to_datetime(val_start), 'end':  pd.to_datetime(val_end)},
            'test': {'start':  pd.to_datetime(test_start), 'end':  pd.to_datetime(test_end)}
        }

        labels = self.open_xarray(self.labels_root_path+"(0.875, 49.375, 1.125, 49.625).nc")
        self.time_dim = labels.dims['time']
        self.obj_name = name

        # load dynamic and static features and information about the number of features
        self.nb_feature_dynamic = 0
        self.all_dynamic_vars = []
        if self.dynamic_features_path is not None:
            self.df_all = self.open_xarray(self.dynamic_features_path)
            self.all_dynamic_vars = list(self.df_all.data_vars)
            self.nb_feature_dynamic = len(self.all_dynamic_vars)
        
        self.nb_feature_static = 0
        self.all_static_vars = []
        if self.static_features_root_path is not None:
            static_features = self.open_xarray(self.static_features_root_path+"(0.875, 49.375, 1.125, 49.625).nc")
            self.all_static_vars = list(static_features.band.values)
            self.nb_feature_static = len(self.all_static_vars)
        self.nb_feature = self.nb_feature_dynamic + self.nb_feature_static



    def open_xarray(self,
                    path:str) -> xr.Dataset:                
        """Open an xarray dataset.
        and convert data varaible to float16 to reduce memory usage.
        Parameters
        ----------
        path : str
            The path to the xarray dataset.

        Returns
        -------
        xr.Dataset
            The opened xarray dataset.
        """
        xarray = xr.open_dataset(path)
        for var in xarray.data_vars:
            xarray[var] = xarray[var].astype('float16')
        return xarray

    def process_mini_cube(self, 
                          labels_path: str, 
                          static_features_path: str):
        """Load a mini datacube static, dynamics and labels. Also replace nan values in labels if needed.
        Parameters
        ----------
        labels_path : str
            Path to the labels xarray. 
        static_features_path : str
            Path to the static features xarray.
        """

        # labels
        self.labels = xr.open_dataset(labels_path)

        # replace nan values with "nan_value" in labels useful when nan indicates no flood
        if self.replace_nan:
            nan_mask = np.isnan(self.labels['__xarray_dataarray_variable__'].values)
            self.labels['__xarray_dataarray_variable__'].data[nan_mask] = self.nan_value   

        # static features
        self.static_features = self.open_xarray(static_features_path)

        # dynamic features
        if self.dynamic_features_path is not None:
            bbox = min(self.labels.x.values), min(self.labels.y.values), max(self.labels.x.values), max(self.labels.y.values)
            self.dynamic_features = self.df_all.sel(y=slice(bbox[3],bbox[1]), x=slice(bbox[0],bbox[2]))

        # set dimensions of the minicube dataset (x_dim_FR and y_dim_FR are the dimensions of the full resolution dataset)
        
        self.x_dim = self.labels.dims['x']
        self.y_dim = self.labels.dims['y']


    def parse_dates(self, 
                    band_str: Union[str, np.datetime64]) -> Tuple[pd.Timestamp, Optional[pd.Timestamp]]:
        """Conversion of string information into datetime objects. It's designed to
        work with either datetime64 objects or strings in the format "YYYYMMDD-YYYYMMDD (for ERA5 dataset)".
        Most likely it is only used for the ERA5 dataset.

        Parameters
        ----------
        band_str : Union[str, np.datetime64]
            The band information, either as a datetime64 object or a string.

        Returns
        -------
        Tuple[pd.Timestamp, Optional[pd.Timestamp]]
            A tuple containing the start and end dates as pandas Timestamps.
        """
        if isinstance(band_str, np.datetime64):
            start_date = band_str
            end_date = None  
        else:
            start_str, end_str = band_str.split('-')
            start_date = pd.to_datetime(start_str)
            end_date = pd.to_datetime(end_str)
        return start_date, end_date # return the start and the end date if any
    

    def index_splitter(self, 
                       input_list: List[int]) -> Tuple[List[int], List[int]]:
        """Splits indices into static and dynamic feature indices.

        Parameters
        ----------
        input_list : List[int]

        Returns
        -------
        Tuple[List[int], List[int]]
            A tuple containing two lists: the first list for static indices and the second list for dynamic indices.
        """
        static = [x for x in input_list if 0 <= x < self.nb_feature_static]
        dynamic = [x for x in input_list if self.nb_feature_static <= x < self.nb_feature_static + self.nb_feature_dynamic]

        if sorted(static + dynamic) != sorted(input_list):
            raise ValueError("Error non existing features indices.")
        dynamic = [x - self.nb_feature_static for x in dynamic]

        return static, dynamic # list of static and dynamic indices used in the model (feature selection)

    
    def train_flood_prediction_model(self,
                                    n_estimators: int,  # number of estimators for the random forest
                                    max_depth: int,  # max depth of the random forest
                                ) -> RandomForestClassifier:
        """Trains a Random Forest Classifier model.
        Parameters
        ----------
        n_estimators : int
            The number of trees in the random forest.
        max_depth : int
            The maximum depth of each tree.
        is_GA : bool, optional
            Whether the model is being trained within a genetic algorithm loop.
        Returns
        -------
        RandomForestClassifier
            The trained Random Forest Classifier model.
        """
        # train the random forest
        model = RandomForestClassifier(
            random_state=self.seed,
            n_estimators=n_estimators,
            max_depth=max_depth,
            n_jobs=20,
            criterion="entropy",
            class_weight='balanced' 
        )

        self.model = model.fit(self.X_train[:,self.selected_features], self.y_train)

        return model


    # NEED TO BE BREAK INTO SMALLER FUNCTION
    def prepare_data(self,
                     compute_full_test_set:bool = False):
        """Prepares data by loading labels and model scores, selecting coordinates
        of the mini data cubes used for training, testing, and validation.
        and splitting data into training, validation, and test datasets.                 

        Args:
            compute_full_test_set (bool, optional): Are we computing the full 
            test set if so we don't need to remove the non france data. 
            Defaults to False.
        """
        
        final_label = xr.open_dataset(self.labels_FR_path)

        model1_score = self.df_all
        total = 0
        coord = {"train":{"TP":[], "FP":[]}, "test":{"TP":[], "FP":[]}, "val":{"TP":[], "FP":[]}}
        final_label_era5 = xr.open_dataset(self.labels_ERA5_path)

        if compute_full_test_set:
            datasets = ["test"]
        else:
            datasets = ["train", "test", "val"]

        for latitude in model1_score.y.values:
            for longitude in model1_score.x.values:
                bbox=(longitude-.125, latitude-.125, longitude+.125, latitude+.125)
                if((bbox[2]>min(final_label.x.values)) &
                   (bbox[0]<max(final_label.x.values)) & 
                   (bbox[3]>min(final_label.y.values)) & 
                   (bbox[1]<max(final_label.y.values))):
                    total +=1
                    for dataset in datasets:
                        label_array = final_label_era5.sel(x=longitude, 
                                                        y=latitude, 
                                                        time = slice(self.dataset_limits[dataset]['start'], 
                                                                    self.dataset_limits[dataset]['end']))['__xarray_dataarray_variable__'].values   

                        if (dataset == "test") & (compute_full_test_set == True):
                            if (label_array.sum()>0):
                                coord[dataset]["TP"].append((longitude,latitude))
                            else:
                                coord[dataset]["FP"].append((longitude,latitude))
                        else:
                            score_array = model1_score.sel(x=longitude, 
                                y=latitude, 
                                time = slice(self.dataset_limits[dataset]['start'], 
                                            self.dataset_limits[dataset]['end']))['M1_score'].values
                            
                            is_flood = label_array>0
                            is_detected = score_array>self.min_score_model1

                            if (is_detected.sum() > 0): 
                                is_detected_flood = np.logical_and(is_flood, is_detected) 
                                if (is_detected_flood.sum() > 0):
                                    coord[dataset]["TP"].append((longitude,latitude))
                                else:
                                    coord[dataset]["FP"].append((longitude,latitude))

        for dataset in datasets:
            print(f"# TP {dataset} :",len(coord[dataset]["TP"]))
            print(f"# FP {dataset} :",len(coord[dataset]["FP"]))

        nb_minicube = {}
        coord_selected = {"train":[], "test":[], "val":[]}
        for dataset in datasets:
            if (dataset == "test") & (compute_full_test_set == True):
                coord_selected[dataset] = coord[dataset]["FP"] + coord[dataset]["TP"]
            else:
                nb_minicube[dataset] = int(min(len(coord[dataset]["TP"]), len(coord[dataset]["FP"]), (self.nb_train_minicube/2)))
                random_indices_TP = np.random.choice(len(coord[dataset]["TP"]), nb_minicube[dataset], replace=False)
                bbox_TP = [coord[dataset]["TP"][i] for i in random_indices_TP]
                random_indices_FP = np.random.choice(len(coord[dataset]["FP"]), nb_minicube[dataset], replace=False)
                bbox_FP = [coord[dataset]["FP"][i] for i in random_indices_FP]
                coord_selected[dataset] = bbox_TP + bbox_FP

        coord_all = coord_selected["train"] + coord_selected["test"] + coord_selected["val"]

        coord_union = []
        for coords in coord_all:
            if coords not in coord_union:
                coord_union.append(coords)

        X_combined = {"train":None, "test":None, "val":None}
        y_combined = {"train":None, "test":None, "val":None}

        for longitude,latitude in coord_union:
            bbox_text=str((longitude-.125, latitude-.125, longitude+.125, latitude+.125))

            self.process_mini_cube(self.labels_root_path+bbox_text+".nc", 
                              self.static_features_root_path+bbox_text+".nc")
            
            features_all_bands = []
            targets_all_bands = []
            static_indices = list(range(self.nb_feature_static))
            dynamic_indices = list(range(self.nb_feature_dynamic))
            num_statics = len(static_indices)

            if (len(self.dynamic_features.x.values) >0) & (len(self.dynamic_features.y.values) >0):
                for band in self.labels.time.values:
                    start_date = pd.to_datetime(band)
                    filtered_dynamic_features = self.dynamic_features.sel(time=start_date)
                    # compute the total number of features
                    total_features = num_statics
                    if len(dynamic_indices) > 0:
                        for _ in dynamic_indices:
                            total_features += 1

                    # create an empty array for features
                    feature_array = np.empty((self.x_dim * self.y_dim, total_features))
                    feature_index = 0

                    # add static features if any
                    for static_index in static_indices:
                        feature_array[:, feature_index] = self.static_features['__xarray_dataarray_variable__'][static_index].data.reshape(self.x_dim * self.y_dim)
                        feature_index += 1

                    # add average and derivative of dynamic features if any
                    for var_index in dynamic_indices:
                        var = self.all_dynamic_vars[var_index]
                        feature_index, feature_array = self.process_dynamic_feature(filtered_dynamic_features[var].values, feature_array, feature_index)
                    feature_temp = feature_array if total_features > 0 else None

                    features_all_bands.append(feature_temp)

                    target = self.labels['__xarray_dataarray_variable__'].sel(time=band).data.reshape(-1)
                    targets_all_bands.append(target)

                
            # split train test
                for dataset in datasets:
                    #print(dataset)
                    if (longitude,latitude) in coord_selected[dataset]:
                        X, y = self.split_subset_date(
                            features_all_bands,
                            targets_all_bands,
                            self.df_all.sel(x=longitude, y=latitude, time=slice(self.dataset_limits[dataset]['start'], 
                                          self.dataset_limits[dataset]['end']))['M1_score'],
                            start_date = self.dataset_limits[dataset]['start'],
                            end_date = self.dataset_limits[dataset]['end'],
                            dataset = dataset,
                            compute_full_test_set = compute_full_test_set)
                        X_combined[dataset], y_combined[dataset] = utils.stack_if_exists(X_combined[dataset], y_combined[dataset], X, y)


        if compute_full_test_set:
            self.Full_X_test = X_combined["test"]
            self.Full_y_test = y_combined["test"]
        else:
            self.X_test = X_combined["test"]
            self.y_test = y_combined["test"]
            self.X_val = X_combined["val"]
            self.y_val = y_combined["val"]
            self.X_train = X_combined["train"]
            self.y_train = y_combined["train"]


    def load_FullRez(self):
        """Loads full resolution dynamic, static features, and labels into the class instance.
        Usefull for displaying the predictions map.
        """
        self.dynamic_features_FR = self.open_xarray(self.dynamic_features_FR_path)
        self.static_features_FR = self.open_xarray(self.static_features_FR_path)
        self.labels_FR = xr.open_dataset(self.labels_FR_path)
        self.x_dim_FR = self.labels_FR.dims['x']
        self.y_dim_FR = self.labels_FR.dims['y']

    def load_InfRez(self):
        """Loads full resolution dynamic, static features, and labels into the class instance.
        Usefull for displaying the predictions map.
        """
        self.dynamic_features_FR = self.open_xarray(self.inf_dynamic_features_FR_path)
        self.static_features_FR = self.open_xarray(self.static_features_FR_path)
        self.x_dim_FR = self.dynamic_features_FR.dims['x']
        self.y_dim_FR = self.dynamic_features_FR.dims['y']

    def prepare_data_one_band(self, 
                              band: str,
                              compute_labels: bool = True):                     
        """Prepares data for a single time point week by combining static and dynamic features.
        
        Parameters
        ----------
        band : str
            The week for which the predictions are being prepared.

        Notes
        -----
        - Assumes dynamic and static features have already been loaded with 'load_FullRez' method.
        """                    
        
        static_indices = list(range(self.nb_feature_static))
        dynamic_indices = list(range(self.nb_feature_dynamic))
        num_statics = len(static_indices)

        start_date = pd.to_datetime(band)
        filtered_dynamic_features = self.dynamic_features_FR.sel(time=start_date)
        # compute the total number of features
        total_features = num_statics
        if len(dynamic_indices) > 0:
            for _ in dynamic_indices:
                total_features += 1

        # create an empty array for features
        feature_array = np.empty((self.x_dim_FR * self.y_dim_FR, total_features))
        feature_index = 0

        # add static features if any
        for static_index in static_indices:
            feature_array[:, feature_index] = self.static_features_FR['__xarray_dataarray_variable__'][static_index].data.reshape(self.x_dim_FR * self.y_dim_FR)
            feature_index += 1


        # add average and derivative of dynamic features if any
        for var_index in dynamic_indices:
            var = self.all_dynamic_vars[var_index]
            feature_index, feature_array = self.process_dynamic_feature_band(filtered_dynamic_features[var].values, feature_array, feature_index)
        if compute_labels:
            target = self.labels_FR['__xarray_dataarray_variable__'].sel(time=band).data.reshape(-1)
        else:
            target = np.zeros((self.x_dim_FR * self.y_dim_FR))

        self.X_band = feature_array
        self.y_band = target
    


    def process_dynamic_feature_band(self, 
                                     feature: np.ndarray, # dynamic feature
                                     feature_array: np.ndarray, # output array of features
                                     feature_index: int # index of the current feature
                                     ) -> Tuple[int, np.ndarray]:
        """Reshape a single dynamic feature for a particular week. 
        And add it to the feature array.

        Parameters
        ----------
        feature : np.ndarray
            The dynamic feature to be added.
        feature_array : np.ndarray
            The array of features to which the dynamic feature will be added.
        feature_index : int
            The index at which the dynamic feature should be added.

        Returns
        -------
        Tuple[int, np.ndarray]
            The updated feature index and the feature array.
        """

        feature_array[:, feature_index] = feature.reshape(self.x_dim_FR * self.y_dim_FR)
        feature_index += 1

        return feature_index, feature_array

    def process_dynamic_feature(self, 
                                value: np.ndarray, # dynamic feature
                                feature_array: np.ndarray, # output array of features
                                feature_index: int # index of the current feature
                                ) -> Tuple[int, np.ndarray]:
        """Processes a single dynamic feature. 
        Expands the dynamics feature from ERA5 resolution to the full resolution.

        Parameters
        ----------
        value : np.ndarray
            The value of the dynamic feature to be added.
        feature_array : np.ndarray
            The array of features to which the dynamic feature value will be added.
        feature_index : int
            The index at which the dynamic feature value should be added.

        Returns
        -------
        Tuple[int, np.ndarray]
            The updated feature index and the feature array.
        """

        feature_array[:, feature_index] = value *np.ones((1,self.x_dim * self.y_dim))
        feature_index += 1

        return feature_index, feature_array
    

    def save_dataset_to_disk(self,
                    dateset_list: list = ["train", "val", "test"],
                    name: str = None):
        """Saves test/val/train data and trained model to disk.

        Parameters
        ----------
        name : str, optional
            The base name for the saved files. If not provided, uses the object's name attribute.
        dataset_list : list, optional
            The list of datasets to be saved.
        """
        if not name:
            name = self.obj_name
        for dataset in dateset_list:
            np.save(f"localdata/models/{name}X_{dataset}", getattr(self, f"X_{dataset}"))
            np.save(f"localdata/models/{name}y_{dataset}", getattr(self, f"y_{dataset}"))

    def load_dataset_from_disk(self,
                    dataset_list: list = ["train", "val", "test"],
                    name: str = None):
        """Loads test/val/train data and trained model from disk.

        Parameters
        ----------
        name : str, optional
            The base name for the saved files. If not provided, uses the object's name attribute.
        dataset_list : list, optional
            The list of datasets to be loaded.
        """
        if not name:
            name = self.obj_name
        for dataset in dataset_list:
            setattr(self, f"X_{dataset}", np.load(f"localdata/models/{name}X_{dataset}.npy", allow_pickle=True))
            setattr(self, f"y_{dataset}", np.load(f"localdata/models/{name}y_{dataset}.npy", allow_pickle=True))

    def save_to_disk(self,
                    name: str = None):
        """Saves test/val/train data and trained model to disk.

        Parameters
        ----------
        name : str, optional
            The base name for the saved files. If not provided, uses the object's name attribute.

        """
        if not name:
            name = self.obj_name
        save_path = "localdata/models/"
        isExist = os.path.exists(save_path)
        if not isExist:
            os.makedirs(save_path)        
        with open(f'{save_path}{name}_features.json', 'w') as f:
            json.dump(self.selected_features, f)

        self.save_dataset_to_disk(name=name,dateset_list=["train", "val", "test"])

        dump(self.model, f"{save_path}{name}.joblib")
        print(self.selected_features)

    def load_from_disk(self, 
                       name: str) -> 'BaseLineModel':
        """Loads model data and trained model from disk.

        Parameters
        ----------
        name : str
            The base name for the files to be loaded.

        Returns
        -------
        BaseLineModel

        """
        self.load_dataset_from_disk(name=name,dataset_list=["train", "val", "test"])
        self.model = load(f"localdata/models/{name}.joblib")
        
        with open(f'localdata/models/{name}_features.json', 'r') as f:
            self.selected_features = json.load(f)

        self.obj_name = name
        self.nb_feature = self.X_train.shape[1]

        print(self.selected_features)

        return self

    def save_full_test_to_disk(self,name=""):
        """Saves the test dataset to disk.
        """
        np.savez_compressed(f"localdata/models/Full_data{name}.npz", Full_X_test=self.Full_X_test, Full_y_test=self.Full_y_test)


    def load_full_test_from_disk(self,name=""):
        """Loads the Full Test dataset from disk.
        """
        data = np.load(f"localdata/models/Full_data{name}.npz")
        self.Full_X_test = data['Full_X_test']
        self.Full_y_test = data['Full_y_test']

    def get_flood_bands(self, 
                        all_bands: list,
                        labels: xr.DataArray) -> list:
        """Filters bands based on the flood condition.

        Parameters
        ----------
        all_bands : list 
            A list of valid weeks
        labels : xarray.DataArray
            The data structure containing flood labels.

        Returns
        -------
        list
            A list of bands where there is at least a flood event.

        """
        flood_bands = []
        for band_date in all_bands:
            band_data = labels['__xarray_dataarray_variable__'].sel(time=band_date).values
            if np.any(band_data != 0):
                flood_bands.append(band_date)
        return flood_bands

    def split_subset_date(self, 
                          features_filtered: np.ndarray, 
                          target_source: np.ndarray, 
                          score_array: np.ndarray, 
                          start_date: str = "2003-09-01", 
                          end_date: str = "2003-09-01", 
                          dataset: str = "train",
                          compute_full_test_set:bool = False) -> Tuple[np.ndarray, np.ndarray]:
        """Create train/test/val dataset based on date and score 
        (filter out weeks under minimum score for model 1).

        Parameters
        ----------
        features_filtered : np.ndarray
            Features to be split.
        target_source : np.ndarray
            The corresponding labels.
        score_array : np.ndarray
            Scores from Model 1.
        start_date : str, optional
            The start date for the subset.
        end_date : str, optional
            The end date for the subset.
        dataset : str, optional
            The dataset name ( "train", "test", or "val").

        Returns
        -------
        Tuple[np.ndarray, np.ndarray]
            The reshaped and filtered features and targets arrays.

        """
        start, end = pd.to_datetime(start_date), pd.to_datetime(end_date)
        if (dataset == "test") & (compute_full_test_set == True):
            all_bands = self.labels.time.values
        else:
            condition_met = score_array.where(score_array > self.min_score_model1, drop=True)
            all_bands = condition_met.time.values

        if self.reduce_train:
            all_bands = self.get_flood_bands(all_bands, self.labels)

        bands = [band for band in all_bands if start < band <= end]
        band_to_index = {band: idx for idx, band in enumerate(all_bands)}

        num_samples = len(bands)
        feature_shape = features_filtered[0].shape[:2]

        X_tmp = np.empty((num_samples, *feature_shape))
        y_tmp = np.empty((num_samples, features_filtered[0].shape[0]))

        for band in bands:
            index = band_to_index[band]
            feature, target = features_filtered[index], target_source[index]
            idx = bands.index(band)
            X_tmp[idx] = feature.reshape(*feature_shape)
            y_tmp[idx] = target

        y = y_tmp.reshape(-1)
        X = np.reshape(X_tmp, (X_tmp.shape[0] * X_tmp.shape[1], X_tmp.shape[2]))
        
        X, y = self.remove_non_france_data(X, y)
        
        return X, y

    def split_subset_date_band(self, 
                               features: np.ndarray, 
                               target_source: np.ndarray, 
                               time_band) -> Tuple[np.ndarray, np.ndarray]:
        """Create train/test/val dataset for a specific time band / week. 

        Parameters
        ----------
        features : np.ndarray
            The array of all features to be split.
        target_source : np.ndarray
            The corresponding targets for the features.
        time_band
            The specific time band for which the data is being prepared.

        Returns
        -------
        Tuple[np.ndarray, np.ndarray]
            The reshaped feature and target arrays.

        """
        band_to_index = {band: idx for idx, band in enumerate(self.labels.time.values)}

        feature_shape = features[0].shape[:2]

        X_tmp = np.empty((1, *feature_shape))
        y_tmp = np.empty((1, features[0].shape[0]))

        index = band_to_index[time_band]
        feature, target = features[index], target_source[index]
        X_tmp[0] = feature.reshape(*feature_shape)
        y_tmp[0] = target

        y = y_tmp.reshape(-1)
        X = np.reshape(X_tmp, (X_tmp.shape[0] * X_tmp.shape[1], X_tmp.shape[2]))
                
        return X, y

    def remove_non_france_data(self, 
                               X: np.ndarray, 
                               y: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Filters out data not relevant for France.
        (data inside France but outside France's borders are labeled as -1)

        Parameters
        ----------
        X : np.ndarray
            The feature array.
        y : np.ndarray
            The target array.

        Returns
        -------
        Tuple[np.ndarray, np.ndarray]
            The filtered feature and target arrays.
        """
        valid_indices = (y != -1)
        return X[valid_indices], y[valid_indices]

    def load_indiv(self, 
                   individual: list, 
                   is_GA: bool = False) -> Any:
        """Transforms a genetic algorithm individual into model parameters and checks/loads the model.
        This method is typically used within a genetic algorithm loop to transform an individual's representation
        into model parameters and then load or check the model accordingly.

        Parameters
        ----------
        individual : list
            The list representation of an individual from a genetic algorithm, including feature selection and model parameters.
        save_model : bool, optional
        """
        print("load_indiv",individual)
        return self.check_load(indices = [i for i, use in enumerate(individual[:self.nb_feature]) if use], 
                                  n_estimators = individual[self.nb_feature], 
                                  max_depth = individual[self.nb_feature+1], 
                                  is_GA = is_GA) 

    def evalModel(self, 
                  individual: list, 
                  save_model: bool = False) -> Tuple[float]:
        """Evaluates a genetic algorithm individual by loading the individual and returning its auc.

        Parameters
        ----------
        individual : list
            The list representation of an individual from a genetic algorithm.
        save_model : bool, optional

        Returns
        -------
        Tuple[float]
            The AUC of the individual (computed on validation data or with a Xval), 
            used as fitness in genetic algorithms.
        """
        return self.load_indiv(individual, save_model), #comma is important to return a tuple

    def select_features(self, individual: list):
        """Selects features based on the selected indices.
        """

        self.selected_features = [i for i, use in enumerate(individual[:self.nb_feature]) if use]

        
    def check_load(self, 
                   indices: list, 
                   n_estimators: int, 
                   max_depth: int, 
                   is_GA: bool = False, 
                   debug: bool = False) -> Tuple[Any, float]:
        """Checks and loads a model based on provided indices, n_estimators, and max_depth.

        If the individual is valid, the model is trained; otherwise, it returns 0. Used within a
        genetic algorithm to evaluate individuals.

        Parameters
        ----------
        indices : list
            The indices of selected features.
        n_estimators : int
            The number of estimators for the RandomForest.
        max_depth : int
            The maximum depth for the RandomForest.
        is_GA : bool, optional
            Whether the model is being trained within a genetic algorithm loop.
        debug : bool, optional
            If True, prints out the parameters and the auc score.

        Returns
        -------
        Tuple[Any, float]
            The trained model and its AUC score.

        """
        if debug:
            print("indices",indices)
            print("n_estimators",n_estimators)
            print("max_depth",max_depth)
        
        self.selected_features = indices
        
  
        if is_GA:
            if len(indices) == 0:
                auc = 0
            else:
                if self.Cross_Val:
                    model = RandomForestClassifier(
                        random_state=self.seed,
                        n_estimators=n_estimators,
                        max_depth=max_depth,
                        n_jobs=20,
                        criterion="entropy",
                        class_weight='balanced' 
                    )   
                    score = self.cross_val(model)
                else:
                    self.train_flood_prediction_model(n_estimators, max_depth)
                    score = self.simple_val()
        else:
            self.train_flood_prediction_model(n_estimators, max_depth)
            score = self.simple_val()
        print("score",score)
        return score
    
    def cross_val(self,
                  model: RandomForestClassifier, 
                  mode:str = "roc")-> float:
        """_summary_

        Args:
            model (RandomForestClassifier): RF model
            mode (str, optional): metric to use for cross validation. Defaults to "roc".

        Returns:
            float: mean of the cross validation scores
        """
        if mode == "":
            mode = self.score_mode

        if mode == "roc":
            scores = cross_val_score(model, self.X_train[:,self.selected_features], self.y_train, cv=5, scoring='roc_auc', n_jobs=-1)
        elif mode == "acc":
            scores = cross_val_score(model, self.X_train[:,self.selected_features], self.y_train, cv=5, scoring='accuracy', n_jobs=-1)
        elif mode == "f1":
            scores = cross_val_score(model, self.X_train[:,self.selected_features], self.y_train, cv=5, scoring='f1', n_jobs=-1)
        elif mode == "precision":
            scores = cross_val_score(model, self.X_train[:,self.selected_features], self.y_train, cv=5, scoring='precision', n_jobs=-1)
        elif mode == "recall":
            scores = cross_val_score(model, self.X_train[:,self.selected_features], self.y_train, cv=5, scoring='recall', n_jobs=-1)
        elif mode == "AP":
            scores = cross_val_score(model, self.X_train[:,self.selected_features], self.y_train, cv=5, scoring='average_precision', n_jobs=-1)
        else:
            raise ValueError("mode must be either roc, acc, f1, precision or recall")
        return scores.mean()

 
    def simple_val(self, 
                   mode:str = "", 
                   dataset:str = "Val"):
        """Computes the score of the model on pre-defined data set (by default the validation set). 
        It can be used as a less computationally expensive (but less precise) alternative to cross validation.

        Args:
            mode (str, optional): mode to use for validation when not using cross validation. 
            dataset (str, optional): dataset to use for validation when not using cross validation. Defaults to "Val".

        Returns:
            _type_: _description_
        """
        if mode == "":
            mode = self.score_mode

        print(dataset)
        if dataset == "Val":
            X = self.X_val[:,self.selected_features]
            y = self.y_val
        elif dataset == "Test":
            X = self.X_test[:,self.selected_features]
            y = self.y_test
        elif dataset == "Full_Test":
            X = self.Full_X_test[:,self.selected_features]
            y = self.Full_y_test
        elif dataset == "Train":
            X = self.X_train[:,self.selected_features]
            y = self.y_train
        elif dataset == "Band":
            X = self.X_band[:,self.selected_features]
            y = self.y_band
        else:
            raise ValueError("dataset must be either Val, Train or Test")

        if mode == "roc":
            np.unique(X)
            y_pred_proba_val = self.model.predict_proba(X)[:, 1]
            score = roc_auc_score(y, y_pred_proba_val)
        elif mode == "AP":
            y_pred_proba_val = self.model.predict_proba(X)[:, 1]
            score = average_precision_score(y, y_pred_proba_val)
        elif mode == "acc":
            y_pred_val = self.model.predict(X)
            score = accuracy_score(y, y_pred_val)
        elif mode == "f1":
            y_pred_val = self.model.predict(X)
            score = f1_score(y, y_pred_val)
        elif mode == "precision":
            y_pred_val = self.model.predict(X)
            score = precision_score(y, y_pred_val)
        elif mode == "recall":
            y_pred_val = self.model.predict(X)
            score = recall_score(y, y_pred_val)
        else:
            raise ValueError("mode must be either roc, acc, f1, precision or recall")
        return score   

    def stack_if_exists(self, X_combined, y_combined, X, y):
        """Stacks the new set of features and targets onto the existing combined arrays.
        
        Parameters
        ----------
        X_combined : np.ndarray or None
            The existing array of combined features, or None if not yet initialized.
        y_combined : np.ndarray or None
            The existing array of combined targets, or None if not yet initialized.
        X : np.ndarray
            The new set of features to add.
        y : np.ndarray
            The new set of targets to add.

        Returns
        -------
        Tuple[np.ndarray, np.ndarray]
            The updated combined feature and target arrays.

        """
        nb_positives = y.sum()
        nb_all = len(y)

        target_ratio = self.target_positive_label_ratio
        target_total = self.target_total_train

        local_tr = nb_positives/nb_all

        missing_label = int((target_ratio - local_tr)*nb_all)

        if X_combined is None:
            X_combined = X
            y_combined = y
        else:
            X_combined = np.vstack((X_combined, X))
            y_combined = np.hstack((y_combined, y))

        self.nb_positives = y_combined.sum()
        self.nb_all = len(y_combined)

        return X_combined, y_combined


    def predict_prob(self, 
                     dataset: str = "Train") -> np.ndarray:
        """Predicts the probability of flood for a given dataset.

        Parameters
        ----------
        dataset : str, optional
            The dataset for which to predict probabilities ("Train", "Test", "Val", or (single) "Band").

        Returns
        -------
        np.ndarray
            The predicted probabilities of flood for the given dataset.

        """
        # predict the probability of flood for the train or test dataset
        if dataset == "Train":
            X = self.X_train[:,self.selected_features]
        elif dataset == "Test":
            X = self.X_test[:,self.selected_features]
        elif dataset == "Full_Test":
            X = self.Full_X_test[:,self.selected_features]
        elif dataset == "Val":
            X = self.X_val[:,self.selected_features]
        elif dataset == "Band":
            X = self.X_band[:,self.selected_features]
        else:
            raise ValueError("dataset must be either Val, Train or Test")
        
        y_pred_proba_test = self.model.predict_proba(X)[:, 1]
        return y_pred_proba_test
    

    def compute_full_grid_with_labels(self,
                                      compute_labels:bool=True) -> Tuple[np.ndarray, np.ndarray]:
        """Computes a grid of predicted probabilities alongside the corresponding labels on the full resolution dataset.

        Returns
        -------
        Tuple[np.ndarray, np.ndarray]
            The full grid of predicted probabilities and the corresponding labels reshaped to the original data's spatial dimensions.
        """
        factor = self.y_dim_FR * self.x_dim_FR
        size = int(self.X_band.shape[0]/factor)
        full_grid = np.full((size, self.y_dim_FR, self.x_dim_FR), np.nan)
        full_grid_flat = full_grid.reshape(-1)  
        full_grid_flat = self.predict_prob("Band")
        if compute_labels:
            mask = self.y_band == -1
            full_grid_flat[mask] = -1
            full_labels = self.y_band.reshape(size, self.y_dim_FR, self.x_dim_FR)
        else:
            full_labels = np.full((size, self.y_dim_FR, self.x_dim_FR), -1)
        full_grid = full_grid_flat.reshape(size, self.y_dim_FR, self.x_dim_FR)
        full_scores = self.X_band[:,-1].reshape(size, self.y_dim_FR, self.x_dim_FR)
        return full_grid, full_labels, full_scores


    def print_feature_importance(self):
        """Prints the importance of each selected feature in the model.
        """
        importances = self.model.feature_importances_

        feature_importances = zip(self.selected_features, importances)
        sorted_feature_importances = sorted(feature_importances, key=lambda x: x[1], reverse=True)

        feature_names = []
        for feature, importance in sorted_feature_importances:
            feature_names.append(self.getfname(feature))
            print(f"{self.getfname(feature)}: {importance}")
        
        feature_names = feature_names
        sorted_feature_importances = sorted_feature_importances

        features, importance_values = zip(*sorted_feature_importances)
        plt.figure(figsize=(12, 6))
        bars = plt.bar(feature_names, importance_values, capsize=5, color='skyblue', edgecolor='black', alpha=0.7)
        plt.xlabel('Features')
        plt.ylabel('Importance')
        plt.title('Feature Importance with Confidence Intervals')
        plt.xticks(rotation=45, ha='right')  # Rotate x-axis labels for better readability


        plt.tight_layout()
        plt.show()
        
    # Fonctions d'affichages

    def verb_indiv(self, 
                   individual: list):
        """Verbosely prints a genetic algorithm individual as model parameters.

        Parameters
        ----------
        individual : list
            The list representation of an individual from a genetic algorithm.

        Returns
        -------
        Any
            The result from verb_name method, typically a model and its accuracy or other metrics.
        """
        # print a GA individual as model parameters
        print(individual)
        return self.verb_name(indices = [i for i, use in enumerate(individual[:self.nb_feature]) if use], 
                                n_estimators = individual[self.nb_feature+0], 
                                max_depth = individual[self.nb_feature+1]) 

    
    def process_AUC_metrics(self, 
                                  dataset:str="Full_Test", 
                                  filter:bool = True, 
                                  scoreMinM1:int = -1):
        """ Computes Average Precision, ROC and Brier Score for the Model 2 
        or the combination of Model 1 and 2.

        Args:
            dataset (str, optional): Dataset to use for evaluation. Defaults to "Full_Test".
            filter (bool, optional): Whether to filter out predictions from Model 1. Defaults to True.
            scoreMinM1 (int, optional): Threshold used for Model 1
        """

        if scoreMinM1 == -1:
            scoreMinM1 = self.min_score_model1

        if dataset == "Test":
            X = self.X_test[:,self.selected_features]
            y = self.y_test
            model1_score = self.X_test[:,-1] #M1_score column index
        elif dataset == "Full_Test":
            if self.Full_X_test is None:
                print("No loaded Full Test Set data")
                print("Auto-Loading of the full Test Set named 'Full'")
                self.load_full_test_from_disk(name="Full")
            X = self.Full_X_test[:,self.selected_features]
            y = self.Full_y_test
            model1_score = self.Full_X_test[:,-1] #M1_score column index
        elif dataset == "Val":
            X = self.X_val[:,self.selected_features]
            y = self.y_val
            model1_score = self.X_val[:,-1] #M1_score column index
        elif dataset == "Train":
            X = self.X_train[:,self.selected_features]
            y = self.y_train
            model1_score = self.X_train[:,-1] #M1_score column index
        else:
            raise ValueError("dataset must be either Val, Train or Test")

        rejected_predictions = model1_score < scoreMinM1
        y_pred_proba_val = utils.batch_predict(self.model,X)
        if filter:
            y_pred_proba_val[rejected_predictions] = 0
        for eval_metric in ["roc", "AP", "BrierScore"]:
            score = self.compute_metric_proba(y_pred_proba_val,
                                         y, 
                                         rejected_predictions=rejected_predictions, 
                                         mode=eval_metric)
            print(f"{eval_metric} : {score}")

    def process_prediction_metrics(self, 
                                   dataset:str="Full_Test", 
                                   filter:bool = True, 
                                   scoreMinM1:int = -1):
        """ Computes F1, Precision, Recall and Accuracy for the Model 2
        or the combination of Model 1 and 2.

        Args:
            dataset (str, optional): Dataset to use for evaluation. Defaults to "Full_Test".
            filter (bool, optional): Whether to filter out predictions from Model 1. Defaults to True.
            scoreMinM1 (int, optional): Threshold used for Model 1.

        """
        if scoreMinM1 == -1:
            scoreMinM1 = self.min_score_model1
        if dataset == "Test":
            X = self.X_test[:,self.selected_features]
            y = self.y_test
            model1_score = self.X_test[:,-1] #M1_score column index
        elif dataset == "Full_Test":
            X = self.Full_X_test[:,self.selected_features]
            y = self.Full_y_test
            model1_score = self.Full_X_test[:,-1] #M1_score column index
        elif dataset == "Val":
            X = self.X_val[:,self.selected_features]
            y = self.y_val
            model1_score = self.X_val[:,-1] #M1_score column index
        elif dataset == "Train":
            X = self.X_train[:,self.selected_features]
            y = self.y_train
            model1_score = self.X_train[:,-1] #M1_score column index
        else:
            raise ValueError("dataset must be either Val, Train or Test")

        rejected_predictions = model1_score < scoreMinM1
        y_pred_val = utils.batch_predict(X, is_proba=False)
        if filter:
            y_pred_val[rejected_predictions] = 0
        for eval_metric in ["f1", "precision", "recall", "acc"]:
            score = self.compute_metric_val(y_pred_val,
                                         y, 
                                         rejected_predictions=rejected_predictions, 
                                         mode=eval_metric)
            print(f"{eval_metric} : {score}")

    def compute_metric_proba(self, y_pred_proba_val, y, mode = "roc", rejected_predictions = []):
        print(mode)
        if mode == "roc":
            score = roc_auc_score(y, y_pred_proba_val)
        elif mode == "AP":
            score = average_precision_score(y, y_pred_proba_val)
        elif mode == "BrierScore":
            score = brier_score_loss(y, y_pred_proba_val)
        else:
            raise ValueError("mode must be either roc, acc, f1, precision or recall")
        return score 

    def compute_metric_val(self, y_pred_proba_val,y_pred_val, y, mode = "roc", rejected_predictions = []):
        print(mode)
        if mode == "acc":
            score = accuracy_score(y, y_pred_val)
        elif mode == "f1":
            score = f1_score(y, y_pred_val)
        elif mode == "precision":
            score = precision_score(y, y_pred_val)
        elif mode == "recall":
            score = recall_score(y, y_pred_val)
        else:
            raise ValueError("mode must be either roc, acc, f1, precision or recall")
        return score  

    def verb_name(self, 
                  indices: list, 
                  n_estimators: int, 
                  max_depth: int):
        """Prints the model hyper-parameters in a verbose manner.

        Parameters
        ----------
        indices : list
            The indices of selected features.
        n_estimators : int
            The number of estimators for the RandomForest.
        max_depth : int
            The maximum depth for the RandomForest.
        """
        static_indices, dynamic_indices = self.index_splitter(indices)

        print("static_indices",static_indices)
        selected_static = [self.static_features.band.values[x] for x in static_indices]
        print("static_names",selected_static)
        print("dynamic_indices",dynamic_indices)
        selected_temp = [self.all_dynamic_vars[x] for x in dynamic_indices]
        print("dynamic_names",selected_temp)
        print("n_estimators",n_estimators)
        print("max_depth",max_depth)

    def auc_graph(self, 
                  dataset = "Train", 
                  metrics = "",
                  key_thresholds = [0.001,0.005,0.01,0.05,0.1, 0.2,0.3, 0.5, 0.9]):
        # plot the ROC curve and the AUC for the train or test dataset
        if dataset == "":
            datasets = ["Train", "Test", "Val"]
        else:
            datasets = [dataset]
        
        if metrics == "":
            metrics = ["roc_auc"]
        else:
            metrics = [metrics]

        for dataset in datasets:
            if dataset == "Train":
                print("Train")
                X = self.X_train[:,self.selected_features]
                y = self.y_train
            elif dataset == "Test":
                print("Test")
                X = self.X_test[:,self.selected_features]
                y = self.y_test
            elif dataset == "Full_Test":
                print("Full_Test")
                X = self.Full_X_test[:,self.selected_features]
                y = self.Full_y_test
            elif dataset == "Val":
                print("Val")
                X = self.X_val[:,self.selected_features]
                y = self.y_val
            else:
                raise ValueError("dataset must be either Val, Train or Test")
            
            y_pred_proba_test = utils.batch_predict(self.model, X)

            for metric in metrics:
                plt.figure(figsize=(8, 6))
                if metric == "roc_auc":
                    fpr, tpr, thresholds = roc_curve(y, y_pred_proba_test)
                    auc = roc_auc_score(y, y_pred_proba_test)
                    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
                    plt.plot(fpr, tpr, color='darkorange', lw=2, label='ROC curve (area = %0.3f)' % auc)
                    plt.xlabel(f'False Positive Rate {dataset}')
                    plt.ylabel(f'True Positive Rate {dataset}')
                    for thresh in key_thresholds:
                        idx = np.where(thresholds >= thresh)[0][-1]
                        plt.plot(fpr[idx], tpr[idx], 'o', markersize=10, label=f'Threshold {thresh:.3f}')

                elif  metric == "average_precision":
                    precision, recall, thresholds = precision_recall_curve(y, y_pred_proba_test)
                    auc =  average_precision_score(y, y_pred_proba_test)
                    plt.plot([0, 1], [1, 0], color='navy', lw=2, linestyle='--')
                    plt.plot(precision, recall, color='darkorange', lw=2, label='PR curve (area = %0.3f)' % auc)
                    plt.xlabel(f'Recall {dataset}')
                    plt.ylabel(f'Precision {dataset}')               


                plt.xlim([0.0, 1.0])
                plt.ylim([0.0, 1.0])
                plt.title(f'Receiver Operating Characteristic {dataset}')
                plt.legend(loc="lower right")
                plt.show()

                print(f"AUC: {auc}")
    
    
    def auc_graph_old(self, 
                  dataset: str = "Train", 
                  key_thresholds: list = [0.001,0.005,0.01,0.05,0.1, 0.2,0.3, 0.5, 0.9]):
        """Plots the ROC curve and computes the AUC for the given dataset.

        Parameters
        ----------
        dataset : str, optional
            The dataset to plot ("Train", "Test", or "Val").
        key_thresholds : list, optional
            A list of key thresholds to mark on the ROC curve.

        """
        # plot the ROC curve and the AUC for the train or test dataset
        if dataset == "Train":
            print("Train")
            X = self.X_train[:,self.selected_features]
            y = self.y_train
        elif dataset == "Test":
            print("Test")
            X = self.X_test[:,self.selected_features]
            y = self.y_test
        elif dataset == "Full_Test":
            print("Full_Test")
            X = self.Full_X_test[:,self.selected_features]
            y = self.Full_y_test
        elif dataset == "Val":
            print("Val")
            X = self.X_val[:,self.selected_features]
            y = self.y_val
        else:
            raise ValueError("dataset must be either Val, Train or Test")
        
        y_pred_proba_test = utils.batch_predict(self.model, X)

        print("# data points",y.shape)
        auc = roc_auc_score(y, y_pred_proba_test)
        print("auc",auc)
        fpr, tpr, thresholds = roc_curve(y, y_pred_proba_test)

        plt.figure(figsize=(8, 6))
        plt.plot(fpr, tpr, color='darkorange', lw=2, label='ROC curve (area = %0.3f)' % auc)
        plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')

        for thresh in key_thresholds:
            idx = np.where(thresholds >= thresh)[0][-1]
            plt.plot(fpr[idx], tpr[idx], 'o', markersize=10, label=f'Threshold {thresh:.3f}')

        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel(f'False Positive Rate {dataset}')
        plt.ylabel(f'True Positive Rate {dataset}')
        plt.title(f'Receiver Operating Characteristic {dataset}')
        plt.legend(loc="lower right")
        plt.show()

        print(f"AUC: {auc}")
    
            
    def getfname(self, 
                 index: int) -> str:
        """Retrieves the feature name for the given index.

        Parameters
        ----------
        index : int
            The index of the feature.

        Returns
        -------
        str
            The name of the feature corresponding to the provided index.
        """    
        
        var_names = self.all_static_vars + self.all_dynamic_vars
        
        return var_names[index]

    def GA_optimisation(self, 
                        ngen:int = 40, 
                        pop:int = 40, 
                        outputfile:str ="Model2", 
                        best_individuals = []):
        """Performs a genetic algorithm to find the best model parameters.
        Parameters
        ----------
        ngen : int, optional
            The number of generations for the genetic algorithm.
        pop : int, optional
            The population size for the genetic algorithm.
        outputfile : str, optional
            The name of the file to save the best model parameters.
        
        """
        creator.create("FitnessMax", base.Fitness, weights=(1.0,))  
        creator.create("Individual", list, fitness=creator.FitnessMax)
        random.seed(self.seed)

        toolbox = base.Toolbox()
        for i in range(self.nb_feature):  
            toolbox.register(f"attr_temp_index{i}", random.choice, [True, False])

        toolbox.register("attr_n_estimators", random.randint, 10, 250) 
        toolbox.register("attr_max_depth", random.randint, 5, 20)

        attributes = [toolbox.__getattribute__(f'attr_temp_index{i}') for i in range(self.nb_feature)] + \
                    [toolbox.attr_n_estimators, toolbox.attr_max_depth]

        toolbox.register("individual", tools.initCycle, creator.Individual, tuple(attributes), n=1)
        toolbox.register("population", tools.initRepeat, list, toolbox.individual)
        toolbox.register("evaluate", self.evalModel)
        toolbox.register("mate", tools.cxUniform, indpb=0.4)
        nbBooleans = self.nb_feature
        toolbox.register("mutate", tools.mutUniformInt, 
                        low=[False]*nbBooleans + [10, 4], 
                        up=[True]*nbBooleans + [350, 21], 
                        indpb=0.3)
        toolbox.register("select", tools.selTournament, tournsize=4)

        stats = tools.Statistics(key=lambda ind: ind.fitness.values)
        stats.register("max", np.max) 

        population = toolbox.population(n=pop)

        if len(best_individuals)>0:
            population += [creator.Individual(new_ind) for new_ind in best_individuals]

        result, log = algorithms.eaSimple(population, 
                                        toolbox, 
                                        cxpb=0.5, 
                                        mutpb=0.2, 
                                        ngen=ngen, 
                                        stats=stats, 
                                        verbose=True)


        best_ind = tools.selBest(population, k=1)[0]
        print("Best Individual: ", best_ind)
        best_fitness_indiv =  best_ind.fitness.values[0]
        print("Best Fitness: ", best_fitness_indiv)
        with open(f"best_fitness_{outputfile}.txt", "w") as file:
            file.write(f"Best Fitness: {best_ind}\n")

    def print_TNTPFN(self, 
                     isTP:bool = True, 
                     isFP:bool = True, 
                     isFN1:bool = True, 
                     isFN2:bool = True, 
                     isTN1:bool = False, 
                     isTN2:bool = False, 
                     thresholdM1:float = -1,
                     thresholdM2:float = 0.5,
                     save_path = "predictions_TPFN_full_test/"):
        """Prints prediction map with 
        True Positives, False Positives, False Negatives, and True Negatives (for both models).

        Parameters
        ----------
        TP : bool, optional
            Whether to print the number of True Positives.
        FP : bool, optional
            Whether to print the number of False Positives.
        FN : bool, optional
            Whether to print the number of False Negatives.
        TN1 : bool, optional
            Whether to print the number of True Negatives for Model 1.
        TN2 : bool, optional
            Whether to print the number of True Negatives for Model 2.
        """
        font_size = 16
        if thresholdM1 == -1:
            thresholdM1 = self.min_score_model1

        for band in self.labels_FR.time.values:
            print(band)
            self.prepare_data_one_band(band)
            score_2d, labels_2d, scoreM1_2d = self.compute_full_grid_with_labels()

            # Create masks for each labels
            mask_positive = labels_2d == 1
            mask_negative = labels_2d == 0
            mask_detected_M2 = score_2d > thresholdM1
            mask_detected_M1 = scoreM1_2d > thresholdM2

            score_2d[score_2d == -1] = np.nan  # remove non-france data

            grid = score_2d[0, :, :]

            plt.figure(figsize=(20, 8))
            cmap = plt.cm.gray_r
            cmap.set_bad('#A5E0E4', 1.)

            im = plt.imshow(grid, cmap=cmap, interpolation='none')

            # Overlay
            if isTN1:
                TN1 = (~mask_positive & ~mask_detected_M1)
                _, y_TN1, x_TN1 = np.where(TN1)
                plt.scatter(x_TN1, y_TN1, color='yellow', label='True Negative Model 1', marker='s', s=1)

            if isTN2:
                TN2 = (~mask_positive & ~mask_detected_M2 & mask_detected_M1)
                _, y_TN2, x_TN2 = np.where(TN2)
                plt.scatter(x_TN2, y_TN2, color='blue', label='True Negative Model 2', marker='s', s=1)

            if isFP:
                FP = (mask_detected_M2 & mask_negative) & (mask_detected_M1)
                _, y_FP, x_FP = np.where(FP)
                plt.scatter(x_FP, y_FP, color='orange', label='False Positive', marker='s', s=1)

            if isFN1:
                FN1 = (~mask_detected_M1 & mask_positive)
                _, y_FN1, x_FN1 = np.where(FN1)
                plt.scatter(x_FN1, y_FN1, color='red', label='False Negative Model 1', marker='s', s=1)

            if isFN2:
                FN2 = (~mask_detected_M2 & mask_positive) & mask_detected_M1
                _, y_FN2, x_FN2 = np.where(FN2)
                plt.scatter(x_FN2, y_FN2, color='purple', label='False Negative Model 2', marker='s', s=1)

            if isTP:
                TP = (mask_detected_M2 & mask_positive) & (mask_detected_M1)
                _, y_TP, x_TP = np.where(TP)
                plt.scatter(x_TP, y_TP, color='green', label='True Positive', marker='s', s=1)

            plt.colorbar(im, label='Predicted Flood Probability')
            plt.title(f'2D Map of Predicted Flood Probabilities - Time Slice {band}')
            plt.xlabel('X Coordinate')
            plt.ylabel('Y Coordinate')
            plt.legend(fontsize=font_size)

            # Adjusting tick label font size
            plt.xticks(fontsize=font_size)
            plt.yticks(fontsize=font_size)

            isExist = os.path.exists(save_path)
            if not isExist:
                os.makedirs(save_path)
            plt.savefig(f"{save_path}{band}.png")
            plt.close()
    
    
    def print_proba(self, 
                     isTP:bool = False, 
                     isFP:bool = False, 
                     isFN1:bool = False, 
                     isFN2:bool = False, 
                     isTN1:bool = False, 
                     isTN2:bool = False, 
                     thresholdM1:float = -1,
                     thresholdM2:float = 0.5,
                     save_path = "predictions_TPFN_full_test/"):
        """Prints prediction map with 
        True Positives, False Positives, False Negatives, and True Negatives (for both models).

        Parameters
        ----------
        TP : bool, optional
            Whether to print the number of True Positives.
        FP : bool, optional
            Whether to print the number of False Positives.
        FN : bool, optional
            Whether to print the number of False Negatives.
        TN1 : bool, optional
            Whether to print the number of True Negatives for Model 1.
        TN2 : bool, optional
            Whether to print the number of True Negatives for Model 2.
        """
        font_size = 16
        if thresholdM1 == -1:
            thresholdM1 = self.min_score_model1

        for band in self.labels_FR.time.values:
            print(band)

            self.prepare_data_one_band()
            score_2d, labels_2d, scoreM1_2d = self.compute_full_grid_with_labels()

            # Create masks for each labels 
            mask_positive = labels_2d == 1
            mask_negative = labels_2d == 0

            labels_2d[labels_2d == -1] = np.nan
            mask_detected_M2 = score_2d > thresholdM1
            mask_detected_M1 = scoreM1_2d > thresholdM2

            score_2d[score_2d == -1] = np.nan  # remove non-france data

            grid = score_2d[0, 2500:, 3500:]

            fig, axs = plt.subplots(2, 1, figsize=(32, 16))
            cmap = plt.cm.gray_r
            cmap.set_bad('#A5E0E4', 1.)

            axs[0].imshow(labels_2d[0, 2500:, 3500:], cmap=cmap, interpolation='none')

            axs[0].set_title(f'Label', fontsize=font_size)


            axs[1].imshow(grid, cmap=cmap, interpolation='none')
            axs[1].set_title('M2 Flood Probabilities', fontsize=font_size)

            # Overlay
            if isTN1:
                TN1 = (~mask_positive & ~mask_detected_M1)
                _, y_TN1, x_TN1 = np.where(TN1)
                plt.scatter(x_TN1, y_TN1, color='yellow', label='True Negative Model 1', marker='s', s=1)

            if isTN2:
                TN2 = (~mask_positive & ~mask_detected_M2 & mask_detected_M1)
                _, y_TN2, x_TN2 = np.where(TN2)
                plt.scatter(x_TN2, y_TN2, color='blue', label='True Negative Model 2', marker='s', s=1)

            if isFP:
                FP = (mask_detected_M2 & mask_negative) & (mask_detected_M1)
                _, y_FP, x_FP = np.where(FP)
                plt.scatter(x_FP, y_FP, color='orange', label='False Positive', marker='s', s=1)

            if isFN1:
                FN1 = (~mask_detected_M1 & mask_positive)
                _, y_FN1, x_FN1 = np.where(FN1)
                plt.scatter(x_FN1, y_FN1, color='red', label='False Negative Model 1', marker='s', s=1)

            if isFN2:
                FN2 = (~mask_detected_M2 & mask_positive) & mask_detected_M1
                _, y_FN2, x_FN2 = np.where(FN2)
                plt.scatter(x_FN2, y_FN2, color='purple', label='False Negative Model 2', marker='s', s=1)

            if isTP:
                TP = (mask_detected_M2 & mask_positive) & (mask_detected_M1)
                _, y_TP, x_TP = np.where(TP)
                plt.scatter(x_TP, y_TP, color='green', label='True Positive', marker='s', s=1)

            #plt.colorbar(im, label='Predicted Flood Probability', fontsize=font_size)
            plt.title(f'2D Map of Predicted Flood Probabilities - Time Slice {band}', fontsize=font_size)
            plt.xlabel('X Coordinate', fontsize=font_size)
            plt.ylabel('Y Coordinate', fontsize=font_size)
            plt.legend(fontsize=font_size)

            # Adjusting tick label font size
            plt.xticks(fontsize=font_size)
            plt.yticks(fontsize=font_size)

            isExist = os.path.exists(save_path)
            if not isExist:
                os.makedirs(save_path)
            plt.savefig(f"{save_path}{band}.png")
            plt.close()
    
    def print_proba_inf(self, 
                     save_path = "predictions_TPFN_full_test/"):
        """Prints prediction map with 
        True Positives, False Positives, False Negatives, and True Negatives (for both models).

        Parameters
        ----------
        TP : bool, optional
            Whether to print the number of True Positives.
        FP : bool, optional
            Whether to print the number of False Positives.
        FN : bool, optional
            Whether to print the number of False Negatives.
        TN1 : bool, optional
            Whether to print the number of True Negatives for Model 1.
        TN2 : bool, optional
            Whether to print the number of True Negatives for Model 2.
        """
        font_size = 16

        for band in self.dynamic_features_FR.time.values:
            print(band)

            self.prepare_data_one_band(band,compute_labels=False)
            score_2d, _, _ = self.compute_full_grid_with_labels(compute_labels=False)

            # Create masks for each labels 

            score_2d[score_2d == -1] = np.nan  # remove non-france data

            grid = score_2d[0, :, :]

            
            cmap = plt.cm.gray_r
            cmap.set_bad('#A5E0E4', 1.)

            plt.figure(figsize=(32, 16))

            plt.imshow(grid, cmap=cmap, interpolation='none')
            
            #plt.colorbar(im, label='Predicted Flood Probability', fontsize=font_size)
            plt.title(f'Flood Score - Time Slice {band}', fontsize=font_size)
            plt.xlabel('X Coordinate', fontsize=font_size)
            plt.ylabel('Y Coordinate', fontsize=font_size)
            plt.legend(fontsize=font_size)

            # Adjusting tick label font size
            plt.xticks(fontsize=font_size)
            plt.yticks(fontsize=font_size)

            isExist = os.path.exists(save_path)
            if not isExist:
                os.makedirs(save_path)
            plt.savefig(f"{save_path}{band}.png")
            plt.close()


    def save_full_pred(self, 
                     save_path = "predictions_TPFN_full_test/"):
        """Prints prediction map with 
        True Positives, False Positives, False Negatives, and True Negatives (for both models).

        Parameters
        ----------
        TP : bool, optional
            Whether to print the number of True Positives.
        FP : bool, optional
            Whether to print the number of False Positives.
        FN : bool, optional
            Whether to print the number of False Negatives.
        TN1 : bool, optional
            Whether to print the number of True Negatives for Model 1.
        TN2 : bool, optional
            Whether to print the number of True Negatives for Model 2.
        """

        full_cube = np.full((self.dynamic_features_FR.time.shape[0], self.y_dim_FR, self.x_dim_FR), np.nan)
        iter = 0
        for band in self.dynamic_features_FR.time.values:
            print(band)

            self.prepare_data_one_band(band,compute_labels=False)
            score_2d, _, _ = self.compute_full_grid_with_labels(compute_labels=False)

            # Create masks for each labels 

            score_2d[score_2d == -1] = np.nan  # remove non-france data

            grid = score_2d[0, :, :]
            full_cube[iter, :, :] = grid
            iter += 1
        
        print(full_cube.shape)


        xr_array_score = xr.DataArray(full_cube, 
                                    dims=["time", "y", "x"],
                                    coords={"time": self.dynamic_features_FR.time, 
                                            "x": self.dynamic_features_FR.x, 
                                            "y": self.dynamic_features_FR.y},
                                    name="M2_score")

        xr_array_score = xr_array_score.astype('float32')

        xr_array_score.to_netcdf('localdata/Model2_Score_Full_Rez_inf.nc')


    def save_prediction_map_and_labels(self, 
                            save_path = "graph/model1_AP/label_and_pred/"):
        font_size = 32
        for k, band_index in enumerate(self.labels.band.values):
            labelmap = self.labels['__xarray_dataarray_variable__'][k].values

            
            grid_2d = self.full_grid_all[k, :, :]
            grid_2d[labelmap == -1] = np.nan

            labelmap[labelmap == -1] = np.nan

            fig, axs = plt.subplots(1, 2, figsize=(32, 16))
            cmap = plt.cm.gray
            cmap.set_bad('#A5E0E4', 1.)


            # Increase label size for axis ticks
            for ax in axs:
                ax.tick_params(axis='both', which='major', labelsize=font_size)

            plt.tight_layout()

            isExist = os.path.exists(save_path)
            if not isExist:
                os.makedirs(save_path)
            plt.savefig(f"{save_path}{band_index}.png")
            plt.close(fig)

