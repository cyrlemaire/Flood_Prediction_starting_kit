import xarray as xr
from deap import base, creator, tools, algorithms
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_curve, roc_auc_score, balanced_accuracy_score, precision_recall_curve, average_precision_score, brier_score_loss, accuracy_score, f1_score, precision_score, recall_score
import matplotlib.pyplot as plt
import pandas as pd
import pickle
import random
from sklearn.model_selection import cross_val_score
from typing import Union, Tuple, List, Optional, Any
import os
from matplotlib.colors import ListedColormap, BoundaryNorm
import src.Utils as utils

class BaseLineModel:
    """Base class for the baseline model.
    """
    number_of_instances = 0
    
    def __init__(self, 
                 labels_path:str, # path to the labels xarray dataset
                 dynamic_features_path:str = None, # path to the dynamic features xarray dataset
                 static_features_path:str = None, # path to the static features xarray dataset
                 replace_nan:bool=True, # replace nan values in labels by a value (useful when nan indicates no flood)
                 nan_value:int=0, # value to replace nan values in labels
                 reduce_train:bool=False, # reduce the train to take only band with flood 
                 train_start:str = "2002-08-03", # date where to split train test
                 train_end:str = "2003-01-1", # date where to split train test
                 test_start:str = "2003-01-01", # date where to split train test
                 test_end:str = "2003-03-17", # date where to split train test
                 val_start:str = "2003-01-1", # date where to split train test
                 val_end:str = "2003-03-17", # date where to split train test
                 inf_start:str = "2003-11-1", # date where to split train test
                 inf_end:str = "2004-01-01", # date where to split train test
                 name:str = "Model_01_default",
                 seed:int = 42,
                 is_cross_val:bool = True,
                 eval_metric_GA:str = "roc_auc"):
        """Constructor for the baseline model

        Args:
            labels_path (str): path to the labels xarray dataset
            dynamic_features_path (str, optional): path to the dynamic features xarray dataset. Defaults to None.
            static_features_path (str, optional): path to the static features xarray dataset. Defaults to None.
            replace_nan (bool, optional): replace nan values in labels by a value (useful when nan indicates no flood).
            nan_value (int, optional): value to replace nan values in labels. Defaults to 0.
            reduce_train (bool, optional): reduce the train to take only band with flood. Defaults to False.
            train_start (str, optional): date where to split train test. Defaults to "2002-08-03".
            train_end (str, optional): date where to split train test. Defaults to "2003-01-1".
            test_start (str, optional): date where to split train test. Defaults to "2003-01-01".
            test_end (str, optional): date where to split train test. Defaults to "2003-03-17".
            val_start (str, optional): date where to split train test. Defaults to "2003-01-1".
            val_end (str, optional): date where to split train test. Defaults to "2003-03-17".
            inf_start (str, optional): date where to split train test. Defaults to "2003-11-1".
            inf_end (str, optional): date where to split train test. Defaults to "2004-01-01".
            name (str, optional): . Defaults to "Model_01_default".
            seed (int, optional): Path to the dynamic features Xarray. Defaults to 42.
            is_cross_val (bool, optional): . Defaults to True.
            eval_metric_GA (str, optional): _description_. Defaults to "roc_auc".

        Raises:
            ValueError: _description_
        """

        self.eval_metric_GA = eval_metric_GA
        self.is_cross_val = is_cross_val
        self.seed = seed
        self.obj_name = name

        self.dataset_limits = {
            'train': {'start':  pd.to_datetime(train_start), 'end':  pd.to_datetime(train_end)},
            'val': {'start':  pd.to_datetime(val_start), 'end':  pd.to_datetime(val_end)},
            'test': {'start':  pd.to_datetime(test_start), 'end':  pd.to_datetime(test_end)},
            'all': {'start':  pd.to_datetime(train_start), 'end':  pd.to_datetime(test_end)},
            'inf': {'start':  pd.to_datetime(inf_start), 'end':  pd.to_datetime(inf_end)}
        }
        
        if (labels_path is None) | (
            (dynamic_features_path is None) & (static_features_path is None)
        ):
            raise ValueError("You must provide a path to the labels and either the dynamic or static features")

        self.labels = xr.open_dataset(labels_path)

        # replace nan values with "nan_value" in labels useful when nan indicates no flood
        #nan_mask = np.isnan(self.labels['__xarray_dataarray_variable__'].data)
        nan_mask = np.isnan(self.labels['__xarray_dataarray_variable__'].values)
        if replace_nan:
            self.labels['__xarray_dataarray_variable__'].data[nan_mask] = nan_value

        # load dynamic and static features and information about the number of features
        self.nb_feature_dynamic = 0
        self.all_dynamic_vars = []
        if dynamic_features_path is not None:
            self.dynamic_features = xr.open_dataset(dynamic_features_path)
            self.nb_feature_dynamic = len(self.dynamic_features.data_vars)
            self.all_dynamic_vars = list(self.dynamic_features.data_vars)
            self.nb_feature_dynamic = 2
            self.all_dynamic_vars = [self.all_dynamic_vars[0], self.all_dynamic_vars[1]]

        self.nb_feature_static = 0
        if static_features_path is not None:
            self.static_features = xr.open_dataset(static_features_path)
            self.all_static_vars = list(self.static_features.data_vars)
            self.nb_feature_static = len(self.static_features.data_vars)

        self.nb_feature = self.nb_feature_dynamic + self.nb_feature_static

        # dimensions of the dataset
        self.x_dim = self.labels.dims['x']
        self.y_dim = self.labels.dims['y']
        self.time_dim = self.labels.dims['time']

        # split train test mode
        self.reduce_train=reduce_train
        # train test split date
    

    def index_splitter(self, 
                       input_list:List[int], 
                       histodept:int, 
                       histodept2:int, 
                       use_derivative:bool)->Tuple[List[int], List[int]]:
        """ convert a list of indices into a list of static indices and a list of dynamic indices
        using the info about the structure of the dataset. Also add the derivative of the dynamic features if needed.
        And compute the name of the dynamic features.

        Args:
            input_list (List[int]): list of indices of the features to use (static and dynamic)
            histodept (int): history depth for averaging the dynamic features
            histodept2 (int): history depth for averaging the dynamic features (second part)
            use_derivative (bool): Activate the computation of the derivative of the dynamic features

        Returns:
            Tuple[List[int], List[int]]: list of static indices and list of dynamic indices
        """

        self.all_dynamic_vars_names = []
        static = [x for x in input_list if 0 <= x < self.nb_feature_static]
        dynamic = [x for x in input_list if self.nb_feature_static <= x < self.nb_feature_static + self.nb_feature_dynamic]

        i_dyna = self.nb_feature_static

        dynamic = [x - self.nb_feature_static for x in dynamic]
        dynamic_names = [self.all_dynamic_vars[x] for x in dynamic]
        self.selected_features = static.copy()
        if len(dynamic)>0:
            for dyna_var in dynamic_names:
                if histodept > 0:
                    self.all_dynamic_vars_names.append(dyna_var+"_"+str(histodept))
                    self.selected_features.append(i_dyna)
                    i_dyna = i_dyna +1
                    if use_derivative:
                        self.all_dynamic_vars_names.append(dyna_var+"_"+str(histodept)+"_deriv")
                        self.selected_features.append(i_dyna)
                        i_dyna = i_dyna +1
                if histodept2 > 0:
                    self.all_dynamic_vars_names.append(dyna_var+"_"+str(histodept2))
                    self.selected_features.append(i_dyna)
                    i_dyna = i_dyna +1
                    if use_derivative:
                        self.all_dynamic_vars_names.append(dyna_var+"_"+str(histodept2)+"_deriv")
                        self.selected_features.append(i_dyna)
        return static, dynamic # list of static and dynamic indices used in the model (feature selection)
    

    
    def train_flood_prediction_model(self, 
                                     indices:List[int], 
                                     use_derivative:bool,
                                     n_estimators:int, 
                                     max_depth:int, 
                                     histodept:int, 
                                     histodept2:int,
                                     is_GA=False)->Tuple[RandomForestClassifier, float]:
        """Train the random forest model

        Args:
            indices (List[int]): list of indices of the features to use (static and dynamic)
            use_derivative (bool): compute the derivative of the dynamic features
            n_estimators (int): number of estimators for the random forest
            max_depth (int): max depth of the random forest
            histodept (int): number of layers to average for the dynamic features (first part)
            histodept2 (int): number of layers to average for the dynamic features (second part)
            is_GA (bool, optional): Are we in a GA optimisation. Defaults to False. (if True we don't save results)

        Returns:
            Tuple[RandomForestClassifier, float]: Trained random forest model and the score of the model
        """

        features_all_bands = []
        targets_all_bands = []
        static_indices, dynamic_indices = self.index_splitter(indices, histodept, histodept2, use_derivative)
        num_statics = len(static_indices)
        for band in self.labels.time.values:
            start_date = pd.to_datetime(band)

            filtered_dynamic_features = self.dynamic_features.sel(time=slice(None, start_date))
            # compute the total number of features
            total_features = num_statics
            if len(dynamic_indices) > 0:
                for _ in dynamic_indices:
                    if histodept > 0:
                        total_features += 1
                    if histodept2 > 0:
                        total_features += 1 
                    if use_derivative & (histodept > 1):
                        total_features += 1 
                    if use_derivative & (histodept2 > 1):
                        total_features += 1

            # create an empty array for features
            feature_array = np.empty((self.x_dim * self.y_dim, total_features))
            feature_index = 0

            # add static features if any
            for static_index in static_indices:
                var = self.all_static_vars[static_index]
                feature_array[:, feature_index] = self.static_features[var].data.reshape(self.x_dim * self.y_dim)
                feature_index += 1

            # add average and derivative of dynamic features if any
            for var_index in dynamic_indices:
                var = self.all_dynamic_vars[var_index]
                feature_index, feature_array = self.process_dynamic_feature(filtered_dynamic_features[var], histodept, feature_array, feature_index, use_derivative)
                feature_index, feature_array = self.process_dynamic_feature(filtered_dynamic_features[var], histodept2, feature_array, feature_index, use_derivative)

            feature_temp = feature_array if total_features > 0 else None

            features_all_bands.append(feature_temp)

            target = self.labels['__xarray_dataarray_variable__'].sel(time=band).data.reshape(-1)
            targets_all_bands.append(target)

        # split train test
        temp_data = {'train': {'X':  None, 'y':  None}, 
                     'val': {'X':  None, 'y':  None}, 
                     'test': {'X':  None, 'y':  None}, 
                     'all': {'X':  None, 'y':  None}, 
                     'inf': {'X':  None, 'y':  None}}
        
        for dataset in ['train', 'val', 'test', 'all', 'inf']:
            if dataset in ['all','inf']:
                france_only = False
            else:
                france_only = True
            temp_data[dataset]['X'], temp_data[dataset]['y'] = self.split_subset_date(
                features_all_bands,
                targets_all_bands,
                start_date = self.dataset_limits[dataset]['start'],
                end_date = self.dataset_limits[dataset]['end'],
                france_only = france_only)

        # train the random forest
        model = RandomForestClassifier(
            random_state=self.seed,
            n_estimators=n_estimators,
            max_depth=max_depth,
            n_jobs=-1,
            criterion="entropy",
            class_weight='balanced' 
        )        

        if (~is_GA) or (~self.is_cross_val):
            model.fit(temp_data['train']['X'], temp_data['train']['y'])
        if self.is_cross_val:
            scores = cross_val_score(model, temp_data['train']['X'], temp_data['train']['y'], cv=5, scoring=self.eval_metric_GA, n_jobs=-1)
        else:
            y_pred_proba_val = model.predict_proba(temp_data['val']['X'])[:, 1]
            if self.eval_metric_GA == "average_precision":
                #fpr, tpr, thresholds = precision_recall_curve(y, y_pred_proba_test)
                scores =  average_precision_score(temp_data['val']['y'], y_pred_proba_val)
            elif self.eval_metric_GA == "auc_roc":
                scores = roc_auc_score(temp_data['val']['y'], y_pred_proba_val)


        # keep the model and the data if needed (sometime we just want to know the AUC without saving the model)
        if ~is_GA:
            self.model = model
            self.X_train = temp_data['train']['X']
            self.X_val = temp_data['val']['X']
            self.X_test = temp_data['test']['X']
            self.X_all = temp_data['all']['X']
            self.X_inf = temp_data['inf']['X']
            self.y_train =  temp_data['train']['y']
            self.y_val = temp_data['val']['y']
            self.y_test = temp_data['test']['y']
            self.y_all = temp_data['all']['y']
            self.mask = ~ np.isnan(target)
        return (model, 
                scores.mean())

    def process_dynamic_feature(self, 
                                feature: xr.DataArray, # dynamic feature
                                histodepth: int, # history depth for averaging the dynamic features
                                feature_array: np.ndarray, # array of features
                                feature_index: int, # index of the feature in the array
                                use_derivative: bool # compute the derivative of the dynamic features
                                )->Tuple[int, np.ndarray]:
        """
        Process a dynamic feature by averaging histodepth layers also compute the derivative if needed.
        Args:
            feature (xr.DataArray): dynamic feature
            histodepth (int): history depth for averaging the dynamic features
            feature_array (np.ndarray): array of features
            feature_index (int): index of the feature in the array
            use_derivative (bool): compute the derivative of the dynamic features

        Returns:
            Tuple[int, np.ndarray]: index of the feature in the array and the array of features
        """
        if histodepth > 0:
            last_layers = feature.isel(time=slice(-histodepth, None))
            averaged_feature = last_layers.mean(dim='time')
            feature_array[:, feature_index] = averaged_feature.data.reshape(self.x_dim * self.y_dim)
            feature_index += 1

            if use_derivative and histodepth > 1:
                average_derivative = np.gradient(last_layers, axis=0).mean(axis=0)
                feature_array[:, feature_index] = average_derivative.reshape(self.x_dim * self.y_dim)
                feature_index += 1

        return feature_index, feature_array

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

        features, importance_values = zip(*sorted_feature_importances)
        plt.figure(figsize=(12, 6))
        bars = plt.bar(feature_names, importance_values, capsize=5, color='skyblue', edgecolor='black', alpha=0.7)
        plt.xlabel('Features')
        plt.ylabel('Importance')
        plt.title('Feature Importance')
        plt.xticks(rotation=45, ha='right')  # Rotate x-axis labels for better readability


        plt.tight_layout()
        plt.show()

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
        
        var_names = self.all_static_vars + self.all_dynamic_vars_names

#        reduce_list = [var_names[i] for i in self.selected_features]
        
        return var_names[index]

    def get_flood_bands(self, 
                        all_bands: List[str], 
                        labels: xr.Dataset) -> List[str]:
        """ Get the bands with flood, use to reduce the train dataset to only the bands with flood 
        when (reduce_train=True) by default we keep all the bands.

        Args:
            all_bands (List[str]): list of all the bands
            labels (xr.Dataset): labels dataset
        Returns:
            List[str]: list of bands with flood
        """
        flood_bands = []
        for band_date in all_bands:
            band_data = labels['__xarray_dataarray_variable__'].sel(time=band_date).values
            if np.any(band_data != 0):
                flood_bands.append(band_date)
        return flood_bands

    def split_subset_date(self, 
                          features_filtered: List[np.ndarray], 
                          target_source: List[np.ndarray], 
                          start_date:str="2003-09-01", 
                          end_date:str="2003-09-01",
                          france_only:bool = False,
                          is_inf:bool = False)-> Tuple[np.ndarray, np.ndarray]:
        """_summary_

        Args:
            features_filtered (List[np.ndarray]): Filtered features array.
            target_source (List[np.ndarray]): Filtered target array.
            start_date (str, optional): start date for the subset. Defaults to "2003-09-01".
            end_date (str, optional): end date for the subset. Defaults to "2003-09-01".
            france_only (bool, optional): Filtering out data not relevant for France. Defaults to False.
            is_inf (bool, optional): is it the inference dataset (if so there is no labels). Defaults to False.

        Returns:
            Tuple[np.ndarray, np.ndarray]: filtered feature and label arrays
        """
        start, end = pd.to_datetime(start_date), pd.to_datetime(end_date)
        all_bands = self.labels.time.values

        if self.reduce_train:
            all_bands = self.get_flood_bands(all_bands, self.labels)

        bands = [band for band in all_bands if start < band <= end]
        band_to_index = {band: idx for idx, band in enumerate(self.labels.time.values)}

        num_samples = len(bands)
        feature_shape = features_filtered[0].shape[:2]

        X_tmp = np.empty((num_samples, *feature_shape))
        if ~is_inf:
            y_tmp = np.empty((num_samples, features_filtered[0].shape[0]))

        for band in bands:
            index = band_to_index[band]
            feature = features_filtered[index]
            if ~is_inf:
                target = target_source[index]
            idx = bands.index(band)
            X_tmp[idx] = feature.reshape(*feature_shape)
            if ~is_inf:
                y_tmp[idx] = target

        if is_inf:
            y = None
        else:
            y = y_tmp.reshape(-1)
        X = np.reshape(X_tmp, (X_tmp.shape[0] * X_tmp.shape[1], X_tmp.shape[2]))
        
        if france_only:
            X, y = self.remove_non_france_data(X, y)

        return X, y


    def remove_nan(self,
                   X: np.ndarray,
                   y: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Filters out data with nan values.

        Args:
            X (np.ndarray): feature array
            y (np.ndarray): label array

        Returns:
            Tuple[np.ndarray, np.ndarray]: filtered feature and label arrays
        """
        valid_indices = ~np.isnan(y) 
        return X[valid_indices], y[valid_indices]

    def train_model(self,
                   individual: List[Any], 
                   is_GA:bool=False)-> float:
        """Transform GA individual into model parameters

        Args:
            individual (List[]): GA individual
            is_GA (bool, optional): Are we in a GA optimisation. Defaults to False. (if True we don't save results)
        
        Returns:
            float: score of the model
        """
        print("train_model",individual)
        return self.check_load(indices = [i for i, use in enumerate(individual[:self.nb_feature]) if use], 
                                  use_derivative = individual[self.nb_feature], 
                                  n_estimators = individual[self.nb_feature+1], 
                                  max_depth = individual[self.nb_feature+2], 
                                  histodept = individual[self.nb_feature+3], 
                                  histostrt = individual[self.nb_feature+4], 
                                  is_GA = is_GA) 


    def evalModel(self,
                  individual: List[Any],
                  is_GA:bool=True)-> Tuple[float]:
        """Evaluate the model with the GA individual

        Args:
            individual (List[]): GA individual
            is_GA (bool, optional): Are we in a GA optimisation. Defaults to True. (if True we don't save results)

        Returns:
            Tuple[float]: score of the model (used as fitness for the GA)
        """
        _, auc = self.train_model(individual, is_GA)
        return auc,


    def check_load(self, 
                   indices: List[int], 
                   use_derivative: bool, 
                   n_estimators: int,
                   max_depth: int, 
                   histodept: int, 
                   histostrt: int, 
                   is_GA:bool=False, 
                   debug:bool=False):
        """ Check if the individual is valid and launch the training of the model if it is valid.
        Set the auc to 0 if the individual is not valid (useful for the GA)

        Args:
            indices (_type_): selected features indices
            use_derivative (_type_): _description_
            n_estimators (_type_): _description_
            max_depth (_type_): _description_
            histodept (_type_): _description_
            histostrt (_type_): _description_
            is_GA (bool, optional): _description_. Defaults to False.
            debug (bool, optional): _description_. Defaults to False.

        Returns:
            _type_: _description_
        """
        if debug:
            print("indices",indices)
            print("use_derivative",use_derivative)
            print("n_estimators",n_estimators)
            print("max_depth",max_depth)
            print("histodept",histodept)
            print("histostrt",histostrt)
            print("is_GA",is_GA)
        
        if(len(indices)>0):
            model, score = self.train_flood_prediction_model(
                indices, 
                use_derivative, 
                n_estimators, 
                max_depth,
                histodept,
                histostrt,
                is_GA)
        else:
            model = None
        if model is None:
            score = 0
        print("score",score)

        return model,score


    def predict_prob(self, dataset = "Train"):
        """ Predict the probability of flood for the train, val, test, all or inference dataset

        Args:
            dataset (str, optional): . Defaults to "Train".

        Returns:
            _type_: _description_
        """
        if dataset == "Train":
            X = self.X_train
        elif dataset == "Test":
            X = self.X_test
        elif dataset == "Val":
            X = self.X_val
        elif dataset == "All":
            X = self.X_all
        elif dataset == "Inf":
            X = self.X_inf
        else:
            raise ValueError("dataset must be either Val, Train, All or Test")
        y_pred_proba_test = self.model.predict_proba(X)[:, 1]
        return y_pred_proba_test

    def compute_full_grid(self):
        """Computes a grid of predicted probabilities alongside the corresponding labels on the full resolution dataset.

        Returns
        -------
        np.ndarray
            The full grid of predicted probabilities.
        """
        factor = self.y_dim * self.x_dim

        all_size = int(self.y_all.shape[0]/factor)
        self.full_grid_all = np.full((all_size, self.y_dim, self.x_dim), np.nan)
        full_grid_flat = self.full_grid_all.reshape(-1)
        full_grid_flat = self.predict_prob("All")
        self.full_grid_all = full_grid_flat.reshape(all_size, self.y_dim, self.x_dim)

        inf_size = int(self.X_inf.shape[0]/factor)
        self.full_grid_inf = np.full((inf_size, self.y_dim, self.x_dim), np.nan)
        full_grid_flat = self.full_grid_inf.reshape(-1)
        full_grid_flat = self.predict_prob("Inf")
        self.full_grid_inf = full_grid_flat.reshape(inf_size, self.y_dim, self.x_dim)

    def save_FP_FN_map(self, 
                            save_path = "graph/model1_AP/compare_threshold/",
                            thresholds = [0.3, 0.35, 0.4, 0.45, 0.5, 0.6]):
        """ Predictions map at different thresholds with false positive and false negative.

        Args:
            save_path (str, optional): Saving path. Defaults to "graph/model1_AP/compare_threshold/".
            thresholds (list, optional): Prediction threshold to define positive / negative predictions.
        """
        cmap = ListedColormap(['white', 'grey', 'black', 'red'])
        
        n_rows = len(thresholds)
        boundaries = [-0.5, 0.5, 1.5, 2.5, 3.5]
        norm = BoundaryNorm(boundaries, cmap.N, clip=True)

        for k, band_index in enumerate(self.labels.sel(time=slice(
                            self.dataset_limits["train"]["start"],
                            self.dataset_limits["train"]["end"],
                            )).time.values):
            fig, axs = plt.subplots(n_rows + 2, 1, figsize=(20, 40))

            grid_2d = self.full_grid_all[k, :, :]
            x_coord_0 = grid_2d.shape[1] * 0.8

            axs[0].imshow(grid_2d, cmap='viridis', interpolation='none')
            axs[0].set_title('Predicted Flood Probabilities')
            axs[0].axvline(x=x_coord_0, color='red', linestyle='--')

            labelmap = self.labels['__xarray_dataarray_variable__'][k]

            axs[1].imshow(labelmap, cmap='viridis')
            axs[1].set_title('Label')
            axs[1].axvline(x=x_coord_0, color='red', linestyle='--')

            for i, threshtest in enumerate(thresholds):
                grid_2d2_tmp = grid_2d.copy()

                grid_2d2_tmp[grid_2d < threshtest] = 0
                grid_2d2_tmp[grid_2d >= threshtest] = 1

                classification_results = np.zeros_like(grid_2d2_tmp)
                classification_results[(grid_2d2_tmp == 0) & (labelmap == 0)] = 0  # TN
                classification_results[(grid_2d2_tmp == 1) & (labelmap == 0)] = 1
                classification_results[(grid_2d2_tmp == 1) & (labelmap == 1)] = 2
                classification_results[(grid_2d2_tmp == 0) & (labelmap == 1)] = 3  # TP

                im = axs[2 + i].imshow(classification_results, cmap=cmap, norm=norm, interpolation='none')
                axs[2 + i].set_title(f"Threshold: {threshtest}")

            plt.tight_layout()

            isExist = os.path.exists(save_path)
            if not isExist:
                os.makedirs(save_path)
            plt.savefig(f"{save_path}{band_index}.png")
            plt.close(fig)


    def save_prediction_map(self, 
                            save_path:str = "graph/model1_AP/predictions/"):
        """ Predictions map at different thresholds.

        Args:
            save_path (str, optional): Saving path. Defaults to "graph/model1_AP/predictions/".
        """
        for specific_time_slice in range(self.full_grid_all.shape[0]):
            grid_2d = self.full_grid_all[specific_time_slice, :, :]
            plt.figure(figsize=(20, 8))
            plt.imshow(grid_2d, cmap='viridis', interpolation='none')
            plt.colorbar(label='M1 Flood Probability')
            plt.title(f'Flood Probabilities - Time Slice {specific_time_slice}')
            plt.xlabel('X Coordinate')
            plt.ylabel('Y Coordinate')

            isExist = os.path.exists(save_path)
            if not isExist:
                os.makedirs(save_path)
            plt.savefig(f"{save_path}{specific_time_slice}.png")
            plt.close()

    def save_prediction_map_and_labels(self, 
                            save_path:str = "graph/model1_AP/label_and_pred/"):
        """ Predictions map at different thresholds with labels.

        Args:
            save_path (str, optional): Saving path. Defaults to "graph/model1_AP/label_and_pred/".
        """
        font_size = 32

        for k, band_index in enumerate(self.labels.sel(time=slice(
                            self.dataset_limits["train"]["start"],
                            self.dataset_limits["train"]["end"],
                            )).time.values):
            labelmap = self.labels['__xarray_dataarray_variable__'][k].values

            grid_2d = self.full_grid_all[k, :, :]
            grid_2d[labelmap == -1] = np.nan

            labelmap[labelmap == -1] = np.nan

            fig, axs = plt.subplots(1, 2, figsize=(32, 16))
            cmap = plt.cm.gray_r
            cmap.set_bad('#A5E0E4', 1.)

            axs[0].imshow(grid_2d, cmap=cmap, interpolation='none')
            axs[0].set_title('M1 Flood Probabilities', fontsize=font_size)

            axs[1].imshow(labelmap, cmap=cmap, interpolation='none')
            axs[1].set_title(f'Label', fontsize=font_size)

            # Increase label size for axis ticks
            for ax in axs:
                ax.tick_params(axis='both', which='major', labelsize=font_size)

            plt.tight_layout()

            isExist = os.path.exists(save_path)
            if not isExist:
                os.makedirs(save_path)
            plt.savefig(f"{save_path}{band_index}.png")
            plt.close(fig)



    def save_error_map(self, 
                            save_path:str = "graph/model1_AP/save_error_map/"):
        """ Predictions map at different thresholds with labels.

        Args:
            save_path (str, optional): Saving path. Defaults to "graph/model1_AP/label_and_pred/".
        """
        font_size = 32

        for k, band_index in enumerate(self.labels.sel(time=slice(
                            self.dataset_limits["train"]["start"],
                            self.dataset_limits["train"]["end"],
                            )).time.values):
            labelmap = self.labels['__xarray_dataarray_variable__'][k].values

            grid_2d = self.full_grid_all[k, :, :]
            grid_2d[labelmap == -1] = np.nan

            labelmap[labelmap == -1] = np.nan

            fig, axs = plt.subplots(1, 2, figsize=(32, 16))
            cmap = plt.cm.bwr
            cmap.set_bad('#A5E0E4', 1.)

            Errors = grid_2d - labelmap
            plt.figure(figsize=(20, 8))
            plt.imshow(grid_2d, cmap=cmap, interpolation='none')



            plt.tight_layout()

            isExist = os.path.exists(save_path)
            if not isExist:
                os.makedirs(save_path)
            plt.savefig(f"{save_path}{band_index}.png")
            plt.close(fig)

    def verb_indiv(self, 
                   individual: List[Any]):
        """ Print the model parameters from a GA individual. 
        Breaks the individual into model parameters.

        Args:
            individual (List[Any]): GA individual
            save_model (bool, optional): 

        """
        print(individual)
        print("indices", [i for i, use in enumerate(individual[:self.nb_feature]) if use])
        print(self.nb_feature)
        self.verb_name(indices = [i for i, use in enumerate(individual[:self.nb_feature]) if use], 
                                use_derivative = individual[self.nb_feature], 
                                n_estimators = individual[self.nb_feature+1], 
                                max_depth = individual[self.nb_feature+2], 
                                histodept = individual[self.nb_feature+3], 
                                histostrt = individual[self.nb_feature+4]) 

    def verb_name(self, 
                  indices: List[int], 
                  use_derivative: bool, 
                  n_estimators:int, 
                  max_depth:int, 
                  histodept:int, 
                  histostrt:int):
        """ Print the model parameters.

        Args:
            indices (List[int]): List of indices of the features to use (static and dynamic)
            use_derivative (bool): compute the derivative of the dynamic features
            n_estimators (int): number of estimators for the random forest
            max_depth (int): max depth of the random forest
            histodept (int): history depth for averaging the dynamic features
            histostrt (int): history depth for averaging the dynamic features (second part)
        """
        static_indices, dynamic_indices = self.index_splitter(indices, histodept, histostrt, use_derivative)

        print("static_indices",static_indices)
        print("len static_indices",len(static_indices))
        
        selected_static = [self.all_static_vars[x] for x in static_indices]
        print("static_names",selected_static)
        print("len static_names",len(selected_static))
        print("dynamic_indices",dynamic_indices)
        selected_temp = [self.all_dynamic_vars[x] for x in dynamic_indices]
        print("dynamic_names",selected_temp)
        print("use_derivative",use_derivative)
        print("n_estimators",n_estimators)
        print("max_depth",max_depth)
        print("histodept",histodept)
        print("histostrt",histostrt)


    def compute_all_metrics(self):
        """Compute all the metrics for the train, val and test dataset"""
        for dataset in ["Train", "Test", "Val"]:
            print(f"{dataset} :")
            if dataset == "Train":
                X = self.X_train
                y = self.y_train
            elif dataset == "Test":
                X = self.X_test
                y = self.y_test
            elif dataset == "Val":
                X = self.X_val
                y = self.y_val
            else:
                raise ValueError("dataset must be either Val, Train or Test")
            y_pred_proba_val = utils.batch_predict(self.model,X, is_proba = True)
            y_pred_val = utils.batch_predict(self.model,X, is_proba = False)
            for metric in ["roc", "AP", "BrierScore", "f1", "precision", "recall", "acc"]:
                score = self.compute_metric(y_pred_proba_val,
                                            y_pred_val,
                                            y, 
                                            mode=metric)
                print(f"{metric} : {score}")

            print(f"")


    def compute_metric(self, 
                       y_pred_proba_val: np.ndarray,
                       y_pred_val: np.ndarray, 
                       y: np.ndarray, 
                       mode = "roc")->float:
        """ Compute a metric for the model

        Args:
            y_pred_proba_val (np.ndarray): model prediction probabilities
            y_pred_val (np.ndarray): model prediction
            y (np.ndarray): true labels
            mode (str, optional): metric to compute. Defaults to "roc".

        Returns:
            float: score of the metric
        """

        if mode == "roc":
            score = roc_auc_score(y, y_pred_proba_val)
        elif mode == "AP":
            score = average_precision_score(y, y_pred_proba_val)
        elif mode == "BrierScore":
            score = brier_score_loss(y, y_pred_proba_val)
        elif mode == "acc":
            score = accuracy_score(y, y_pred_val)
        elif mode == "bacc":
            score = balanced_accuracy_score(y, y_pred_val)
        elif mode == "f1":
            score = f1_score(y, y_pred_val)
        elif mode == "precision":
            score = precision_score(y, y_pred_val)
        elif mode == "recall":
            score = recall_score(y, y_pred_val)
        else:
            raise ValueError("mode must be either roc, acc, f1, precision or recall")
        return score  
    

    def process_full_predictions(self, 
                                 dataset="Full_Test"):
        """Computes the prediction from the succession of both Model 1 and Model 2.
        """
        if dataset == "Test":
            X = self.X_test[:,self.selected_features]
            y = self.y_test
            model1_score = self.X_test[:,-1] #M1_score column index
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
        rejected_predictions = model1_score < 0.5
        y_pred_proba_val = utils.batch_predict(self.model,X, is_proba = True)
        y_pred_proba_val[rejected_predictions] = 0
        y_pred_val = utils.batch_predict(self.model,X, is_proba = False)
        y_pred_val[rejected_predictions] = 0
        for eval_metric in ["roc", "AP", "BrierScore", "f1", "precision", "recall", "acc"]:
            score = self.compute_metric(y_pred_proba_val,
                                         y_pred_val,
                                         y, 
                                         rejected_predictions=rejected_predictions, 
                                         mode=eval_metric)
            print(f"{eval_metric} : {score}")

    def auc_graph(self, 
                  dataset:List[str] = ["Train"], 
                  metrics:List[str] = ["auto"],
                  key_thresholds:List[float] = [0.001,0.005,0.01,0.05,0.1, 0.2,0.3, 0.5, 0.9]):
        """ Plot the ROC curve and the AUC for the train or test dataset

        Args:
            dataset (List[str], optional): dataset to plot. Defaults to ["Train"].
            metrics (List[str], optional): metrics to plot. Defaults to "auto".
            key_thresholds (List[float], optional): Key thresholds to plot for the AUC-ROC

        """
        # plot the ROC curve and the AUC for the train or test dataset
        if len(dataset)==0:
            datasets = ["Train", "Test", "Val"]
        else:
            datasets = dataset
        if len(metrics)==0:
            metrics = [self.eval_metric_GA]
        else:
            metrics = [metrics]

        for dataset in datasets:
            for metric in metrics:
                if dataset == "Train":
                    print("Train")
                    X = self.X_train
                    y = self.y_train
                elif dataset == "Test":
                    print("Test")
                    X = self.X_test
                    y = self.y_test
                elif dataset == "Val":
                    print("Val")
                    X = self.X_val
                    y = self.y_val
                else:
                    raise ValueError("dataset must be either Val, Train or Test")
                
                y_pred_proba_test = self.model.predict_proba(X)[:, 1]

                plt.figure(figsize=(8, 6))
                if metric == "roc_auc":
                    fpr, tpr, thresholds = roc_curve(y, y_pred_proba_test)
                    auc = roc_auc_score(y, y_pred_proba_test)
                    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
                    plt.plot(fpr, tpr, color='darkorange', lw=2, label='ROC curve (area = %0.3f)' % auc)
                    plt.xlabel(f'False Positive Rate {dataset}')
                    plt.ylabel(f'True Positive Rate {dataset}')
                    plt.title(f'Receiver Operating Characteristic {dataset}')

                elif  metric == "average_precision":
                    precision, recall, thresholds = precision_recall_curve(y, y_pred_proba_test)
                    auc =  average_precision_score(y, y_pred_proba_test)
                    plt.plot([0, 1], [1, 0], color='navy', lw=2, linestyle='--')
                    plt.plot(precision, recall, color='darkorange', lw=2, label='PR curve (area = %0.3f)' % auc)
                    plt.xlabel(f'Recall {dataset}')
                    plt.ylabel(f'Precision {dataset}')
                    plt.title(f'Precision Recall Curve {dataset}')


                if metric == "roc_auc":
                    for thresh in key_thresholds:
                        idx = np.where(thresholds >= thresh)[0][-1]
                        plt.plot(fpr[idx], tpr[idx], 'o', markersize=10, label=f'Threshold {thresh:.3f}')
            

                plt.xlim([0.0, 1.0])
                plt.ylim([0.0, 1.0])
                plt.legend(loc="lower right")
                plt.show()

                print(f"AUC: {auc}")
    
    def save_to_disk(self, 
                     name:str = ""):
        """Save the model to disk

        Args:
            name (bool, optional): Name of the model.
        """
        if name == "":
            name = self.obj_name

        with open(self.obj_name+'.pkl', 'wb') as file: 
            pickle.dump(self, file)



    def load_from_disk(self, 
                       name:str)->object:
        """ Load the model from disk

        Args:
            name (str): Name of the model.

        Returns:
            object: model
        """
        with open(name+'.pkl', 'rb') as file:  
            loaded = pickle.load(file)

        self = loaded

        return loaded

    def GA_optimisation(self, 
                        ngen:int = 40, 
                        pop:int = 40, 
                        best_individuals:List[List] = None):
        """Launch the GA optimisation

        Args:
            ngen (int, optional): Number of generations. Defaults to 40.
            pop (int, optional): Number of individuals in the population. Defaults to 40.
            best_individuals (List[List], optional): List of individuals to add to the population. Defaults to None.
        """
        creator.create("FitnessMax", base.Fitness, weights=(1.0,))  
        creator.create("Individual", list, fitness=creator.FitnessMax)
        random.seed(self.seed)

        toolbox = base.Toolbox()
        for i in range(self.nb_feature):  
            toolbox.register(f"attr_temp_index{i}", random.choice, [True, False])

        toolbox.register("attr_derivative", random.choice, [True, False])
        toolbox.register("attr_n_estimators", random.randint, 10, 300) 
        toolbox.register("attr_max_depth", random.randint, 5, 20)
        toolbox.register("attr_histodept", random.randint, 3, 90)
        toolbox.register("attr_histostrt", random.randint, 2, 30)

        attributes = [toolbox.__getattribute__(f'attr_temp_index{i}') for i in range(self.nb_feature)] + \
                    [toolbox.attr_derivative] + \
                    [toolbox.attr_n_estimators, toolbox.attr_max_depth, toolbox.attr_histodept, toolbox.attr_histostrt]

        toolbox.register("individual", tools.initCycle, creator.Individual, tuple(attributes), n=1)
        toolbox.register("population", tools.initRepeat, list, toolbox.individual)
        toolbox.register("evaluate", self.evalModel)
        toolbox.register("mate", tools.cxUniform, indpb=0.4)
        nbBooleans = self.nb_feature +1
        toolbox.register("mutate", tools.mutUniformInt, 
                        low=[False]*nbBooleans + [10, 5, 2, 2], 
                        up=[True]*nbBooleans + [350, 20, 80, 30], 
                        indpb=0.35)
        toolbox.register("select", tools.selTournament, tournsize=4)

        stats = tools.Statistics(key=lambda ind: ind.fitness.values)
        stats.register("max", np.max) 

        population = toolbox.population(n=pop)

        if best_individuals is not None:
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
        with open("best_fitness.txt", "w") as file:
            file.write(f"Best Fitness: {best_ind}\n")