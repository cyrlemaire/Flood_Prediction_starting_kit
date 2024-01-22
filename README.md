# Flood Map Prediction Challenge: Starting kit

This reposirory contains everything you need to start the Flood Map Prediction Challenge starting on 22nd January 2023. The challenge is hosted on Codabench at this adress:
https://www.codabench.org/competitions/1567/

If you haven't already, you can create an account and register!

## What is the purpose of this starting kit?

This repository allows every participant to run a complete pipeline to process the data, train a model, and create a submission file to predict the flood risk maps on the secret test set. After following the instruction on this Readme page, you will obtain a .csv file that you can submit on the Codabench challenge page to appear on the leaderboard.

The model proposed on this starting kit is our baseline model. Your objective is to improve it or create a completely different model to beat our score and get the highest ranking possible!

For all the information on the challenge and the submission process, please refer to challenge page: https://www.codabench.org/competitions/1567/

If you have any question, or if you experience any difficulty, we recomand that you post on the challenge forum: https://www.codabench.org/forums/1486/. You can also contact us at hackathon@capgemini.com

## Starting kit content

This kit contains 3 notebooks presenting the 3 steps of our baseline pipeline. We first process the raw data to prepare it for modeling and we train 2 sequential models. The idea behind this method is to use a first model with a low resolution (alligned with the ERA5 climate data - tiles of 31 sqkm ) that uses the climate variables to predict where and when floods are likely to happen. The output of the first models allow the second model (the actual high resolution flood map predictor) to only focus on the areas likely to be flooded.

This approach allows us to be frugal by not training a high resolution model on the entire dataset (climate + geodata) at once. 


## Get started

Before running the notebooks, create a python environement of your choice with python 3.8.18. You can then install the requirements by running:
```
pip install -r requirements.txt
```

You can then open the 3 notebooks and run them in order:
* 1_datapreparator_minicube.ipynb : Download the data cubes, compute new features, input missing data and preprocess data into xarray at two different reslution for the two models
* 2_baseline_model01.ipynb: Train the first model using climate data. This model uses an agregated label at the ERA5 resolution (tiles of 31 sqkm) and predicts a M1 score for each tile. This score quantifies the amount of flood pixel present in the tile for the given week. The M1 score is used as an output feature for the second model. The model is then evaluated and heat maps are created to visualize the results
* 3_baseline_model02.ipynb: Train the second model using the actual flood maps. This model is trained on a subset of the dataset chosen using a threshold on the M1 score (*min_score_model1* attribute of the BaseLineModel class). Once trained, the model is evaluated on a subset of the data. 
* 4_submission.ipynb: generate the submission file for the codabench leaderboard from a netcdf prediction file.

In the current state, the model is trained and tested on subsets of the training data. To obtain the full baseline performance, you should train the model on the full provided dataset.



