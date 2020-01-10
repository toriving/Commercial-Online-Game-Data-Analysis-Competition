# [Commercial Online Game Data Analysis Competition @ GIST](https://gist-leaderboard.com/competition/1)
Design for Online Game Churn Prediction Model for considering residual value using the Commercial Online Game Data

## Team information
Team : Ironman (¹Úµ¿ÁÖ)  
4th place  

## NER model architecture

This is a simple network with several fully connected layers.  
You can see model's detail architecture in [create_model.py](https://github.com/toriving/Commercial-Online-Game-Data-Analysis-Competition/model/create_model.py)


## Requirements

python==3.6  
tensorflow-gpu==1.13.0  
pandas>=0.23.4  
numpy>=1.15.4  
scikit-learn>=0.19.1  


## Run
__First, unzip all the zip files in folder raw.__
```
>> cd Ironman  
>> cd preprocess  
>> python preprocess.py  
>> cd ..  
```

__If you want to train model__  
```
>> cd model  
>> python create_model.py  
```

__Predict__
```
>> cd predict  
>> python predict.py  
```
