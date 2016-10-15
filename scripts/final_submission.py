#
# File: final_submission.py
#
# Script for final submission to the 
# Taxi Destination Competition at kaggle.com 
#
# @author mchasse: Matthew Chasse
#
####################################################################

import json
import zipfile
import pandas as pd
import numpy as np
import csv as csv
import matplotlib.pyplot as plt
import datetime


#return a text parser for the large file, with chunk size "sz"
def load_train(sz=100):
    zf = zipfile.ZipFile("../data/train.csv.zip")
    return pd.read_csv(zf.open('train.csv'), chunksize=sz, iterator=True, converters={'POLYLINE': lambda x: json.loads(x)})



#remove rows with empty POLYLINE field
def rem_empty_polyline(X):
    empty_rows=[]
    for j in range(len(X['POLYLINE'])):      
        entry=X['POLYLINE'].values[j]
        if(entry==[]):
                  empty_rows.append(j)
    return X.drop(X.index[empty_rows])
    

#remove rows with incomplete GPS data (only ten cases)
def rem_missing(X):
    empty_rows=[]
    for j in range(len(X['MISSING_DATA'])):      
        entry=X['MISSING_DATA'].values[j]
        if(entry=="True"):
            empty_rows.append(j)
    return X.drop(X.index[empty_rows])    
    
    

#add the last latitude and longitude from the POLYLINE field to main dataframe X and return it 
def lat_long_last(X):

    latitudes=[]
    longitudes=[]
    
    for j in range(len(X['POLYLINE'])):      
        entry=X['POLYLINE'].values[j]
        if(len(entry)==0):
            latitudes.append(-999)
            longitudes.append(-999)
        else:
            last=entry[-1]           
            latitudes.append(last[0])
            longitudes.append(last[1])
            
    X['LAST_LAT']=longitudes 
    X['LAST_LON']=latitudes
    
    return X



#add the first latitude and longitude from the POLYLINE field to main dataframe X and return it 
def lat_long_first(X):
    
    latitudes=[]
    longitudes=[]
    
    for j in range(len(X['POLYLINE'])):      
        entry=X['POLYLINE'].values[j]
        if(len(entry)==0):
            latitudes.append(-999)
            longitudes.append(-999)
        else:
            last=entry[0]           
            latitudes.append(last[0])
            longitudes.append(last[1])
            
    X['FIRST_LAT']=longitudes 
    X['FIRST_LON']=latitudes
    
    return X



# add the second to last latitude and longitude from the POLYLINE field to main dataframe X and return it 
def lat_long_2ndToLast(X):
    
    latitudes=[]
    longitudes=[]
    
    for j in range(len(X['POLYLINE'])):      
        entry=X['POLYLINE'].values[j]
        if(len(entry)==0):
            latitudes.append(-999)
            longitudes.append(-999)
        elif(len(entry)==1):
            last=entry[-1]           
            latitudes.append(last[0])
            longitudes.append(last[1])            
        else:
            last=entry[-2]           
            latitudes.append(last[0])
            longitudes.append(last[1])
            
    X['S2L_LAT']=longitudes 
    X['S2L_LON']=latitudes
    
    return X



# truncate the polyline data if the length is >1 using a flat distribution
def trunc_polyline(X):
    
    for j in range(len(X['POLYLINE'])):      
        entry=X['POLYLINE'].values[j]
        m=len(entry)
        if m>1:
          cut=np.random.randint(1,m+1)
          X['POLYLINE'].values[j]=entry[:cut]  

    return X


#
# Evaluation metric -- appears to be off by factor of 2!? no time, figure out later
#
#  phi_i are latitudes and lambda_j are longitudes
#
d_2_rad=np.pi/180.0

#compute haversine distance between two coordinates (phi_1,lambda_1) and (phi_2,lambda_2)
def haversine(phi_1,lambda_1,phi_2,lambda_2):
    r=6371  #kilometers
    #r=3959 #miles
    a= np.sin(d_2_rad*(phi_2-phi_1))**2+np.cos(d_2_rad*phi_1)*np.cos(d_2_rad*phi_2)*np.sin(d_2_rad*(lambda_2-lambda_1))**2
    return 2*r*np.arctan(np.sqrt(a/(1-a)))



#compute the mean haversine distance between -- not safe, make sure all array dimensions are the same
def mean_haversine(phi_1s,lambda_1s,phi_2s,lambda_2s):

    total=0
    m=len(phi_1s)
    for j in range(m):
        #print haversine(phi_1s[j],lambda_1s[j],phi_2s[j],lambda_2s[j])
        total+=haversine(phi_1s[j],lambda_1s[j],phi_2s[j],lambda_2s[j])

    return total/m
        

    

# add the last distance delta from the POLYLINE field to main dataframe X and return it 
def lastDelta(X):
    
    deltas=[]
    
    for j in range(len(X['POLYLINE'])):      
        entry=X['POLYLINE'].values[j]
        if(len(entry)==0):  #this should not happen if length zero paths are excluded
            deltas.append(-999)
        elif(len(entry)==1):          
            deltas.append(0)            
        else:
            last=entry[-1]
            Sec2Last=entry[-2]
            deltas.append(haversine(last[0],last[1],Sec2Last[0],Sec2Last[1]))
            
            
    X['LDELTA']=deltas 
    
    return X


# add the number of points in polyline
def num_points(X):
       
    n_points=[]
    
    for j in range(len(X['POLYLINE'])):      
        entry=X['POLYLINE'].values[j]
        n_points.append(len(entry))
                        
    X['NPOINTS']=n_points 
    
    return X
    

# add the hour of the day 
def hour(X):
    
    hour=[]
    
    for j in range(len(X['TIMESTAMP'])):
        entry=X['TIMESTAMP'].values[j]
        value=datetime.datetime.fromtimestamp(entry)
        daytime_in_hours=float(value.hour)+float(value.minute)/60.0
        hour.append(daytime_in_hours)
        
    X['HOUR']=hour
    
    return X

# Fuzzy version of hour of day that gives better fit
def hour2(X):
    
    hour=[]
    
    for j in range(len(X['TIMESTAMP'])):
        entry=X['TIMESTAMP'].values[j]
        value=datetime.datetime.fromtimestamp(entry)
        daytime_in_hours=float(value.hour)+float(value.minute)/60.0
        if( (daytime_in_hours>5) and (daytime_in_hours<10)):
            hour.append('M')
        elif( (daytime_in_hours>=10) and (daytime_in_hours<18)):
            hour.append('D')
        else:
            hour.append('E')
            
    X['HOUR']=hour
    X['HOUR']=pd.factorize(X['HOUR'])[0]
    
    return X


#The vector between the last and second to last data point in the POLYLINE data
def last_delta_vec(X):
    
    x_deltas=[]
    y_deltas=[]
    
    for j in range(len(X['POLYLINE'])):      
        entry=X['POLYLINE'].values[j]
        if(len(entry)==0):  #this should not happen if length zero paths are excluded
            x_deltas.append(0)
            y_deltas.append(0)
        elif(len(entry)==1):          
            x_deltas.append(0)    
            y_deltas.append(0)
        else:
            last=entry[-1]
            first=entry[-2]
            x_deltas.append((last[0]-first[0]))
            y_deltas.append((last[1]-first[1]))
            
    X['LDLAT']=x_deltas 
    X['LDLON']=y_deltas
        
    return X


# truncate the polyline -- fancy version, designed to match the distribution of lengths in the test data
def trunc_polyline2(X):
    
    for j in range(len(X['POLYLINE'])):      
        entry=X['POLYLINE'].values[j]
        m=len(entry)

        if m>65:    
            a=np.random.uniform()
            z=int(1/(a+1.0/m))+1
            cut=m-z
        else:
            cut=min(np.random.geometric(float(1/55.0)),m)
            
            
        X['POLYLINE'].values[j]=entry[:cut] 
            
    return X

# final fit: use only first point, last point, last_delta_vec, DAY_TYPE (as determined by 50/50 CV error)
#

from sklearn import ensemble 
#from sklearn.neighbors import KNeighborsRegressor 


train_parser=load_train(1000)



target=pd.DataFrame()
train_predictors=pd.DataFrame()

first_chunk=True
k=0
cutoff=50
clf=[] 

for chunk in train_parser:
    chunk=rem_empty_polyline(chunk)    
    chunk=rem_missing(chunk)
    chunk=lat_long_last(chunk)

    if(first_chunk):
        target=chunk[['LAST_LAT','LAST_LON']]
    else:
        target=pd.concat([target,chunk[['LAST_LAT','LAST_LON']]])
        
    #
    # MODIFY by randomly truncating POLYLINE for training
    #  and recompute predictor features
    #
    chunk=trunc_polyline(chunk)
    chunk=lat_long_last(chunk)
    chunk=lat_long_first(chunk)
    chunk=last_delta_vec(chunk)
    chunk=hour2(chunk)
    chunk=chunk[['LAST_LAT','LAST_LON','FIRST_LAT','FIRST_LON','LDLON','LDLAT','DAY_TYPE','HOUR']]
    chunk['DAY_TYPE']=pd.factorize(chunk['DAY_TYPE'])[0]
    if(first_chunk):
        train_predictors=chunk
        first_chunk=False
    else:
        train_predictors=pd.concat([train_predictors,chunk])
    
    k+=1
    if(k==cutoff):
        clf.append(ensemble.RandomForestRegressor(n_estimators=15)) 
        m=len(clf)-1
        clf[m].fit(train_predictors.values,target.values)
        first_chunk=True
        k=0


## 
## Compute ~50/50 CV error
##
'''
l=0
m=len(clf)
cut=round(float(m)/2)
cv_errors=[]
cv_chunks=50

# re-initialize the train data parser
train_parser=load_train(1000)


if(m<2):
    print "only one segment, cannot compute CV error"
else:
    
    for chunk in train_parser:
        chunk=rem_empty_polyline(chunk)    
        chunk=rem_missing(chunk)
        chunk=lat_long_last(chunk)
        chunk=lat_long_first(chunk)
        
        target=chunk[['LAST_LAT','LAST_LON']]

        chunk=trunc_polyline(chunk)
        chunk=lat_long_last(chunk)
        chunk=lat_long_first(chunk)
        chunk=last_delta_vec(chunk)  
        chunk=hour2(chunk)
        chunk=chunk[['TRIP_ID','LAST_LAT','LAST_LON','FIRST_LAT','FIRST_LON','LDLON','LDLAT','DAY_TYPE','HOUR']]
        chunk['DAY_TYPE']=pd.factorize(chunk['DAY_TYPE'])[0]
        
        chunks_models=cut*cutoff
        if (l>=chunks_models) and (l<chunks_models+cv_chunks) :
         
            predict=0
            for j in range(1,int(cut)):  
                predict +=clf[j].predict(chunk.values[:,1:])
            
            predict = predict/float(cut-1)
                
            mean_dist=mean_haversine( predict[:,0],predict[:,1],target['LAST_LAT'].values,target['LAST_LON'].values)
            cv_errors.append(mean_dist)
        l+=1
        
    #print "cv_errors: " + str(cv_errors)
    print "avg cv_error: " + str(np.array(cv_errors).mean())
'''
#
# END of CV calculation
#
        
        
zft = zipfile.ZipFile("../data/test.csv.zip")
test = pd.read_csv(zft.open('test.csv'), chunksize=100, iterator=True, converters={'POLYLINE': lambda x: json.loads(x)}) 

first_chunk=True

for chunk in test:
    chunk=lat_long_last(chunk)
    chunk=lat_long_first(chunk)
    chunk=last_delta_vec(chunk)  
    chunk=hour2(chunk)
    chunk['DAY_TYPE']=pd.factorize(chunk['DAY_TYPE'])[0]
    chunk=chunk[['TRIP_ID','LAST_LAT','LAST_LON','FIRST_LAT','FIRST_LON','LDLON','LDLAT','DAY_TYPE','HOUR']]
    
    
    m=len(clf)
    predict=clf[0].predict(chunk.values[:,1:])
    for j in range(1,m):  
        predict +=clf[j].predict(chunk.values[:,1:])
    
    predict = predict/float(m)
    
    submit_df=pd.DataFrame({'TRIP_ID':chunk['TRIP_ID'], 'LONGITUDE':predict[:,1], 
                            'LATITUDE':predict[:,0]})
    if first_chunk:
        submit_df.to_csv('../data/final1_forest.csv', mode='w', columns=['TRIP_ID','LATITUDE','LONGITUDE'], index=False)
        first_chunk=False
    else:
        submit_df.to_csv('../data/final1_forest.csv', mode='a', columns=['TRIP_ID','LATITUDE','LONGITUDE'], index=False, header=False)
    
        
