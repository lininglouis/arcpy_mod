from scipy.stats.stats import pearsonr
import sys
import csv
import numpy as np
import pandas as pd
import os
import shutil
import re
import errno
from matplotlib import pyplot as plt
from datetime import datetime as dt

import random
 
import arcpy
arcpy.env.overwriteOutput = False
 
import arcpy
from arcpy import env
from arcpy.sa import *
import arcpy, arcinfo
 

import arcpy
import fileinput
import string
import os
import glob
import time
import statsmodels.api as sm




    
def myJoin(dirName, filename):
    
    fullPath = os.path.join(dirName, filename).replace('\\','/')
    return fullPath

    
def multipleJoin(*pathElement):
    path = ""
    for ele in pathElement:    
        path = os.path.join(path, ele)
    return path
            
    
    
def createBuffer(inputFeature, outputBuffer, distanceField, dissolveType):
 
    sideType = "FULL"
    endType = "ROUND"
    arcpy.Buffer_analysis(inputFeature, outputBuffer, distanceField, sideType, endType, dissolveType)

    
def getPointForRasterArrayFunction(point, window_actual_length_meters):
    
    correctX = point.X - window_actual_length_meters/2.0
    correctY = point.Y - window_actual_length_meters/2.0
    return arcpy.Point(correctX, correctY)

    
def gaussian(x, mean, sigma):
    return ( np.exp(-np.power(x - mean, 2.) / (2 * np.power(sigma, 2.)))  ) /  (sigma*np.sqrt(2*np.pi))
 

def getGroupCodeToCDLCode(df_mapping, targetGroupMethod, groupNameToGroupCode):

    # add GroupCode
    classCodeToGroupCode = df_mapping[targetGroupMethod].apply(lambda groupName: groupNameToGroupCode[groupName])
    full_map_dataframe = df_mapping[['Formal_CDL_NLCD_Code', targetGroupMethod]]
    full_map_dataframe.loc[:, 'targetGroupCode'] = classCodeToGroupCode

    map_df = full_map_dataframe[:]
    map_df.index = full_map_dataframe['targetGroupCode']   
    return  map_df['Formal_CDL_NLCD_Code']


def getCDLCodeToGroupCode(df_mapping, targetGroupMethod, groupNameToGroupCode):

    classCodeToGroupCode = df_mapping[targetGroupMethod].apply(lambda groupName: groupNameToGroupCode[groupName])
    full_map_dataframe = df_mapping[['Formal_CDL_NLCD_Code', targetGroupMethod]]
    full_map_dataframe.loc[:, 'targetGroupCode'] = classCodeToGroupCode
    
    map_df = full_map_dataframe[:]
    map_df.index = full_map_dataframe['Formal_CDL_NLCD_Code']   
    return  map_df['targetGroupCode']
 

def findIfCodeInCodeList(cdlCode, r):
 
    if cdlCode in r:
        return 1
    else:
        return 0
    
 
        
class Configurer:
    
    def __init__(self, targetGroupMethod, weightType, weightKernelLengthMeters):
        
        map_path = r'D:\Bee\BeeMetadata\CDL_NLCD_merge_ranklist_correct.csv'
        self.df_map = pd.read_csv(map_path)
    
        self.targetGroupMethod = targetGroupMethod
        
        self.groupNameList = list(set(self.df_map[targetGroupMethod]))
        self.groupCodeList = np.arange(len(self.groupNameList))
    
        self.groupNameToGroupCode = { className : classCode for classCode, className in zip(self.groupCodeList, self.groupNameList)  }
        self.groupCodeToGroupName = { className : classCode for classCode, className in zip(self.groupCodeList, self.groupNameList)  }
    
        self.groupCodeToCDLCode = getGroupCodeToCDLCode(self.df_map, targetGroupMethod, self.groupNameToGroupCode)
        self.cdlCodeToGroupCode = getCDLCodeToGroupCode(self.df_map, targetGroupMethod, self.groupNameToGroupCode)        
        
 
        self.weightKernelLengthMeters = weightKernelLengthMeters
        self.weightType =  weightType
        self.yearResolutionDict = { 2006:56, 2007:56, 2008:56, 2009:56, 2010:30, 2011:30, 2012:30, 2013:30, 2014:30, 2015:30, 2016:30}

 
 
 
    
    
    

def getGroupedLayersFromCdlParcel(cdlMatrix, Configurer):
 
    layerArrs = []
    for groupName in Configurer.groupNameList:
        
        
        if groupName != 'other':

            groupCode= Configurer.groupNameToGroupCode[groupName]
            cdlCode = Configurer.groupCodeToCDLCode[groupCode]
    
            cdlCodeList = []     
            if type(cdlCode) is np.int64:
                cdlCodeList.append(cdlCode)
            if type(cdlCode) is pd.core.series.Series:
                cdlCodeList = list(cdlCode)
                
            #for code in codeList:
            
            whole_bool_arr = np.zeros(cdlMatrix.shape, dtype=bool)
            for cdlCode in cdlCodeList:
                whole_bool_arr = np.bitwise_or(whole_bool_arr, cdlMatrix == cdlCode)
                
            whole_bool_arr = whole_bool_arr.astype(int)

      
            #getBinaryArrByCodeList_vec = np.vectorize(findIfCodeInCodeList, excluded='r')
            #getBinaryArrByCodeList_vec(cdl_parcel, []
            #binary_arr = getBinaryArrByCodeList_vec(cdlMatrix, r = cdlCodeList)       
            
#            print cdlCodeList, t()-t1
            layerArrs.append(whole_bool_arr)
            
    return layerArrs
 
 
    
    
def getWeightMatrix_Gaussian(windowLength, sigma = None, center=None):
    """ Make a square gaussian kernel.
    size is the length of a side of the square
    fwhm is full-width-half-maximum, which
    can be thought of as an effective radius.
    """

    x = np.arange(0, windowLength, 1, float)
    y = x.reshape(-1, 1)
    
    x0 = y0 = windowLength // 2

    x_dist = x-x0
    y_dist = y-y0
    true_dist = np.sqrt((x_dist**2 + (y_dist)**2))

    if sigma is None:
        mat = gaussian(true_dist, 0, sigma = windowLength//4)
        return mat/mat.sum()
        
    else:
        mat = gaussian(true_dist, 0, sigma = sigma)
        return mat/mat.sum()
        
        
        
def getExponentialDecay(distanceMatrix):
    
    lamda = 0.0009
    N0 = 1
    return  N0 * np.exp( -  lamda * distanceMatrix)
 


        
def getWeightMatrix_Exponential(windowLength):
    """ Make a square gaussian kernel.
    size is the length of a side of the square
    fwhm is full-width-half-maximum, which
    can be thought of as an effective radius.
    """

    x = np.arange(0, windowLength, 1, float)
    y = x.reshape(-1, 1)
    
    x0 = y0 = windowLength // 2
    x_dist = x-x0
    y_dist = y-y0
    true_dist = np.sqrt((x_dist**2 + (y_dist)**2))
    true_dist_in_meter = true_dist * 30

    decayedMat = getExponentialDecay(true_dist_in_meter)
    return  decayedMat 
        

        
        
            
def transform_Point_2_RasterArray(pointTuple, raster_path, windowPixelSize = 100, resolution=30):
 
    exactPoint = arcpy.Point(pointTuple[0], pointTuple[1])
    cdlRaster_NAD = raster_path
 
    window_actual_length_meters = windowPixelSize * resolution
    correctPoint = getPointForRasterArrayFunction(exactPoint,  window_actual_length_meters) 
    rasterParcel = arcpy.arcpy.RasterToNumPyArray(cdlRaster_NAD, correctPoint, windowPixelSize, windowPixelSize)
    
    return rasterParcel
    
 
    

    
 

def transform_CDLParcel_2_WeightedLandcoverSeries(cdlParcel, my_Configurer, weightMatrix):
    
    if cdlParcel.shape[0] != weightMatrix.shape[0] :
        raise ValueError('Please make sure the weightMatrix and CDL parcel have the same size')
    
    layersBinaryArrList =  getGroupedLayersFromCdlParcel(cdlParcel, my_Configurer)   
    layerNameWeightedSum = {}
    groupNameList = [groupName for groupName in my_Configurer.groupNameList if groupName != 'other']
    for groupName, binaryArrOneLayer in zip(groupNameList, layersBinaryArrList):
        weightedSum = np.sum (binaryArrOneLayer *  weightMatrix )
        layerNameWeightedSum[groupName] = weightedSum    

    series_groupedAcres = pd.Series(layerNameWeightedSum)
    return    series_groupedAcres
    


def getBeeHivesInfoByYear(year):
                          
    df = pd.read_csv(r'D:\Bee\BeeSites\Registered Apiary Sites Location\vacancy.csv')
    df_present = df.filter(regex='SITE|Presence.*') 

    df_2013_related = df.filter(regex='SITE|.*{}.*'.format(year))
    df_2013_present  = df_2013_related.loc[df_2013_related['Presence{}'.format(year)]==1.0]
    target_columns = ['SITE', 'PalletMin{}'.format(year), 'ColonyMin{}'.format(year)]
    df_2013_pallete = df_2013_present.loc[df_2013_present['PalletMin{}'.format(year)]>0][target_columns]
    df_2013_colony  = df_2013_present.loc[df_2013_present['ColonyMin{}'.format(year)]>0][target_columns]
    
    df_2013_pallete.to_csv('df_{}_pallete.csv'.format(year)) 
    df_2013_colony.to_csv('df_{}_colony.csv'.format(year)) 
    df_2013_present[['SITE', 'Presence{}'.format(year)]].to_csv('df_{}_present.csv'.format(year))
    
    return df_2013_present[['SITE', 'Presence{}'.format(year)]],  df_2013_pallete, df_2013_colony
    
                            
def getWeightedInfoByXY(pointRecord, year, my_Configurer):
    

    if type(pointRecord) is pd.Series:        
        pointTuple =  pointRecord[['X_coor', 'Y_coor']].values
    else:
        pointTuple = pointRecord
 
    yearResolution = my_Configurer.yearResolutionDict[year] 
    windowPixelSize = int(float(my_Configurer.weightKernelLengthMeters)/float(yearResolution))
    
    
    weightMatrix = getWeightKernel(windowLength = windowPixelSize, kernelType = my_Configurer.weightType)   #plain
    #print 'year',year, 'yearResolution', yearResolution, 'windowPixelSize', windowPixelSize, 'weightMatrix', weightMatrix  
    
    cdlParcel = transform_Point_2_RasterArray(pointTuple, getCDLDataPath(year), windowPixelSize = windowPixelSize, resolution=yearResolution)

    #layersBinaryArrList =  getGroupedLayersFromCdlParcel(cdlParcel, my_Configurer)   
    series_groupedLandcoverAcre= transform_CDLParcel_2_WeightedLandcoverSeries(cdlParcel, my_Configurer, weightMatrix)
    #print (series_groupedLandcoverAcre *  (my_Configurer.yearResolutionDict[year] ** 2)).sum(),  cdlParcel.shape
    
    return series_groupedLandcoverAcre *  (my_Configurer.yearResolutionDict[year] ** 2)                            



def cdl_Reprojection(cdl_raster_path, cdl_NAD_output_path):

    spatial_ref = arcpy.SpatialReference(26914)    
    arcpy.ProjectRaster_management(cdl_raster_path, cdl_NAD_output_path, spatial_ref, 'NEAREST', 30)

    
    
# CDL information
#cdl_Reprojection(r'D:\Bee\bee_binary_data\binaryLayerGeneration\CDL_RAW\CDL_NDSD_20091.tif', 
#                 r'D:\Bee\bee_binary_data\binaryLayerGeneration\CDL_NAD\CDL_NDSD_20091_NAD.tif')

    
    
def getPosPointDataByYear(shp_pos_points, year, ifRemoveNearest = True):
 
    df_record = pd.read_csv(r'D:\Bee\BeeSites\Registered Apiary Sites Location\vacancy.csv')                            
    df_joint_shp_record = selectPresent.getJoin_shp_record(shp_pos_points, df_record)
    df_present_year, df_colony_year, df_pallate_year = selectPresent.getBeeHivesInfoByYear(df_joint_shp_record, year)
    
    if ifRemoveNearest:
        df_present_year = selectPresent.removeNearest(df_present_year, X_columnName = 'X_coor', y_columnName='Y_coor', distanceThresh = 3200)

    return df_present_year
    
    
    
def getGoodPointsListFromClintByYear(shp_pos_points, year, ifRemoveNearest = True):
#Good Point 
    df_present_year = getPosPointDataByYear(shp_pos_points, year, ifRemoveNearest = ifRemoveNearest)
    goodPoint_list = df_present_year[['X_coor', 'Y_coor']].values

    return df_present_year, goodPoint_list


def getBadPointsList(shp_points_neg):
        
    #Bad Point   
    spatial_ref = arcpy.SpatialReference(26914)  
    with arcpy.da.SearchCursor(shp_points_neg, field_names = ["SHAPE@XY"], spatial_reference = spatial_ref)  as cursor:
        badPoint_list = [row[0] for row in cursor ]
    
    random.seed(42)
    random.shuffle(badPoint_list)
    
    return badPoint_list


def getPointsInfoFromShapefile(shapefile_path, ifRemoveNearest):
    
    spatial_ref = arcpy.SpatialReference(26914)  
    with arcpy.da.SearchCursor(shapefile_path, field_names = ["SHAPE@XY"], spatial_reference = spatial_ref)  as cursor:
        points_List = [row[0] for row in cursor ]

    points_Arr = np.array(points_List)
    if ifRemoveNearest:
        df = selectPresent.removeNearestPoints(points_Arr, distanceThresh = 3200)
        return df
    else:
        return points_Arr
    

    
def getWeightKernel(windowLength = 100, kernelType= 'gaussian'):
    
    if(kernelType == 'gaussian'):  
        weightMatrix = getWeightMatrix_Gaussian(windowLength = windowLength, sigma=16)  
    
    if(kernelType == 'plain'):  
        weightMatrix = np.ones((windowLength, windowLength)) 
        
    if(kernelType =='exponential'):
        weightMatrix = getWeightMatrix_Exponential(windowLength = windowLength)  

    #print kernelType, '----------------------------------' 
    return weightMatrix.copy()

 
def getBadPointListFromBottomDensity(goodPointsNum):
    
    bottomPointsPath = r'D:\Bee\bee_binary_data\2013_tabulate\binary_data_generate\bottomDensity\bottomPoints_save\bottom_point.txt'
    bottomPointsArray = np.loadtxt(bottomPointsPath)
    badPoint_list = bottomPointsArray[:goodPointsNum, :2] 
    return badPoint_list
    

def if_in_May_Sep(tm):
    if(tm <= dt.strptime('9/30/2011', '%m/%d/%Y')  and tm >= dt.strptime('5/1/2011', '%m/%d/%Y')):
        return True
    else:
        return False
        

def func(row, year):

    from datetime import datetime as dt
    if(np.isnan(row['Presence{}'.format(year)])):
        return 1
    else:
        #print row['{}T1'.format(year)], row['{}T2'.format(year)]
        pass
    
    time_T1 = dt.strptime(row['{0}T1'.format(year)], '%m/%d/%Y')
    time_T2 = dt.strptime(row['{0}T2'.format(year)], '%m/%d/%Y')
    if (if_in_May_Sep(time_T2)  or if_in_May_Sep(time_T1))  and row['Presence2011'] == 0:
        return -1
    else:
        return 1
    
        
        
def getGoodPoints_alwaysGood(yearList, goodYears = 2, ifRemoveNearest= True):
    
    
    df_record = pd.read_csv(r'D:\Bee\BeeSites\Registered Apiary Sites Location\vacancy.csv')                            
    shp_pos_points = r'D:\Bee\bee_binary_data\binaryLayerGeneration\weightedLayers\clint_full_points.shp'
    df_joint_shp_record =  selectPresent.getJoin_shp_record(shp_pos_points, df_record)
 
    res = []
    for  year in yearList:      
        year_non_present = df_joint_shp_record.filter(regex='Presence{0}|{0}T2|{0}T1.*'.format(year)).apply(lambda row: func(row, year), axis=1)
        res.append(year_non_present)
     
    judge_array = np.vstack(res).T
    judge_bool_non_presence = (judge_array==-1).any(axis=1)
    df_joint_shp_record = df_joint_shp_record[~judge_bool_non_presence]
     
    
    str_Presence = ''
    for i in yearList:
        str_Presence += 'Presence{}|'.format(i)
    str_Presence = str_Presence[:-1]

    print str_Presence

    df_four_years = df_joint_shp_record.filter(regex=str_Presence)
    bool_larger_than_3 = df_four_years.sum(axis=1)>=goodYears
    df_joint_shp_record = df_joint_shp_record[bool_larger_than_3]

    if(ifRemoveNearest):
        df_joint_shp_record = selectPresent.removeNearest(df_joint_shp_record, X_columnName = 'X_coor', y_columnName='Y_coor', distanceThresh = 3200)           
     
    return df_joint_shp_record
        

def getCDLDataPath(year):    
    return  r'D:\Bee\bee_binary_data\binaryLayerGeneration\CDL_NAD\CDL_NDSD_{}1_NAD.tif'.format(year)

    
def runLogitModel(X, y):
    import statsmodels.api as sm
    ols = sm.Logit(y, sm.add_constant(X))
    result =  ols.fit()
    print result.summary()    
    
def runLogitModel_PosNeg(pos_mat, neg_mat):
    y_pos_mat = np.ones(pos_mat.shape[0])
    y_neg_mat = np.zeros(neg_mat.shape[0])
    runLogitModel(pd.concat([pos_mat, neg_mat], axis=0), np.hstack([y_pos_mat, y_neg_mat]))


    

def generatePointsShpFromPositon(pointsInfo, shp_path, spatial_ref= arcpy.SpatialReference(26914)    ):
    pt = arcpy.Point()
    ptGeoms = []
    for p in pointsInfo:
        pt.X = p[0]
        pt.Y = p[1]
        ptGeoms.append(arcpy.PointGeometry(pt, spatial_ref))
    
    arcpy.CopyFeatures_management(ptGeoms, shp_path)
 

def getCorePointLocation():
    corePointsFullInfo = selectPresent.getWholeCoreGoodPoints()    
    corePointsMarkInfo = corePointsFullInfo[['site_name', 'X_coor', 'Y_coor' ,'XY_coor']]
    return corePointsMarkInfo
    
def getCorePointPresenceInfo():
    
    corePointsFullInfo = selectPresent.getWholeCoreGoodPoints()    
    return corePointsFullInfo.filter(regex = 'site_name| X_coor|Y_coor|\d+.*' )
    
def filterMaytoSep(func): 
    
    def wrapper(d, regex):
        listFiles = func(d, regex)
        filteredList = []
        for p in listFiles:
            if re.match('.*20140[5-9].*', p):
                filteredList.append(p)   
                
        print filteredList
        return filteredList
                
    return wrapper
            
    
@filterMaytoSep
def listdir_fullpath(d, regex = None):
    
    res = []
    for f in os.listdir(d):
        if not regex:
            res.append(os.path.join(d, f))
        else:
            if re.match(regex, f):
                res.append(os.path.join(d, f))    
    
    return res


def listdir_fullpath_org(d, regex = None):
    
    res = []
    for f in os.listdir(d):
        if not regex:
            res.append(os.path.join(d, f))
        else:
            if re.match(regex, f):
                res.append(os.path.join(d, f))    
    
    return res
    
    
            
            
def getND_climateDataset():   
    
    prep_dir = r'D:\Bee\BeeClimateDataset\PRISM_ppt_stable_4kmM3_2014_all_bil'
    prep_list = listdir_fullpath_org(prep_dir, '.*bil$')
    tdmean_dir = r'D:\Bee\BeeClimateDataset\PRISM_tdmean_stable_4kmM1_2014_all_bil'
    tdmean_list = listdir_fullpath_org(tdmean_dir, '.*bil$')
    tmean_dir = r'D:\Bee\BeeClimateDataset\PRISM_tmean_stable_4kmM2_2014_all_bil'
    tmean_list = listdir_fullpath_org(tmean_dir, '.*bil$')
    processClimateData(prep_list, 'getNDPart')
    processClimateData(tdmean_list, 'getNDPart')
    processClimateData(tmean_list, 'getNDPart')




def getClimateSpatialRef():
    ras_path = r'D:\Bee\BeeClimateDataset\PRISM_ppt_stable_4kmM3_2014_all_bil\PRISM_ppt_stable_4kmM3_2014_bil.bil'
    ras = arcpy.Raster(ras_path)
    climateSpatialRef = ras.spatialReference
    return climateSpatialRef


def reproject_default(FROM, TO, Resolution=30):

    spatial_ref = arcpy.SpatialReference(26914)    
    arcpy.ProjectRaster_management(FROM, TO, spatial_ref, 'NEAREST', Resolution)


    
    
    
def processClimateData(rasterList, operation):
    
    if operation == 'reproject':
        dir_prefix = 'UTM_'
        
    if operation == 'getNDPart':
        dir_prefix = 'ND_'
    
    
    for ras in rasterList:
        
        if re.findall('.*PRISM_(.*)_stable', ras):
            ras_basename = os.path.basename(ras)
            output_ras_basename = dir_prefix + ras_basename + '.tif'
            code = re.findall('.*PRISM_(.*)_stable', ras)[0]
            dir_repj = os.path.join('D:\Bee\BeeClimateDataset', dir_prefix + code)
            
            output_ras_path = os.path.join(dir_repj, output_ras_basename)
            
            if operation == 'reproject':
                reproject_default(FROM = ras, TO = output_ras_path) 
                
            mask_shp = r'F:\Apiary2\Boundary\boundary_ND\boundary_ND.shp'
            if operation == 'getNDPart':
                nd_raster = ExtractByMask(ras, mask_shp)
                nd_raster.save(output_ras_path)
                
                
        else:
            raise ValueError('wrong!')    


    
def generateClimateData(dir_climateTable, pointsLocations ):

     # Temperature Information
    prep_dir = r'D:\Bee\BeeClimateDataset\PRISM_ppt_stable_4kmM3_2014_all_bil'
    prep_list = listdir_fullpath_org(prep_dir, '.*bil$') 
    
    tdmean_dir = r'D:\Bee\BeeClimateDataset\PRISM_tdmean_stable_4kmM1_2014_all_bil'
    tdmean_list = listdir_fullpath_org(tdmean_dir, '.*bil$')
    
    tmean_dir = r'D:\Bee\BeeClimateDataset\PRISM_tmean_stable_4kmM2_2014_all_bil'
    tmean_list = listdir_fullpath_org(tmean_dir, '.*bil$')
    
    # to UTM meters unit
    processClimateData([ prep_list + tdmean_list + tmean_list], 'reproject')

    
    # process UTM dataset
    UTM_prep_dir = r'D:\Bee\BeeClimateDataset\UTM_ppt'
    UTM_prep_list = listdir_fullpath_org(UTM_prep_dir, '.*bil.tif$') 
    UTM_tdmean_dir = r'D:\Bee\BeeClimateDataset\UTM_tdmean'
    UTM_tdmean_list = listdir_fullpath_org(UTM_tdmean_dir, '.*bil.tif$')
    UTM_tmean_dir = r'D:\Bee\BeeClimateDataset\UTM_tmean'
    UTM_tmean_list = listdir_fullpath_org(UTM_tmean_dir, '.*bil.tif$')
    
     
    climate_list =[UTM_prep_list, UTM_tdmean_list, UTM_tmean_list]
    label_list = ['prep','tdmean','tmean']
    
    climateDict = {'prep':[],'tdmean':[],'tmean':[]}
    for climate_label, climate_list in zip(label_list, climate_list):
        for rasterPath in climate_list:     
            climateData = []
            month = re.findall('2014(.*)_', rasterPath)[0]
            for idx, point in enumerate(pointsLocations):
                windowSize = 3
                climateParcel = transform_Point_2_RasterArray(point, rasterPath, windowPixelSize = windowSize)
                mean_temp = np.mean(climateParcel.ravel())
                climateData.append([idx, month, mean_temp])
            climateDict[climate_label].append(climateData)
     
     

    for key, value in climateDict.items():
        
        dataNumeric_oneClimate = np.array([ np.array(data)[:,2] for data in climateDict[key] ]).astype(float).T
        df_one_climate = pd.DataFrame(data=dataNumeric_oneClimate, columns= [ 'month_' + str(i) for i in range(5,10)])
        df_one_climate.to_csv(os.path.join(dir_climateTable, key + '.csv'), index_label='Grid_ID') 
     
        
        
def timeit(func):
    
    def wrapper(self):
        start = time.time()
        res = func(self)  # no parameter need to put in
        end = time.time()
        print 'time spent', end - start
        if res:
            return res
        
    return wrapper
    

 

 
def getShannonIndex(cdlParcelArray):
    unique, counts = np.unique(cdlParcelArray, return_counts=True)    
    p = counts/float(np.sum(counts))        
    log2P = np.log2(p)
    entropy = -np.sum(p* log2P)
    return entropy

def getSimplesonIndex(cdlParcelArray):
    unique, counts = np.unique(cdlParcelArray, return_counts=True)    
    p = counts/float(np.sum(counts))        
    simpsonIndex = 1 - np.sum(1 - p**2)
    return simpsonIndex
        

def getInfoFromShapefile(shp_path, fields, spatialRefCode=None):
    
    if not spatialRefCode:
        spatialCode = 26914 
        print 'using default projection sys  UTM 14N'
        print '--------------------------------------'
    else:
        spatialCode = spatialRefCode
 
 
    with arcpy.da.SearchCursor(shp_path, field_names = fields, spatial_reference = arcpy.SpatialReference(spatialCode))  as cursor:
        result =  [ row   for row in cursor ]

    
    return zip(*result) 
    
def t():
    return time.time()    
    
def getSpatialReference(sp_string):
    
    if sp_string=='UTM_14N':
        spatial_ref = arcpy.SpatialReference(26914)
        return  spatial_ref
    if sp_string=='WGS84':    
        return arcpy.SpatialReference(4326)


def getXYFromShapefile(shp_path, spatialRefCode=None):
    
    if not spatialRefCode:
        spatialCode = 26914 
        print 'using default projection sys  UTM 14N'
        print '--------------------------------------'
    else:
        spatialCode = spatialRefCode
    
    with arcpy.da.SearchCursor(shp_path, field_names = ["SHAPE@XY"], spatial_reference = arcpy.SpatialReference(spatialCode))  as cursor:
        cursorList = [ list(i[0]) for i in cursor ]
    df_gridPoints = np.array(cursorList)
    return df_gridPoints


def createShapefileFromXY(pointData, shp_path, spatialRefCode=None):

    if not spatialRefCode:
        spatialCode = 26914 
        print 'using default projection sys  UTM 14N'
        print '--------------------------------------'
    else:
        spatialCode = spatialRefCode
    
    with arcpy.da.SearchCursor(shp_path, field_names = ["SHAPE@XY"], spatial_reference = arcpy.SpatialReference(spatialCode))  as cursor:
        cursorList = [ list(i[0]) for i in cursor ]
    df_gridPoints = np.array(cursorList)
    return df_gridPoints


def getFieldNames(shapePath):
    return [f.name for f in arcpy.ListFields(shapePath)   ]

def deleteFields(shapePath, fields):
    return arcpy.DeleteField_management(shapePath, fields)
            
class NegBeeDataFrame:
    
    def __init__(self, df_landcover, df_positionDensity):
        self.position = df_positionDensity[['site_name', 'X_coor', 'Y_coor'] ]
        self.landcover = self.removePositionInfo(df_landcover)
        self.bigTable = self.mergeAll(self.position, self.landcover)
        
    def removePositionInfo(self, df_frame):
        positionCols = [u'X_coor', u'Y_coor', u'XY_coor']
        removedCols = [col for col in df_frame.columns if not (col in positionCols) ]
        return df_frame[removedCols]

    def getPostion_AllNegBee(self):
        
        df_Neg_positions = self.position[[u'X_coor', u'Y_coor']].copy()
        df_Neg_positions.columns = ['position_X_coor', 'position_Y_coor']
        return df_Neg_positions
 
    def mergeAll(self, df_position, df_landcover):        
        df_position = df_position.copy()
        df_landcover = df_landcover.copy() 
        df_position.columns = ['position_'+ col if col != 'site_name' and col!='Density' else col for col in df_position.columns  ]
        df_landcover.columns = ['landcover_'+ col if col != 'site_name'  else 'site_name' for col in df_landcover.columns  ]
        df_first =pd.merge(left=df_position, right=df_landcover, how='inner', on='site_name')        
        return df_first
 
    def generateLandcoverYear(self, year):
        return self.bigTable.filter(regex='landcover_{}.*'.format(year))
        
    def getPresenceYear(self, year):
        return self.bigTable.filter(regex='record_{}.*'.format(year))
        
    def getYearPresent_AllInfo(self, year):
        return self.bigTable.filter(regex = self.getRegexForYear(year))
    
    def getYearPresent_Landcover(self, year):
        return self.bigTable.filter(regex = self.getRegexForYear(year)).filter(regex='landcover.*')
        
    def getRegexForYear(self, year):
        return 'site_name|position.*|landcover.*{}'.format(year)
        
        
 

def getRows_Shapefile(shapeFilePath):  
    cursor = arcpy.da.SearchCursor(shapeFilePath, ['OBJECTID'])
    rows = len([ 1 for row in cursor ])
    return rows


def efficientJoin(inputShapefile, newJointColumn, newColumnData, ifReplace=False, field_type=None):


    if not field_type:
        print ('available type:  TEXT, FLOAT, SHORT, LONG') 
        raise ValueError('Sorry, you need to set field type\n available type:  TEXT, FLOAT, SHORT, LONG\n')
        
    if newJointColumn not in getFieldNames(inputShapefile):
        arcpy.AddField_management(inputShapefile, newJointColumn, field_type=field_type, field_precision=6 ) 
    elif ifReplace:
        deleteFields(inputShapefile, [ newJointColumn] )
        arcpy.AddField_management(inputShapefile, newJointColumn, field_type=field_type, field_precision=6, field_is_nullable="NULLABLE" )  
    else:
        raise ValueError('Field names exits, set ifReplace to be True, if you want to overright') 

    # rows_shapefile = getRows_Shapefile(inputShapefile)
    # rows_csv = newColumnData.shape[0]
    # if rows_shapefile != rows_csv:
    #     raise ValueError('added data and current shapefile, size not match')

    wholeRows = len(newColumnData)
    sk = 0
    with arcpy.da.UpdateCursor(inputShapefile, field_names=[newJointColumn], spatial_reference=getSpatialReference('UTM_14N')) as cursor:
        data_iter = iter(newColumnData)
        for idx, row in  enumerate(cursor):
            print idx
            row[0] = data_iter.next()
            cursor.updateRow(row)  
            sk += 1
            per = (float(sk) / wholeRows) * 100
            if per % 10 == 0:
                print (float(sk) / wholeRows) * 100, '%'
    

# def efficientJoin(inputShapefile, csvPath, csvField, newJointColumn, ifReplace=False,  field_type=None):
     
#     if not field_type:
#         raise ValueError('Sorry, you need to set field type')
#         print ('available type:  TEXT, FLOAT, SHORT, LONG') 
        
#     rows_shapefile = getRows_Shapefile(inputShapefile)
#     rows_csv = pd.read_csv(csvPath).sjhape[0]
#     if rows_shapefile != rows_csv:
#         raise ValueError('added data and current shapefile, size not match')

    
#     with arcpy.da.UpdateCursor(inputShapefile, field_names=[newJointColumn], spatial_reference=bl.getSpatialReference('UTM_14N')) as cursor:
#         if newJointColumn not in bl.getFieldNames(inputShapefile):
#             arcpy.AddField_management(inputShapefile, newJointColumn, field_type=field_type, field_precision=6, field_is_nullable='NULLABLE' ) 
#         elif ifReplace:
#             bl.deleteFields(inputShapefile, [ newJointColumn] )
#             arcpy.AddField_management(inputShapefile, newJointColumn, field_type=field_type, field_precision=6, field_is_nullable='NULLABLE' )  
#             print '--------------delete Field finished----------------'
#         else:
#             raise ValueError('Field names exits, set ifReplace to be True, if you want to overright') 
#             #raise ValueError('Field names exits, set ifReplace to be True, if you want to overright') 
    
#         print '-------------start writing------------'
#         freader = csv.reader(open(csvPath), delimiter=",")
#         head = freader.next()
#         nirr_index = head.index(csvField)
    
#         sk = 0
#         for idx, row in  enumerate(cursor):
#             row[0] = freader.next()[nirr_index]
#             cursor.updateRow(row)  
#             sk += 1
#             if sk % 10000 == 0:
#                 print sk

def fastLeftJoin(shapefilePath, left_on, dataFrame,  right_on, addedFieldName):
    # shp_table = shapefilePath
    # print 'joining start'
    # df_merge_res = pd.merge(left=shp_table, right=dataFrame,  left_on=left_on, right_on=right_on, how='left')
    # if df_merge_res.shape[0] != shp_table.shape[0]:
    #     print 'left join result:', df_merge_res.shape[0], 'origin shp result: ', shp_table.shape[0]
    #     raise ValueError( 'left join result rows  !=  shapefile table rows')
    # df_merge_res = df_merge_res.fillna(-9999)
    # print 'joining ended'
    # efficientJoin(cultivated_CLUs, newJointColumn=addedFieldName, newColumnData=df_merge_res[addedFieldName])
    pass