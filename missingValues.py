#-------------------------------------
# file: missingValues.py



# This script fills in all missing values in both train and test datasets

# To Use: Save the training and testing datasets to a folder named 'data' in your current directory
# then run it in ipython with: %run missingValues.py

# First let us import the required packages
import numpy as np
import pandas as pd

# Let's read-in the training dataset and the testing dataset
trainData = pd.read_csv('data/train.csv')
testData = pd.read_csv('data/test.csv')
    
# Now dealing with missing values in groups

# 1. The basement group:

# It will be useful to create a basement index for the relevant categories
indBase = pd.Index(['BsmtQual','BsmtCond', 'BsmtExposure', 'BsmtFinType1', 'BsmtFinType2',
                    'BsmtFinSF1', 'BsmtFinSF2', 'BsmtUnfSF', 'TotalBsmtSF', 'BsmtFullBath', 'BsmtHalfBath'])

# Here we see only one obs in the testset with Nas in the numerical vars describing the basement size
# We can assume this should be zero i.e. no basement, let's set these to zero:
testData.loc[660, indBase[[5,6,7,8]] ] = testData.loc[660, indBase[[5,6,7,8]] ].fillna(0.0)


# If we look at the problem observations again:
# We can see that the four size variables 'BsmtFinSF1' 'BsmtFinSF2' 'BsmtUnfSF' and 'TotalBsmtSF'
# are set to zero, confirming that these missing values are indeed No Basement.
# Here 'TotalBsmtSF' is presumably a sum of: 'BsmtFinSF1' 'BsmtFinSF2' 'BsmtUnfSF'.
# So we can write something like: when 'TotalBsmtSF' == 0, set all other basement NaNs to 'NoBasement'

# First create logical vector:
zeroB_train = trainData['TotalBsmtSF'] == 0
zeroB_test = testData['TotalBsmtSF'] == 0
# Fill in missing values for the categorical vars and the two 'BsmtFull/HalfBath' vars:
trainData.loc[zeroB_train, indBase[0:5] ] = trainData.loc[zeroB_train, indBase[0:5] ].fillna("NoBasement")
testData.loc[zeroB_test, indBase[0:5] ] = testData.loc[zeroB_test, indBase[0:5] ].fillna("NoBasement")
trainData.loc[zeroB_train, indBase[[9,10]]] = trainData.loc[zeroB_train, indBase[[9,10]]].fillna(0.0)
testData.loc[zeroB_test, indBase[[9,10]]] = testData.loc[zeroB_test, indBase[[9,10]]].fillna(0.0)

# Now we are left with the following NaNs:
# BsmtQual 0 2, BsmtCond 0 3, BsmtExposure 1 2, BsmtFinType2 1 0
# These properties clearly have a basement, so we cannot set 'NoBasement'.

# BsmtQual 0 2: the most typical values are TA and Gd, we could just assume TA
testData.loc[:, 'BsmtQual'] = testData.loc[:,'BsmtQual'].fillna("TA")

# BsmtCond 0 3: the most typical value is TA, we could just assume TA as well
testData.loc[:, 'BsmtCond'] = testData.loc[:,'BsmtCond'].fillna("TA")

# BsmtExposure 1 2: Refers to walkout or garden level walls.
# The most typical value is 'No' No Exposure, we could simply assume No as well
trainData.loc[:, 'BsmtExposure'] = trainData.loc[:,'BsmtExposure'].fillna("No")
testData.loc[:, 'BsmtExposure'] = testData.loc[:,'BsmtExposure'].fillna("No")

# BsmtFinType2 1 0: Rating of basement finished area (if multiple types):
# We could guess here that the cat is the same as 'BsmtFinType1' i.e. 'GLQ' Good Living Quarters
trainData.loc[:, 'BsmtFinType2'] = trainData.loc[:, 'BsmtFinType2'].fillna("GLQ")

# That's it for the Basement group!

# 2. The garage group:

# We start by creating a Garage index for the relevant categories
indGarage = pd.Index(['GarageType', 'GarageFinish', 'GarageQual', 'GarageCond',
                      'GarageYrBlt', 'GarageCars', 'GarageArea'])

# We see one strange obs with missing values for the garage size, but with 'GarageType' = 'Detchd'
# which means Detached from home. We can see thats it's not a second garage in Misc.
# We can assume this should be zero i.e. no Garage, let's set these sizes to zero:
testData.loc[1116, indGarage[[5,6]] ] = testData.loc[1116, indGarage[[5,6]] ].fillna(0.0)
testData.loc[1116, indGarage[[0]] ] = 'NaN' # clear the (probably) wrong category

# Assume if 'GarageArea' is zero, these properties are without garages. Then all categories should be
# set to "NoGarage".

# First create logical vector:
zeroG_train = trainData['GarageArea'] == 0
zeroG_test = testData['GarageArea'] == 0

# Fill in missing values for the categorical vars:
trainData.loc[zeroG_train, indGarage[0:4] ] = trainData.loc[zeroG_train, indGarage[0:4] ].fillna("NoGarage")
testData.loc[zeroG_test, indGarage[0:4] ] = testData.loc[zeroG_test, indGarage[0:4] ].fillna("NoGarage")

# And fill in the 'GarageYrBlt' with zeros:
# To set the missing values in GarageYrBlt (Year garage was built), which is a continuous variable,
# we could either set all dates to category and create new NoGarage cat, or set this missing value to zero.
# Here we set it to zero. This is the same as creating a new variable 'WithGarage' that takes value 1 for yes,
# and 0 for no, and then an interaction term between the two.
trainData.loc[zeroG_train, indGarage[4] ] = trainData.loc[zeroG_train, indGarage[4] ].fillna(0)
testData.loc[zeroG_test, indGarage[4] ] = testData.loc[zeroG_test, indGarage[4] ].fillna(0)
# After adding these missing values, we have to be careful, the correlations matrix changes!
# Another option here is to make GarageYrBlt = YearBuilt for these missing values,
# since these variables are highly correlated

# We are left with one obs in the test dataset which is a 'GarageType' = 'Detchd' for 'GarageCars' = 1.
#testData['GarageCond'].value_counts()
# Assume most common values: 'GarageFinish'=Unf, 'GarageQual'='GarageCond'=TA,
# Assume 'GarageYrBlt' = 'YearBuilt'
testData.loc[666, indGarage[[1]] ] = "Unf"
testData.loc[666, indGarage[[2,3]] ] = testData.loc[666, indGarage[[2,3]] ].fillna("TA")
testData.loc[666, indGarage[[4]] ] = testData.loc[666, 'YearBuilt' ]

# There's one obs that say "NaN" as a string, change to cat
testData.loc[1116, 'GarageType' ] = "NoGarage" # change value

# That's it for the Garage group!

# 3. The exterior group:

# We start by creating a Exterior index for the relevant categories
indExt = pd.Index(['MasVnrType', 'MasVnrArea', 'Exterior1st', 'Exterior2nd'])

# Here we assume that missing values for 'MasVnrArea' correspond to No Veneer, see data description
trainData.loc[:, 'MasVnrArea'] = trainData.loc[:, 'MasVnrArea'].fillna(0.0)
testData.loc[:, 'MasVnrArea'] = testData.loc[:, 'MasVnrArea'].fillna(0.0)

# Assume if 'MasVnrArea' is zero, then MasVnrType should be set to "None".

# First create logical vector:
zeroV_train = trainData['MasVnrArea'] == 0
zeroV_test = testData['MasVnrArea'] == 0

# Fill in missing values for the categorical vars:
trainData.loc[zeroV_train, indExt[[0]] ] = trainData.loc[zeroV_train, indExt[[0]] ].fillna("None")
testData.loc[zeroV_test, indExt[[0]] ] = testData.loc[zeroV_test, indExt[[0]] ].fillna("None")

# We are then left with two obs with missing values:
# The most common 'MasVnrType' after 'None' is 'BrkFace', we can assume this.
testData.loc[:, 'MasVnrType'] = testData.loc[:, 'MasVnrType'].fillna("BrkFace")

# The most common cat for 'Exterior1st' and 'Exterior2nd' is 'VinylSd', we can assume these.
testData.loc[:, indExt[[2, 3]] ] = testData.loc[:, indExt[[2, 3]] ].fillna("VinylSd")

# Note: Here perhaps it's better to use some other imputing method, such as nearest neighbours

# That's it for this group!

# 4. The other features group:

# LotFrontage 259 227: Linear feet of street connected to property (continuous)
# Assume missing data corresponds to zero
trainData.loc[:, 'LotFrontage'] = trainData.loc[:, 'LotFrontage'].fillna(0)
testData.loc[:, 'LotFrontage'] = testData.loc[:, 'LotFrontage'].fillna(0)

# Alley 1369 1352: Type of alley access. (Cat, NA No alley access)
# There is no NoAlley category, let's create one
trainData.loc[:, 'Alley'] = trainData.loc[:, 'Alley'].fillna("NoAlley")
testData.loc[:, 'Alley'] = testData.loc[:, 'Alley'].fillna("NoAlley")

# FireplaceQu 690 730: Fireplace quality. (Cat, NA No Fireplace)
# There is no NoFireplace category, let's create one
trainData.loc[:, 'FireplaceQu'] = trainData.loc[:, 'FireplaceQu'].fillna("NoFire")
testData.loc[:, 'FireplaceQu'] = testData.loc[:, 'FireplaceQu'].fillna("NoFire")

# PoolQC 1453 1456: Pool quality. (Cat, NA No Pool)
# There is no NoPool category, let's create one
trainData.loc[:, 'PoolQC'] = trainData.loc[:, 'PoolQC'].fillna("NoPool")
testData.loc[:, 'PoolQC'] = testData.loc[:, 'PoolQC'].fillna("NoPool")

# Fence 1179 1169: Fence quality. (Cat, NA No Fence)
# There is no NoFence category, let's create one
trainData.loc[:, 'Fence'] = trainData.loc[:, 'Fence'].fillna("NoFence")
testData.loc[:, 'Fence'] = testData.loc[:, 'Fence'].fillna("NoFence")

# MiscFeature 1406 1408: Miscellaneous feature not covered in other categories. (Cat, NA None)
# There is no NoMisc category, let's create one
trainData.loc[:, 'MiscFeature'] = trainData.loc[:, 'MiscFeature'].fillna("NoMisc")
testData.loc[:, 'MiscFeature'] = testData.loc[:, 'MiscFeature'].fillna("NoMisc")

# Electrical 1 0: Electrical system. (Cat, Mix Mixed)
# Here it is not sure what a missing value means
# One option is to assume the most common type i.e. 'SBrkr' for this one missing value
trainData.loc[:, 'Electrical'] = trainData.loc[:, 'Electrical'].fillna("SBrkr")

# MSZoning 0 4: Identifies the general zoning classification of the sale. (Cat)
# RL most common
testData.loc[:, 'MSZoning'] = testData.loc[:, 'MSZoning'].fillna("RL")

# Utilities 0 2: Type of utilities available (Cat)
# AllPub most common
testData.loc[:, 'Utilities'] = testData.loc[:, 'Utilities'].fillna("AllPub")

# KitchenQual 0 1: Kitchen quality (Cat)
# TA most common
testData.loc[:, 'KitchenQual'] = testData.loc[:, 'KitchenQual'].fillna("TA")

# Functional 0 2: Home functionality (Assume typical unless deductions are warranted) (Cat)
# Typ most common
testData.loc[:, 'Functional'] = testData.loc[:, 'Functional'].fillna("Typ")

# SaleType 0 1: Type of sale (Cat)
# WD most common
testData.loc[:, 'SaleType'] = testData.loc[:, 'SaleType'].fillna("WD")

# That's it for this group!

# There should be no more missing values in both datasets
# Check again the number of missing values per column
for i in list(testData.columns) :
    k = sum(pd.isnull(trainData[i]))
    k2 = sum(pd.isnull(testData[i]))
    if (k != 0) | (k2 != 0): # print only columns that have missing values
        print(i, k, k2)
