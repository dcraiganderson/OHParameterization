# OH Parameterization
Tools to use the machine learning parameterization of OH described in Anderson et al (2022)
The files included here allow for the creation of a training dataset for an OH parameterization
using output from the NASA GSFC MERRA2 GMI (https://acd-ext.gsfc.nasa.gov/Projects/GEOSCCM/MERRA2GMI/) simulation.  
MERRA2 GMI output is available with an account on the discover server or through opendap.
Files included here are:
1. CreateTrainingSet_MERRA2GMI.py: Creates the training dataset.  Directories will need to be modified by the user.
2. CreateParameterization.py: Creates a sample parameterization of OH with XGBoost using files 3 and 4.

Contact daniel.c.anderson (at sign) nasa.gov with any questions.
