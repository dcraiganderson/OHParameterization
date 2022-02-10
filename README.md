# OH Parameterization
Tools to use the machine learning parameterization of OH described in Anderson et al (2022)
The files included here allow for the creation of a training dataset for an OH parameterization
using output from the NASA GSFC MERRA2 GMI (https://acd-ext.gsfc.nasa.gov/Projects/GEOSCCM/MERRA2GMI/) simulation.  
MERRA2 GMI output is available with an account on the discover server or through opendap.
Files included here are:
1. CreateTrainingSet_MERRA2GMI.py: Creates the training dataset.  Directories will need to be modified by the user.
2. CreateParameterization.py: Creates a sample parameterization of OH with XGBoost using files 3 and 4.
3. SampleTrainingSet.dat: A sample training set of inputs for July that can be input into CreateParameterization.py
4. SampleTrainingTargets.dat: The training targets (OH) that correspond to the SampleTrainingSet.dat Input values.
5. SampleParameterization_MO7.joblib.dat: A parameterization of OH for July that was used and described in Anderson et al (2022). 
