# This script creates the training dataset to be input into CreateParameterization.py.
# The code is structured to work with output from the NASA GEOS MERRA2 GMI simulation
# and should be relatively easily modifiable to apply to other models and simulations. 
# Multiple files are output by the script including (see the ReadMe file for more details):
# 1. OHMerge_StartYear_StopYear_mmdd.nc4: 
#    A consolidation of OH values from all years to be used in the parameterization.  One
#    file is output for each day of the month.
# 2. MaskMerge_StartYear_StopYear_mmdd.nc4:
#    A file that creates a mask to indicate values that should not be included in the 
#    parameterization including OH not in the troposphere and years to be omitted from
#    the training dataset.
# 3. RandomOH_Indices_StartYear_StopYear_mm.nc:
#    This file contains the randomly selected OH targets to be used in training along with
#    indices that indicate temporal and spatial coordinates of the individual target.
# 4. ParameterizationInputs_StartYear_StopYear_All_mm.nc
#    This file contains the parameterization inputs corresponding to the OH targets found
#    in Indices_StartYear_StopYear_mm.nc4.  All potential parameterization inputs can
#    be included here.
# 5. ParameterizationInputs_StartYear_StopYear_mm.dat:
#    This file is formatted to be input into the CreateParameterization.py script to generate
#    the parameterization.  Is created from a subset of the variables in the .nc4 Inputs file.
# 6. ParameterizationTargets_StartYear_StopYear_mm.dat:
#    The training targets to be input into CreateParameterization.py.

# import necessary modules
import numpy as np
import netCDF4 as nc
import time

startTime = time.time()
# Variables to be set by user
Month = 7 #Month of the parameterization to be created
DayList = np.arange(1,3,1) #Number of days in month + 1
OmitYears = [1982] #Omit listed years for model validation 
StartYear = 1980
StopYear = 1983   #non-inclusive
CreateTPause = 0   #1 or 0: Creates the troposphere mask and combines OH from different years into one file
numinbin = 6700  #Number of datapoints to include in each percentile bin for each day.  Should be 6700 when doing all years
SaveOHMerge = 0 #1 or 0: Creates a netcdf file with unfiltered OH and the mask. Only works if CreateTPause = 1
SubsetOH = 0    #1 or 0: Randomly samples the OH in percentile bins and creates the indices for the OH from the 4D merged file
SaveIndices = 0 #1 or 0: Saves the randomly sampled OH and the corresponding indices as a netcdf file. SubsetOH must = 1
SubsetInputs = 0 #1 or 0: Makes a 4D merged file for each variable in InputList and then creates a 1D subset from the saved indices
ApendMe = 0  #If 0, creates a new netcdf file for Inputs, else adds to existing netcdf file             
WritedatFile = 1 #Creates the .dat files needed to create the parameterization.

# List of variables to be added to the ParameterizationInputs_StartYear_StopYear_All_mm.nc4 
# Note that SZA is not included here because it was not output for the MERRA2 GMI simulation.
# SZA values were calculated separately and added to the .dat file
InputList = ['NO2', 'O3', 'CH4', 'CO', 'ISOP', 'ACET', 'C2H6', 'C3H8', 'PRPE', 'ALK4', 'MP', 'H2O2','PL','QV','T','Lat','CLOUD','TAUCLWUP','TAUCLIUP','TAUCLWDWN','TAUCLIDWN','ALBUV','GMISTRATO3','CH2O','AODDWN','AODUP']

#VarList is the variables that will be written to the ParameterizationInputs.dat file.
#These are the variables the RT model will be trained on
#VarList = ['Lat','PL','T','NO2','O3','CH4','CO','ISOP','ACET','C2H6','C3H8','PRPE','ALK4','MP','H2O2','TAUCLWDWN','TAUCLIDWN','TAUCLIUP','TAUCLWUP','CLOUD','QV','GMISTRATO3','ALBUV','AODUP','AODDWN','CH2O','SZA']

VarList = ['Lat','NO2','GMISTRATO3']

##################
#After initial setup of directories, model specific information, no need to modify below here.

# These lists indicate in which file the MERRA2 GMI output is stored
# ALBUV and GMISTRATO3 are the only variables in their files.  Lat is calculated otherwise.
DACList = ['NO2', 'O3', 'CH4', 'CO', 'ISOP', 'ACET', 'C2H6', 'C3H8', 'PRPE', 'ALK4', 'MP', 'H2O2', 'CH2O']
MetList = ['PL', 'QV', 'T']
CloudList = ['CLOUD', 'TAUCLWUP', 'TAUCLIUP', 'TAUCLWDWN', 'TAUCLIDWN']
PMList = ['AODDWN', 'AODUP']


Years = np.arange(StartYear,StopYear)
Days = np.arange(0,np.size(DayList))
if Month < 10:
    MonthName = '0' + str(Month)
else:
    MonthName = str(Month)

# Create mask to block out stratospheric values. Mask value of 1 indicates troposphere. Also import OH. 
if CreateTPause == 1:
    print('Creating OH merge and mask files ...')
    for dd in Days:
        if DayList[dd] < 10:
            DayName = '0' + str(DayList[dd])
        else:
            DayName = str(DayList[dd])
        for ii in Years:
            YearName = str(ii)
            dirname = '/discover/nobackup/projects/gmao/merra2_gmi/pub/Y' + YearName + '/M' + MonthName + '/'

            #Import tropopause information
            TPauseName = dirname + 'MERRA2_GMI.tavg24_2d_dad_Nx.' + YearName + MonthName + DayName + '.nc4'
            TPauseFile = nc.Dataset(TPauseName)
            TROPPB = TPauseFile['TROPPB'][:]
            TPauseFile.close()
            TROPPB = np.squeeze(TROPPB)    #Blended tropopause; units are Pa

            #Import pressure information
            PName = dirname + 'MERRA2_GMI.tavg3_3d_met_Nv.' + YearName + MonthName + DayName + '.nc4'
            PFile = nc.Dataset(PName)
            PL = PFile['PL'][:]
            PFile.close()
            PL = np.mean(PL,0)           #Mid-pressure level; units are Pa
            sizePL = np.shape(PL)

            #Determine below tropopause locations
            TROPPB_Rep = np.tile(TROPPB, (sizePL[0], 1, 1))
            PDif = PL - TROPPB_Rep
            Mask = np.zeros(sizePL)

            #Mask out above tropopause and years to omit from training             
            if ii in OmitYears:
                Mask[:] = 0
            else:
                Mask[PDif>0] = 1

            #Import OH information             
            OHName = dirname + 'MERRA2_GMI.tavg24_3d_dac_Nv.' + YearName + MonthName + DayName + '.nc4'
            OHFile=nc.Dataset(OHName)
            OH = OHFile['OH'][:]
            OHFile.close()
            OH=np.squeeze(OH)

            # Consolidate data for each year into one large matrix.
            if ii==StartYear:
                MaskSummary = np.zeros((StopYear-StartYear,sizePL[0],sizePL[1],sizePL[2]))
                OHSummary = np.zeros((StopYear-StartYear,sizePL[0],sizePL[1],sizePL[2]))
            MaskSummary[ii-StartYear, :, :, :] = Mask
            OHSummary[ii-StartYear, :, :, :] = OH

        if SaveOHMerge == 1:
            dims_all = np.shape(MaskSummary)
            directory = '/user/dir/'
            savename = directory + 'MaskMerge_' + str(StartYear) + '_' + str(StopYear-1) + '_' + MonthName + DayName + '.nc'
            print('saving ' + savename)
            ncfile = nc.Dataset(savename, mode='w', format = 'NETCDF4_CLASSIC')
            dim1 = ncfile.createDimension('Year',dims_all[0])
            dim2 = ncfile.createDimension('lev',dims_all[1])
            dim3 = ncfile.createDimension('lat',dims_all[2])
            dim4 = ncfile.createDimension('lon',dims_all[3])
            OHme = ncfile.createVariable('Mask', np.float32,('Year','lev','lat','lon',))
            OHme[:] = MaskSummary
            ncfile.close()

            directory = '/user/dir/'
            savename = directory + 'OHMerge_' + str(StartYear) + '_' + str(StopYear-1) + '_' + MonthName + DayName + '.nc'
            print('saving ' + savename)
            ncfile = nc.Dataset(savename, mode='w', format = 'NETCDF4_CLASSIC')
            dim1 = ncfile.createDimension('Year',dims_all[0])
            dim2 = ncfile.createDimension('lev',dims_all[1])
            dim3 = ncfile.createDimension('lat',dims_all[2])
            dim4 = ncfile.createDimension('lon',dims_all[3])
            OHme = ncfile.createVariable('OH', np.float32,('Year','lev','lat','lon',))
            OHme[:] = OHSummary
            ncfile.close()

MergeTime = (time.time() - startTime)
print('Time to Merge OH and Mask Files: ', MergeTime)

# Subset OH across 20 equally spaced percentile bins for each day and then merge the days
startTime2 = time.time()
if SubsetOH == 1: # If 1, create a new file; otherwise, load an existing file
    print('Subsetting OH ...')
    gg = 0
    for dd in Days:
        if DayList[dd] < 10:
            DayName = '0' + str(DayList[dd])
        else:
            DayName = str(DayList[dd])
        directory = '/user/dir/'
        loadname = directory + 'MaskMerge_' + str(StartYear) + '_' + str(StopYear-1) + '_' + MonthName + DayName + '.nc'
        MaskFile = nc.Dataset(loadname)
        MaskSummary = MaskFile['Mask'][:]
        MaskFile.close()                                                                                                    

        directory = '/user/dir/'
        loadname = directory + 'OHMerge_' + str(StartYear) + '_' + str(StopYear-1) + '_' + MonthName + DayName + '.nc'
        OHFile = nc.Dataset(loadname)
        OHSummary = OHFile['OH'][:]
        OHFile.close()

        #Apply the mask
        tropindices = np.nonzero(MaskSummary == 1)
        OHMasked = OHSummary
        OHMasked = OHMasked[tropindices]
        b=np.shape(OHMasked)

        #Calculate percentiles for individual days
        percentile_me = np.arange(5,100,5)
        OH_percentile = np.zeros(np.size(percentile_me)+2)
        jj = 0
        for ii in percentile_me:
            OH_percentile[jj+1] = np.percentile(OHMasked,percentile_me[jj])
            jj = jj + 1
        #Add min and max values                                                                                                               
        OH_percentile[0]=np.min(OHMasked)
        OH_percentile[-1]=np.max(OHMasked)
        
        #Initialize some variables        
        if gg == 0:
            OHall = np.zeros([np.size(DayList),numinbin*(np.size(OH_percentile)-1)]) #Will contain all randomly sampled OH for each day
            #Indicates the position along dimensions 1, 2, 3, and 4 of the OH value from the OHMerge file.
            indices1all = np.zeros([np.size(DayList),numinbin*(np.size(OH_percentile)-1)])
            indices2all = np.zeros([np.size(DayList),numinbin*(np.size(OH_percentile)-1)])
            indices3all = np.zeros([np.size(DayList),numinbin*(np.size(OH_percentile)-1)])
            indices4all = np.zeros([np.size(DayList),numinbin*(np.size(OH_percentile)-1)])
            DayMat = np.zeros([np.size(DayList),numinbin*(np.size(OH_percentile)-1)])
            
        DayMat[dd,:]=DayList[dd]
        
        #Do the random sampling
        jj=0
        for ii in range(np.size(OH_percentile)-1):
            indices = np.nonzero((MaskSummary == 1) & (OHSummary>OH_percentile[ii]) & (OHSummary<OH_percentile[ii+1]))
            indexlength=np.shape(indices)
            chosefrom = np.arange(0,indexlength[1])
            randomindices = np.random.choice(chosefrom,numinbin,0)
            indices1 = indices[0]
            indices1 = indices1[randomindices]
            indices2 = indices[1]
            indices2 = indices2[randomindices]
            indices3 = indices[2]
            indices3 = indices3[randomindices]
            indices4 = indices[3]
            indices4 = indices4[randomindices]
            OHrandom = OHSummary[indices1, indices2, indices3, indices4]
            indices1all[gg,jj:(jj+numinbin)] = indices1
            indices2all[gg,jj:(jj+numinbin)] = indices2
            indices3all[gg,jj:(jj+numinbin)] = indices3
            indices4all[gg,jj:(jj+numinbin)] = indices4
            OHall[gg,jj:(jj+numinbin)] = OHrandom
            jj = jj + numinbin

        gg = gg+1
    
    if SaveIndices == 1:
        OHsize = np.shape(OHall)
        savefile = '/user/dir/RandomOH_Indices_' + str(StartYear) + '_' + str(StopYear-1) + '_' + MonthName + '.nc'
        ncfile = nc.Dataset(savefile, mode='w', format = 'NETCDF4_CLASSIC')
        dim1 = ncfile.createDimension('Dim1',OHsize[0])
        dim2 = ncfile.createDimension('Dim2',OHsize[1])
        OHme = ncfile.createVariable('OH', np.float32,('Dim1','Dim2',))
        SaveIndices1 = ncfile.createVariable('Indices1',np.float32,('Dim1','Dim2',))
        SaveIndices2 = ncfile.createVariable('Indices2',np.float32,('Dim1','Dim2',))
        SaveIndices3 = ncfile.createVariable('Indices3',np.float32,('Dim1','Dim2',))
        SaveIndices4 = ncfile.createVariable('Indices4',np.float32,('Dim1','Dim2',))
        DayMatMe = ncfile.createVariable('DayMatrix',np.float32,('Dim1','Dim2',))
        OHme[:] = OHall
        SaveIndices1[:] = indices1all
        SaveIndices2[:] = indices2all
        SaveIndices3[:] = indices3all
        SaveIndices4[:] = indices4all
        DayMatMe[:] = DayMat
        ncfile.close()

MergeTime = (time.time() - startTime2)
print('Time to Subset OH: ', MergeTime)

startTime3 = time.time()
if SubsetInputs == 1:
    if SubsetOH != 1:
        #Load indices from previous run if haven't just created them
        loadname = '/user/dir/RandomOH_Indices_' + str(StartYear) + '_' + str(StopYear-1) + '_' + MonthName + '.nc'
        SummaryFile = nc.Dataset(loadname)
        indices1all = SummaryFile['Indices1'][:]
        indices1all = np.array(indices1all)
        indices2all = SummaryFile['Indices2'][:]
        indices2all = np.array(indices2all)
        indices3all = SummaryFile['Indices3'][:]
        indices3all = np.array(indices3all)
        indices4all = SummaryFile['Indices4'][:]
        indices4all = np.array(indices4all)
        DayMat = SummaryFile['DayMatrix'][:]
        SummaryFile.close()

    # Convert indices to correct type
    indices1all = indices1all.astype(int)
    indices2all = indices2all.astype(int)
    indices3all = indices3all.astype(int)
    indices4all = indices4all.astype(int)
    
    #Begin amassing parameterization inputs into large matrices and then subsample
    for jj in range(np.size(InputList)):
        SpeciesMe = InputList[jj]
        gg = 0
        print('Subsetting ', SpeciesMe, ' ...')
        for dd in Days:
            if DayList[dd] < 10:
                DayName = '0' + str(DayList[dd])
            else:
                DayName = str(DayList[dd])
            print('Loading July ', DayName) 
            for ii in Years:
                YearName = str(ii)
                #Directory where MERRA2 GMI data are housed
                dirname = '/discover/nobackup/projects/gmao/merra2_gmi/pub/Y' + YearName + '/M' + MonthName + '/'

                #Import Variable information                                                                                                  
                if SpeciesMe in  DACList:
                    SpeciesName = dirname + 'MERRA2_GMI.tavg24_3d_dac_Nv.' + YearName + MonthName + DayName + '.nc4'
                elif SpeciesMe in MetList:
                    SpeciesName = dirname + 'MERRA2_GMI.tavg3_3d_met_Nv.' + YearName + MonthName + DayName + '.nc4'
                elif SpeciesMe in CloudList:
                    SpeciesName = dirname + 'MERRA2_GMI.tavg3_3d_cld_Nv.' + YearName + MonthName + DayName + '.nc4'
                elif SpeciesMe == 'GMISTRATO3':
                    SpeciesName = dirname + 'MERRA2_GMI.tavg24_2d_dad_Nx.' + YearName + MonthName + DayName + '.nc4'
                elif SpeciesMe == 'ALBUV':
                    SpeciesName = '/discover/nobackup/dcanders/QuickChem/Data/OMILER_345nm_M2GMIGrid.nc'
                elif SpeciesMe in PMList:
                    SpeciesName = dirname + 'MERRA2_GMI.tavg24_3d_adf_Nv.' + YearName + MonthName + DayName + '.nc4'
                    SpeciesName2 = dirname + 'MERRA2_GMI.tavg3_3d_met_Nv.' + YearName + MonthName + DayName + '.nc4'                

                if SpeciesMe == 'Lat':
                    Species = np.arange(-90,90.5,.5) #The latitude values for the simulation
                    sizePL = np.size(Species) 
                elif SpeciesMe == 'ALBUV':
                    SpeciesFile = nc.Dataset(SpeciesName)
                    Species = SpeciesFile['LER'][:]
                    SpeciesFile.close()
                    Species = Species[Month-1,:,:]
                    Species = np.squeeze(Species)
                    sizePL = np.shape(Species)
                else:
                    SpeciesFile=nc.Dataset(SpeciesName)

                    if SpeciesMe == 'GMISTRATO3':
                        Species = SpeciesFile['GMITO3'][:]
                        Species2 = SpeciesFile['GMITTO3'][:]
                        Species2 = np.squeeze(Species2)
                    elif SpeciesMe == 'TAUCLIUP' or SpeciesMe == 'TAUCLIDWN':
                        Species = SpeciesFile['TAUCLI'][:]
                    elif SpeciesMe == 'TAUCLWUP' or SpeciesMe =='TAUCLWDWN':
                        Species = SpeciesFile['TAUCLW'][:]
                    elif SpeciesMe in PMList:
                        SpeciesShort1 = SpeciesFile['BCSCACOEF'][:]+SpeciesFile['DUSCACOEF'][:]+SpeciesFile['NISCACOEF'][:]+SpeciesFile['OCSCACOEF'][:] + SpeciesFile['SSSCACOEF'][:]+SpeciesFile['SUSCACOEF'][:]
                        SpeciesFile2 = nc.Dataset(SpeciesName2)
                        # Need to convert scattering coefficient to AOD.  Calculate grid box height.
                        HeightMe = SpeciesFile2['H'][:]
                        HeightMe = np.squeeze(HeightMe)
                        SpeciesShort1 = np.squeeze(SpeciesShort1)
                        SpeciesFile2.close()
                        HeightMe = np.squeeze(np.mean(HeightMe,0))
                        LayerDepth = np.zeros(np.shape(SpeciesShort1))
                        for ttt in range(np.shape(LayerDepth)[0]):
                            if ttt == 0:
                                LayerDepth[ttt,:,:] = (HeightMe[ttt,:,:]-HeightMe[ttt+1,:,:])
                            elif ttt == (np.shape(LayerDepth)[0]-1):
                                LayerDepth[ttt,:,:] = HeightMe[ttt-1,:,:]-HeightMe[ttt,:,:]
                            else:
                                LayerDepth[ttt,:,:] = (HeightMe[ttt,:,:]-HeightMe[ttt+1,:,:])/2 + (HeightMe[ttt-1,:,:]-HeightMe[ttt,:,:])/2
                        Species = LayerDepth*SpeciesShort1
                    else:
                        Species = SpeciesFile[SpeciesMe][:]
                    SpeciesFile.close()

                if (SpeciesMe in MetList) or (SpeciesMe in CloudList): #These are output every 3 hours.  Calculate 24 hr average
                    Species = np.mean(Species,0)

                Species=np.squeeze(Species)
                sizePL=np.shape(Species)

                if SpeciesMe == 'GMISTRATO3':
                    Species=Species-Species2
                # Calculate the Optical Depth above a grid point
                elif SpeciesMe == 'TAUCLIUP' or SpeciesMe == 'TAUCLWUP' or SpeciesMe == 'AODUP':
                    SpeciesSize = np.shape(Species)
                    Species2 = np.zeros(SpeciesSize)
                    for xx in range(SpeciesSize[0]):
                        if xx > 0:
                            ExtractedSubset=Species[0:xx,:,:]
                            Species2[xx,:,:]=np.sum(ExtractedSubset,axis=0)
                    Species = Species2
                # Calculate the Optical Depth below a grid point
                elif SpeciesMe == 'TAUCLIDWN' or SpeciesMe == 'TAUCLWDWN' or SpeciesMe == 'AODDWN':
                    SpeciesSize = np.shape(Species)
                    Species2 = np.zeros(SpeciesSize)
                    for xx in range(SpeciesSize[0]-1):
                        ExtractedSubset = Species[(xx+1):SpeciesSize[0],:,:]
                        Species2[xx,:,:] = np.sum(ExtractedSubset,axis=0)
                    Species = Species2

                # Consolidate data for each year into one large matrix.                                                                       
                if ii==StartYear:
                    if (SpeciesMe in DACList) or (SpeciesMe in MetList) or (SpeciesMe in CloudList) or (SpeciesMe in PMList):
                        SpeciesSummary = np.zeros(((StopYear-StartYear),sizePL[0],sizePL[1],sizePL[2]))
                    elif (SpeciesMe == 'GMISTRATO3') or (SpeciesMe == 'ALBUV'):
                        SpeciesSummary = np.zeros(((StopYear-StartYear),sizePL[0],sizePL[1]))
                    elif SpeciesMe == 'Lat':
                        SpeciesSummary = np.zeros(((StopYear-StartYear),361))

                if (SpeciesMe in DACList) or (SpeciesMe in MetList) or (SpeciesMe in CloudList) or (SpeciesMe in PMList):
                    SpeciesSummary[ii-StartYear, :, :, :] = Species
                elif (SpeciesMe == 'GMISTRATO3')  or (SpeciesMe == 'ALBUV'):
                    SpeciesSummary[ii-StartYear,:,:] = Species
                elif SpeciesMe == 'Lat':
                    SpeciesSummary[ii-StartYear,:] = Species
        
                #END loop creating merge variable   
            #Extract input data
            if gg == 0:
                SpeciesExtracted = np.zeros(np.shape(indices1all))
            if (SpeciesMe in DACList) or (SpeciesMe in MetList) or (SpeciesMe in CloudList) or (SpeciesMe in PMList):
                SpeciesExtracted[gg,:] = SpeciesSummary[indices1all[gg,:], indices2all[gg,:], indices3all[gg,:], indices4all[gg,:]]
            elif (SpeciesMe == 'GMISTRATO3')  or (SpeciesMe == 'ALBUV'):
                SpeciesExtracted[gg,:] = SpeciesSummary[indices1all[gg,:], indices3all[gg,:], indices4all[gg,:]]
            elif SpeciesMe == 'Lat':
                SpeciesExtracted[gg,:] = SpeciesSummary[indices1all[gg,:], indices3all[gg,:]]
            gg = gg+1
        
        SpeciesExtracted = SpeciesExtracted.flatten()
        SpeciesSize = np.size(SpeciesExtracted)

        #Save input data
        if ApendMe == 0:
            if jj == 0:
                #Create new file                                                                                                                                   
                savedir = '/user/dir/'
                savename = savedir + 'ParameterizationInputs_' + str(StartYear) + '_' + str(StopYear-1) + '_' + MonthName + '.nc'
                ncfile = nc.Dataset(savename, mode='w', format = 'NETCDF4_CLASSIC')
                dim1 = ncfile.createDimension('Dim1',SpeciesSize)
            Tempsavevar = ncfile.createVariable(SpeciesMe, np.float32,('Dim1',))
            Tempsavevar[:] = SpeciesExtracted
        else:
            if jj == 0:
                #Open existing file
                savedir = '/user/dir/'
                savename = savedir + 'ParameterizationInputs_' + str(StartYear) + '_' + str(StopYear-1) + '_' + MonthName + '.nc'
                ncfile = nc.Dataset(savename, mode='a')
            Tempsavevar = ncfile.createVariable(SpeciesMe, np.float32,('Dim1',))
            Tempsavevar[:] = SpeciesExtracted
    ncfile.close()

MergeTime = (time.time() - startTime3)
print('Time to Subset Inputs: ', MergeTime)

if WritedatFile == 1:
    print('Writing .dat files...')
    savedir = '/user/dir/'
    netcdfname = savedir + 'ParameterizationInputs_' + str(StartYear) + '_' + str(StopYear-1) + '_' + MonthName + '.nc'
    # SZA file is only needed if it's not saved in the netcdfname file
    #SZAname = savedir + 'FilteredInputs_SZA_' + str(StartYear) + '_' + str(StopYear-1) + '_' + MonthName + '.nc'
    
    #Create 2D array that combines all the information
    NetSumFile = nc.Dataset(netcdfname)
    #SZAFile = nc.Dataset(SZAname)
    DimMe = np.shape(NetSumFile['NO2'][:])
    for ii in range(len(VarList)):
        if VarList[ii]=='SZA':
            Temp = SZAFile['SZA'][:]
        else:
            Temp = NetSumFile[VarList[ii]][:]
        if VarList[ii] == 'PL':
            Temp = Temp/100 #Converts pressure to hPa

        if ii == 0:
            Summary = Temp
        else:
           Summary = np.vstack((Summary,Temp))

    #SZAFile.close()
    NetSumFile.close()

    # Create parameterization inputs file
    Summary=Summary.T
    datfilename = savedir + 'ParameterizationInputs_' + str(StartYear) + '_' + str(StopYear-1) + '_' + MonthName + '.dat'
    datfile = open(datfilename, 'w')
    line1 = str(len(VarList)) + ',3 \n'
    line3 = str(StartYear) + '_' + str(StopYear-1) + '_M' + MonthName + '\n'

    datfile.write(line1)
    for ii in range(len(VarList)):
        if ii < (len(VarList)-1):
            temp = VarList[ii] + ','
            datfile.write(temp)
        else:
            temp = VarList[ii] + '\n'
            datfile.write(temp)

    datfile.write(line3)
    np.savetxt(datfile, Summary, fmt='%.7e', delimiter=' ')
    datfile.close()

    # Create parameterization targets file from OH saved in nc file
    savedir = '/user/dir/'
    datfilename = savedir + 'ParameterizationOutputs_' + str(StartYear) + '_' + str(StopYear-1) + '_' + MonthName + '.dat'
    datfile = open(datfilename, 'w')
    line1 = '1,3 \n'
    line2 = 'OH \n'
    line3 = str(StartYear) + '_' + str(StopYear-1) + '_M' + MonthName + '\n'

    savedir = '/uers/dir/'
    netcdfname = savedir + 'RandomOH_Indices_' + str(StartYear) + '_' + str(StopYear-1) + '_' + MonthName + '.nc'
    OHfile = nc.Dataset(netcdfname)
    OH = OHfile['OH'][:]
    #OH = OH[GoodVals]
                      
    OH = OH.flatten()
                             
    datfile.write(line1)
    datfile.write(line2)
    datfile.write(line3)
    np.savetxt(datfile, OH, fmt='%.7e', delimiter=' ')
    datfile.close()

AllTime = (time.time() - startTime)
