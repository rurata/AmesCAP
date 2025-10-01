#!/usr/bin/env python3

#
#Assumes the companion scripts are in the same directory
#Load generic Python Modules
import argparse #parse arguments
import sys
import os
import numpy as np
import time       # monitor processing time
from netCDF4 import Dataset

#===========
from amescap.Ncdf_wrapper import Ncdf
#from amesgcm.FV3_utils import regrid_Ncfile #regrid source
from amescap.Script_utils import prYellow,prCyan,prRed,find_tod_in_diurn,FV3_file_type,filter_vars,get_longname_units
from amescap.FV3_utils import find_n,interp_KDTree,axis_interp,expand_index
#==========


#======================================================
#                  ARGUMENTS PARSER
#======================================================
parser = argparse.ArgumentParser(description="""\033[93m MarsObs  regridding of FV3 diurn files onto MCS diurnal obervations \n \033[00m""",
                                formatter_class=argparse.RawTextHelpFormatter)

parser.add_argument('input_file', nargs='+',
                                help='***.nc file or list of ***.nc files. Must be time-shifted to uniform local time and pressure interpolated ****.atmos.diurn_T_pstd.nc')

parser.add_argument('-my','--my',type=int,required=True,
                 help=""" Mars Years to reggrid the datafile. Supported years are [29-34]  \n"""
                      """>  Usage: MCS_Sim.py ****.atmos.diurn_T_pstd.nc -my 32 \n""") 
                      
parser.add_argument('-noNan','--noNan',action='store_true',default=False,
                 help=""" Do not put NaN in interpolated file \n"""
                      """>  Usage: MCS_Sim.py ****.atmos.diurn_T_pstd.nc -my 32 -noNaN \n""")                       
                                                             
parser.add_argument('-include','--include',nargs='+',
                     help="""For data reduction, filtering, time-shift, only include listed variables. Dimensions and 1D variables are always included \n"""
                         """> Usage: MarsObsp.py ****.atmos.diurn_Tpstd.nc -my 32 --include temp dust    \n"""
                         """\033[00m""")               
                                                                
parser.add_argument('--debug',  action='store_true', help='Debug flag: release the exceptions')





# NAS default 
obs_directory='/u/mkahre/MCMC/analysis/obsdata/amesgcmOBS/'
temp_dust_ice_NaN_in_MCS=['temp','dust','wice'] 


#================USER INPUT===============================
temp_dust_ice_NaN_in_GCM=['temp','dust_mass','ice_mass']
#=========================================================
def main():
    
    #Get Mars year
    my=parser.parse_args().my
    name_target='%s/MCSdata_binned_MY%i_unoff_diurn_T_pstd.nc'%(obs_directory,my)
    print('Using>>>',name_target)
    out_ext='_regrid_MCS%i'%(my)
    
    file_list=parser.parse_args().input_file
    cwd=os.getcwd()
    path2data=os.getcwd()
    
    #=================================================================
    #========================  Regrid  files =========================
    #=================================================================    
        
    #Add path unless full path is provided

    fNcdf_t=Dataset(name_target,'r')

    for filei in file_list:
        start_time = time.time()
        #Add path unless full path is provided
        if not ('/' in filei):
            fullnameIN = path2data + '/' + filei
        else:
            fullnameIN=filei
        fullnameOUT = fullnameIN[:-3]+out_ext+'.nc'
        
        f_in = Dataset(fullnameIN, 'r', format='NETCDF4_CLASSIC')
                
        var_list = filter_vars(f_in,parser.parse_args().include) # get all variables in input file
        
        
        #List of axis to copy. Handle the special case of diurn files that may have have different dimension name (e.g. time_of_day_4 and time_of_day_24)
        
        axis_list_to_copy=['lat', 'lon','pstd','time','areo']
        f_type_t,_=FV3_file_type(fNcdf_t)
        if f_type_t=='diurn':
            tod_name_t=find_tod_in_diurn(fNcdf_t)
            axis_list_to_copy.append(tod_name_t)
        
        
        fnew = Ncdf(fullnameOUT) # define a Ncdf object from the Ncdf wrapper module
        
        #Copy all dims  from the target file to the new file
        fnew.copy_all_dims_from_Ncfile(fNcdf_t)
        
        #Copy time_of_day_2 array (not present in f_in file)
        fnew.copy_Ncaxis_with_content(fNcdf_t.variables[tod_name_t])
        prCyan("Copying axis from target file: %s..."%(tod_name_t))

        #Loop over all variables in file
        for ivar in var_list:
            varNcf     = f_in.variables[ivar]
            longname_txt,units_txt=get_longname_units(f_in,ivar)
            if  ivar in axis_list_to_copy :
                    prCyan("Copying axis from target file: %s..."%(ivar))
                    fnew.copy_Ncaxis_with_content(fNcdf_t.variables[ivar])
            elif varNcf.dimensions[-2:]==('lat', 'lon'): #Ignore variables like  'time_bounds', 'scalar_axis' or 'grid_xt_bnds'...
                prCyan("Regridding: %s..."%(ivar))
                var_OUT=regrid_to_OBS(varNcf,f_in,fNcdf_t)  
                #===Rename the time_of_day dimensions to the target file as necessary===
                dim_name_OUT=varNcf.dimensions
                if f_type_t=='diurn':dim_name_OUT=(dim_name_OUT[0],tod_name_t)+dim_name_OUT[2:]
                #======Filter NaN====
                #This is only done when the bypass flag no-NaN is unsued and if the variable is in the provided input at the begining of this file
                if parser.parse_args().noNan is False and ivar in temp_dust_ice_NaN_in_GCM:
                    ii=temp_dust_ice_NaN_in_GCM.index(ivar) #get the mapping index for the variable name in GCM to variable name in MCS
                    VAR_MCS=fNcdf_t.variables[temp_dust_ice_NaN_in_MCS[ii]][:]
                    var_OUT+=0*VAR_MCS  #Add GCM + 0*MCS This has the effect of Mmsking NaN values
                
                fnew.log_variable(ivar,var_OUT,dim_name_OUT,longname_txt,units_txt)
        fnew.close()
        fNcdf_t.close()
        print("Completed in %.3f sec" % (time.time() - start_time))
            

#END of script
#============Import modules==============


def interp_sampler(varIN,Llev,Lfull,type_int='log',reverse_input=False,index=None,modulo=None):
    '''
    Linear or logarithmic interpolation along a 1D axis, to a ND grid. This can be seen as the reverse operation of the vertical interpolation   Alex Kling 8-10-21
    Args:
        varIN: variable to interpolate (N-dimensional array with INTERPOLATION AXIS FIRST,                    (e.g temp[tod,time,lat,lon])
        Llev : INPUT  levels as a 1D array. Same size as varIN[0,...] (e.g tod)                                          v    v   v   v
        Lfull: ND axis to be interpolated, same DIMENSIONS as varIN  with interpolated axis FIRST (e.g time_of_day_full[ 6 ,  1 , 2 , 48]). (Shape may be different than varIN)
        reverse_input (boolean) : reverse input arrays, if monotonically decreasing with N
        type_int : 'log' for logarithmic (typically pressure), 'lin' for linear (typically altitude)
        index: indices for the interpolation, already processed as [klev,Ndim] 
               Indices will be recalculated if not provided.
        modulo  (float) : apply a modulo for interpolation to allow loop around. e.g. 360 for longitudes, 24 to time of day       
    Returns:
        varOUT: variable interpolated from Llev pressure or altitude levels
        
    with A = log(Llev/pn+1)/log(pn/pn+1) in 'log' mode     
         A =    (zlev-zn+1)/(zn-zn+1)    in 'lin' mode
         
    '''
    dimsIN=varIN.shape               #get input variable dimensions
    Nfull=Lfull.shape[0]
    Nlev=len(Llev)

    dimsOUT=tuple(np.append(Nfull,dimsIN[1:]))
    Ndim= np.int(np.prod(dimsIN[1:]))          #Ndim is the product  of all dimensions but the vertical axis
    varIN= np.reshape(varIN, (Nlev, Ndim))    #flatten the other dimensions to (Nfull, Ndim)
    Lfull= np.reshape(Lfull, (Nfull, Ndim) )   #flatten the other dimensions to (Nfull, Ndim)
    varOUT=np.zeros((Nfull, Ndim))
    Ndimall=np.arange(0,Ndim)                   #all indices (does not change)
    
    if reverse_input:
        Llev=Llev[::-1,:] 
        varIN=varIN[::-1,:]
    
    for k in range(0,Nfull):
        #Find nearest layer to Llev[k]
        if np.any(index):
            #index have been pre-computed:  
            n= index[k,:]
        else:
            # Compute index on the fly for that layer. 
            # Note that inverse_input is always set to False as if desired, Lfull was reversed earlier
            n= np.squeeze(find_n(Llev[k],Lfull,False)) 
        #==Slower method (but explains what is done below): loop over Ndim======
        for ii in range(Ndim):
            if n[ii]<Nlev-1:
                alpha=(Lfull[k,ii]-Llev[n[ii]+1])/(Llev[n[ii]]-Llev[n[ii]+1])
                varOUT[k,ii]=varIN[n[ii],ii]*alpha+(1-alpha)*varIN[n[ii]+1,ii]
        
        #=================    Fast method  no loop  =======================
        #TODO Can we adapt the original (and faster) implementation in FV3_utils.py  to work with unstructured interpolation?
    return np.reshape(varOUT,dimsOUT)
    
        
def regrid_to_OBS(VAR_Ncdf,file_Nc_in,file_Nc_target):
    '''
    Regrid a Ncdf variable from one file's structure to match another file  [Alex Kling , May 2021]
    Args:
        VAR_Ncdf: A netCDF4 variable OBJECT, e.g. 'f_in.variables['temp']' from the source file
        file_Nc_in: The opened netcdf file object  for that input variable, e.g f_in=Dataset('fname','r')
        file_Nc_target: Anopened netcdf file object  for the target grid t e.g f_out=Dataset('fname','r')  
    Returns:
        VAR_OUT: the VALUES of VAR_Ncdf[:], interpolated on the grid for the target file. 
    while the closest points in the vertical are a few 10's -100's meter in the PBL, which would results in excessive weighting in the vertical.
    '''
    ftype_in,zaxis_in=FV3_file_type(file_Nc_in)
    ftype_t,zaxis_t=FV3_file_type(file_Nc_target)
    
    tod_name_in=find_tod_in_diurn(file_Nc_in)
    #Sanity check

    if ftype_in !=ftype_t:
        print("""*** Warning*** in regrid_Ncfile, input file  '%s' and target file '%s' must have the same type"""%(ftype_in,ftype_t))
        
    if zaxis_in!=zaxis_t:
        print("""*** Warning*** in regrid_Ncfile, input file  '%s' and target file '%s' must have the same vertical grid"""%(zaxis_in,zaxis_t))
        
    if zaxis_in=='pfull' or zaxis_t=='pfull':
        print("""*** Warning*** in regrid_Ncfile, input file  '%s' and target file '%s' must be vertically interpolated"""%(zaxis_in,zaxis_t))
        
    
    #Load data for target file
    variableNames_MCS = file_Nc_target.variables.keys();
    areo_t=file_Nc_target.variables['areo'][:]
    lat_t=file_Nc_target.variables['lat'][:]
    lon_t=file_Nc_target.variables['lon'][:]
    lev_t=file_Nc_target.variables['pstd'][:]
    timeave=file_Nc_target.variables['timeave'][:] #Time min and max vary significantly (few hours) only at the poles: see dtimemax-dtimemin

    #file_Nc_target.close()
    
    # Load source data
    tod_in=file_Nc_in.variables[tod_name_in][:]
    lat_in=file_Nc_in.variables['lat'][:]
    lon_in=file_Nc_in.variables['lon'][:]
    lev_in=file_Nc_in.variables['pstd'][:]
    areo_in=file_Nc_in.variables['areo'][:]
    
    #Get array elements
    var_OUT=VAR_Ncdf[:]
    #file_Nc_in.close()
    
    
    #STEP 1: Lat/lon interpolation always performed unless target lon and lat are identical
    if not (np.all(lat_in==lat_t) and np.all(lon_in==lon_t)) :
        var_OUT=interp_KDTree(var_OUT,lat_in,lon_in,lat_t,lon_t,4) 
    
    print('Lon/lat interp done...')
    
    #STEP 2: Linear or log interpolation if there is a vertical axis
    zaxis_in='pstd'
    zaxis_t='pstd'
    
    if zaxis_in in VAR_Ncdf.dimensions:
        pos_axis=VAR_Ncdf.dimensions.index(zaxis_in) #Get position: 'pstd' position is 1 in ('time', 'pstd', 'lat', 'lon')
        lev_in=file_Nc_in.variables[zaxis_in][:]
        lev_t=file_Nc_target.variables[zaxis_t][:]
        #Check if the input need to be reversed. The values are reversed if increasing upward (yes, this is counter intuitive), because we are re-using 
        #the find_n() function  which was designed for pressure interpolation (decreases upward)
        #so 
        if lev_in[0]>lev_in[-1]:
            reverse_input=True
        else:
            reverse_input=False
        if zaxis_in in ['zagl','zstd'] :
            intType='lin'
        elif zaxis_in=='pstd':
            intType='log'  
        var_OUT=axis_interp(var_OUT, lev_in,lev_t, pos_axis, reverse_input=reverse_input, type_int=intType)
    
    print('Vertical interp done...')
    
    #STEP 3: Linear interpolation in Ls
    areo_in=np.mean(areo_in,axis=1)    
    if 'time' in VAR_Ncdf.dimensions:
        pos_axis=0
        var_OUT=axis_interp(var_OUT, np.squeeze(areo_in)%360,np.squeeze(areo_t)%360, pos_axis, reverse_input=False, type_int='lin',modulo=360)   
    
    print('Time interp done...')
    
    #STEP 4: Linear interpolation  from regularly-spaced to irregular local time:
    #It is impractical (and redundant) to compute the indices for all altitudes. We therefore only compute those for each time, lat and lon and expands those.
    if ftype_in =='diurn':
        
        #Put time_of_day axis first
        Lfull=timeave.swapaxes(0,1)
        """
        #We will re-use the indices for each files, this speeds-up the calculation
        compute_indices=True    
        for ivar in var_list: 
            if (fNcdf.variables[ivar].dimensions==('time','pfull', 'lat', 'lon') or
             fNcdf.variables[ivar].dimensions==('time',tod_name,'pfull', 'lat', 'lon')):
                if compute_indices:
                    prCyan("Computing indices ...")
                    index=find_n(L_3D_P,lev_in,reverse_input=need_to_reverse)
                    compute_indices=False
        """
        index=find_n(tod_in,Lfull)  #index.reshape([2,len(areo_t), len(lat_t), len(lon_t)])
    
        var_OUT=var_OUT.swapaxes(0,1).copy() #Put time_of_day axis first
        
        
        #This is a 3D variable, expand indices and LFULL from (tod,time,lat,lon) to (tod,time,LEV,lat,lon)
        if zaxis_in in VAR_Ncdf.dimensions:
            INDEX=expand_index(index,var_OUT.shape,2) 
            LFULL=np.repeat(Lfull[:,:,np.newaxis,...],len(lev_t),axis=2)
            var_OUT=interp_sampler(var_OUT,tod_in,LFULL,type_int='lin',reverse_input=False,index=INDEX,modulo=24.).swapaxes(0,1)
        #This is a 2D variable  
        else: 
            var_OUT=interp_sampler(var_OUT,tod_in,Lfull,type_int='lin',reverse_input=False,index=index,modulo=24.).swapaxes(0,1)
    return var_OUT   
    
if __name__ == "__main__":
    main()



