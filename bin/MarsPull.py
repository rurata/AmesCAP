#!/usr/bin/env python3

# Load generic Python modules
import sys        # system command
import os         # access operating systems function

# make print statements appear in color
def prCyan(skk): print("\033[96m{}\033[00m".format(skk))
def prYellow(skk): print("\033[93m{}\033[00m".format(skk))

# try to import specific scientic modules
try:
    import numpy as np
    import argparse     # parse arguments
    import requests

except ImportError as error_msg:
    prYellow("Error while importing modules")
    print(f"Error was: {error_msg.message}")
    exit()

except Exception as exception:
    # output unexpected Exceptions
    print(exception, False)
    print(f"{exception.__class__.__name__}: {exception.message}")
    exit()

# ======================================================
#                  ARGUMENT PARSER
# ======================================================
parser = argparse.ArgumentParser(
    description=("""\033[93mUilities for accessing files on the MCMC 
                NAS Data Portal \033[00m """), 
                formatter_class=argparse.RawTextHelpFormatter)

parser.add_argument('-id', '--id', type=str,
                    help=("Query data by simulation identifier "
                    "corresponding to \n"
                    "a subdirectory of legacygcmdata/:\n"
                    "\033[96mhttps://data.nas.nasa.gov/legacygcm/"
                    "data_legacygcm.php?dir=/legacygcmdata\033[00m\n"
                    "Current options include: "
                    "'\033[93mACTIVECLDS\033[00m', "
                    "'\033[93mINERTCLDS\033[00m', "
                    "and '\033[93mACTIVECLDS_NCDF\033[00m'\n"
                    "> Usage: MarsPull.py -id  INERTCLDS \n\n"))

parser.add_argument('-ls', '--ls', nargs='+', type=float,
                    help="Query data by solar longitude (Ls) - requires"
                    " a simulation identifier (--id)\n"
                    "> Usage: MarsPull.py -id ACTIVECLDS -ls 90.\n"
                    ">        MarsPull.py -id ACTIVECLDS -ls [start] "
                    "[stop] \n\n")

parser.add_argument('-f', '--filename', nargs='+', type=str,
                    help=("Query data by filename - requires"
                    " a simulation identifier (--id)\n"
                    "> Usage: MarsPull.py -id ACTIVECLDS_NCDF -f "
                    "fort.11_0730 fort.11_0731"))

# ======================================================
#                  DEFINITIONS
# ======================================================
saveDir = (f"{os.getcwd()}/")

# available files by Ls:
lsStart = np.array([  0,   5,  10,  15,  19,  24,  29,  34,  38,  43,
                    48,  52,  57,  61,  66,  70,  75,  79,  84,  88,
                    93,  97, 102, 106, 111, 116, 121, 125, 130, 135,
                   140, 146, 151, 156, 162, 167, 173, 179, 184, 190, 
                   196, 202, 209, 215, 221, 228, 234, 241, 247, 254, 
                   260, 266, 273, 279, 286, 292, 298, 304, 310, 316,
                   322, 328, 333, 339, 344, 350, 355])

lsEnd = np.array([  4,   9,  14,  19,  24,  29,  33,  38,  42,  47,
                    52,  56,  61,  65,  70,  74,  79,  83,  88,  92,
                    97, 101, 106, 111, 115, 120, 125, 130, 135, 140,
                   145, 150, 156, 161, 167, 172, 178, 184, 190, 196,
                   202, 208, 214, 221, 227, 233, 240, 246, 253, 259,
                   266, 272, 279, 285, 291, 297, 304, 310, 316, 321,
                   327, 333, 338, 344, 349, 354,   0])


def download(url, filename):
    """
    Downloads a file from  https://data.nas.nasa.gov.
    
    The file to download is specified by appending the above URL with
    the legacy gcm subdirectory + the filename. The filename can be 
    provided by the user directly or determined based on the 
    user-requested solar longitude (Ls). The simulation identifier (ID)
    must always be provided.
    
    Parameters
    ----------
    URL: str
        The URL to download from. This is built from:
        https://data.nas.nasa.gov/legacygcm/download_data_legacygcm.php?file=/legacygcmdata/
        by appending the simulation ID to the end of the URL.
    filename: str
        The name of the file to download
    
    Raises
    ------
    rsp.status_code
        A file-not-found error
    
    Returns
    -------
    downloaded file
    """

    ### _, fname = os.path.split(filename)
    
    # use a context manager to make an HTTP request and file
    rsp = requests.get(url, stream=True)
    
    # get the total size, in bytes, from the response header
    total_size = rsp.headers.get('content-length')

    if rsp.status_code == 404:
        print(f"File not found! Error code: {rsp.status_code}")
    
    else:
        
        # if the header is found, file size known. Return progress bar
        if total_size is not None:
            with open(filename, 'wb') as f:
                downloaded = 0
                if total_size:
                    
                    # define the size of the chunk to iterate over (Mb)
                    chunk_size = max(int(total_size/1000), 1024*1024)
                
                # iterate over every chunk and calculate % of total_size
                for chunk in rsp.iter_content(chunk_size=chunk_size):
                    downloaded += len(chunk)
                    f.write(chunk)
                    
                    # calculate current %
                    status = int(50*downloaded/total_size)
                    
                    # print progress to console then flush console
                    sys.stdout.write('\r[{}{}]'.format('#' * status, '.' * (50 - status)))
                    sys.stdout.flush()
            sys.stdout.write('\n')
        
        ## else:
        ##     # If the header is not found, skip the progressbar
        ##     print('Downloading %s ...' % (fname))
        ##     with open(local_file, 'wb')as f:
        ##         f.write(data.content)
        ##     print('%s Done' % (fname))


# ======================================================
#                  MAIN PROGRAM
# ======================================================
def main():
    simID = parser.parse_args().id

    if simID is None:
        prYellow(
            "***Error*** Simulation identifier [-id, --id] required."
            "Use 'MarsPull.py -h' for additional help.")
        exit()

    URLbase = ("https://data.nas.nasa.gov/legacygcm/"
               "download_data_legacygcm.php?file=/legacygcmdata/"
               + simID + "/")

    if parser.parse_args().ls:
        lsInput = np.asarray(parser.parse_args().ls)
        
        if len(lsInput) == 1:
            # Query the file corresponding to this Ls
            i_start = np.argmin(np.abs(lsStart-lsInput))
            if lsInput < lsStart[i_start]:
                i_start -= 1
            
            i_request = np.arange(i_start, i_start+1)

        elif len(lsInput) == 2:
            # Query the files between [start] & [stop], inclusive
            i_start = np.argmin(np.abs(lsStart-lsInput[0]))
            if lsInput[0] < lsStart[i_start]:
                i_start -= 1

            i_end = np.argmin(np.abs(lsEnd-lsInput[1]))
            if lsInput[1] > lsEnd[i_end]:
                i_end += 1

            i_request = np.arange(i_start, i_end+1)

        print(f"Saving {len(i_request)} files in {saveDir}")
        for ii in i_request:
            # Legacy .nc files
            if simID == 'ACTIVECLDS_NCDF':
                fName = 'LegacyGCM_Ls%03d_Ls%03d.nc' % (lsStart[ii], lsEnd[ii])
            # fort.11 files
            else:
                fName = 'fort.11_%04d' % (670+ii)
                print(f"FILENAME = {fName}")
                fName2 = f"fort.11_{(670+ii):.4d}"
                print(f"FILENAME2 = {fName2}")

            url = URLbase+fName
            filename = saveDir + fName
            print(f"Downloading {url}...")
            download(url, filename)

    elif parser.parse_args().filename:
        f_input = np.asarray(parser.parse_args().filename)
        for ff in f_input:
            url = URLbase+ff
            filename = saveDir+ff
            print(f"Downloading {url}...")
            download(url, filename)
    else:
        prYellow("No data requested. Use -ls or -f to specify data to download.")

# ======================================================
#                  END OF PROGRAM
# ======================================================

if __name__ == '__main__':
    main()
