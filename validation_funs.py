# validation_funs.py

# Created by David Miller, dmil1991@gmail.com
# Updated: August 4, 2014

# Description and Usage:
""" These are many functions that used to validate the outputs of LandTrendr against a reference. First usage is designed
to compare LandTrendr disturbance outputs to labels from TimeSync. The functions here will be called by other scripts for
specific use cases."""

import os, sys, csv, numpy, gdal, math, glob
from gdalconst import *

def csv2list(file):
    # Returns an imported csv as a list
    return list(csv.reader(open(file,"rU")))

def extract_kernel(spec_ds,x,y,width,height,band,transform):
    # Modified original code from Zhiqiang Yang (read_spectral) at Oregon State University
    """read spectral value from band centered around [x,y] with width and height"""
    xoffset = int(x - transform[0])/30 - width/2
    yoffset = int(y - transform[3])/-30 - height/2

    # plot is outside the image boundary
    if xoffset <0 or yoffset > spec_ds.RasterYSize - 1:
        return [-9999]
    this_band = spec_ds.GetRasterBand(band)
    specs = this_band.ReadAsArray(xoffset, yoffset, width, height)
    return specs

def extract_kernel_and_coords(spec_ds,x,y,width,height,band,transform):
    """read spectral value from band centered around [x,y] with width and height"""
    xoffset = int(x - transform[0])/30 - width/2
    yoffset = int(y - transform[3])/-30 - height/2

    x_indeces = numpy.arange(xoffset, xoffset+width)
    y_indeces = numpy.arange(yoffset, yoffset+height)
    x_coords = x_indeces * transform[1] + transform[0] 
    y_coords = y_indeces * transform[5] + transform[3] 
    all_coords = numpy.zeros([x_coords.size,y_coords.size,2])
    for ind, i in enumerate(x_coords):
        for jnd, j in enumerate(y_coords):
            all_coords[jnd,ind] = (i,j) 

    # plot is outside the image boundary
    if xoffset <0 or yoffset > spec_ds.RasterYSize - 1:
        return [-9999]
    this_band = spec_ds.GetRasterBand(band)
    specs = this_band.ReadAsArray(xoffset, yoffset, width, height)
    return specs, all_coords

def list2csv(filepath,arg):
    # exports a list to a csv
    ofile = open(filepath,'wb')
    writer = csv.writer(ofile, dialect='excel')
    for line in arg:
        writer.writerow(line)
    ofile.close()

def sixDigitTSA(pathrow):
    # make TSA six digits for searching directories
    # pass pathrow, first coerce to string if not already
    if type(pathrow) != str: pathrow = str(pathrow)
    # check length, and make TSA six digit
    # e.g. for 4529, --> 045029
    pathrow = pathrow.strip()
    if len(pathrow) < 4:
        sys.exit("Enter TSA with at least 4 digits")
    elif len(pathrow) == 4:
        pathrow = '0' + pathrow[:2] + '0' + pathrow[2:]
    elif len(pathrow) == 5:
        if pathrow[0] == '0':
            pathrow = pathrow[:3] + '0' + pathrow[3:]
        elif pathrow[2] == '0':
            pathrow = '0' + pathrow
        else:
            sys.exit("Provide TSA of form PPRR e.g. 4529")
    return pathrow

def fourDigitTSA(pathrow):
    # call to convert to four digit version to look up TSA in plots table
    pathrow6 = sixDigitTSA(pathrow)
    pathrow4 = pathrow6[1:3] + pathrow6[4:]
    return pathrow4

def getInputFileVert(pathrow):
    """ Get input vertyrs and vertvals files from locations 
    base on the pathrow (TSA) """
    
    # change pathrow to six-digit
    pathrow = sixDigitTSA(pathrow)
    dir = '/projectnb/trenders/scenes/{0}/outputs/nbr/'.format(pathrow)
    os.chdir(dir)
    
    # get vertyrs
    globCheck = '*paramset01_*[0-9]_vertyrs.bsq'
    filelistyrs = glob.glob(globCheck)
    if len(filelistyrs) > 1: sys.exit("Only one reference file expected in {0}".format(dir))
    elif len(filelistyrs) == 0:
        dir = '/auto/nfs-archive/ifs/noreplica/project/trenders/scenes/{0}/outputs/nbr/'.format(pathrow)
        os.chdir(dir)
        file = glob.glob(globCheck)
        if len(filelistyrs) == 0: sys.exit("No applicable files found")
    
    # get vertvals
    globCheck = '*paramset01_*[0-9]_vertvals.bsq'
    filelistvals = glob.glob(globCheck)
    print filelistvals 
    if len(filelistvals) > 1: sys.exit("Only one reference file expected in {0}".format(dir))
    elif len(filelistvals) == 0:
        dir = '/auto/nfs-archive/ifs/noreplica/project/trenders/scenes/{0}/outputs/nbr/'.format(pathrow)
        os.chdir(dir)
        file = glob.glob(globCheck)
        if len(filelistvals) == 0: sys.exit("No applicable files found")
    
    return os.path.join(dir,filelistyrs[0]), os.path.join(dir,filelistvals[0])

def getInputFile(pathrow, search_string):
    #pathrow = sys.argv[1]
    # change pathrow to six-digit
    # set up for vertyrs...
    pathrow = sixDigitTSA(pathrow)
    dir = '/projectnb/trenders/scenes/{0}/outputs/nbr/nbr_lt_labels/'.format(pathrow)
    os.chdir(dir)
    filelist = glob.glob(search_string)
    if len(filelist) > 1: 
        filelist = filelist[0]
    elif len(filelist) == 0:
        dir = '/projectnb/trenders/scenes/{0}/outputs/nbr/nbr_lt_labels_mr227/'.format(pathrow)
        os.chdir(dir)
        filelist = glob.glob(search_string)
        if len(filelist) > 1: 
            filelist = filelist[0]
        elif len(filelist) == 0: 
            sys.exit("No applicable files found")
    else:
        filelist = filelist[0]
    return os.path.join(dir,filelist)

def string2list(string):
    """ split our string-lists into lists """
    string = string.strip()
    string = string.strip('[')
    string = string.strip(']')
    elements = string.split(',')
    return elements

def string2listplus(string,yorn):
    # coerce a string to a list
    # choose whether or not you want the values to be coerced to ints
    string = string.strip('[] ')
    elements = string.split(',')
    type(elements)
    if yorn == 'y':
        try:
            for i in range(len(elements)): 
                elements[i] = int(elements[i])
        except ValueError:
            pass
    return elements

def interpolate(year1, year2, val1, val2):
    """get a list of interpolated values between val1 and val2 based on year1 and year2,
    does not include change in values for year1"""
    n_elements = year2 - year1
    dif_vals = val2 - val1
    dif_by_yr = dif_vals / n_elements
    out_list = []
    out_list_slope = []
    new_val = val1
    for i in range(n_elements):
        new_val = new_val + dif_by_yr
        out_list.append(new_val)
        out_list_slope.append(dif_by_yr)
    return out_list, out_list_slope

# YEAR BASED ANALYSIS FUNCTIONS
def kernel_extractor_verts_fn(inputParams, ignore_vals, tsa_list, file_info):
    """ extracts kernel values from the vertyrs and vertvals based on the
    locations given in the file in file info"""
    
    current_path = os.getcwd()
    print '\nExtracting vertyrs and vertvals kernel values'
    
    # coerce ignore vals to floats
    for i in range(0,len(ignore_vals)): ignore_vals[i] = float(ignore_vals[i])
    
    # Open reference list
    csv_path = file_info[0]
    pointlist = csv2list(csv_path)
    
    # Get locations of important columns
    xcol = pointlist[0].index(file_info[1])
    ycol = pointlist[0].index(file_info[2])
    tsacol = pointlist[0].index(file_info[3])
    
    # Get reference list as an array for finding TSA
    pointarray = numpy.array(pointlist)
    
    # lowest numbered band to loop through, default 1
    lowband = 1 
    
    # Prepare header of output list
    outlist = []
    outlist.append(pointlist[0])
    
    # Execute these if only on first loop of pathrow
    pathrowAppend = True
    
    # Preallocate index_lengths
    index_lengths = [0]
    
    # Loop through all the input pathrows
    for pathrow in tsa_list:
        pathrow = str(pathrow)
        # Print which TSA
        print "\n"
        print "TSA: " + pathrow
        
        # Get the BSQ path for yrs and vals
        yrsinputfile, valsinputfile = getInputFileVert(pathrow)
        
        # Open the BSQ and get transform for yrs
        spec_ds = gdal.Open(yrsinputfile)
        transform = spec_ds.GetGeoTransform()
        # Count number of bands in BSQ
        nbands = spec_ds.RasterCount
        
        # Open the vals BSQ
        spec_ds_vals = gdal.Open(valsinputfile)
        
        # Get 4-digit pathrow to look up indices
        pathrow4 = fourDigitTSA(pathrow)
        # Get indices of pathrow (TSA) from reference list
        tsa_indices = list(numpy.where(pointarray[:,tsacol]==pathrow4)[0])
        
        # Execute these if only on first loop of a given testset
        testsetAppend = True
        
        # Loop through all the kernel, rule combos
        for testset in inputParams:
            # Get kernel_type and rules from inputs
            kernel_type, rule = testset.split()
            # import rules from validation_kernelRules.py and kernel type from validation_kernelTypes.py
            os.chdir(current_path)
            exec 'from validation_kernelRules import {0} as thisrule'.format(rule)
            exec 'from validation_kernelTypes import {0} as thiskernel'.format(kernel_type)
            in_kernel = thiskernel()
            
            print 'Kernel type: ' + kernel_type
            print 'Rule: ' + rule
            
            # Execute these if only on first loop of band
            bandAppend = True
            
            # Loop through all bands in TSA by rule & kernel
            for band in range(lowband,nbands + 1):
                # append headers only once per band
                if pathrowAppend == True: outlist[0].append('Yrs_band_' + str(band) + '_' + kernel_type + '_' + rule)
                if pathrowAppend == True and rule == 'kernelMajorityVerts': outlist[0].extend(['Vals_band_' + str(band) + '_' + kernel_type + '_' + rule,'Valsmean_band_' + str(band) + '_' + kernel_type + '_' + rule])
                if pathrowAppend == True and rule == 'kernelReturnAll': outlist[0].append('Vals_band_' + str(band) + '_' + kernel_type + '_' + rule)
                print "Band:",band
                
                # Execute these if only on first loop of indices
                indicesAppend = True
                
                # Loop through all indices
                for i in tsa_indices: # or read in TSA
                
                    # Coordinates in Albers Conical NA
                    x = float(pointlist[i][xcol])
                    y = float(pointlist[i][ycol])
                
                    # use the kernel_type to extract the kernel surrounding the point from the raster
                    height = in_kernel.shape[0]
                    width = in_kernel.shape[1]
                    
                    # Extract the kernel info
                    kernelyrs = extract_kernel(spec_ds, x, y, width, height, band, transform)
                    if rule == 'kernelMajorityVerts' or rule == 'kernelReturnAll': kernelvals = extract_kernel(spec_ds_vals, x, y, width, height, band, transform)
                    
                    # append reference values to array only first time through TSA
                    if testsetAppend == True and bandAppend == True: outlist.append(pointlist[i][0:4])
                    
                    # set position for the output list
                    outlist_pos = 1 + tsa_indices.index(i) + sum(index_lengths)
                    if rule == 'kernelMajorityVerts' or rule == 'kernelReturnAll':
                        outlist[outlist_pos].extend(thisrule(kernelyrs, kernelvals, ignore_vals))
                    else:
                        outlist[outlist_pos].append(thisrule(kernelyrs, ignore_vals))

                    # End of first loop of indices
                    indicesAppend = False
                    
                # End of first loop of band
                bandAppend = False
            
            # End of first loop of testset
            testsetAppend = False
        
        # End of first loop of pathrow
        pathrowAppend = False
        
        # Add to the length of indices
        index_lengths.append(len(tsa_indices))
        #print index_lengths
        
    return outlist

def interpolate_kernel_ltverts(ltverts, slpORval, yrscols, valscols):
    
    """ interpolates the kernel value outputs from the output of
    kernel_extractor_verts_fn."""
    
    # slpORval:
    # Use slope for dNBR, val for NBR
    
    # Append out all years from 1984 to 2012 (hardwired for now)
    outlist = [ltverts[0][0:4][:]]
    outlist[0].extend(['value by pixel','Years by pixel'])
    
    # loop through all the plots
    for row in ltverts[1:]:
        # create list of lists for each pixel through years
        pixelThruYears = []
        interpolatedYears = []
        try:
            valscol1 = string2list(row[valscols[1]])
        except AttributeError:
            valscol1 = row[valscols[1]]
        
        for i in range(len(valscol1)):
            pixelThruYears.append([int(valscol1[i])])
            interpolatedYears.insert(0,[])
        
        # loop through each "band" of years and values, each time taking a year and the next greater available
        for band in range(len(yrscols) - 1):
            # get list of years and values for kernel
            try:
                yrscol1 = string2list(row[yrscols[band]])
                yrscol2 = string2list(row[yrscols[band+1]])
                valscol1 = string2list(row[valscols[band]])
                valscol2 = string2list(row[valscols[band+1]])
            except AttributeError:
                yrscol1 = row[yrscols[band]]
                yrscol2 = row[yrscols[band+1]]
                valscol1 = row[valscols[band]]
                valscol2 = row[valscols[band+1]]
            
            # loop over every "pixel" in the "kernel"
            for pixel in range(len(yrscol1)):
                #print "pixel",pixel
                year1 = int(yrscol1[pixel])
                year2 = int(yrscol2[pixel])
                val1 = int(valscol1[pixel])
                val2 = int(valscol2[pixel])
                # set up an index for years if on the first "band"
                if band == 0: interpolatedYears[pixel] = [year1]
                if year2 != 0:
                    value, slope = interpolate(year1, year2, val1, val2)
                    
                    # Use slope for dNBR, val for NBR
                    if slpORval == 'slope' or slpORval == 'slp':
                        pixelThruYears[pixel].extend(slope)
                    elif slpORval == 'value' or slpORval == 'val':
                        pixelThruYears[pixel].extend(value)
                    
                    value, slope = interpolate(year1, year2, year1, year2)
                    interpolatedYears[pixel].extend(value)
        
        outlist.append(row[0:4])
        outlist[-1].extend([pixelThruYears,interpolatedYears])
    
    return outlist

def repeat_timesync_vertex(vertex_file_info, foundplots, which_column):
    """ repeats the values between the given years from the timesync
    vertex.csv file """
    
    vertexpath = vertex_file_info[0]
    vertexList = list(csv.reader(open(vertexpath,'rU')))
    vertexArray = numpy.array(vertexList)
    vtsaInd = vertexList[0].index(vertex_file_info[1])
    vplotidInd = vertexList[0].index(vertex_file_info[2])
    vYearInd = vertexList[0].index(vertex_file_info[3])
    vChangeProcessInd = vertexList[0].index(which_column)
    
    outlist = [foundplots[0][0:4][:]]
    new_col_header = which_column + ' by year'
    outlist[0].extend([new_col_header, 'Years by pixel'])
    
    fptsaInd = foundplots[0].index('tsa')
    fpplotidInd = foundplots[0].index('plotid')
    
    for row in foundplots[1:]:
        inclusive = 1
        outItemList = []
        
        tsa = row[fptsaInd]
        plotid = row[fpplotidInd]
        vplotInds = numpy.where((vertexArray[:,vtsaInd] == tsa) & (vertexArray[:,vplotidInd] == plotid))[0]
        vRows = []
        interpolatedYears = []
        for vRowInd in vplotInds:
            vRows.append(vertexList[vRowInd])
        
        # Need to make sure the rows are in the correct order
        year_col = 2
        vRows.sort(key=lambda x: x[year_col])
        interpolatedYears.append(int(vRows[0][year_col]))
        
        for vRowInd in range(len(vRows)-1):
            year1 = int(vRows[vRowInd][vYearInd])
            year2 = int(vRows[vRowInd+1][vYearInd])
            n_elements = year2 - year1
            if inclusive == 1: n_elements = n_elements + 1
            #print n_elements
            change_val = vRows[vRowInd+1][vChangeProcessInd]
            try:
                change_val = int(change_val)
            except ValueError:
                pass
            change_val_list = [change_val]*n_elements
            outItemList.extend(change_val_list)
            interpolatedYears.extend(range(year1+1, year2+1))
            inclusive = 0
        
        outlist.append(row[0:4])
        outlist[-1].extend([outItemList,interpolatedYears])
    
    return outlist

def yearly_columns(in_array, first_val, interpolate_val, years):
    """ Takes outputs of interpolate_kernel_ltverts or
    repeat_timesync_vertex and formats them into having a
    single column for every year"""
    
    # default setup
    outlist = [in_array[0][0:4][:]]
    outlist[0].extend(years)
    
    # Loop through each row (ie plot)
    for row in in_array[1:]:
        outlist.append(row[0:4])
        
        try:
            val_list = string2list(row[4])
            yr_list = string2list(row[5])
        except AttributeError:
            val_list = row[4]
            yr_list = row[5]
        
        if len(val_list) != len(yr_list):
            sys.exit('Not equal number of pixels for years and values')
        
        year_list = []
        for i in years:
            year_list.insert(0,[])
        
        # Loop through each pixel
        
        if type(val_list[0]) == list:
            pixCountList = range(len(val_list))
        else:
            pixCountList = [0]
        
        for pixInd in pixCountList:
            
            if type(val_list[0]) == list:
                try:
                    val_list_pix = string2list(val_list[pixInd])
                    yr_list_pix = string2list(yr_list[pixInd])
                except AttributeError:
                    val_list_pix = val_list[pixInd]
                    yr_list_pix = yr_list[pixInd]
            else:
                val_list_pix = val_list
                yr_list_pix = yr_list
            
            if len(val_list_pix) != len(yr_list_pix):
                sys.exit('Not equal number of years and values for given pixel')
            
            # Loop through all the years
            for i in range(len(val_list_pix)):
                # get year and value
                val = val_list_pix[i]
                try:
                    val = int(val)
                except ValueError:
                    pass
                yr = int(yr_list_pix[i])
                
                # find where to append to in year_list
                col = outlist[0].index(yr) - 4
                year_list[col].append(val)
        outlist[-1].extend(year_list)
    
    # column headers
    outlist[0][4] = str(outlist[0][4]) + '_' + first_val
    for i in range(5,len(outlist[0])):
        outlist[0][i] = str(outlist[0][i]) + '_' + interpolate_val
    
    return outlist

def compile_full_list(reference_info_range, year_range, *args):
    """ takes all of the inputs from *args and compiles a master list
    for each plot by year. 
    
    reference_info_range is used to specify the columns that should be
    the same between all of the inputs *args
    
    year_range is the range of years to be used from each input *args,
    which will be identified using the column headers"""
    
    # Check to see if the length of input files are all equal
    n_plots = len(args[0])
    for arg in args:
        if len(arg) != n_plots:
            #sys.exit('Number of plots in yearly column inputs not the same')
            print 'Number of plots in yearly column inputs not the same'
    
    # Output list
    out_list = []
    
    # Loop over all rows in the input files
    for row in range(n_plots):
        
        # Get the common reference info and check to make sure all is equal
        common_ref = []
        for i in reference_info_range:
            common_ref.append(args[0][row][i])
            for arg in args[1:]:
                common_ref_test = arg[row][i]
                if common_ref_test != common_ref[-1]:
                    #sys.exit('Indices do not match at row ' + str(row))
                    print 'Indices do not match at row ' + str(row)
        
        # Work with headers if it is the first row
        if row == 0:
            # Indices of year column
            yr_indices_by_arg = []
            # headers from each of these columns
            headers_by_arg = []
            
            for arg in args:
                
                yr_indices_by_arg.append([])
                headers_by_arg.append([])
                
                for yr in year_range:
                    for i in range(len(arg[0])):
                        split_lst = arg[0][i].split('_')
                        
                        try:
                            test_yr = int(split_lst[0])
                        except ValueError:
                            test_yr = split_lst[0]
                        
                        if yr == test_yr:
                            # append to each yr indices 
                            yr_indices_by_arg[-1].append(i)
                            header = split_lst[1]
                            if len(split_lst) > 2:
                                header = '_'.join(split_lst[1:])
                            headers_by_arg[-1].append(header)
                
            # Test to see if all of the headers for each input data arg is
            # same (e.g. NBR for 1984, but dNBR from 1985 onward... no can do!)
            for header_list in headers_by_arg:
                for h in header_list:
                    if h != header_list[0]:
                        print 'Warning: header list not all equal for ' + header_list[0]
                        print 'Headers may be incorrect'
            
            # Set up first row of out_list with headers
            out_list.append(common_ref[:])
            out_list[-1].append('year')
            for arg_ind in range(len(args)):
                list_to_append = args[arg_ind][1][yr_indices_by_arg[arg_ind][0]]
                for i in range(len(list_to_append)):
                    if len(list_to_append) > 1:
                        new_header = headers_by_arg[arg_ind][0] + '_' + str(i + 1)
                    else:
                        new_header = headers_by_arg[arg_ind][0]
                    out_list[-1].append(new_header)
                
        else:
            # Build output list for each plot at each select year
            for i in range(len(year_range)):
                # Append common reference information and the year in question
                out_list.append(common_ref[:])
                out_list[-1].append(year_range[i])
                
                # Loop over each argument and append to out_list
                for arg_ind in range(len(args)):
                    col_index = yr_indices_by_arg[arg_ind][i]
                    values = args[arg_ind][row][col_index]
                    if values == []: values = ['no_data']
                    out_list[-1].extend(values[:])
    
    return out_list

def compute_columns(lst, search_prefix, rule):
    
    """ Generates a new column to append to the lst. The new column will be
    calculated by applying the rule to the columns that were specified by the
    search_prefix. The search_prefix is a string that is in each of the column
    headers that you want to apply the rule to in order to generate a new
    column. """
    
    rule = rule.lower()
    
    # Get column indices
    inds = []
    for i in range(len(lst[0])):
        header_list = lst[0][i].split('_')
        h0 = '_'.join(header_list[:-1])
        if h0 == search_prefix:
            inds.append(i)
    
    # Add header
    new_header = rule + '_' + search_prefix
    lst[0].append(new_header)
    
    # Loop over all non-header rows
    for row in range(1,len(lst)):
        #print row
        # Make list of float values
        value_list = []
        for i in inds:
            value_list.append(float(lst[row][i]))
        
        # Apply rule to values
        try:
            if rule == 'mean':
                out_value = numpy.mean(value_list)
            elif rule == 'median':
                out_value = numpy.median(value_list)
            elif rule == 'max':
                out_value = max(value_list)
            elif rule == 'min':
                out_value = min(value_list)
            elif rule == 'npixgt0':
                out_value = len([i for i in value_list if i > 0])
            elif rule == 'npixlt0':
                out_value = len([i for i in value_list if i < 0])
        except IndexError:
            print 'Problematic number of columns at row ' + str(row)
        
        # Append value to row
        lst[row].append(out_value)
    
    return lst

# COMPARING RESULTS WITH CONFUSION MATRICES

def bin_label(bins, element):
    """ label an element to be within one of the bins """
    
    lower = max([x for x in bins if x <= element])
    upper = min([x for x in bins if x > element])
    label = 'Bin Range: {0} <= x < {1}'.format(lower,upper)
    return label

def relabel_quantitative(bins, elements):
    """ places the list in elements into the bins from bins
    with string labels. outputs the strings used in bin_labels,
    as well as the list of element labels reclassified into these
    bins"""
    
    bin_labels = []
    element_labels = []
    
    # Make the list of bin labels
    for i in range(len(bins)-1):
        label = 'Bin Range: {0} <= x < {1}'.format(bins[i],bins[i+1])
        bin_labels.append(label)
    
    # Classify elements in to ranges of the bins
    # min <= element < max
    for i in elements:
        label = bin_label(bins,i)
        element_labels.append(label)
    
    return bin_labels, element_labels

def relabel_multiple_element_list(*args):
    """ will concatenate multiple lists of elements in order to
    get a single list of one concatenated element. also will return
    the new label names, even if only one list is given as input"""
    
    header_labels = []
    element_labels = []
    
    for i in range(len(args[0])):
        label = args[0][i]
        if len(args) > 1:
            for j in range(1,len(args)):
                label = '{0}_{1}'.format(label,args[j][i])
        element_labels.append(label)
        if label not in header_labels: header_labels.append(label)
    
    header_labels.sort()
    
    return header_labels, element_labels

# need to re-do this script
def organize_comparisons(L, col_args, col_bins, row_args, row_bins):
    """ take columns of interest and organize so that they of
    a two-column format with categorical variables, set up for
    building a contingency table"""
    
    # inputs
    # L = 2D, list of lists, with rows and columns
    # headers are columns to be used
    # col_args = list of the headers to be used for columns
    # If more than 1, the results will be concatenated
    # row_args = list of the headers to be used for rows
    # If more than 1, the results will be concatenated
    
    # quantitative columns can be binned
    
    # Get indices for arguments in first row (header line) of L
    col_inds = []
    for i in col_args: col_inds.append(L[0].index(i))
    row_inds = []
    for i in row_args: row_inds.append(L[0].index(i))
    
    # Cannot assume rows will be quantitative or columns will need to be concatenated...
    # will need to modify this part of the code to make it more general
    
    # Organize quantitative variables into bins
    elements = []
    for i in L[1:]: elements.append(float(i[row_inds[0]]))
    row_headers, row_labels = relabel_quantitative(row_bins, elements)
    
    # Concatenate multiple column inputs
    elements1 = []
    for i in L[1:]: elements1.append(i[col_inds[0]])
    elements2 = []
    for i in L[1:]: elements2.append(i[col_inds[1]])
    col_headers, col_labels = relabel_multiple_element_list(elements1, elements2)
    
    return row_headers, row_labels, col_headers, col_labels

def confusion_with_headers(row_headers, row_list, col_headers, col_list):
    """ runs the confusion_matrix function from sklearn and then provides
    column and row headers, while removing extraneous rows and columns
    resulting from the confusion_matrix function"""
    
    from sklearn.metrics import confusion_matrix
    labels = row_headers + col_headers
    out1 = confusion_matrix(row_list, col_list, labels)
    
    out2 = [['']]
    out2[0].extend(col_headers)
    for i in range(len(row_headers)):
        out2.append([row_headers[i]])
        out2[-1].extend(out1[i][len(row_headers):])
    
    return out2

# EVENT BASED ANALYSIS FUNCTIONS
def kernel_extractor_labels_fn(inputParams, ignore_vals, tsa_list, file_info, search_string):
    
    """almost the same as kernel_extractor_verts_fn, but pulls information from event-based,
    labeled files, like greatest_disturbance.bsq, greatest_fast_disturbance.bsq, etc."""
    
    current_path = os.getcwd()
    
    # ignore_vals as a list of values in a single string, separated by spaces
    for i in range(0,len(ignore_vals)): ignore_vals[i] = float(ignore_vals[i])
    
    # Open reference list
    csv_path = file_info[0]
    pointlist = csv2list(csv_path)
    
    # Get locations of important columns
    xcol = pointlist[0].index(file_info[1])
    ycol = pointlist[0].index(file_info[2])
    tsacol = pointlist[0].index(file_info[3])
    
    # Get reference list as an array for finding TSA
    pointarray = numpy.array(pointlist)
    
    # Prepare header of output list
    outlist = []
    outlist.append(pointlist[0])
    
    
    # Execute these if only on first loop of pathrow
    pathrowAppend = True
    
    # Preallocate index_lengths
    index_lengths = [0]
    
    # Loop through all the input pathrows
    for pathrow in tsa_list:
        # Print which TSA
        print "\n"
        print "TSA: " + pathrow
        
        # Get the BSQ path
        inputfile = getInputFile(pathrow, search_string)
        
        # Open the BSQ and get transform
        spec_ds = gdal.Open(inputfile)
        transform = spec_ds.GetGeoTransform()
        # Count number of bands in BSQ
        nbands = spec_ds.RasterCount
        
        # Get 4-digit pathrow to look up indices
        pathrow4 = fourDigitTSA(pathrow)
        # Get indices of pathrow (TSA) from reference list
        tsa_indices = list(numpy.where(pointarray[:,tsacol]==pathrow4)[0])
        
        # Execute these if only on first loop of a given testset
        testsetAppend = True
        
        # Loop through all the kernel, rule combos
        for testset in inputParams:
            # Get kernel_type and rules from inputs
            kernel_type, rule = testset.split()
            os.chdir(current_path)
            # import rules from validation_kernelRules.py and kernel type from validation_kernelTypes.py
            exec 'from validation_kernelRules import {0} as thisrule'.format(rule)
            exec 'from validation_kernelTypes import {0} as thiskernel'.format(kernel_type)
            in_kernel = thiskernel()
            
            print 'Kernel type: ' + kernel_type
            print 'Rule: ' + rule
            
            # Execute these if only on first loop of band
            bandAppend = True
            
            # Loop through all bands in TSA by rule & kernel
            for band in [1,2,3]:
                # append headers only once per band
                if pathrowAppend == True: outlist[0].append('band_' + str(band) + '_' + kernel_type + '_' + rule)
                
                print "Band:",band
                
                # Execute these if only on first loop of indices
                indicesAppend = True
                
                # Loop through all indices
                for i in tsa_indices: # or read in TSA
                
                    # Coordinates in Albers Conical NA
                    x = float(pointlist[i][xcol])
                    y = float(pointlist[i][ycol])
                
                    # use the kernel_type to extract the kernel surrounding the point from the raster
                    height = in_kernel.shape[0]
                    width = in_kernel.shape[1]
                    
                    # Extract the kernel info
                    kernel = extract_kernel(spec_ds, x, y, width, height, band, transform)
                    
                    # append reference values to array only first time through TSA
                    if testsetAppend == True and bandAppend == True: outlist.append(pointlist[i][0:4])
                    
                    # set position for the output list
                    outlist_pos = 1 + tsa_indices.index(i) + sum(index_lengths)
                    outlist[outlist_pos].append(thisrule(kernel,ignore_vals))
                    
                    # End of first loop of indices
                    indicesAppend = False
                    
                # End of first loop of band
                bandAppend = False
            
            # End of first loop of testset
            testsetAppend = False
        
        # End of first loop of pathrow
        pathrowAppend = False
        
        # Add to the length of indices
        index_lengths.append(len(tsa_indices))
        #print index_lengths
        
    return outlist

def format_kernel_events_to_vertex(input):
    """ Takes an event-based extracted kernel info table, ie the output
    of kernel_extractor_labels_fn, and formats it to be like the vertex.csv
    file, which will be used as a reference for comparison"""
    
    # year and value columns hard-coded for now
    year_cols = [4]
    mag_cols = [5]
    dur_cols = [6]
    
    # set up output headers
    output = [input[0][0:4][:]]
    output[0].extend(['kernel_pixel_num','year','magnitude','duration'])
    
    # Loop by rows in the input
    for row in input[1:][:]:
        # Get all the kernel strings back into lists
        # for k in range(4,len(row[4:])+4):
            # kernel = row[k]
            # kernel = string2listplus(kernel,'y')
            # row[k] = kernel
        # Loop for each pixel index in the kernel
        for pixel in range(len(row[4])):
            # Loop for each "band"
            for band in range(len(year_cols)):
                # Get the year of vertex
                year = row[year_cols[band]][pixel]
                # Only continue if year > 0
                output.append(row[0:4][:])
                # pixel number, range 1:9 for 3x3
                pixel_num = pixel + 1
                output[-1].append(pixel_num)
                # Magnitude value
                mag = row[mag_cols[band]][pixel]
                dur = row[dur_cols[band]][pixel]
                # Append as new row to output
                output[-1].extend([year,mag,dur])
    
    return output

def repeat_kernel_magnitudes(ltevents, yr_range, yrscol, magcol, durcol, use_dur):
    
    """ repeats the kernel value outputs from the output of
    kernel_extractor_labels_fn for the years given by the year and
    duration information"""
    
    # yr_range = range(1984, 2013) = [1984, ..., 2012]
    outlist = [ltevents[0][0:4][:]]
    outlist[0].extend(['value by pixel','Years by pixel'])
    
    # loop through all the plots
    for row in ltevents[1:]:
        # create list of lists for each pixel through years
        pixelThruYears = []
        yr_range_pixels = []
        
        # Append out for the correct number of pixels
        for i in row[magcol]:
            pixelThruYears.append([])
            yr_range_pixels.append(yr_range)
        
        # loop over every "pixel" in the "kernel"
        for pixel in range(len(pixelThruYears)):
            
            # yod = year of disturbance
            # If use_dur is true, use the duration column to repeat magnitude
            if use_dur.lower()[0] == 't':
                yod_start = row[yrscol][pixel]
                yod_end = yod_start + row[durcol][pixel]
                sub_yrs = range(yod_start, yod_end)
                # If use_dur is false, do not use the duration column to repeat magnitude
                # magnitude is only assigned for the year of disturbance
            else:
                sub_yrs = [row[yrscol][pixel]]
            
            # loop over every year in the range
            for yr in yr_range:
                
                # If the year is within the range of year and duration,
                # append the mag value
                
                if yr in sub_yrs:
                
                    pixelThruYears[pixel].append(row[magcol][pixel])
                
                # If not, then append 0
                
                else:
                    
                    pixelThruYears[pixel].append(0)
        
        # Append the information to the outlist
        outlist.append(row[0:4])
        outlist[-1].extend([pixelThruYears,yr_range_pixels])
    
    return outlist

def sum_greatest_and_second(full_list, kernel_size):
    
    """ Creates a new column that is the sum of the
    greatest and second greatest value for each kernel
    pixel column """
    
    
    # Set up headers
    for i in range(1,kernel_size + 1):
        full_list[0].append("sum_greatest_second_LTmag_" + str(i))
    
    # Loop over kernel pixels
    for i in range(1,kernel_size + 1):
        greatest_ind = full_list[0].index("greatest_LTmag_" + str(i))
        second_ind = full_list[0].index("second_LTmag_" + str(i))
        
        # Loop over all rows
        for row in range(1,len(full_list)):
            s = full_list[row][greatest_ind] + full_list[row][second_ind]
            full_list[row].append(s)
    
    return full_list

# COMPARING EVENT BASED RESULTS WITH CONFUSION MATRICES
def event_comparisons_TOTAL_LIST(L, yr_range, yr_tolerance, tsa_col_header, plotid_col_header, year_col_header, change_process_col_header, other_info_col_header, LTmag_col_header, LTbins):
    """ Method to find the matches between the TimeSync interpretation and the LandTrendr column of choice. Called by the run_accuracy_events(...) function.
    
    Generally run within the validation_events_confusion_matrices...py scripts.
    
    L
        the summarized comparison spreadsheet generated by the validation_events_greatest...py scripts
    
    yr_range
        the range of years that we're comparing the two datasets
        first value should be the first year
        last value should be one greater than the last year
    
    yr_tolerance
        the number of years between one disturbance tag and another that we can call a match
        generally either 0 or 1
    
    tsa_col_header
        the string that is on top of the column that has the TSA values in it
    
    plotid_col_header
        the string that is on top of the column that has the plotid values in it
    
    year_col_header
        the string that is on top of the column that has the year values in it
    
    change_process_col_header
        the string that is on top of the column that has the TimeSync change process
        calls in it
    
    other_info_col_hader
        the string that is on top of another column that should be used in tandem with
        the TimeSync change process label in order to separate results. Usually, this
        will be the relative_magnitude column.
    
    LTmag_col_header
        the string that is on top of the column that has the values of LandTrendr magnitude
        values
    
    LTbins
        a list of the bins that will be used to break up the LTmag values
        example: [0, 1, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100, 200]
    
    End of explanation section. """
    
    # Get column indices from the locations of the headers
    tsa_col = L[0].index(tsa_col_header)
    plotid_col = L[0].index(plotid_col_header)
    year_col = L[0].index(year_col_header)
    change_process_col = L[0].index(change_process_col_header)
    try:
        other_info_col = L[0].index(other_info_col_header)
    except ValueError:
        other_info_col = 'none'

    LTmag_col = L[0].index(LTmag_col_header)

    # LT_mag info
    LTmag_labels = []

    # TimeSync info
    TS_change_labels = []
    if type(other_info_col) == int: TS_other_info_col = []
    
    # No match rows - rows without a match
    no_match_rows = []
    
    # ...and loop through all rows
    r = 1
    while r < len(L):
        
        # Set tsa and plotid for particular plot
        tsa = L[r][tsa_col]
        plotid = L[r][plotid_col]
        
        # Set up subset of list for each plot
        plot_list = []
        for yr in yr_range:
            if L[r][year_col] == str(yr) and L[r][tsa_col] == tsa and L[r][plotid_col] == str(plotid):
                plot_list.append(L[r])
            r = r + 1
        
        # Go through all of the years for a plot
        change_process1 = '' #keep this to check for consecutive, same type disturbances
        for plot_row in range(len(plot_list)):
            change_process = plot_list[plot_row][change_process_col]
            LTmag = float(plot_list[plot_row][LTmag_col])
            
            # If the change process is a disturbance and not consecutive (ie same process over multiple years)
            if (change_process != 'Stable' and change_process != 'no_data' and 
                change_process != 'Other non-disturbance' and change_process != 'Recovery' and 
                change_process != change_process1):
                
                # Append the change_process and other_info_col (if necessary)
                TS_change_labels.append(change_process)
                if type(other_info_col) == int: TS_other_info_col.append(plot_list[plot_row][other_info_col])
                
                # Test years in order of distance from given year
                test_yr_order = range(-yr_tolerance,yr_tolerance+1)
                test_yr_order.sort(key = lambda x: abs(int(x)))
                appender = 0
                for i in test_yr_order:
                    
                    plot_row_test = plot_row + i
                    # If the plot_row_test index is non-negative (otherwise would index from end of list) and not larger than available number of years
                    # and the magnitude is greater than 0
                    if plot_row_test >= 0 and plot_row_test < len(yr_range) and float(plot_list[plot_row_test][LTmag_col]) > 0:
                        LTmag_labels.append(float(plot_list[plot_row_test][LTmag_col]))
                        appender = 1
                        break
                
                # If no LTmag >0 was found, then append a zero value
                if appender == 0: LTmag_labels.append(0)
                
            # or the LT_mag is greater than 0
            # This will only work appropriately without any duration values
            elif float(LTmag) > 0:
                
                LTmag_labels.append(float(LTmag)) # need to figure out multiple assignment issue...  
                
                # Test years in order of distance from given year
                test_yr_order = range(-yr_tolerance,yr_tolerance+1)
                test_yr_order.sort(key = lambda x: abs(int(x)))
                appender = 0
                for i in test_yr_order:
                    
                    plot_row_test = plot_row + i
                    # If the plot_row_test index is non-negative (otherwise would index from end of list) and not larger than available number of years
                    # and the change_process is a disturbance
                    if (plot_row_test >= 0 and plot_row_test < len(yr_range) and plot_list[plot_row_test][change_process_col] != 'Stable' and plot_list[plot_row_test][change_process_col] != 'no_data' and 
                        plot_list[plot_row_test][change_process_col] != 'Other non-disturbance' and plot_list[plot_row_test][change_process_col] != 'Recovery' > 0 and plot_list[plot_row_test][change_process_col] != change_process1):
                        
                        change_process = plot_list[plot_row_test][change_process_col]
                        
                        TS_change_labels.append(change_process)
                        if type(other_info_col) == int: TS_other_info_col.append(plot_list[plot_row_test][other_info_col])
                        appender = 1
                        break
                
                # If no TS disturbance was found, then append non-disturbance and other value of 0
                if appender == 0:
                    TS_change_labels.append('Non-disturbance')
                    if type(other_info_col) == int: TS_other_info_col.append('0')
            
            else:
                # no match indices
                thisrow = r - len(yr_range) + plot_row
                no_match_rows.append(thisrow)
            
            # Assign current change process to previous one
            change_process1 = change_process
    
    # Relabel the elements, if necessary
    if type(other_info_col) == int:
        TS_header_labels, TS_element_labels = relabel_multiple_element_list(TS_change_labels, TS_other_info_col)
    else:
        TS_element_labels = TS_change_labels[:]
        TS_header_labels = list(set(TS_change_labels))
    
    LT_header_labels, LT_element_labels = relabel_quantitative(LTbins, LTmag_labels)
    
    LT_element_labels2 = []
    TS_element_labels2 = []
    no_match_rows_ind = 0
    dist_ind = 0
    for i in range(1,len(L)):
        if no_match_rows[no_match_rows_ind] == i:
            LT_element_labels2.append('no_match')
            TS_element_labels2.append('no_match')
            no_match_rows_ind = no_match_rows_ind + 1
        else:
            LT_element_labels2.append(LT_element_labels[dist_ind])
            TS_element_labels2.append(TS_element_labels[dist_ind])
            dist_ind = dist_ind + 1
    
    return LT_header_labels, LT_element_labels, LT_element_labels2, TS_header_labels, TS_element_labels, TS_element_labels2

def event_comparisons_TOTAL_LIST_new(L, yr_range, yr_tolerance, tsa_col_header, plotid_col_header, year_col_header, change_process_col_header, other_info_col_header, LTmag_col_header, LTbins):
    """ NEW VERSION OF event_comparisons_TOTAL_LIST(...) - DOES NOT WORK AS OF YET, NEEDS DEBUGGING
    This was an attempt to clean up some of the matching issues between disturbance events that became apparent while using the previous script
    
    Method to find the matches between the TimeSync interpretation and the LandTrendr column of choice. Called by the run_accuracy_events(...) function.
    
    Generally run within the validation_events_confusion_matrices...py scripts.
    
    L
        the summarized comparison spreadsheet generated by the validation_events_greatest...py scripts
    
    yr_range
        the range of years that we're comparing the two datasets
        first value should be the first year
        last value should be one greater than the last year
    
    yr_tolerance
        the number of years between one disturbance tag and another that we can call a match
        generally either 0 or 1
    
    tsa_col_header
        the string that is on top of the column that has the TSA values in it
    
    plotid_col_header
        the string that is on top of the column that has the plotid values in it
    
    year_col_header
        the string that is on top of the column that has the year values in it
    
    change_process_col_header
        the string that is on top of the column that has the TimeSync change process
        calls in it
    
    other_info_col_hader
        the string that is on top of another column that should be used in tandem with
        the TimeSync change process label in order to separate results. Usually, this
        will be the relative_magnitude column.
    
    LTmag_col_header
        the string that is on top of the column that has the values of LandTrendr magnitude
        values
    
    LTbins
        a list of the bins that will be used to break up the LTmag values
        example: [0, 1, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100, 200]
    
    End of explanation section. """
    
    # Get column indices from the locations of the headers
    tsa_col = L[0].index(tsa_col_header)
    plotid_col = L[0].index(plotid_col_header)
    year_col = L[0].index(year_col_header)
    change_process_col = L[0].index(change_process_col_header)
    try:
        other_info_col = L[0].index(other_info_col_header)
    except ValueError:
        other_info_col = 'none'

    LTmag_col = L[0].index(LTmag_col_header)

    # LT_mag info
    LTmag_labels = []

    # TimeSync info
    TS_change_labels = []
    if type(other_info_col) == int: TS_other_info_col = []
    
    # No match rows - rows without a match
    no_match_rows = []
    
    # ...and loop through all rows
    r = 1
    while r < len(L):
        
        # Set tsa and plotid for particular plot
        tsa = L[r][tsa_col]
        plotid = L[r][plotid_col]
        
        # Set up subset of list for each plot
        plot_list = []
        for yr in yr_range:
            if L[r][year_col] == str(yr) and L[r][tsa_col] == tsa and L[r][plotid_col] == str(plotid):
                plot_list.append(L[r][:])
            r = r + 1
        
        
        # Test years in order of distance from given year
        test_yr_order = range(-yr_tolerance,yr_tolerance+1)
        test_yr_order.sort(key = lambda x: abs(int(x)))
        
        for dif in test_yr_order:
            
            if dif <= 0:
                start_ind = -dif
                end_ind = len(plot_list)
            else:
                start_ind = 0
                end_ind = len(plot_list) - dif
            
            print "start_ind " + str(start_ind)
            print "end_ind " + str(end_ind) + "\n"
            
            # Go through all of the years for a plot
            for i in range(start_ind, end_ind):
                change_process = plot_list[i][change_process_col]
                LTmag = plot_list[i][LTmag_col]
                # need to check comparisons between change_process and LTmag at the intervals specified by dif...
        
    
    # Relabel the elements, if necessary
    if type(other_info_col) == int:
        TS_header_labels, TS_element_labels = relabel_multiple_element_list(TS_change_labels, TS_other_info_col)
    else:
        TS_element_labels = TS_change_labels[:]
        TS_header_labels = list(set(TS_change_labels))
    
    LT_header_labels, LT_element_labels = relabel_quantitative(LTbins, LTmag_labels)
    
    LT_element_labels2 = []
    TS_element_labels2 = []
    no_match_rows_ind = 0
    dist_ind = 0
    for i in range(1,len(L)):
        if no_match_rows[no_match_rows_ind] == i:
            LT_element_labels2.append('no_match')
            TS_element_labels2.append('no_match')
            no_match_rows_ind = no_match_rows_ind + 1
        else:
            LT_element_labels2.append(LT_element_labels[dist_ind])
            TS_element_labels2.append(TS_element_labels[dist_ind])
            dist_ind = dist_ind + 1
    
    return LT_header_labels, LT_element_labels, LT_element_labels2, TS_header_labels, TS_element_labels, TS_element_labels2

# Running events accuracy table
def run_accuracy_events(input_path, output_path, yr_range, yr_tolerance, tsa_col_header, plotid_col_header, year_col_header, change_process_col_header, other_info_col_header, LTmag_col_header, LTbins):
    """Creates an accuracy table for the given inputs at the years of disturbance, and appends the resulting labels to the original input table,
    as well as outputting an accuracy table .csv"""
    
    L = csv2list(input_path)
    row_headers, row_labels1, row_labels2, col_headers, col_labels1, col_labels2 = event_comparisons_TOTAL_LIST(L, yr_range, yr_tolerance, tsa_col_header, plotid_col_header, year_col_header, change_process_col_header, other_info_col_header, LTmag_col_header, LTbins)
    L[0].extend([LTmag_col_header + '_yrtol_' + str(yr_tolerance) + '_Bin_Range', LTmag_col_header + '_yrtol_' + str(yr_tolerance) + '_TimeSync_label'])
    for i in range(1,len(L)):
        L[i].append(row_labels2[i-1])
        L[i].append(col_labels2[i-1])
    list2csv(input_path, L)

    output = confusion_with_headers(row_headers, row_labels1, col_headers, col_labels1)
    list2csv(output_path, output)