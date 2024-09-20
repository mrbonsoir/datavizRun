# Tool for reading gpx file

import numpy as np
import gpxpy
import pandas as pd
import geopy.distance
import matplotlib.pyplot as plt
import os 

# Some data
list_starting_date =["2017-09-14","2018-09-14","2019-09-14","2020-09-14","2021-09-14","2022-09-14","2023-09-14","2024-09-14","2025-09-14"]

startingPoint_df = pd.DataFrame(columns=["name","startingDate","latitude","longitude"], 
                                index=np.arange(5))
startingPoint_df.iloc[:,0] = ["Origin","Park Girouard","Park Doyon","Espace Ciele","Summit Circle"]
startingPoint_df.iloc[:,1] = ["2017-09-14", "2020-06-18","2022-11-21", "2022-11-24", "2022-11-24"]
startingPoint_df.iloc[:,2] = [45.46604,  45.471895, 45.4776919, 45.489528, 45.491468847148674]
startingPoint_df.iloc[:,3] = [-73.6194,  -73.613628, -73.62141,  -73.567539, -73.60451710657861]

def fun_gpx2pd(gpx_path):
    """
    The function load a gpx data file and return a pandas dataFrame with info extracted from the gpx file
    """
    #print(gpx_path)
    with open(gpx_path) as f:
        gpx = gpxpy.parse(f)
    
    # Convert to a dataframe one point at a time.
    points = []
    for segment in gpx.tracks[0].segments:
        for p in segment.points:
            points.append({
                #'time': p.time,
                'latitude': p.latitude,
                'longitude': p.longitude,
                'elevation': p.elevation,
            })
    df = pd.DataFrame.from_records(points)

    # Cumulative distance.
    coords = [(p.latitude, p.longitude) for p in df.itertuples()]
    df['distance'] = [0] + [geopy.distance.distance(from_, to).m for from_, to in zip(coords[:-1], coords[1:])]
    df['cumulative_distance'] = df.distance.cumsum()
    
    
    # Timing.
    #df['duration'] = df.time.diff().dt.total_seconds().fillna(0)
    #df['cumulative_duration'] = df.duration.cumsum()
    #df['pace_metric'] = pd.Series((df.duration / 60) / (df.distance / 1000)).bfill()
    
    return df

def fun_listPath_gpx2pd(list_data_path):
	"""
	Function to convert list of path with gpx data to a panda dataFrame 
	"""

	list_gpxpath_df = []
	single_gpxpath_df = []

	for c, d in enumerate(list_data_path):
	    #print(d,c)
		single_gpxpath_df = fun_gpx2pd(d)
		
		# update list of gpx dataFrame
		list_gpxpath_df.append(single_gpxpath_df)

	return list_gpxpath_df

def fun_display_list_data_path(list_path_df, run_year = 2018, figsize_window=(4,4)):
	"""
		Function to display all path from a list of path.
	"""

	plt.figure(figsize=figsize_window)

	for c in np.arange(len(list_path_df)):
	    df = list_path_df[c]	
	    plt.plot(df['longitude'], df['latitude'],color=(1,1,1))


	plt.title('GPX paths of all RunRite routes in year {1:1.0f} of RunRite'.format(len(list_path_df), run_year), size=10);
	plt.xlabel('Latitude')
	plt.ylabel('Longitiude')
	plt.grid(False)

	ax = plt.gca()
	ax.set_aspect('equal', adjustable='box')
	ax.set_facecolor((0,0,0))
	plt.axis([ -73.650, -73.55, 45.45, 45.55])

	#plt.show()

def fun_DownSample_gpx(gpx_df, number_of_sample = 100):
    """
    The function will down sample a DataFrame that contains a gpx track.
        In:eee
            - gpx_df (DataFrame): obtained from fun_gpx2pd
            - number_of_sample (int): the new number of samples that defined the gpx track
        Out:
            - gpx_DownSample_df (DataFrame) with same columns as in gpx_df
    """
    gpx_DownSample_df = gpx_df
    
    vec_slice_df      = np.linspace(0,gpx_df.shape[0]-1,number_of_sample).astype(int)
    gpx_DownSample_df = gpx_DownSample_df.iloc[vec_slice_df,:]

    return gpx_DownSample_df

def fun_get_average_distance_run(list_run_df):
    """"
    The function takes a list of DataFramne as input and returns the average run lenght 
    of the corresponding runs.
    """
    
    # get number of run per list
    nb_run = len(list_run_df)
    
    # get average distance for this list of run
    vec_run_distance = np.zeros(nb_run)
    
    for i in np.arange(nb_run):
        vec_run_distance[i] = list_run_df[i].iloc[-1,5]
    
    average_run_distance = np.mean(vec_run_distance / 1000)

    return average_run_distance
	
def fun_center_point_run(list_run_df):
    """"
    The function takes a list of DataFramne as input and returns the average 
    latitute and longitude.
    """
    
    # get number of run per list
    nb_run = len(list_run_df)

    
    # get average distance for this list of run
    vec_latitude_longitude = np.zeros((nb_run,2))

    for i in np.arange(nb_run):
        vec_latitude_longitude[i,0] = list_run_df[i].latitude.mean()
        vec_latitude_longitude[i,1] = list_run_df[i].longitude.mean()

    
    vec_latitude_longitude = np.mean(vec_latitude_longitude, axis=0)

    return vec_latitude_longitude

def fun_clean_trace_start_end(gpsTrace_df, gpsPointLongitude, gpsPointLatitude, index_min_ref=30, debug = False):
    """
    Clean the DataFrame by keeping only the gps points corresponding to the run time.

    Input:
        - index_min_ref is the value from which we need to clean the pass in percentage versus the number of data point.
    It returns a DataFrame
    """
    
    index_min_ref = np.round(len(gpsTrace_df) * (index_min_ref / 100))
    lower_min_ref = np.round(len(gpsTrace_df) * (3 / 100)) # start 3per after the first index

    diff_lat = gpsPointLatitude  - gpsTrace_df["latitude"][:]
    diff_lon = gpsPointLongitude - gpsTrace_df["longitude"][:]
    diff_all = (np.sqrt(diff_lon**2 + diff_lat**2))
    indexMinDiff = np.argmin(diff_all)

    if debug == True:
        print(index_min_ref,indexMinDiff,"-")

    if (lower_min_ref > 0) & (indexMinDiff <= index_min_ref):
        if debug == True:
            print(indexMinDiff,"in {0, 0.3}per")

        gpsTrace_df = gpsTrace_df.drop(np.arange(0, indexMinDiff))
        gpsTrace_df = gpsTrace_df.reset_index()

        gpsTrace_df["cumulative_distance"] = gpsTrace_df["cumulative_distance"] - gpsTrace_df["cumulative_distance"].loc[0]
    
        cleanedGpsTrace_df = gpsTrace_df.copy()
    else:
        cleanedGpsTrace_df = gpsTrace_df.copy()

    return cleanedGpsTrace_df

def fun_create_df_from_list_df(list_all_df, list_all_file_name, startRunDate_df, specialDate="2022-09-01", club_name = "RunRite"):
    """The function takes as input: 
    - a list of DataFrame as input 
    - a list of csv files
    - startRunDate_df a DataFrame of location and starting date of the runs, there are 4 different locations
    - specialDate a single - for now - date for which it need to do re-set the location start

    it returns: 
    - a single DataFrame gathering information by element of the list of provided DataFrane"""

    all_info_df = pd.DataFrame(columns = ["time","cumulative_distance","indexNum","numberDay","indexStartingPoint","numberRunnersPerRun"],
                               index=[np.arange(len(list_all_df))])

    for c, d in enumerate(list_all_df):
        # clean the time value to keep only yy mm dd
        head_tail = os.path.split(list_all_file_name[c])
    
        time_run  = head_tail[1][len(club_name)+1:len(club_name)+11].replace("_","-")

        all_info_df.iloc[c,0] = time_run
        
        # copy the values cumulative distance to the list of run, ie the last cumulative of each run
        all_info_df.iloc[c,1] = list_all_df[c]["cumulative_distance"].iloc[-1]
        
        all_info_df.iloc[c,2] = c

    # format properly the time column
    all_info_df["time"] = pd.to_datetime(all_info_df['time'], format='%Y-%m-%d')
    
    #convert index to time index
    all_info_df = all_info_df.set_index('time') 

    all_info_df['numberDay'] = all_info_df.index.strftime('%j')

    # add a new column as indexNum 0 to number of traces
    all_info_df['indexStartingPoint'] = 0


    select_df = all_info_df[(all_info_df.index >= startRunDate_df["startingDate"][0]) &
                            (all_info_df.index < startRunDate_df["startingDate"][1])].copy()
    index_sel = np.array(select_df["indexNum"].tolist())
    for c in index_sel:
        all_info_df.iloc[c,3] = 0

    select_df = all_info_df[(all_info_df.index >= startRunDate_df["startingDate"][1]) &
                            (all_info_df.index < startRunDate_df["startingDate"][2])].copy()
    index_sel = np.array(select_df["indexNum"].tolist())
    for c in index_sel:
        all_info_df.iloc[c,3] = 1

    select_df = all_info_df[(all_info_df.index >= startRunDate_df["startingDate"][2]) &
                            (all_info_df.index <= startRunDate_df["startingDate"][3])].copy()
    index_sel = np.array(select_df["indexNum"].tolist())
    for c in index_sel:
        all_info_df.iloc[c,3] = 2

    select_df = all_info_df[(all_info_df.index >= startRunDate_df["startingDate"][3])].copy()
    index_sel = np.array(select_df["indexNum"].tolist())
    for c in index_sel:
        all_info_df.iloc[c,3] = 3

    # and special case for 2022 1 of September:
    select_df = all_info_df[(all_info_df.index == pd.to_datetime(specialDate))].copy()
    index_sel = np.array(select_df["indexNum"].tolist())
    all_info_df.iloc[index_sel,3] = 3

    return all_info_df

def fun_traces_before_race_from_lits_df(list_of_run_df, info_about_list_of_run_df, info_about_startingPoint_df, debug = False):
    """
    The function will remove the traces before the begining a run trace if it's too far from the actual start of the run.
    """
    for c, single_trace_df in enumerate(list_of_run_df):
    #--> single_trace_df is equivalent to list_of_run_df[c]
    
        # check value 0, 1, 2, 3 of starting date:
        ref_lon = info_about_startingPoint_df["longitude"][info_about_list_of_run_df.iloc[c,3]] 
        ref_lat = info_about_startingPoint_df["latitude"][info_about_list_of_run_df.iloc[c,3]]
        cleaned_run_df = fun_clean_trace_start_end(single_trace_df, ref_lon, ref_lat, debug = debug)
    
        # update the info all traces df
        if len(cleaned_run_df) < len(single_trace_df):
            if debug == True:
                print(c, "lenght before/after ",len(single_trace_df),"/",len(cleaned_run_df), info_about_list_of_run_df["cumulative_distance"].iloc[c] / 1000, "--->", 
                      cleaned_run_df["cumulative_distance"].iloc[-1]/1000)
        
            list_of_run_df[c] = cleaned_run_df.copy()
            
            # replace cumulative value
            info_about_list_of_run_df.iloc[c,0] = cleaned_run_df["cumulative_distance"].iloc[-1]

    return list_of_run_df, info_about_list_of_run_df

def display_group_runs_by_dates(list_of_run_df, info_about_list_of_run_df, date_interval, 
                                name_of_the_running_club = "RunRite",title_="toto"):
    """
    The function displays a group of run in a date interval.
    """
    select_run_df = info_about_list_of_run_df[(info_about_list_of_run_df.index >= date_interval[0]) &
                                              (info_about_list_of_run_df.index < date_interval[1])].copy()
    index_sel = np.array(select_run_df["indexNum"].tolist())
    
    #plt.figure()
    fig, (ax1) = plt.subplots(1, 1,figsize=(8,8))
    plt.rcParams['figure.figsize'] = [4, 4]

    # plot the traces
    for c, i in enumerate(index_sel):
        single_gps_trace_df = list_of_run_df[index_sel[c]]
        ax1.plot(single_gps_trace_df["longitude"],single_gps_trace_df["latitude"],'k')

    start_Lon = startingPoint_df.iloc[:,3].to_numpy()
    start_Lat = startingPoint_df.iloc[:,2].to_numpy()

    if name_of_the_running_club == "RunRite":
        # draw departure points
        plt.scatter(start_Lon, start_Lat,c='r', alpha=0.75, s=200)
        plt.plot(start_Lon[0:4], start_Lat[0:4],'-',c='r')

        # draw line departure to summit circle
        plt.scatter(start_Lon[-1], start_Lat[-1], c='g', alpha=0.95, s=400)
    elif name_of_the_running_club == "NDGRC":
        # draw departure points
        plt.scatter(start_Lon[1], start_Lat[1],c='r', alpha=0.75, s=200)
    
        # draw line departure to summit circle
        plt.scatter(start_Lon[-1], start_Lat[-1], c='g', alpha=0.95, s=400)
    else:
        # draw line departure to summit circle
        plt.scatter(start_Lon[-1], start_Lat[-1], c='g', alpha=0.95, s=400)


    #for ii in np.arange(4):
    #    plt.plot(start_Lon[[ii,-1]], start_Lat[[ii,-1]],'-',c='b')
    
    #print(start_Lon[[0,-1]], start_Lat[[0,-1]])

    ax1.set_xlim(-73.66, -73.55)    
    ax1.set_ylim(45.45, 45.525)    

    #plt.title(title_+" - "+date_interval[0]+" to "+date_interval[1]+"")
    plt.xlabel("longitude")
    plt.ylabel("latitude")  
    ax1.set_axis_off()

def display_RefeferencePoint():
    """
    Function to display only the reference points.
    """
    fig, (ax1) = plt.subplots(1, 1,figsize=(8,8))
    plt.rcParams['figure.figsize'] = [4, 4]
    
    # draw departure points
    start_Lon = startingPoint_df.iloc[:,3].to_numpy()
    start_Lat = startingPoint_df.iloc[:,2].to_numpy()
    plt.scatter(start_Lon, start_Lat,c='r', alpha=0.75, s=200)
    plt.plot(start_Lon[0:4], start_Lat[0:4],'-',c='r')

    # draw line departure to summit circle
    plt.scatter(start_Lon[-1], start_Lat[-1], c='g', alpha=0.95, s=400)

    ax1.set_xlim(-73.66, -73.55)    
    ax1.set_ylim(45.45, 45.525)    

    plt.xlabel("longitude")
    plt.ylabel("latitude")  
    ax1.set_axis_off()


