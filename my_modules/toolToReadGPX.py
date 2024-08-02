# Tool for reading gpx file

import numpy as np
import gpxpy
import pandas as pd
import geopy.distance
import matplotlib.pyplot as plt

def fun_gpx2pd(gpx_path):
    """
    The function load a gpx data file and return a pandas dataFrame with info extracted from the gpx file
    """
    with open(gpx_path) as f:
        gpx = gpxpy.parse(f)
    
    # Convert to a dataframe one point at a time.
    points = []
    for segment in gpx.tracks[0].segments:
        for p in segment.points:
            points.append({
                'time': p.time,
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
    df['duration'] = df.time.diff().dt.total_seconds().fillna(0)
    df['cumulative_duration'] = df.duration.cumsum()
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





