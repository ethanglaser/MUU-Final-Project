import pandas as pd
import matplotlib.pyplot as plt
import datetime
import numpy as np
import seaborn as sns
from collections import Counter


def df_prep(df):
    # Filter input data, ensure desired data types, drop weekends
    df = df[['ended_at', 'started_at', 'start_station_id', 'end_station_id']]
    df['ended_at'] = pd.to_datetime(df['ended_at'])
    df['started_at'] = pd.to_datetime(df['started_at'])
    df['start_station_id'] = pd.to_numeric(df['start_station_id'], errors='coerce')
    df['end_station_id'] = pd.to_numeric(df['end_station_id'], errors='coerce')
    df = df[(df['started_at'].dt.weekday != 5) & (df['started_at'].dt.weekday != 6)]
    return df

def get_results_times(df, station_id):
    # For a given station, get the time and outcome of every row in the data corresponding to that station
    df = df[(df['start_station_id'] == station_id) | (df['end_station_id'] == station_id)]

    df_start = df[df['start_station_id'] == station_id]
    df_end = df[df['end_station_id'] == station_id]
    df_start['result'] = -1
    df_end['result'] = 1
    df_start['time'] = df_start['started_at']
    df_end['time'] = df_end['ended_at']
    df_start = df_start[['result', 'time']]
    df_end = df_end[['result', 'time']]
    df_all = pd.concat([df_start, df_end])

    df_all = df_all.sort_values(by=['time'])
    return df_all['result'].tolist(),  df_all['time'].tolist()

def create_dict(results, times):
    #creates dictionary with keys for each day and values including the events, counts (cumulation of events), and times
    current_day = ''
    min_index = 0
    x = []
    x_dict = {}
    for i in range(len(results)):
        if times[i].day != current_day:
            if current_day != '':
                x_dict[current_day]['range'] = max(x_dict[current_day]['counts']) - min(x_dict[current_day]['counts'])
                plt.figure()
                plt.plot([t.hour + t.minute / 60.0 for t in times[min_index:i]], x[min_index:i])
                plt.savefig('Figures/Days/day' + str(current_day))
            current_day = times[i].day
            min_index = i
            x_dict[current_day] = {'counts': [], 'times': [], 'events': [], 'range': 0}
        x_dict[current_day]['counts'].append(sum(results[min_index:i+1]))
        x_dict[current_day]['events'].append(results[i])
        x_dict[current_day]['times'].append(times[i])
        x.append(sum(results[min_index:i+1]))
    #plt.figure()
    #plt.plot(times,x)
    #plt.savefig('results2')
    return x_dict

def plot_everyday(start_hour, end_hour, data_dict):
    plt.figure()
    for key in data_dict:
        x = []
        y = []
        for i in range(len(data_dict[key]['times'])):
            if data_dict[key]['times'][i].hour >= start_hour and data_dict[key]['times'][i].hour < end_hour:
                x.append(data_dict[key]['times'][i].hour + data_dict[key]['times'][i].minute / 60)
                y.append(data_dict[key]['counts'][i])
        plt.plot(x, y)
    plt.xlabel('Time (hours)')
    plt.ylabel('Relative capacity (0 at 00 hrs)')
    plt.savefig('Figures/Days/everyday' + str(start_hour) + '-' + str(end_hour))


def get_activity(x_dict, time_interval, start_hour, end_hour):
    # for every day calculate sum of events within time interval over the course of start_hour to end_hour
    activity = []
    for current_interval in range(start_hour * 60, end_hour * 60, time_interval):
        for key in x_dict:
            current_activity = 0
            for i in range(len(x_dict[key]['times'])):
                if x_dict[key]['times'][i].time() >= datetime.time(current_interval // 60, current_interval % 60) and x_dict[key]['times'][i].time() < datetime.time((current_interval + time_interval) // 60, (current_interval + time_interval) % 60):
                    current_activity += x_dict[key]['events'][i]
            activity.append(current_activity)

    return activity

def create_tm(capacity, activity):
    # create transition matrix
    tm = n_trans = np.zeros([capacity+1,capacity+1])

    freq = {}
    for delta in activity:
        freq[delta] = activity.count(delta)

    for state in range(capacity+1):
        for delta in freq:
            delta = int(delta)
            if delta < 0 and -delta <= state:
                n_trans[state,int(state+delta)] = freq[delta]
            elif delta > 0 and delta + state <= capacity:
                n_trans[state,state+delta] = freq[delta]
        n_trans[state,state] = freq[0]

    for state in range(capacity+1):
        row_sum = np.sum(n_trans[state,:])
        for end_state in range(capacity+1):
            tm[state,end_state] = n_trans[state,end_state]/row_sum

    return tm

def get_stationary_distribution(markov):
    # convert markov into stationary distribution
    dim = np.shape(markov)[0]
    A = (markov - np.identity(dim)).T
    A = np.append(A, np.ones([1, dim]), axis=0)
    B = np.zeros(dim + 1)
    B[-1] = 1
    return np.linalg.lstsq(A, B)[0]

def get_lra(stationary):
    # determine long run average from stationary distribution
    long_run_average = 0
    for index, item in enumerate(stationary):
        long_run_average += index*item
    return long_run_average

def create_station_markov(df, station_id, start_hour, end_hour, time_interval, capacity):
    # go through entire process to get stationary distribution and long run average for given station, time frame, interval by reading in processed file or calling above functions    

    #results, times = get_results_times(df, station_id)
    #my_dict = create_dict(results, times)
    #activity = get_activity(my_dict, time_interval, start_hour, end_hour)
    processed = pd.read_csv('processed/processed_station_' + ''.join(str(station_id).split('.')) + '_timestep_' + str(time_interval) + '.csv')
    processed['hour'] = pd.to_datetime(processed['time']).dt.strftime('%H').astype(int)
    processed = processed[(processed['hour'] >= start_hour) & (processed['hour'] < end_hour)]
    activity = processed['total'].tolist()
    markov = create_tm(capacity, activity)

    # save as csv
    time_frame = 'morning'
    if start_hour >= 12:
        time_frame = 'evening'

    # save as heatmap
    plt.figure(figsize=[20, 20])
    plt.title("Heatmap for " + str(station_id) + " from " + str(start_hour) + " to " + str(end_hour))
    sns.heatmap(markov)
    plt.savefig('heatmaps/' + str(station_id) + '_' + time_frame + '_' + str(time_interval) + '.svg')

    stationary = get_stationary_distribution(markov)
    stat_df = pd.DataFrame(stationary, columns = ['Stationary Distribution'])
    stat_df.to_csv('stationary_distributions/' + str(station_id) + '_' + time_frame + '_' + str(time_interval) + '.csv')
    plt.figure()
    plt.title("Stationary distribution for station " + str(station_id) + " in " + time_frame)
    plt.xlabel("Available bikes")
    plt.xlabel("Likelihood of outcome (stationary distribution value)")
    plt.scatter(stat_df.index.values, stat_df['Stationary Distribution'].values)
    plt.savefig('stationary_distributions/' + str(station_id) + '_' + time_frame + '_' + str(time_interval) + '.png')
    #np.savetxt('stationary_distributions/' + str(station_id) + '_' + time_frame + '_' + str(time_interval) + ".csv", stationary, delimiter=",")
    expectation = get_lra(stationary)
    print("Expectation", station_id, capacity, time_frame, time_interval, expectation)
    #return stationary, expectation
    return {'station_id': station_id, 'capacity': capacity, 'time_frame': time_frame, 'time_interval': time_interval, 'expectation': expectation}

if __name__ == '__main__':
    df = pd.read_csv('Data/202107-citibike-tripdata.csv')

    # initialize 3 stations of interest with their capacities
    station_id1 = 6140.05
    station_id2 = 5980.07
    station_id3 = 5329.03

    capacity1 = 50
    capacity2 = 66
    capacity3 = 31

    start_hour_morn = 8
    end_hour_morn = 12
    start_hour_even = 16
    end_hour_even = 20
    time_interval = 10
    df = df_prep(df)
    pre_df = []
    for time_interval in [5, 10]:
        pre_df.append(create_station_markov(df, station_id1, start_hour_morn, end_hour_morn, time_interval, capacity1))
        pre_df.append(create_station_markov(df, station_id1, start_hour_even, end_hour_even, time_interval, capacity1))
        pre_df.append(create_station_markov(df, station_id2, start_hour_morn, end_hour_morn, time_interval, capacity2))
        pre_df.append(create_station_markov(df, station_id2, start_hour_even, end_hour_even, time_interval, capacity2))
        pre_df.append(create_station_markov(df, station_id3, start_hour_morn, end_hour_morn, time_interval, capacity3))
        pre_df.append(create_station_markov(df, station_id3, start_hour_even, end_hour_even, time_interval, capacity3))

    pd.DataFrame(pre_df).to_csv('results.csv')
