import pandas as pd
import matplotlib.pyplot as plt
import datetime
import numpy as np
import seaborn as sns
from collections import Counter


def df_prep(df):
    df = df[['ended_at', 'started_at', 'start_station_id', 'end_station_id']]
    df['ended_at'] = pd.to_datetime(df['ended_at'])
    df['started_at'] = pd.to_datetime(df['started_at'])
    df = df[(df['started_at'].dt.weekday != 5) & (df['started_at'].dt.weekday != 6)]
    return df

def get_results_times(df, station_id):
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
    current_day = ''
    min_index = 0
    x = []
    x_dict = {}
    for i in range(len(results)):
        if times[i].day != current_day:
            if current_day != '':
                x_dict[current_day]['range'] = max(x_dict[current_day]['counts']) - min(x_dict[current_day]['counts'])
                plt.figure()
                plt.plot(times[min_index:i], x[min_index:i])
                plt.savefig('Figures/Days/day' + str(current_day))
            current_day = times[i].day
            min_index = i
            x_dict[current_day] = {'counts': [], 'times': [], 'events': [], 'range': 0}
        x_dict[current_day]['counts'].append(sum(results[min_index:i+1]))
        x_dict[current_day]['events'].append(results[i])
        x_dict[current_day]['times'].append(times[i])
        x.append(sum(results[min_index:i+1]))
    print([x_dict[key]['range'] for key in x_dict])
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
    activity = []
    for current_interval in range(start_hour * 60, end_hour * 60, time_interval):
        for key in x_dict:
            current_activity = 0
            for i in range(len(x_dict[key]['times'])):
                if x_dict[key]['times'][i].time() >= datetime.time(current_interval // 60, current_interval % 60) and x_dict[key]['times'][i].time() < datetime.time((current_interval + time_interval) // 60, (current_interval + time_interval) % 60):
                    current_activity += x_dict[key]['events'][i]
            activity.append(current_activity)

    return activity

'''
Martin function here
'''

def create_tm(capacity, activity):
    tm = n_trans = np.zeros([capacity+1,capacity+1])
    current_state = 0

    freq = {}
    for delta in activity:
        freq[delta] = activity.count(delta)

    for state in range(capacity+1):
        for delta in freq:
            if delta < 0 and -delta <= state:
                n_trans[state,state+delta] = freq[delta]
            elif delta > 0 and delta + state <= capacity:
                n_trans[state,state+delta] = freq[delta]
        n_trans[state,state] = freq[0]

    for state in range(capacity+1):
        row_sum = np.sum(n_trans[state,:])
        for end_state in range(capacity+1):
            tm[state,end_state] = n_trans[state,end_state]/row_sum

    return tm


def create_station_markov(df, station_id, start_hour, end_hour, time_interval, capacity):
    results, times = get_results_times(df, station_id)
    my_dict = create_dict(results, times)
    activity = get_activity(my_dict, time_interval, start_hour, end_hour)
    # martin
    markov = create_tm(capacity, activity)
    return markov

if __name__ == '__main__':
    df = pd.read_csv('Data/202107-citibike-tripdata.csv')

    station_id = 5980.07
    capacity = 10
    start_hour = 8
    end_hour = 12
    time_interval = 5
    df = df_prep(df)
    m = create_station_markov(df, station_id, start_hour, end_hour, time_interval, capacity)

    print(m)

    plt.figure(figsize=[20, 20])
    sns.heatmap(m)
    plt.savefig('colormap2.svg')