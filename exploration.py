import pandas as pd
import matplotlib.pyplot as plt
import datetime
from collections import Counter

df = pd.read_csv('Data/202107-citibike-tripdata.csv')
df = df[['ended_at', 'started_at', 'start_station_id', 'end_station_id']]
df['ended_at'] = pd.to_datetime(df['ended_at'])
df['started_at'] = pd.to_datetime(df['started_at'])
#df['weekday'] = df['started_at'].dt.weekday
df = df[(df['started_at'].dt.weekday != 5) & (df['started_at'].dt.weekday != 6)]
df = df[(df['start_station_id'] == 5980.07) | (df['end_station_id'] == 5980.07)]

df_start = df[df['start_station_id'] == 5980.07]
df_end = df[df['end_station_id'] == 5980.07]
df_start['result'] = -1
df_end['result'] = 1
df_start['time'] = df_start['started_at']
df_end['time'] = df_end['ended_at']
df_start = df_start[['result', 'time']]
df_end = df_end[['result', 'time']]
df_all = pd.concat([df_start, df_end])

df_all = df_all.sort_values(by=['time'])
results = df_all['result'].tolist()
times = df_all['time'].tolist()
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
plt.figure()
plt.plot(times,x)
plt.savefig('results2')

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

# plot_everyday(6, 10, x_dict)
# plot_everyday(6, 11, x_dict)
# plot_everyday(12, 20, x_dict)
plot_everyday(17, 23, x_dict)

activity = get_activity(x_dict, 5, 17, 23)
plt.figure()
plt.hist(activity)
plt.savefig('Figures/activity_hist')
