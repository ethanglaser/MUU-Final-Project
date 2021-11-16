import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv('Data/202107-citibike-tripdata.csv')
df = df[['ended_at', 'started_at', 'start_station_id', 'end_station_id']]
df['ended_at'] = pd.to_datetime(df['ended_at'])
df['started_at'] = pd.to_datetime(df['started_at'])
#df['weekday'] = df['started_at'].dt.weekday
print(len(df.index))
df = df[(df['started_at'].dt.weekday != 5) & (df['started_at'].dt.weekday != 6)]
print(len(df.index))
df = df[(df['start_station_id'] == 5980.07) | (df['end_station_id'] == 5980.07)]
print(len(df.index))

df_start = df[df['start_station_id'] == 5980.07]
df_end = df[df['end_station_id'] == 5980.07]
df_start['result'] = -1
df_end['result'] = 1
df_start['time'] = df_start['started_at']
df_end['time'] = df_end['ended_at']
df_start = df_start[['result', 'time']]
df_end = df_end[['result', 'time']]
df_all = pd.concat([df_start, df_end])
print(len(df_all.index))

df_all = df_all.sort_values(by=['time'])
results = df_all['result']
x = [sum(results[:i+1]) for i in range(len(results))]
plt.figure()
plt.plot(x)
plt.savefig('results')

