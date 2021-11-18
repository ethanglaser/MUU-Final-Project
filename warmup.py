import pandas as pd
import matplotlib.pyplot as plt


df = pd.read_csv('Data/202107-citibike-tripdata.csv')
df['ended_at'] = pd.to_datetime(df['ended_at'])
df['started_at'] = pd.to_datetime(df['started_at'])
df['duration_mins'] = (df['ended_at'] - df['started_at']).dt.seconds.div(60)

def warmup1(df, cutoff=120, bins=10):
    plt.figure()
    plt.title('Histogram of ride durations under '+ str(cutoff) + ' mins')
    plt.ylabel('Frequency')
    plt.xlabel('Duration (min)')
    plt.hist(df[df['duration_mins'] < cutoff]['duration_mins'], bins=bins)
    plt.savefig('Figures/warmup1')

def warmup2(df):
    print("Expected duration:", df['duration_mins'].mean())
    print("Duration variance:", df['duration_mins'].var())
    print("Probability of ride > 20 mins:", len(df[df['duration_mins'] > 20].index) / len(df.index))

def warmup3(df):
    temp_df = df[df['member_casual'] == 'member']
    print("Probability of ride > 20 mins given member:", len(temp_df[temp_df['duration_mins'] > 20].index) / len(temp_df.index))

def warmup4(df):
    temp_df = df[df['duration_mins'] > 25]
    print("Probability of being member given > 25 mins ride:", len(temp_df[temp_df['member_casual'] == 'member'].index) / len(temp_df.index))

print("Warmup:")
warmup1(df)
print("2).")
warmup2(df)
print("3).")
warmup3(df)
print("4).")
warmup4(df)