# %% Step 1-2: Get Data
import requests
import pandas as pd
import matplotlib.pyplot as plt
import time

# Grabbing asteroid data from NASA's API, wrapped in a function so it retries if it fails
def fetch_asteroids(api_key="DEMO_KEY", max_retries=3, timeout=10):
    url = "https://api.nasa.gov/neo/rest/v1/neo/browse"
    params = {"api_key": api_key}

    for attempt in range(1, max_retries + 1):
        try:
            response = requests.get(url, params=params, timeout=timeout)
            response.raise_for_status()  
            return response.json()
        except requests.exceptions.Timeout:
            print(f"Attempt {attempt}: timed out, retrying...")
        except requests.exceptions.ConnectionError:
            print(f"Attempt {attempt}: connection error, retrying...")
        except requests.exceptions.HTTPError as e:
            # bad request/key, just stop here
            print(f"HTTP error: {e}")
            return None
        time.sleep(2)

    print("Failed after all retries.")
    return None


data = fetch_asteroids()
if data:
    print("Success:", len(data['near_earth_objects']), "asteroids fetched")
else:
    print("Fetch failed.")


# %% Step 3: Clean Data
# each asteroid has a list of close approaches inside it
# one row per asteroid gets turned into many rows, one per close approach it ever had

all_approaches = []
for asteroid in data["near_earth_objects"]:
    # these two values belong to the asteroid itself, so I copied them onto every approach row
    name = asteroid['name']
    is_hazardous = asteroid['is_potentially_hazardous_asteroid']
    for approach in asteroid['close_approach_data']:
        date = approach['close_approach_date']
        miss_km = approach['miss_distance']['kilometers']
        velocity_kmh = approach['relative_velocity']['kilometers_per_hour']
        row = {'name': name, 'is_hazardous': is_hazardous, 'date': date, 'miss_km': miss_km, 'velocity_kmh': velocity_kmh}
        all_approaches.append(row)

print("Total approach events:", len(all_approaches))
print("Sample row:", all_approaches[0])
df = pd.DataFrame(all_approaches)
print("DataFrame shape:", df.shape)
print("Column types before conversion:\n", df.dtypes)
print(df.head())

# %% Step 4: Transform Data

df['date'] = pd.to_datetime(df['date'])
df['miss_km'] = pd.to_numeric(df['miss_km'])
df['velocity_kmh'] = pd.to_numeric(df['velocity_kmh'])

print("Column types after conversion:\n", df.dtypes)

# pulling out year and decade so I can group approaches by time period later
df['year'] = df['date'].dt.year
df['decade'] = (df['year'] // 10 * 10)

print("Preview with year/decade columns added:")
print(df[['name', 'date', 'year', 'decade', 'miss_km', 'velocity_kmh']].head())


# %% Checkpoint: Verify Cleaning
# verifying the cleaning worked - checking for missing values, duplicates, and any weird outliers
print("Missing values per column:\n", df.isna().sum())
print("\nDuplicate rows:", df.duplicated().sum())
print("\nSummary stats (checking for anything that looks wrong):")
print(df.describe())

# %% Step 5: Analyze Data

# 1. What's the single closest an asteroid has ever come to Earth?
closest = df.sort_values('miss_km').head(1)
print("Insight 1 - Closest approach ever recorded:\n", closest)

# 2. Do hazardous asteroids actually pass closer on average than non-hazardous ones?
hazard_comparison = df.groupby('is_hazardous')['miss_km'].mean()
print("Insight 2 - Avg miss distance, hazardous vs non-hazardous:\n", hazard_comparison)

# 3. Which decade has the most recorded close approaches in this dataset?
most_approaches_per_decade = df.groupby('decade').size().sort_values(ascending=False)
print("Insight 3 - Approaches per decade (busiest first):\n", most_approaches_per_decade)

# 4. What's the fastest recorded approach, and which asteroid was it?
fastest = df.sort_values('velocity_kmh', ascending=False).head(1)
print("Insight 4 - Fastest approach ever recorded:\n", fastest)

# 5. Is there a relationship between how fast an asteroid is going and how close it gets?
correlation = df['velocity_kmh'].corr(df['miss_km'])
print(f"Insight 5 - Correlation (velocity vs miss distance): {correlation:.3f}")
# Negative number = faster asteroids tend to pass closer

# %% Step 6: Reports

# report 1 - quick comparison table: hazardous vs non-hazardous asteroids
report1 = df.groupby("is_hazardous").agg(
    hazardous_count =("name", "count"),
    avg_miss_km =("miss_km", "mean"),
    avg_velocity_kmh =("velocity_kmh", "mean"),
).round(1).sort_values("avg_miss_km", ascending=False)

print("Report 1: Hazardous vs Non-Hazardous Summary\n", report1, "\n")


# report 2 - the 10 closest calls Earth has ever had (or will have) in this dataset
report2 = df.sort_values('miss_km').head(10)[['name', 'date', 'miss_km', 'velocity_kmh', 'is_hazardous']].reset_index(drop=True)
print("Report 2: Top 10 Closest Approaches\n", report2)

# %% Step 6.1: Saving Reports

report1.to_csv("Reports/report1_Hazardous_vs_Non-Hazardous_Summary.csv")
report2.to_csv("Reports/report2_top10_closest_approaches.csv", index=False)

# Also save the full cleaned dataset
df.to_csv("Dataset/asteroids_clean_dataset.csv", index=False)
print("Saved: report1, report2, clean dataset as csv")

# %% Step 7: Visualization

# chart 1 - visual version of insight #2, easier to see the gap at a glance
print("Rendering Chart 1: Hazardous vs Non-Hazardous avg distance...")

fig, ax = plt.subplots(figsize=(7,5))
report1['avg_miss_km'].plot(kind='bar', ax=ax, color='steelblue')
ax.set_title('Average Miss Distance: Hazardous vs Non-Hazardous')
ax.set_xlabel('Is Hazardous')
ax.set_ylabel('Average Miss Distance (km)')
plt.xticks(rotation=0)
plt.tight_layout()
plt.savefig('Visualisations/chart1_hazard_vs_distance.png', dpi=150)
plt.show()

# chart 2 - visual version of insight #5, red dots = hazardous asteroids
print("Rendering Chart 2: Velocity vs Miss Distance scatter...")
fig, ax = plt.subplots(figsize=(7,5))
ax.scatter(df['velocity_kmh'], df['miss_km'], alpha=0.6, c=df['is_hazardous'].map({True: 'red', False: 'gray'}))
ax.set_xlabel('Velocity (km/h)')
ax.set_ylabel('Miss Distance (km)')
ax.set_title('Velocity vs Miss Distance (red = hazardous)')
plt.tight_layout()
plt.savefig('Visualisations/chart2_velocity_vs_distance.png', dpi=150)
plt.show()


# %% Step 8: Error Handling Test
# quick test to prove the error handling actually works - using a fake API key on purpose

def fetch_asteroids_broken_test():
    try:
        response = requests.get(
            "https://api.nasa.gov/neo/rest/v1/neo/browse",
            params={"api_key": "INVALID_KEY_TEST"},
            timeout=10
        )
        response.raise_for_status()
        return response.json()
    except requests.exceptions.HTTPError as e:
        print("Handled gracefully:", e)
        return None

test_result = fetch_asteroids_broken_test()
print("Result of broken-key test:", test_result)
# %%
