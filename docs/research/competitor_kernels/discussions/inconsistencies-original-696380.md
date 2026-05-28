# topic 696380 — inconsistencies-original

fetched: 2026-05-28 ; n_messages=6

## msg 0 — unknown (? votes)

The "original" dataset is supposed to be based on data provided by the [FastF1](https://github.com/theOehrly/Fast-F1) library, but the creator of the dataset also derived additional features from the existing features.

Each combination of `Driver`, `Race` and `Year` corresponds to a time series indexed by `LapNumber` \\(i\\). As such, the derived features are supposed to satisfy some consistency requirements. For example,

\\[
LapTime\\_Delta_i=LapTime_i-LapTime_{i-1}\quad\text{if }PitStop_i\not=1
\\]

\\[
Position\\_Change_i=Position_{i-1}-Position_i
\\]

\\[
PitNextLap_i=1\iff PitStop_{i+1}=1
\\]

It seems that these do not hold for all the time series. It is unclear what is causing this, possibly bugs.
```python
def test_consistency(driver, race, year):
    print(F'>>> Test started for {driver=}, {race=}, {year=}')
    mask = (original.Driver==driver)&(original.Race==race)&(original.Year==year)
    assert sum(mask)>0
    X = original[mask].set_index('LapNumber')
    # LapTime_Delta test
    for i, r in X.iterrows():
        try:
            delta = r['LapTime (s)'] - X.loc[i-1, 'LapTime (s)']
            if not r['PitStop'] and not np.isclose(delta, r['LapTime_Delta']):
                print('*** LapTime_Delta consistency test failed.')
                print(F'Expected: {delta}, Actual: {r['LapTime_Delta']}')
                break
        except:
            pass

    # Position_Change test
    for i, r in X.iterrows():
        try:
            delta = X.loc[i-1, 'Position'] - r['Position']
            if not np.isclose(delta, r['Position_Change']):
                print('*** Position_Change consistency test failed.')
                print(F'Expected: {delta}, Actual: {r['Position_Change']}')
                break
        except:
            pass

    # PitStop/PitNextLap test
    for i, r in X.iterrows():
        try:
            pit_next_lap = X.loc[i+1, 'PitStop']
            if pit_next_lap^r['PitNextLap'] :
                print('*** PitStop/PitNextLap consistency test failed.')
                print(F'PitNextLap={r.PitNextLap}, Next PitStop={pit_next_lap}')
                break
        except:
            pass
    print(F'<<< Test completed for {driver=}, {race=}, {year=}')

test_consistency('ALB', 'Abu Dhabi Grand Prix', 2023);print()
test_consistency('ALB', 'Abu Dhabi Grand Prix', 2024);print()
test_consistency('ALB', 'Abu Dhabi Grand Prix', 2025);print()
test_consistency('ZHO', 'Abu Dhabi Grand Prix', 2023);print()
test_consistency('ZHO', 'Abu Dhabi Grand Prix', 2024);print()
```
```text
>>> Test started for driver='ALB', race='Abu Dhabi Grand Prix', year=2023
<<< Test completed for driver='ALB', race='Abu Dhabi Grand Prix', year=2023

>>> Test started for driver='ALB', race='Abu Dhabi Grand Prix', year=2024
*** LapTime_Delta consistency test failed.
Expected: 19.001000000000005, Actual: -26.10499999999999
*** Position_Change consistency test failed.
Expected: 0, Actual: 1.0
*** PitStop/PitNextLap consistency test failed.
PitNextLap=0, Next PitStop=1
<<< Test completed for driver='ALB', race='Abu Dhabi Grand Prix', year=2024

>>> Test started for driver='ALB', race='Abu Dhabi Grand Prix', year=2025
*** LapTime_Delta consistency test failed.
Expected: -8.047000000000011, Actual: 1.8819999999999908
*** Position_Change consistency test failed.
Expected: 0, Actual: 2.0
*** PitStop/PitNextLap consistency test failed.
PitNextLap=0, Next PitStop=1
<<< Test completed for driver='ALB', race='Abu Dhabi Grand Prix', year=2025

>>> Test started for driver='ZHO', race='Abu Dhabi Grand Prix', year=2023
<<< Test completed for driver='ZHO', race='Abu Dhabi Grand Prix', year=2023

>>> Test started for driver='ZHO', race='Abu Dhabi Grand Prix', year=2024
*** LapTime_Delta consistency test failed.
Expected: 19.108999999999995, Actual: -26.033999999999992
*** Position_Change consistency test failed.
Expected: 0, Actual: 1.0
*** PitStop/PitNextLap consistency test failed.
PitNextLap=0, Next PitStop=1
<<< Test completed for driver='ZHO', race='Abu Dhabi Grand Prix', year=2024
```

---

## msg 1 — unknown (? votes)

Great deep dive into the PitStop vs PitNextLap logic. It’s interesting that the mutual information remains low despite the consistency violations. Do you think these derivation errors are systemic across the whole 'original' dataset, or just specific to certain years/races like the Abu Dhabi examples?

---

## msg 2 — unknown (? votes)

Here is another look at how problematic the "original" dataset is. Obviously, among all the variables derived from the fastf1 raw data, the most important ones are `PitNextLap` and `PitStop`. Let's compare what these variables really should be from the raw fastf1 data.

```python
import fastf1, os, logging

logging.disable(logging.CRITICAL)

cache_dir = './f1'
if not os.path.exists(cache_dir):
    os.makedirs(cache_dir)

fastf1.Cache.enable_cache(cache_dir)

original = pd.read_csv('/kaggle/input/datasets/aadigupta1601/f1-strategy-dataset-pit-stop-prediction/f1_strategy_dataset_v4.csv')

def diff_data(driver, race, year):
    # X is raw fastf1 data
    session = fastf1.get_session(year, race, 'R')
    session.load()
    df = session.laps
    df = df[df.Driver==driver]
    assert not df.empty
    X = df[['LapNumber', 'Compound', 'Stint', 'Position']].convert_dtypes()
    X['LapTime (s)'] = df['LapTime'].dt.total_seconds()
    X['PitNextLap'] = df['PitInTime'].notna().astype(int)
    X['PitStop'] = df['PitOutTime'].notna().astype(int)
    # X0 is from "original" dataset
    X0 = original[(original['Driver']==driver) & (original['Race']==race) & (original['Year']==year)]
    cols = ['LapNumber', 'Compound', 'Stint', 'Position', 'LapTime (s)', 'PitNextLap', 'PitStop']
    X0 = X0[cols]

    X['indicator'] = 'fastf1'
    X0['indicator'] = 'original'

    df_tmp = pd.concat([X, X0]).drop_duplicates(cols, keep=False)
    if df_tmp.empty:
        print('FastF1 data and "original" data are identical.')
        return
    X = df_tmp[df_tmp['indicator']=='fastf1'].drop(['indicator'], axis=1)
    X0 = df_tmp[df_tmp['indicator']=='original'].drop(['indicator'], axis=1)
    return X0.merge(
        X, on=['LapNumber', 'Compound', 'Stint', 'Position', 'LapTime (s)'],
        how='left', suffixes=('_orig','_fastf1')
    )
```
```python
d, r, y = 'ALB', 'Abu Dhabi Grand Prix', 2023
diff_data(d,r,y)
```
```text
FastF1 data and "original" data are identical.
```
```python
d, r, y = 'ALB', 'Abu Dhabi Grand Prix', 2024
diff_data(d,r,y)
```
![](https://www.googleapis.com/download/storage/v1/b/kaggle-forum-message-attachments/o/inbox%2F1048539%2F9b3bc8bc604765187eca5860adeb46b6%2FCapture.PNG?generation=1777756488135486&alt=media)

The "original" dataset says the car was pretty much in the pit every lap, which is nonsensical.


---

## msg 3 — unknown (? votes)

These inconsistencies help explain two things I noticed in my analysis. In my [first notebook](https://www.kaggle.com/code/ferariz/f1-pit-stops-does-the-model-know-when-to-pit) the `PitStop` feature showed a borderline leakage pattern that I couldn't fully explain. In my [third notebook](https://www.kaggle.com/code/ferariz/f1-pit-stops-what-s-actually-driving-the-score) the external dataset had a noticeably higher pit rate (0.255 vs 0.199) despite covering the same races — which looked like distribution shift but may actually reflect these derivation errors. I ended up excluding `PitStop` from features independently, and this post suggests that caution was well-placed.

---

## msg 4 — unknown (? votes)

I am Agree for the code Predicting in Pitshop and PitNextLap now in the information has been actually but now you think the systematics for original code that's above

---

## msg 5 — unknown (? votes)

yes also 209359 rows have the inconsistency where the pitstop and stint differ by 2. How can you be on the 2nd stint when having 0 pitstops? is it possible?


---
