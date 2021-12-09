from numpy.core.fromnumeric import mean
import pandas as pd
from pandas.io.parsers import read_csv
import numpy as np
import matplotlib.pyplot as plt 
import scipy.stats as stats

# load data into DataFrame
flying_logbook = pd.read_csv("flying_logbook.csv")
work_hours = pd.read_csv("work_hours.csv")

flying_logbook_df = pd.DataFrame(flying_logbook)
work_hours_df = pd.DataFrame(work_hours)

# merge data
merge_df = flying_logbook_df.merge(work_hours_df, on="Date")
merge_df = pd.DataFrame(merge_df)

merge_df.to_csv("merge.csv")

# clean data 
work_or_flying = []
flying = []
for i in range(len(merge_df)):
    if merge_df.at[i, "Work"] == "No" and merge_df.at[i, "Total Hours"] == 0:
        work_or_flying.append("NaN")
        flying.append("NaN")
    elif merge_df.at[i, "Work"] == "Yes" and merge_df.at[i, "Total Hours"] != 0:
        work_or_flying.append("Yes")
        flying.append("Yes")
    else:
        work_or_flying.append("No")
        if merge_df.at[i, "Work"] == "No" and merge_df.at[i, "Total Hours"] != 0:
            flying.append("Yes")
        else:
            flying.append("No")

merge_df["Fly"] = flying
merge_df["Fly and Work"] = work_or_flying

for i in range(len(work_or_flying)):
    if work_or_flying[i] == "NaN":
        merge_df = merge_df.drop(i)

merge_df.reset_index(inplace=True)
merge_df = merge_df.drop("index", axis=1)
merge_df.to_csv("merge_cleaned.csv")


# Visualizations

# sum of dif categories bar chart? instr, xc, day, night

# split, apply, combine
# group by Work and find average flying time
work_df = merge_df.groupby("Work")
daily_hours_ser = pd.Series(dtype=float)
daily_hours_ser = work_df["Dual Received"].mean()
print(daily_hours_ser)

# scatter plot of amount of landings per flight
plt.figure()
y1 = merge_df["Day Landings"].squeeze()
x1 = merge_df["Total Hours"].squeeze()
plt.scatter(x1, y1, color="red", marker="*")
plt.title("Amount of Landings for Durations of Flying")
plt.xlabel("Flight Duration (hrs)")
plt.ylabel("Number of Landings")
plt.show()

# pi chart of pic or dual
plt.figure()
x = merge_df["Pilot in Command"].squeeze()
y = merge_df["Dual Received"].squeeze()
xbar = mean(x)
ybar = mean(y)
arr = [xbar, ybar]
label_types = ["Pilot in Command", "Dual Received"]
plt.pie(arr, labels=label_types, autopct="%1.1f%%")
plt.show()


# hypthoesis testing

# did i fly more when I had work or when I had no work
both = []
flyonly = []
for i in range(len(merge_df)):
    if merge_df.at[i, "Fly and Work"] == "Yes":
        both.append(merge_df.at[i, "Total Hours"])
    if merge_df.at[i, "Fly and Work"] == "No" and merge_df.at[i, "Fly"] == "Yes":
        flyonly.append(merge_df.at[i, "Total Hours"])
both = pd.Series(both)
flyonly = pd.Series(flyonly)
# Step 1
# H0: flyonly hrs > flyandwork hrs
# H1: flyonly hrs <= flyandwork hrs

# Step 2
# los = 0.05

# Step 3
# two sample, independant one tail
n1 = len(flyonly)
n2 = len(both)
df = n1 + n2 -2

# Step 4
sp2 = ((n1 - 1) * flyonly.std() ** 2 + (n2 - 1) * both.std() ** 2) / df
t = (flyonly.mean() - both.mean()) / np.sqrt(sp2 * (1 / n1 + 1 / n2))
t, p = stats.ttest_ind(flyonly, both)
print("sp2:", sp2)
print("t:", t)
print("p:", p/2)

# Step 5
if p/2 < 0.05:
    print("Reject H0")
else:
    print("Do not reject H0")

# instrument training had longer flights than normal training
# two sample, independant one tail
ifr = []
vfr = []
for i in range(len(merge_df)):
    if merge_df.at[i, "Instrument"] != 0:
        ifr.append(merge_df.at[i, "Total Hours"])
    else:
        vfr.append(merge_df.at[i, "Total Hours"])
ifr = pd.Series(ifr)
vfr = pd.Series(vfr)

n1 = len(ifr)
n2 = len(vfr)
df = n1 + n2 -2

# Step 4
sp2 = ((n1 - 1) * ifr.std() ** 2 + (n2 - 1) * vfr.std() ** 2) / df
t = (ifr.mean() - vfr.mean()) / np.sqrt(sp2 * (1 / n1 + 1 / n2))
t, p = stats.ttest_ind(ifr, vfr)
print("sp2:", sp2)
print("t:", t)
print("p:", p/2)

# Step 5
if p < 0.05:
    print("Reject H0")
else:
    print("Do not reject H0")


# knn
