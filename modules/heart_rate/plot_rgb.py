import pandas as pd
import matplotlib.pyplot as plt

data = pd.read_csv("modules/heart_rate/rgb_data.csv")

plt.figure(figsize=(10, 5))

plt.plot(data["time"], data["red"], label="Red")
plt.plot(data["time"], data["green"], label="Green")
plt.plot(data["time"], data["blue"], label="Blue")

plt.xlabel("Time (seconds)")
plt.ylabel("Pixel Intensity")
plt.title("Forehead RGB Signal")
plt.legend()
plt.grid()

plt.show()