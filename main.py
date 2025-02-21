import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import matplotlib.animation as animation

# Load the dataset and remove rows with NaN values in the YEAR column
file_path = "annual_seasonal_mean.csv"
df = pd.read_csv(file_path)
df = df.dropna(subset=['YEAR'])  # Remove rows where YEAR is NaN

# Convert temperature columns to numeric values
temperature_columns = ["JAN-FEB", "MAR-MAY", "JUN-SEP", "OCT-DEC"]
for col in temperature_columns:
    df[col] = pd.to_numeric(df[col], errors='coerce')

# Define month mapping for seasonal data
season_to_months = {
    "JAN-FEB": [1, 2],
    "MAR-MAY": [3, 4, 5],
    "JUN-SEP": [6, 7, 8, 9],
    "OCT-DEC": [10, 11, 12]
}

# Transform dataset into a month-wise format
monthly_data = []
for _, row in df.iterrows():
    year = int(row["YEAR"])
    for season, months in season_to_months.items():
        # Skip if temperature value is NaN
        if pd.isna(row[season]):
            continue
        temp = row[season]
        for month in months:
            monthly_data.append([year, month, temp])

# Create a new DataFrame
df_monthly = pd.DataFrame(monthly_data, columns=["Year", "Month", "Temperature"])

# Normalize temperature anomaly data
df_monthly["Temp_Anomaly_Norm"] = (df_monthly["Temperature"] - df_monthly["Temperature"].min()) / (
        df_monthly["Temperature"].max() - df_monthly["Temperature"].min()
)

# Define plot with adjusted size and spacing
fig = plt.figure(figsize=(13, 10))
ax = fig.add_subplot(111, projection='polar')

# Create a colorbar axis with adjusted position
cbar_ax = fig.add_axes([0.93, 0.15, 0.02, 0.7])  # [left, bottom, width, height]

# Get the temperature range for consistent colormap
temp_min = df_monthly["Temperature"].min()
temp_max = df_monthly["Temperature"].max()
norm = plt.Normalize(temp_min, temp_max)

# Define animation function
def update(frame):
    ax.clear()
    ax.set_theta_zero_location("N")
    ax.set_theta_direction(-1)

    subset = df_monthly[df_monthly["Year"] <= frame]

    theta = np.radians(subset["Month"] * 30)  # Convert months to angle (30 degrees per month)
    r = subset["Temp_Anomaly_Norm"] + 1  # Normalize radius
    temperatures = subset["Temperature"]

    # Create scatter plot with actual temperatures for colors
    scatter = ax.scatter(theta, r, c=temperatures, cmap='coolwarm',
                        norm=norm, alpha=0.7, s=50)

    # Connect points with lines
    ax.plot(theta, r, color='gray', alpha=0.3, lw=1)

    # Customize grid
    ax.grid(True, alpha=0.3)
    ax.set_xticks(np.radians(np.linspace(0, 330, 12)))
    ax.set_xticklabels(["Jan", "Feb", "Mar", "Apr", "May", "Jun",
                        "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"],
                        fontsize=10)

    # Remove radial labels and set limits
    ax.set_yticklabels([])
    ax.set_ylim(0.8, df_monthly["Temp_Anomaly_Norm"].max() + 1.5)

    # Add temperature annotations at points
    for t, r, temp in zip(theta, r, temperatures):
        if frame == subset["Year"].iloc[-1]:  # Only show temps for the latest year
            ax.text(t, r + 0.05, f'{temp:.1f}°C',
                   ha='center', va='bottom', fontsize=8)

    # Update title with more information
    current_temp = subset.iloc[-1]["Temperature"]
    ax.set_title(f"Temperature Variation in India\n"
                 f"Year: {frame} | Latest: {current_temp:.1f}°C",
                 fontsize=14, fontweight="bold", pad=15)

    # Add explanatory text
    fig.text(0.02, 0.02, "Each point represents monthly temperature.\n"
             "Colors indicate temperature values.\n"
             "Distance from center shows relative temperature variation.",
             fontsize=8, ha='left', va='bottom')

    return scatter

# Create animation with a slightly faster speed
years = sorted(df_monthly["Year"].unique())
ani = animation.FuncAnimation(fig, update, frames=years,
                            repeat=True, interval=50)

# Create colorbar with adjusted font size
cbar = plt.colorbar(plt.cm.ScalarMappable(norm=norm, cmap='coolwarm'),
                   cax=cbar_ax, label='Temperature (°C)')
cbar.ax.tick_params(labelsize=9)
cbar.set_label('Temperature (°C)', fontsize=10)

# Add padding to prevent text cutoff
fig.subplots_adjust(right=0.9)

# Save animation as GIF with higher quality
ani.save("india_climate_spiral.gif", writer="pillow", fps=15, dpi=150)

# Show the animation
plt.show()
