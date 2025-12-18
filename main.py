import matplotlib.pyplot as plt
from os import makedirs
import seaborn as sns
import pandas as pd
import numpy as np
from const import *


def analyze(dataset, column):
    """
    Compute summary statistics and derived metrics for a single column.
    """
    data = dataset[column]

    # Year-over-year absolute change
    yearly_change = np.append([np.nan], np.diff(data)).tolist()
    # Year-over-year percent change
    percent_change = (data.pct_change() * 100).to_list()

    # Simple rolling average with a fixed window
    rolling_window = 5
    convolution = np.convolve(data, np.ones(rolling_window) / rolling_window, mode='valid')
    rolling_avg = np.append([np.nan] * (rolling_window - 1), convolution).tolist()

    # Basic summary stats
    mean = np.mean(data).item()
    std = np.std(data).item()
    median = np.median(data).item()

    # Pack everything into a DataFrame for plotting
    df = pd.DataFrame({
        "Year": dataset["Year"],
        column: data,
        "Yearly Change": yearly_change,
        "Percent Change": percent_change,
        "Rolling Average": rolling_avg
    })

    analysis = {"df": df, "mean": mean, "std": std, "median": median}
    return analysis


class Plot:
    """
    Helper class for generating all climate-related visualizations.
    """
    def __init__(self, dataset, dataset_yearly_avg, analyses):
        # Full merged dataset (monthly or raw records)
        self.dataset = dataset
        # Yearly-averaged dataset
        self.dataset_yearly_avg = dataset_yearly_avg
        # Precomputed analyses for each column
        self.analyses = analyses

    def plot_raw(self):
        """
        Plot raw CO2 and temperature time series.
        """
        fig, axs = plt.subplots(2, 1)
        year = self.dataset["Year"]

        # Raw CO2 levels
        axs[0].plot(year, self.dataset[COLUMNS[0]].dropna(), label=COLUMNS[0], color="mediumslateblue", alpha=0.75)
        axs[0].plot(year, self.dataset[COLUMNS[1]].dropna(), label=COLUMNS[1], color="midnightblue", linestyle="--")
        axs[0].set_title("Carbon Dioxide Levels In Atmosphere")
        axs[0].set_ylabel("Carbon Dioxide (ppm)")
        axs[0].set_xlabel("Year")
        axs[0].legend(fontsize=8)
        axs[0].grid(alpha=0.7)

        # Raw temperature series
        axs[1].plot(year, self.dataset[COLUMNS[2]].dropna(), label=COLUMNS[2], color="mediumslateblue", alpha=0.75)
        axs[1].plot(year, self.dataset[COLUMNS[3]].dropna(), label=COLUMNS[3], color="midnightblue", linestyle="--")
        axs[1].set_title("Global Temperatures")
        axs[1].set_ylabel("Temperature (celsius)")
        axs[1].set_xlabel("Year")
        axs[1].legend(fontsize=8)
        axs[1].grid(alpha=0.7)

        plt.suptitle("Climate Metrics 1958-2015")
        plt.tight_layout()
        # plt.show()
        plt.savefig("Figures/Climate Metrics 1958-2015.png", bbox_inches="tight", dpi=300)

    def plot_avg(self):
        """
        Plot yearly-averaged CO2 and temperature time series.
        """
        fig, axs = plt.subplots(2, 1)
        year = self.dataset_yearly_avg["Year"]

        # Yearly average CO2 levels
        axs[0].plot(year, self.dataset_yearly_avg[COLUMNS_AVG[0]], label=COLUMNS_AVG[0], color="mediumslateblue", alpha=0.75)
        axs[0].plot(year, self.dataset_yearly_avg[COLUMNS_AVG[1]], label=COLUMNS_AVG[1], color="midnightblue", linestyle="--")
        axs[0].set_title("Carbon Dioxide Levels In Atmosphere")
        axs[0].set_ylabel("Carbon Dioxide (ppm)")
        axs[0].set_xlabel("Year")
        axs[0].legend(fontsize=8)
        axs[0].grid(alpha=0.7)

        # Yearly average temperatures
        axs[1].plot(year, self.dataset_yearly_avg[COLUMNS_AVG[2]], label=COLUMNS_AVG[2], color="mediumslateblue", alpha=0.75)
        axs[1].plot(year, self.dataset_yearly_avg[COLUMNS_AVG[3]], label=COLUMNS_AVG[3], color="midnightblue", linestyle="--")
        axs[1].set_title("Global Temperatures")
        axs[1].set_ylabel("Temperature (celsius)")
        axs[1].set_xlabel("Year")
        axs[1].legend(fontsize=8)
        axs[1].grid(alpha=0.7)

        plt.suptitle("Yearly Average Climate Metrics 1958-2015")
        plt.tight_layout()
        # plt.show()
        plt.savefig("Figures/Yearly Average Climate Metrics 1958-2015.png", bbox_inches="tight", dpi=300)

    def plot_analyses(self):
        """
        For each metric, plot:
        - original vs rolling average
        - yearly changes
        - percent changes with a fitted trend line
        - correlation matrix heatmap
        """
        graphs = [
            ("Carbon Dioxide (ppm)", "Yearly Average Carbon Dioxide Levels Analysis"),
            ("Seasonally Adjusted CO2 (ppm)", "Yearly Average Seasonally Adjusted Carbon Dioxide Levels Analysis"),
            ("Land Temperature (celsius)", "Yearly Average Land Temperature Analysis"),
            ("Land and Ocean Temperature (celsius)", "Yearly Average Land and Ocean Temperature Analysis"),
        ]

        for i, analysis in enumerate(self.analyses):
            data, mean, median, std = analysis["df"], analysis["mean"], analysis["median"], analysis["std"]

            # Layout: time series on top, bar chart + scatter/trend below
            fig, axs = plt.subplot_mosaic([['top', 'top'], ['left', 'right']], constrained_layout=True)
            year = data["Year"]

            # Raw + rolling average
            axs["top"].plot(year, data[COLUMNS_AVG[i]], label=COLUMNS_AVG[i], color="lightskyblue", linestyle="--")
            axs["top"].plot(year, data["Rolling Average"], label="Rolling Average", color="blue", alpha=0.7)
            axs["top"].set_ylabel(graphs[i][0])
            axs["top"].set_xlabel("Year")
            axs["top"].grid(alpha=0.7)
            axs["top"].legend(fontsize=8)

            # Yearly change as bars
            axs["left"].bar(year, data["Yearly Change"], label="Yearly Change", color="royalblue", alpha=0.6)
            axs["left"].grid(axis='y', linestyle='--', alpha=0.7)
            axs["left"].set_ylabel("Yearly Change")
            axs["left"].set_xlabel("Year")
            axs["left"].legend(fontsize=8, loc="upper left")

            # Percent change with linear trend over time
            idx = np.isfinite(year) & np.isfinite(data["Percent Change"])
            slope, intercept = np.polyfit(year[idx], data["Percent Change"][idx], 1).tolist()
            line = slope * year + intercept
            axs["right"].scatter(year, data["Percent Change"], s=20, edgecolor="none", label="Percent Change", color="mediumslateblue", alpha=0.5)
            axs["right"].plot(year, line, color="blue", alpha=0.5)
            axs["right"].grid(linestyle='--', alpha=0.7)
            axs["right"].set_ylabel("Percent Change")
            axs["right"].set_xlabel("Year")
            axs["right"].legend(fontsize=8, loc="upper left")

            plt.suptitle(graphs[i][1])
            plt.tight_layout()
            # plt.show()
            plt.savefig(f"Figures/{graphs[i][1]}.png", bbox_inches="tight", dpi=300)

            # Correlation matrix for this metric and its derived columns
            plt.figure(figsize=(8, 6))
            sns.heatmap(
                data.drop("Year", axis=1).rename(columns={COLUMNS_AVG[i]: COLUMNS_ABR[i]}).corr(),
                annot=True,
                cmap='coolwarm',
                fmt=".2f"
            )
            plt.title(f"{COLUMNS_ABR[i]} Correlation Matrix")
            plt.tight_layout()
            # plt.show()
            plt.savefig(f"Figures/Yearly Average {COLUMNS[i]} Correlation Matrix.png", bbox_inches="tight", dpi=300)

    def plot_co2_temp_regression(self):
        """
        Fit linear regression models between yearly average CO2 concentration
        and global temperature metrics, and visualize model fit and residual trends.
        """
        co2_col = COLUMNS_AVG[0]  # "avg. Carbon Dioxide in Atmosphere (ppm)"
        temp_cols = [
            COLUMNS_AVG[2],  # "avg. Land Temperature (celsius)"
            COLUMNS_AVG[3],  # "avg. Land and Ocean Temperature (celsius)"
        ]

        for temp_col in temp_cols:
            # Select and clean data for regression
            data = self.dataset_yearly_avg[[co2_col, temp_col]].dropna()
            x = data[co2_col].values
            y = data[temp_col].values

            # Linear regression using numpy (y = slope * x + intercept)
            slope, intercept = np.polyfit(x, y, 1)
            y_pred = slope * x + intercept
            residuals = y - y_pred

            # R^2 as a simple model quality metric
            ss_res = np.sum(residuals ** 2)
            ss_tot = np.sum((y - np.mean(y)) ** 2)
            r2 = 1 - ss_res / ss_tot

            # Sort x for a clean regression line
            order = np.argsort(x)
            x_sorted = x[order]
            y_pred_sorted = y_pred[order]

            fig, axs = plt.subplots(1, 2, figsize=(12, 5))

            # Left: Model fit (scatter + regression line)
            axs[0].scatter(x, y, alpha=0.6, label="Observed", color="mediumslateblue")
            axs[0].plot(x_sorted, y_pred_sorted, label=f"Linear Fit (R² = {r2:.2f})", color="midnightblue")
            axs[0].set_xlabel("avg. CO$_2$ in Atmosphere (ppm)")
            axs[0].set_ylabel(temp_col)
            axs[0].set_title(f"Linear Regression: {temp_col} vs CO$_2$")
            axs[0].grid(alpha=0.7)
            axs[0].legend(fontsize=8)

            # Right: Residuals vs CO2 (residual trends)
            axs[1].scatter(x, residuals, alpha=0.6, color="mediumslateblue")
            axs[1].axhline(0, linestyle="--", linewidth=1, color="midnightblue")
            axs[1].set_xlabel("avg. CO$_2$ in Atmosphere (ppm)")
            axs[1].set_ylabel("Residuals (°C)")
            axs[1].set_title(f"Residuals: {temp_col} vs CO$_2$")
            axs[1].grid(alpha=0.7)

            plt.suptitle(
                f"CO$_2$ vs {temp_col}: Model Fit and Residual Trends",
                y=1.02
            )
            plt.tight_layout()

            # Make filename safe for saving
            safe_temp_name = temp_col.replace(" ", "_").replace("/", "_").replace(".", "")
            plt.savefig(
                f"Figures/Linear Regression CO2 vs {safe_temp_name}.png",
                bbox_inches="tight",
                dpi=300
            )
            plt.close(fig)


def main():
    """
    Load data, compute yearly averages, run analyses, and generate plots.
    """
    makedirs("Figures", exist_ok=True)

    # CO2 data
    read_co2 = pd.read_csv("Data/CO2LevelsInAtmosphere.csv").dropna()
    filter_co2 = read_co2.filter(["Year", "Carbon Dioxide (ppm)", "Seasonally Adjusted CO2 (ppm)"])

    # Focus on 1958–2015 and compute yearly averages
    co2_data = filter_co2[(filter_co2["Year"] >= 1958) & (filter_co2["Year"] <= 2015)].reset_index(drop=True)
    co2_data_yearly_avg = co2_data.groupby("Year", as_index=False).mean()

    # Temperature data
    read_temp = pd.read_csv("Data/GlobalTemperatures.csv").dropna()
    read_temp["Year"] = pd.to_datetime(read_temp["dt"]).dt.year
    filter_temp = read_temp.filter((["Year", "LandAverageTemperature", "LandAndOceanAverageTemperature"]))

    temp_data = filter_temp[(filter_temp["Year"] >= 1958) & (filter_temp["Year"] <= 2015)].reset_index(drop=True)
    temp_data_yearly_avg = temp_data.groupby("Year", as_index=False).mean()

    # Merge raw datasets
    dataset = pd.merge(co2_data, temp_data, how='inner')
    dataset.columns = ["Year"] + COLUMNS

    # Merge yearly-averaged datasets
    dataset_yearly_avg = pd.merge(co2_data_yearly_avg, temp_data_yearly_avg, how='inner')
    dataset_yearly_avg.columns = ["Year"] + COLUMNS_AVG

    # Precompute analyses for each yearly-average column
    analyses = []
    for column in COLUMNS_AVG:
        analysis = analyze(dataset_yearly_avg, column)
        analyses.append(analysis)

    # Generate all plots
    plot = Plot(dataset, dataset_yearly_avg, analyses)
    plot.plot_raw()
    plot.plot_avg()
    plot.plot_analyses()
    plot.plot_co2_temp_regression()


if __name__ == "__main__":
    main()
