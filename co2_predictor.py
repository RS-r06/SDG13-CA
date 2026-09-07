# CO2 Emission Predictor
#
# Fits a Random Forest and a linear regression to a generated dataset of
# country level economic and energy indicators, reports how well each fits,
# and saves the charts to the charts/ folder.
#
# The data is synthetic. See README.md for what that means for the results.

import argparse
import os

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split

CHART_DIR = "charts"
SEED = 42


def make_dataset(n_samples=1000, seed=SEED):
    """Generate the synthetic dataset.

    CO2 is a fixed linear function of the five features plus noise. That is the
    relationship the models then try to recover.
    """
    rng = np.random.default_rng(seed)

    gdp_per_capita = np.clip(rng.normal(25000, 15000, n_samples), 1000, 80000)
    # Log-normal with median about 20 million, clipped at the size of the
    # largest real country. An earlier version used mean=15 here, which put
    # every value above the clip and made the column a constant.
    population = np.clip(rng.lognormal(3.0, 1.5, n_samples), 0.1, 1400)
    energy_consumption = np.clip(rng.normal(150, 80, n_samples), 10, 500)
    renewable_energy_pct = rng.beta(2, 5, n_samples) * 100
    industrial_production = np.clip(rng.normal(100, 40, n_samples), 20, 300)

    co2_emissions = (
        0.0003 * gdp_per_capita
        + 0.01 * population
        + 0.05 * energy_consumption
        + 0.02 * industrial_production
        - 0.02 * renewable_energy_pct
        + rng.normal(0, 5, n_samples)
    )
    co2_emissions = np.clip(co2_emissions, 0.1, None)

    return pd.DataFrame(
        {
            "GDP_per_capita": gdp_per_capita,
            "Population_millions": population,
            "Energy_consumption_per_capita": energy_consumption,
            "Renewable_energy_percentage": renewable_energy_pct,
            "Industrial_production_index": industrial_production,
            "CO2_emissions_Mt": co2_emissions,
        }
    )


def save(fig, name, show):
    path = os.path.join(CHART_DIR, name)
    fig.savefig(path, dpi=120)
    print(f"  saved {path}")
    if show:
        plt.show()
    plt.close(fig)


def plot_eda(data, show):
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    fig.suptitle("CO2 emissions against each feature", fontsize=14)

    axes[0, 0].hist(data["CO2_emissions_Mt"], bins=30, alpha=0.7, color="red")
    axes[0, 0].set_title("Distribution of CO2 emissions")
    axes[0, 0].set_xlabel("CO2 emissions (Mt)")

    pairs = [
        (axes[0, 1], "GDP_per_capita", "GDP per capita ($)", "tab:blue"),
        (axes[0, 2], "Energy_consumption_per_capita", "Energy consumption per capita", "orange"),
        (axes[1, 0], "Renewable_energy_percentage", "Renewable energy %", "green"),
        (axes[1, 1], "Population_millions", "Population (millions)", "purple"),
    ]
    for ax, col, label, colour in pairs:
        ax.scatter(data[col], data["CO2_emissions_Mt"], alpha=0.6, color=colour, s=12)
        ax.set_title(f"{label} vs CO2")
        ax.set_xlabel(label)
        ax.set_ylabel("CO2 emissions (Mt)")

    sns.heatmap(data.corr(), annot=True, fmt=".2f", cmap="coolwarm", center=0, ax=axes[1, 2])
    axes[1, 2].set_title("Correlation matrix")

    fig.tight_layout()
    save(fig, "eda.png", show)


def plot_importance(importance, show):
    fig, ax = plt.subplots(figsize=(10, 6))
    sns.barplot(
        data=importance, x="importance", y="feature", hue="feature",
        palette="viridis", legend=False, ax=ax,
    )
    ax.set_title("Random Forest feature importance")
    ax.set_xlabel("Importance score")
    ax.set_ylabel("")
    fig.tight_layout()
    save(fig, "feature_importance.png", show)


def plot_performance(y_test, y_pred, r2, show):
    fig, axes = plt.subplots(1, 2, figsize=(15, 6))

    axes[0].scatter(y_test, y_pred, alpha=0.6, color="blue")
    lo, hi = y_test.min(), y_test.max()
    axes[0].plot([lo, hi], [lo, hi], "r--", lw=2)
    axes[0].set_xlabel("Actual CO2 emissions (Mt)")
    axes[0].set_ylabel("Predicted CO2 emissions (Mt)")
    axes[0].set_title(f"Actual vs predicted (R squared = {r2:.3f})")

    residuals = y_test - y_pred
    axes[1].scatter(y_pred, residuals, alpha=0.6, color="green")
    axes[1].axhline(y=0, color="r", linestyle="--")
    axes[1].set_xlabel("Predicted CO2 emissions (Mt)")
    axes[1].set_ylabel("Residual (actual minus predicted)")
    axes[1].set_title("Residuals")

    fig.tight_layout()
    save(fig, "actual_vs_predicted.png", show)


def plot_scenarios(results, show):
    names = [name for name, _ in results]
    values = [value for _, value in results]
    fig, ax = plt.subplots(figsize=(10, 6))
    bars = ax.bar(names, values, color=["green", "orange", "red"], alpha=0.7)
    ax.set_title("Predicted CO2 emissions for three made-up country profiles")
    ax.set_ylabel("Predicted CO2 emissions (Mt)")
    for bar, value in zip(bars, values):
        ax.text(
            bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.5,
            f"{value:.1f} Mt", ha="center", va="bottom", fontweight="bold",
        )
    fig.tight_layout()
    save(fig, "scenarios.png", show)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--show", action="store_true", help="also open each chart in a window")
    args = parser.parse_args()
    if not args.show:
        matplotlib.use("Agg")
    os.makedirs(CHART_DIR, exist_ok=True)

    print("CO2 Emission Predictor")
    print("=" * 60)

    data = make_dataset()
    features = [c for c in data.columns if c != "CO2_emissions_Mt"]
    print(f"\nGenerated {len(data)} rows with features: {features}")

    print("\nDataset statistics")
    print(data.describe().round(2).to_string())

    print("\nCorrelation of each feature with CO2 emissions")
    correlations = data.corr()["CO2_emissions_Mt"].drop("CO2_emissions_Mt").sort_values(ascending=False)
    for feature, corr in correlations.items():
        print(f"  {feature:32s} {corr:+.3f}")

    print("\nCharts")
    plot_eda(data, args.show)

    X = data[features]
    y = data["CO2_emissions_Mt"]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=SEED)

    print("\nModels")
    rf = RandomForestRegressor(n_estimators=100, random_state=SEED)
    rf.fit(X_train, y_train)
    rf_pred = rf.predict(X_test)

    linear = LinearRegression()
    linear.fit(X_train, y_train)
    lin_pred = linear.predict(X_test)

    rows = []
    for name, pred in [("Random Forest", rf_pred), ("Linear regression", lin_pred)]:
        rows.append(
            {
                "model": name,
                "MAE (Mt)": mean_absolute_error(y_test, pred),
                "RMSE (Mt)": np.sqrt(mean_squared_error(y_test, pred)),
                "R squared": r2_score(y_test, pred),
            }
        )
    scores = pd.DataFrame(rows).set_index("model")
    print(scores.round(3).to_string())
    rf_r2 = scores.loc["Random Forest", "R squared"]

    print("\nRandom Forest feature importance")
    importance = (
        pd.DataFrame({"feature": features, "importance": rf.feature_importances_})
        .sort_values("importance", ascending=False)
        .reset_index(drop=True)
    )
    for _, row in importance.iterrows():
        print(f"  {row['feature']:32s} {row['importance']:.3f}")
    plot_importance(importance, args.show)
    plot_performance(y_test, rf_pred, rf_r2, args.show)

    print("\nPredictions for three made-up country profiles")
    scenarios = {
        "High renewables": {
            "GDP_per_capita": 40000,
            "Population_millions": 50,
            "Energy_consumption_per_capita": 200,
            "Renewable_energy_percentage": 80,
            "Industrial_production_index": 120,
        },
        "Developing economy": {
            "GDP_per_capita": 8000,
            "Population_millions": 100,
            "Energy_consumption_per_capita": 80,
            "Renewable_energy_percentage": 20,
            "Industrial_production_index": 60,
        },
        "Industrial heavyweight": {
            "GDP_per_capita": 35000,
            "Population_millions": 200,
            "Energy_consumption_per_capita": 350,
            "Renewable_energy_percentage": 30,
            "Industrial_production_index": 250,
        },
    }
    results = []
    for name, point in scenarios.items():
        prediction = rf.predict(pd.DataFrame([point]))[0]
        results.append((name, prediction))
        print(f"  {name:24s} {prediction:6.2f} Mt")
    plot_scenarios(results, args.show)

    print("\nWhat the fit says")
    top = importance.iloc[0]
    second = importance.iloc[1]
    print(f"  Strongest feature: {top['feature']} ({top['importance']:.2f}), then {second['feature']} ({second['importance']:.2f}).")
    renew = correlations["Renewable_energy_percentage"]
    print(f"  Renewable share correlates {renew:+.2f} with emissions, so more renewables means slightly less CO2 in this data.")
    print(f"  The Random Forest explains {rf_r2:.0%} of the variance in the test set. The rest is the noise term added when the data was generated.")
    print("  These numbers describe the generated data only, not any real country.")


if __name__ == "__main__":
    main()
