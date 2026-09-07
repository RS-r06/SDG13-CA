# CO2 Emission Predictor

A regression model that estimates a country's CO2 emissions from its economic
and energy indicators: GDP per capita, population, energy consumption, share of
renewable energy and industrial production.

Built for a project on UN Sustainable Development Goal 13, Climate Action.

## About the data

The model is trained on a synthetic dataset of 1000 records generated in the
script, not on measured country data. I built it this way to test the modelling
pipeline end to end. The scores below say how well the model fits that
generated data. They are not a claim about real world accuracy.

## What it does

- Generates the dataset and the relationships between the features
- Splits the data into training and test sets
- Fits a Random Forest regressor and a linear regression as a baseline
- Reports mean absolute error, root mean squared error and R squared for both
- Saves charts of the data, the feature importance, the fit and three
  example predictions to the `charts/` folder

## Built with

Python, pandas, numpy, scikit-learn, matplotlib, seaborn

## How to run it

```bash
pip install -r requirements.txt
python co2_predictor.py
```

Add `--show` to open each chart in a window as well as saving it.

## Results

Scores on the held out 20 percent test set:

| Model | MAE (Mt) | RMSE (Mt) | R squared |
|---|---|---|---|
| Random Forest | 4.29 | 5.43 | 0.514 |
| Linear regression | 4.19 | 5.17 | 0.560 |

The two strongest features in the Random Forest were GDP per capita (0.36)
and energy consumption per capita (0.36). Population came third (0.11).

![Actual against predicted](charts/actual_vs_predicted.png)

![Feature importance](charts/feature_importance.png)

## What I learned

The linear model beats the Random Forest here, and it should: the script
generates CO2 as a straight line through the five features plus random noise,
so a linear model is the right shape and the forest can only approximate it.
The R squared of about 0.5 is a ceiling set by that noise, not a fault in the
model.

An earlier version of this project claimed 85 percent accuracy. The script
had never reached that. It reported R squared 0.42, and one of its five
features was a constant because of a wrong parameter in the population
generator, so its row in the correlation matrix was blank. Both are fixed
and the numbers above come from running the current script.

## What I would do differently

Swap the generated data for a real source such as Our World in Data and see
how much worse the model does. Real emissions are not a straight line through
five indicators, so I would expect the Random Forest to earn its place there.

## Author

Rehumile Masego Sechele, rehumiles@gmail.com
