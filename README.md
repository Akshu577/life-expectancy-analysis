# What Affects Life Expectancy? (WHO data, 193 countries, 2000–2015)

**Live page:** https://akshu577.github.io/life-expectancy-analysis/

I wanted to find out which public health and economic factors have the biggest effect on life expectancy, and how few of them you need to predict it well.

Everything is in [`Life_Expectancy_Analysis.ipynb`](Life_Expectancy_Analysis.ipynb).

## What I did

1. **Feature engineering.** I created Population Size groups, a Lifestyle feature (Alcohol × BMI), an Economy feature (GDP × Population) and a Death Ratio.
2. **Cleaning.** I dropped the 10 rows with no target. I treated impossible zeros (0 years of schooling) as missing. I filled missing values with the median *for the same country* first and the overall median only as a fallback.
3. **Transforms.** I log-transformed every column with skewness above 2 (population, GDP, HIV/AIDS, measles and others).
4. **Multicollinearity.** For each pair of predictors with |r| > 0.75, I dropped the one less related to life expectancy.
5. **Outliers.** Deleting every outlier row would have cost more than a third of the data, and those rows were real countries, not errors. So I capped values at the whisker limits instead.
6. **Linear regression** with a 75/25 train/test split.
7. **Forward feature selection** to find the smallest set of features that predicts as well as the full model.

## Results

| | RMSE (years) | MAPE | R² |
|---|---|---|---|
| Train | 3.57 | 4.0% | 0.861 |
| Test | 3.61 | 4.2% | 0.853 |

- Train and test are almost the same, so the model isn't overfitting. 91% of predictions fall within ±10% of the actual value.
- **Only 4 features are needed:** income composition of resources, HIV/AIDS, adult mortality and diphtheria immunization. With those alone, R² is 0.848, about the same as all 17 features.
- **Schooling:** each extra year is linked to about +2.3 years of life expectancy (r = 0.75).
- **Health spending:** in countries below 65 years, spending more had almost no relationship with life expectancy. Targeting HIV and vaccination matters more than the total amount spent.
- **Alcohol** correlates *positively* with life expectancy, but that comes from wealth (richer countries drink more), not from alcohol itself.

## Limitations

- When a country had no values at all for a column, it got the global median. This creates visible vertical lines in some scatter plots.
- Capping HIV/AIDS outliers hides how extreme they were. That's why the model over-predicts for countries like Zimbabwe and Angola in the early 2000s.
- The population column has some values that look like they're missing digits, so I don't trust the population-size results much.

## Data

WHO Global Health Observatory life expectancy data (the Kaggle version), included as `Life_Expectancy_Data.csv`.

## Run it

```bash
pip install pandas numpy matplotlib seaborn scikit-learn jupyter
jupyter notebook Life_Expectancy_Analysis.ipynb
```
