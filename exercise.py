# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: light
#       format_version: '1.5'
#       jupytext_version: 1.11.2
#   kernelspec:
#     display_name: Python 3
#     language: python
#     name: python3
# ---

# # Analysis of Howell's data with pymc3

import numpy as np              # type: ignore
import matplotlib.pyplot as plt # type: ignore
# %matplotlib inline

import pandas as pd             # type: ignore

# Partial census data for !Kung San people (Africa), collected by Nancy Howell (~ 1960), csv from R. McElreath, "Statistical Rethinking", 2020.

# +
howell: pd.DataFrame

try:
    howell = pd.read_csv('https://raw.githubusercontent.com/rmcelreath/rethinking/master/data/Howell1.csv', sep=';', dtype={'male': bool})
except:
    howell = pd.read_csv('Howell1.csv', sep=';', dtype={'male': bool})
# -

# ## A normal model for the height
#
# We want to analyse the hypothesis that the height of adult people is normally distributed, therefore we design this statistical model ($h$ is the height), with an *a priori* normal distribution of the mean, and an *a priori* uniform distribution of the standard deviation.
#
#
# $ h \sim N(\mu, \sigma)$
#
# $ \mu \sim N(170, 20) $
#
# $ \sigma \sim U(0, 50) $
#
#

import pymc3 as pm   # type: ignore

# +
norm_height = pm.Model()

with norm_height:
    mu = pm.Normal('mu_h', 170, 20)
    sigma = pm.Uniform('sigma_h', 0, 50)
    h = pm.Normal('height', mu, sigma)
# -

# ### Exercise 1
#
# Plot the *a priori* densities of the three random variables of the model. You can sample random values with the method `random` of each. For example `mu.random(size=1000)` samples 1000 values from the *a priori* distribution of `sigma`. 
#

# +
fig, ax = plt.subplots(1, 3, figsize = (15,4))

ax[0].set_title('height')
ax[0].hist(h.random(size = 1000), density=True, label='model height', alpha = 0.8)
ax[0].hist(howell['height'], density=True, label='real height', alpha=0.8)
ax[0].legend()
ax[1].set_title('sigma')
ax[1].hist(sigma.random(size=1000), density=True, label='model std', color = 'blue')
ax[1].vlines(howell['height'].std(),0, 0.03 , label='real std', color = 'black')
ax[1].legend()
ax[2].set_title('mu')
ax[2].hist(mu.random(size=1000), density=True, label='model mean', color = 'green')
ax[2].vlines(howell['height'].mean(), 0, 0.03,
             label='real mean', color='black', ls = '--')
_ = ax[2].legend()
# -

# ### Exercise 2
#
# Consider only adult ($\geq 18$) males. Redefine the model above, making the height `h` an **observed** variable, using Howell's data about adult males as observations.

# +
adult_males = howell[(howell['age'] >= 18) & (howell['male'] == True)]

adult_males_model = pm.Model()

with adult_males_model:
    mu = pm.Normal('mu_h', 170, 20)
    sigma = pm.Uniform('sigma_h', 0, 50)
    h = pm.Normal('height', mu, sigma, observed = adult_males['height'])

# -

# ### Exercise 3
#
# Sample values from the posterior, by using `pm.sample()`. Remember to execute this within the context of the model, by using a `with` statement. 

with adult_males_model:
    sample_adult_males = pm.sample(return_inferencedata=True)


# ### Exercise 4
#
# Plot together the density of the posterior `mu_h` and the density of the prior `mu_h`.
#

# +
fig, ax = plt.subplots()

ax.hist(mu.random(size=1000), label='prior', alpha=0.8, density=True)

# since sample_adult_males.posterior.mu_h contains array with shape(4,1000), 
# I thought it's better to combine all values in 1d array
ax.hist(np.array(sample_adult_males.posterior.mu_h).reshape(-1),label='posterior', alpha=0.8, density=True)

ax.set_title('adult males mean height')
_ = ax.legend()
# -

# ### Exercise 5
#
# Plot the posterior densities by using `pm.plot_posterior`.

_ = pm.plot_posterior(sample_adult_males)

# ### Exercise 6
#
# Since `h` is now an observed variable, it is not possible to sample prior values directly from it. You can instead use `pm.sample_prior_predictive`. Compute the sample prior predictive mean of the height.

with adult_males_model:
    sample_prior = pm.sample_prior_predictive()


# ### Exercise 7
#
# Plot together all the posterior height densities, by using all the sampled values for `mu` and `sigma` (Use the `gaussian` function below. You will get many lines! Use a gray color and a linewidth of 0.1). Add to the plot (in red) the posterior height density computed by using the mean for the posterior `mu` and `sigma`. Add to the plot (in dashed blue) the prior height density computed by using the mean for the prior `mu` and `sigma` (used the values computed by solving the previous exercise).     
#

def gaussian(x: np.ndarray, mu: float, sigma: float) -> np.ndarray:
    return (1/(2*np.pi*sigma**2)**.5)*np.exp(-(x - mu)**2/(2*sigma**2))


# +
interval = np.linspace(100, 220, 500)

fig, ax = plt.subplots(figsize = (10,4))

for i in range(len(sample_adult_males.posterior.mu_h)):
    for j in range(len(sample_adult_males.posterior.mu_h[i])):
        ax.plot(interval, gaussian(x=interval, mu=np.array(sample_adult_males.posterior.mu_h)[i][j], 
                                   sigma=np.array(sample_adult_males.posterior.sigma_h)[i][j]),
                color = 'grey', lw = 0.1, label='_nolegend_' if (i>0) | (j>0) else "posterior")  

        
ax.plot(interval, gaussian(x=interval, mu=np.array(sample_adult_males.posterior.mu_h).mean(), 
            sigma=np.array(sample_adult_males.posterior.sigma_h).mean()),
            color = 'red', label = 'posterior mean')


ax.plot(interval,  gaussian(x=interval, mu=sample_prior['mu_h'].mean(), sigma=sample_prior['sigma_h'].mean()), 
        color='blue', ls='dashed', label = 'prior')


_ = ax.legend()
# -


# ## A linear regression model
#
# We want to analyze the relationship between height and weight in adult males. We consider the following model, where $h$ is the height, $w$ is the weight, $\bar w$ is the mean weight.
#
# $ h \sim N(\mu, \sigma)$
#
# $ \mu = \alpha + \beta*(w - \bar w) $
#
# $ \alpha = N(178, 20) $
#
# $ \beta = N(0, 10) $
#
# $ \sigma \sim U(0, 50) $
#

# ### Exercise 8
#
# Define the model `linear_regression` as a `pm.Model()`. Use Howell's data as observations.

# +
linear_regression = pm.Model()

with linear_regression:
    alpha = pm.Normal('alpha', 178, 20)
    beta = pm.Normal('beta', 0, 10)
    mu = alpha+beta*(adult_males['weight'] - adult_males['weight'].mean())
    sigma = pm.Uniform('sigma', 0, 50)
    h = pm.Normal('height', mu, sigma, observed=adult_males['height'])


# -

# ### Exercise 9
#
# Sample the model and plot the posterior densities.
#

with linear_regression:
    lin_regr_sample = pm.sample(return_inferencedata=True)


_ = pm.plot_posterior(lin_regr_sample)

# ### Exercise 10
#
# Plot a scatter plot of heights and the deviations of the weights from the mean. Add to the plot the regression line  using as the parameters the mean of the sampled posterior values.
#

# +
deviation = adult_males['weight'] - adult_males['weight'].mean()
x = np.linspace(deviation.min(), deviation.max(), 100)

fig, ax = plt.subplots()

ax.scatter(deviation, adult_males['height'])
ax.plot(x, np.array(lin_regr_sample.posterior.alpha).mean() + np.array(lin_regr_sample.posterior.beta).mean()*x, 
       label = f'linear regression\n{np.array(lin_regr_sample.posterior.alpha).mean():.2f} + {np.array(lin_regr_sample.posterior.beta).mean():.2f}*x', 
        color = 'red')

ax.set_xlabel('deviation of weight')
ax.set_ylabel('height')

_ = fig.legend()
