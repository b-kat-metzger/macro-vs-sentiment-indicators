# Overview of Project
## Economic-Based Indicators 
Macroeconomic indicators are objective measures which show the true performance of the economy. 
These indicators are typically published by federal government subsidiaries such as the FED and summarize economic performance over the last few months. 
Since these indicators are published months after the time series that they represent, they are categorized as lagging indicators. 
Some Macroeconomic-Based Indicators we will use in this project are inflation, GDP, unemployment rates, CPI, and corporate bond rates. 

## Finance-Based Indicators 
Finance-Based Indicators are emotional measures of how people who own market shares perceive the current economy. 
Such measures are much more indictive to short-term, volatile changes in the market. 
Because of their volatile nature, these indicators will be much noisier and will require more wrangling measures such as PCA to reduce dimensionality. 
Since these indicators are based on current sentiments of the market, they are categorized as leading indicators. 
Some Finance-Based Indicators we will use in this project are the VIX, S&P trading volume/direction, and social media sentiment scores (from an economic API or by computing scores from a Twitter API).

## Models
We use three different regression models to discover which set of features performs the best for predicting S&P 500 monthly close values. These models are trained and tested on the datasets fetched from Yahoo Finance and the Federal Reserve Economic Database (FRED)

### Linear Regression 
Serving as a baseline model, Linear Regression attempts to predict next-month S&P 500 returns as a weighted linear combination of input features. The model computes a coefficient for each indicator, which allows us to examine which economic or financial variables contribute most to the prediction.

* Strengths: very simple, interpretable, and quick to train
* Limitations: as its name suggests, the model assumes linear relationships and cannot capture any non-linear effects and interactions between features

### Random Forest Regressor
An ensemble method composed of multiple decision trees that are trained on subsets of the data. Each tree learns a set of non-linear decision rules, and the forest combines predictions to reduce overfitting.

* Strengths: can capture non-linear relationships, whereas Linear Regression cannot; handles mixed feature types; provides feature importance scores
* Limitations: less interpretable than linear models; can struggle when useful information is weak or noisy.
We build and test additional ensemble models using bagging and gradient boosting to increase range of testing features

### Long Short-Term Memory Neural Network
A type of recurrent neural network designed to learn from sequential data, making it well-suited for financial time series and predicting markets. LSTM neural networks maintain internal “memory cells” that can capture and remember patterns over time, allowing the model to learn any dependencies that linear and tree-based models could possibly miss.

* Strengths: optimized for sequence learning; can capture long-term patterns; can handle non-linearity
* Limitations: needs a lot of data; prone to overfitting on smaller datasets; can be sensitive to hyperparameters
 * Neural Network Layers: As a deep neural network, our LSTM model uses many layers of nodes to process and propagate predictions. Since we are using the Tensorflow library, these layers are heavily abstracted and we only need to define our layers and their specifics. Our model employs two dropout layers with a dropout rate of 20%, two recurrent layers which process the time series data sequentially and vectorizes it, and two activation dense layers which use ReLU activation to prevent linearity in our probability distribution. 




# Project Structure
│<br>
├── README.md<br>          	
├── Makefile - See [Project Setup](#project-setup)<br>          	
├── requirements.txt -- Create a VM and use "pip install -r requirements.txt" <br>
├── results.txt -- Dynamically created log of models if **make all** is used<br>
├── .gitignore           <br>
│
├── data/ <br>
│   ├── clean/ -- Cleaned and merged aggregate data 
│   └── raw/ -- Unmodified downloaded data <br>  	
│<br>
├── src/ -- Python modules <br>
│   ├── data/ <br>
│   │   ├── data_pipeline.py -- FRED / yfinance API pulls, building merged dataset <br>
│   │   └── run_pipeline.py -- Feature creation, further wrangling and cleaning, execution point for fetching <br> 
│   │ <br>
│   ├── models/ <br>
│   │   ├── linear_regression.py -- Linear Regression Model construction and testing <br>
│   │   ├── lstm_model.py -- Long Short-Term Memory Neural Network Architecture, building, training <br>
│   │   └── rand_forest.py -- Random-Forest Regressor Model building, training <br>
│   │ <br>
│   └── utils/ <br>
│       └── charting.py -- Visualizations for models and dataset <br>
│ <br>
│<br>
├── report/ -- Final report documents <br> 
│    ├── ProjectProgressReport.pdf <br>
│    ├── Team32_FinalReport.pdf -- Formal writeup of project <br>
│    └── economic-financial-features-presentation.pdf -- Formal presentation <br>

## Project Setup
* Create an environment file for the FRED API Key
```
touch .env
```
* Open the environment file and paste your indidvidual FRED API Key that can be registered for free at https://fredaccount.stlouisfed.org/
```
FRED_API_KEY=YOURKEY
```
### Manual
* Create a virtual environment and install dependancies
```
python3 -m venv myenv
python3 pip install -r requirements.txt
```
# Automated
```
make all
```
* Run automated setup to create your environment, fetch and clean the datasets, and build the models
* Upon running the automated script, all model outputs can be found in /results.txt 

### Project Members
- Hank Corrion
- Josh Santiago
- Ben Metzger

### Acknowledgements
* This project was created for an academic project at Washington State University (Nov 2025 - Dec 2025)
* Any code writen in this repository can be distributed and shared openly