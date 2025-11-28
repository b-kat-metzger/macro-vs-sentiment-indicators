# Overview of Project
## Macroeconomic-Based Indicators 
Macroeconomic indicators are objective measures which show the true performance of the economy. 
These indicators are typically published by federal government subsidiaries such as the FED and summarize economic performance over the last few months. 
Since these indicators are published months after the time series that they represent, they are categorized as lagging indicators. 
Some Macroeconomic-Based Indicators we will use in this project are inflation, GDP, unemployment rates, CPI, and corporate bond rates. 

## Sentiment-Based Indicators 
Sentiment-Based Indicators are emotional measures of how people who own market shares perceive the current economy. 
Such measures are much more indictive to short-term, volatile changes in the market. 
Because of their volatile nature, these indicators will be much noisier and will require more wrangling measures such as PCA to reduce dimensionality. 
Since these indicators are based on current sentiments of the market, they are categorized as leading indicators. 
Some Sentiment-Based Indicators we will use in this project are the VIX, S&P trading volume/direction, and social media sentiment scores (from an economic API or by computing scores from a Twitter API). 

# Project Structure
│<br>
├── README.md<br>          	
├── requirements.txt -- Create a VM and use "pip install -r requirements.txt" <br>
├── .gitignore           <br>
│
├── data/ <br>
│   ├── raw/ -- Unmodified downloaded data <br>
│   ├── cleaned/ -- Processed downloaded data <br>  
│   └── static -- Downloaded data from another model; **must be in repo** <br>
│<br>
├── notebooks/ -- Contains all testing and developing of models using jupyter notebook <b*r>
│   	<br>
│   	<br>
├── src/ -- Python modules <br>
│   ├── data/ <br>
│   │   ├── fetch_data.py -- FRED / yfinance API pulls <br>
│   │   ├── clean_data.py -- Cleaning and wrangling <br>
│   │   └── features.py -- Feature creation, further wrangling <br> 
│   │ <br>
│   ├── models/ <br>
│   │   ├── log_regress.py -- Logistic Regression Model building, training <br>
│   │   ├── lstm_model.py -- Long Short-Term Memory Neural Network Architecture, building, training <br>
│   │   └── rand_forest.py -- Random-Forest Regressor Model building, training <br>
│   │ <br>
│   └── utils/ <br>
│       ├── config.py -- Parameters, paths, global variables <br>
│       └── charting.py -- Reusable charts for notebook & final visuals <br>
│ <br>
├── saved/ -- Saved trained models <br>
│ <br>
├── visuals/ -- Visualizations & graphs used in final report <br>
│<br>
├── report/ -- Final report documents <br> 
│    ├── ProjectProgressReport.pdf <br>
│    ├── final_report.pdf -- To be added when report is written <br>
│    └── presentation_slides.pptx -- To be added when slides are created <br>

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
make setup
make clean
make fetch
```

### Project Members
- Hank Corrion
- Josh Santiago
- Ben Metzger
