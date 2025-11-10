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
│
├── README.md          	
├── requirements.txt   	     Create a VM and use "pip install -r requirements.txt"
├── .gitignore             	
│
├── data/
│   └── raw/                 Unmodified downloaded data  	
│
├── notebooks/		     Contains all testing and developing of models using jupyter notebook
│   	
│   	
├── src/                    Python modules
│   ├── data/
│   │   ├── fetch_data.py          FRED / yfinance API pulls
│   │   ├── clean_data.py          Cleaning and wrangling
│   │   └── features.py		   Feature creation, further wrangling 
│   │
│   ├── models/
│   │   ├── log_regress.py	   Logistic Regression Model building, training
│   │   ├── lstm_model.py          Long Short-Term Memory Neural Network Architecture, building, training
│   │   └── rand_forest.py	   Random-Forest Regressor Model building, training
│   │
│   └── utils/
│       ├── config.py              Parameters, paths, global variables
│       └── charting.py            Reusable charts for notebook & final visuals
│
├── saved/                         Saved trained models
│
├── visuals/                       Visualizations & graphs used in final report
│
└── report/			   Final report documents
    ├── ProjectProgressReport.pdf
    ├── final_report.pdf	       To be added when report is written
    └── presentation_slides.pptx       To be added when slides are created
### Project Members
- Hank Corrion
- Josh Santiago
- Ben Metzger
