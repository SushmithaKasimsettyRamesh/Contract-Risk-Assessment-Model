# Contract Risk Assessment and Forecasting Tool

A machine learning solution that predicts contract risk scores for live entertainment industry bookings, helping agencies proactively manage operational and financial risks.

## 🎯 Project Overview

This project addresses unpredictability in contract execution within the live entertainment industry by leveraging historical data from over **290,000 contracts** to build a predictive risk scoring system. The solution combines business rule logic with data-driven insights to identify high-risk contracts that may suffer from cancellations, overdue payments, or short lead times.

**Key Achievement**: Deployed production-ready ML model with **86.7% R² accuracy** and **<50ms API response time**.

## 🚀 Business Impact

- **Proactive Risk Management**: Enables booking agents to intervene early in high-risk contracts
- **Improved Forecasting**: Enhanced revenue predictability for leadership teams
- **Operational Efficiency**: Reduced last-minute disruptions and manual interventions
- **Better Customer Experience**: Minimized event cancellations and delays

## 🛠️ Tech Stack

- **Machine Learning**: Python, Scikit-learn, Pandas, NumPy
- **Data Processing**: Azure SQL Server, ODBC connections
- **Model Deployment**: Azure Machine Learning, REST API
- **Environment Management**: Conda, Docker containers

## 📊 Model Performance

| Metric | Value |
|--------|--------|
| **R² Score** | 0.867 |
| **Mean Absolute Error** | 0.66 |
| **Root Mean Squared Error** | 0.94 |
| **API Response Time** | ~20ms |

## 🔍 Key Features Engineered

1. **Temporal Features**: Issue-to-event lead time analysis
2. **Financial Features**: Financial delta calculations (Gross - Artist Net - Commission)
3. **Risk Flags**: First-time presenter identification, overdue deposit tracking
4. **Behavioral Patterns**: Agent performance metrics and historical trends

## 🏗️ Architecture

The solution follows an end-to-end ML pipeline:

1. **Data Ingestion**: Multi-source data from contracts, venues, presenters, and financial systems
2. **Feature Engineering**: Advanced temporal, financial, and behavioral feature creation
3. **Model Training**: HistGradientBoostingRegressor for tabular data excellence
4. **Deployment**: Scalable REST API on Azure Container Instance
5. **Monitoring**: Real-time performance tracking and logging

## 📁 Repository Structure

```
contract-risk-assessment/
├── README.md
├── conda-env.yml                 # Environment specifications
├── Model_Training.ipynb          # Complete ML pipeline notebook
├── score.py                      # Production scoring script
├── docs/
│   └── Mar_FT_1_Project_Report.pdf  # Detailed project report
└── requirements.txt              # Python dependencies
```

## 🚦 Getting Started

### Prerequisites
- Python 3.8+
- Conda package manager
- Azure account (for deployment)

### Installation

1. **Clone the repository**
   ```bash
   git clone https://github.com/yourusername/contract-risk-assessment.git
   cd contract-risk-assessment
   ```

2. **Set up environment**
   ```bash
   conda env create -f conda-env.yml
   conda activate contract-risk-env
   ```

3. **Run the training pipeline**
   ```bash
   jupyter notebook Model_Training.ipynb
   ```

## 🔑 Key Insights

**Top Risk Predictors** (by feature importance):
- **Issue-to-Event Days**: Short lead times strongly correlate with contract failures
- **Status Risk Flags**: Cancelled, on-hold, or pending statuses
- **First-Time Presenters**: Higher risk for new clients
- **Financial Deltas**: Discrepancies in financial calculations
- **Agent Performance**: Historical agent success rates

## 📈 Results

The model successfully identifies high-risk contracts with **86.7% accuracy**, enabling:
- Early intervention strategies
- Improved resource allocation
- Enhanced client satisfaction
- Better financial forecasting

## 🔄 Deployment

The model is deployed as a production REST API on Azure Machine Learning with:
- **Scalability**: Azure Container Instance with 2 vCPUs, 4GB RAM
- **Security**: Token-based authentication
- **Monitoring**: Integrated Application Insights
- **Performance**: Sub-50ms response times

## ⚠️ Limitations

- External factors (weather, health, legal issues) not accounted for
- Risk labels derived from heuristic rules rather than confirmed outcomes
- Potential bias from training data patterns
- Dependency on timely CRM data updates

## 🤝 Team

**Team Members**: Dinakar Murthy, Bhanu Teja Nandina, Sushmitha Ramesh, Piyush Sonawane  
**Advisor**: Archit Garg  
**Program**: ElevateMe Bootcamp

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.


---

*For detailed technical documentation, please refer to the [Project Report](docs/Mar_FT_1_Project_Report.pdf).* 
