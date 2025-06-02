# AgroDashRD - Dominican Agricultural Sector Analysis Dashboard 🌱

## What is AgroDashRD?

AgroDashRD is an interactive tool for visualizing and analyzing agricultural sector data in the Dominican Republic. It's designed to help both farmers and industry professionals make better data-driven decisions.

## Main Features

### For Farmers 👨‍🌾

- **Current Prices**: Check the latest prices of agricultural products in different markets.
- **Harvest Calculator**: Estimate your harvest value based on current market prices.
- **Best Markets**: Discover where you can get better prices for your products.
- **Planting Calendar**: Identify the best times to sell each product.

### For Professionals and Analysts 📊

- **Value Chain Analysis**: Study product and price flow throughout the chain.
- **Market Comparison**: Analyze price differences between markets.
- **Price Forecasts**: View trends and future price predictions.
- **Statistical Analysis**: Access detailed metrics and in-depth data analysis.

## Project Structure

```txt
AgrodashRD(v1.9.2)/
├── app.py                  # Main script to run the app
├── assets/                 # Static files (images, icons, styles)
├── data/                   # Data files and cache
├── src/                    # Main source code
│   ├── dashboard_agricultor.py
│   ├── dashboard_profesional.py
│   ├── ...
│   └── utils/
├── requirements.txt        # Python dependencies
├── pyproject.toml         # Python project configuration
├── Dockerfile             # Docker container configuration
├── Procfile              # Deployment configuration (Heroku/Render)
├── render.yaml           # Render deployment configuration
├── setup.sh             # Initialization script
├── deployment_guide.md   # Detailed deployment guide
├── Next steps.md        # Roadmap and future tasks
└── README.md            # This file
```

## Local Installation and Execution

1. **Clone the repository:**

  ```bash
  git clone <repository-url>
  cd AgrodashRD(v1.8.5)
  ```

2. **Install dependencies:**

  ```bash
  pip install -r requirements.txt
  ```

1. **Run the application:**

  ```bash
  python app.py
  ```

  The application will be available at `http://localhost:8050` or the configured port.

## Deployment

- **Docker:**

  ```bash
  docker build -t agrodashrd .
  docker run -p 8050:8050 agrodashrd
  ```

- **Render/Heroku:**
  Use the `Procfile` and `render.yaml` files for automatic deployment. Check `deployment_guide.md` for detailed instructions.

## Data

Data files are located in the `data/` folder and are used to feed visualizations and analyses. The system uses cache to speed up processing (`data/cache/`).

## Benefits

- **For Farmers:**
  - Make better sales decisions
  - Better plan your plantings
  - Maximize your profits

- **For Professionals:**
  - Analyze market trends
  - Identify improvement opportunities
  - Make data-driven decisions

## Updates

Data is regularly updated to provide accurate and relevant information about the Dominican agricultural market. Datasets are available [here][datasets].

[datasets]: https://drive.google.com/drive/folders/17yLtW6AJeFK46HgnAIFnPCHRALvCReG_?usp=drive_link

## Support

For questions or assistance, contact the support team through the corresponding platform.

---
Developed with ❤️ for the Dominican agricultural sector.
