# QuickCart Stockout Risk Prediction

This project implements a supervised ML pipeline to predict **Safe / At-Risk / Imminent** stockout risk for QuickCart’s dark-store operations.

## 📊 Dataset
- 12 stores × 60 SKUs × 30 days = 21,600 rows
- 5 tables: `dim_stores`, `dim_skus`, `dim_suppliers`, `dim_events`, `fact_inventory_daily`

## ✅ Sanity Numbers
- Safe: 65.42% (14,131 rows)
- At-Risk: 24.01% (5,186 rows)
- Imminent: 10.57% (2,283 rows)
- Festival week spike: Imminent 23.31% vs 9.51% non-festival
- Supplier reliability tiers: 15.83% / 6.30% / 3.82%
- Perishable effect: 12.8% vs 9.3%

## 🛠 Feature Engineering
- `reorder_gap = reorder_point - closing_stock`
- `days_of_cover_ratio = days_of_cover / lead_time_days_expected`
- Cleaned supplier reliability (`N/A` imputed with category median)
- `is_recent_reorder` via groupby-shift
- Category one-hot encoding
- Temporal features: `day_of_month`, `days_since_festival_start`

## 🔀 Train/Test Split
- Time-based: Train Oct 1–23, Test Oct 24–30
- Avoids leakage across store-SKU pairs

## 🤖 Models
- Baseline (majority class)
- Multinomial Logistic Regression
- Random Forest
- Gradient Boosting

## 🎯 Metric Focus
- Recall on **Imminent** class (critical to avoid missed stockouts)
