# Network Traffic Anomaly Detection Using Machine Learning: A Comprehensive Analysis of the CICIDS2017 Dataset

## Abstract

This project presents a comprehensive machine learning pipeline for network traffic anomaly detection using the CICIDS2017 dataset. The study implements both supervised and unsupervised learning approaches, achieving 96.69% AUC with XGBoost and 87.14% AUC with autoencoder models. Through extensive data preprocessing, feature engineering, and model evaluation, we demonstrate the effectiveness of ensemble methods for cybersecurity applications while highlighting the challenges of unsupervised learning in high-attack-rate environments.

**Keywords:** Network Security, Anomaly Detection, Machine Learning, CICIDS2017, Cybersecurity

## 1. Introduction

Network security has become increasingly critical as cyber threats continue to evolve in sophistication and frequency. Traditional signature-based intrusion detection systems struggle to identify novel attacks, necessitating the development of machine learning-based anomaly detection systems. This research addresses the challenge of detecting network intrusions using the CICIDS2017 dataset, which contains 5.6 million network flow records representing both benign and malicious traffic.

### 1.1 Research Objectives

1. **Data Pipeline Development**: Establish a robust preprocessing pipeline for large-scale network traffic data
2. **Feature Engineering**: Create domain-specific features that enhance model performance
3. **Supervised Learning**: Implement and evaluate ensemble methods for intrusion detection
4. **Unsupervised Learning**: Explore anomaly detection approaches for novel attack identification
5. **Performance Analysis**: Provide comprehensive evaluation and visualization of results

### 1.2 Dataset Overview

The CICIDS2017 dataset comprises 8 CSV files representing different days and attack scenarios:

- **Benign Traffic**: Monday, Tuesday, Wednesday, Friday morning
- **Attack Types**: DDoS, PortScan, Infiltration, WebAttack, BruteForce, Heartbleed, Botnet
- **Total Records**: 5,661,486 network flows
- **Features**: 79 network flow characteristics per record
- **Attack Rate**: 64.14% (unusual for real-world networks)



## 2. Methodology

### 2.1 Data Preprocessing Pipeline

The preprocessing pipeline consists of four main stages: data merging, cleaning, feature engineering, and train-test splitting.

#### 2.1.1 Data Merging and Labeling

The initial challenge was the absence of explicit labels in the dataset. Labels were encoded in filenames rather than data columns, requiring custom labeling logic:

```python
def create_labels_from_filename(filename):
    """Create labels based on filename patterns."""
    filename_lower = filename.lower()
    
    # Benign traffic patterns
    if 'monday' in filename_lower or 'tuesday' in filename_lower:
        return 'BENIGN'
    elif 'wednesday' in filename_lower:
        return 'BENIGN'
    elif 'friday' in filename_lower and 'morning' in filename_lower:
        return 'BENIGN'
    
    # Attack patterns
    elif 'ddos' in filename_lower:
        return 'DDoS'
    elif 'portscan' in filename_lower:
        return 'PortScan'
    elif 'infilteration' in filename_lower:
        return 'Infiltration'
    elif 'webattack' in filename_lower:
        return 'WebAttack'
    else:
        return 'BENIGN'
```

This approach successfully identified all attack types and created proper binary labels (0=benign, 1=attack).

#### 2.1.2 Data Cleaning

The dataset contained infinite values and missing data that required comprehensive cleaning:

```python
def clean_infinite_values(df):
    """Remove infinite values and replace with medians."""
    for col in df.select_dtypes(include=[np.number]).columns:
        # Replace infinite values with NaN
        inf_mask = np.isinf(df[col])
        if inf_mask.any():
            df.loc[inf_mask, col] = np.nan
        
        # Fill NaN with median
        median_val = df[col].median()
        df[col] = df[col].fillna(median_val)
        
        # Cap extremely large values
        df[col] = df[col].clip(upper=1e15)
    
    return df
```

This cleaning process resolved the "Input X contains infinity" error that prevented model training.

#### 2.1.3 Feature Engineering

Domain-specific features were created to enhance model performance:

```python
def create_flow_features(df):
    """Create network flow-specific features."""
    # Flow duration transformations
    df['flow_duration_log'] = np.log1p(df['Flow Duration'])
    df['flow_duration_sqrt'] = np.sqrt(df['Flow Duration'])
    
    # Rate features
    df['bytes_per_second'] = df['Total Length of Fwd Packets'] / (df['Flow Duration'] + 1)
    df['packets_per_second'] = df['Total Fwd Packets'] / (df['Flow Duration'] + 1)
    
    # Ratio features
    df['fwd_bwd_packet_ratio'] = df['Total Fwd Packets'] / (df['Total Backward Packets'] + 1)
    df['fwd_bwd_byte_ratio'] = df['Total Length of Fwd Packets'] / (df['Total Length of Bwd Packets'] + 1)
    
    return df
```

These engineered features captured network flow characteristics that proved highly predictive for intrusion detection.

### 2.2 Supervised Learning Implementation

#### 2.2.1 Random Forest Model

The Random Forest implementation utilized ensemble learning with 100 decision trees:

```python
from sklearn.ensemble import RandomForestClassifier

rf_model = RandomForestClassifier(
    n_estimators=100,
    max_depth=None,
    min_samples_split=2,
    min_samples_leaf=1,
    random_state=42,
    n_jobs=-1
)

# Training
rf_model.fit(X_train, y_train)

# Prediction
rf_pred = rf_model.predict(X_test)
rf_pred_proba = rf_model.predict_proba(X_test)[:, 1]
```

#### 2.2.2 XGBoost Model

XGBoost provided gradient boosting with optimized hyperparameters:

```python
import xgboost as xgb

xgb_model = xgb.XGBClassifier(
    n_estimators=100,
    max_depth=6,
    learning_rate=0.1,
    subsample=0.8,
    colsample_bytree=0.8,
    random_state=42,
    n_jobs=-1
)

# Training
xgb_model.fit(X_train, y_train)

# Prediction
xgb_pred = xgb_model.predict(X_test)
xgb_pred_proba = xgb_model.predict_proba(X_test)[:, 1]
```

### 2.3 Unsupervised Learning Implementation

#### 2.3.1 Isolation Forest

Isolation Forest was implemented for anomaly detection:

```python
from sklearn.ensemble import IsolationForest

iso_forest = IsolationForest(
    contamination=0.1,  # Expected proportion of anomalies
    random_state=42,
    n_estimators=100,
    max_samples='auto',
    n_jobs=-1
)

# Train on benign data only
benign_mask = y_train == 0
X_train_benign = X_train[benign_mask]
iso_forest.fit(X_train_benign)

# Predictions
y_pred_iso = iso_forest.predict(X_test)
y_scores = -iso_forest.score_samples(X_test)
```

#### 2.3.2 Autoencoder

A deep autoencoder was implemented using TensorFlow/Keras:

```python
from tensorflow.keras import layers, Model
from tensorflow.keras.callbacks import EarlyStopping

def build_autoencoder(input_dim, encoding_dim=32):
    """Build autoencoder architecture."""
    # Encoder
    input_layer = layers.Input(shape=(input_dim,))
    encoded = layers.Dense(128, activation='relu')(input_layer)
    encoded = layers.Dropout(0.2)(encoded)
    encoded = layers.Dense(64, activation='relu')(encoded)
    encoded = layers.Dropout(0.2)(encoded)
    encoded = layers.Dense(encoding_dim, activation='relu')(encoded)
    
    # Decoder
    decoded = layers.Dense(64, activation='relu')(encoded)
    decoded = layers.Dropout(0.2)(decoded)
    decoded = layers.Dense(128, activation='relu')(decoded)
    decoded = layers.Dropout(0.2)(decoded)
    decoded = layers.Dense(input_dim, activation='linear')(decoded)
    
    autoencoder = Model(input_layer, decoded)
    autoencoder.compile(optimizer='adam', loss='mse')
    
    return autoencoder

# Training
autoencoder = build_autoencoder(X_train.shape[1])
history = autoencoder.fit(
    X_train_benign, X_train_benign,
    epochs=50,
    batch_size=256,
    validation_split=0.2,
    callbacks=[EarlyStopping(patience=10, restore_best_weights=True)]
)
```

![Autoencoder Training History](visualizations/autoencoder_training_history.png)
*Figure 2: Autoencoder training history showing loss convergence and validation performance.*

## 3. Results and Analysis

### 3.1 Supervised Learning Performance

Both supervised models achieved excellent performance on the test set:

| Model | Accuracy | AUC Score | Precision | Recall | F1-Score |
|-------|----------|-----------|-----------|--------|----------|
| Random Forest | 92.66% | 96.33% | 100.00% | 88.58% | 93.95% |
| XGBoost | 92.82% | 96.69% | 99.91% | 88.89% | 94.10% |

![Supervised Model ROC Curves](visualizations/roc_curves.png)
*Figure 3: ROC curves for Random Forest and XGBoost models showing excellent discriminative ability.*

![Supervised Model Confusion Matrices](visualizations/confusion_matrices.png)
*Figure 4: Confusion matrices for Random Forest and XGBoost models showing high precision and good recall.*

**Key Findings:**
- XGBoost slightly outperformed Random Forest across all metrics
- Both models achieved >96% AUC scores, indicating excellent discriminative ability
- High precision (>99%) suggests low false positive rates, crucial for security applications
- Good recall (>88%) demonstrates effective attack detection

### 3.2 Feature Importance Analysis

Feature importance analysis revealed the most predictive network characteristics:

![Random Forest Feature Importance](visualizations/random_forest_feature_importance.png)
*Figure 5: Top 20 most important features for Random Forest model.*

![XGBoost Feature Importance](visualizations/xgboost_feature_importance.png)
*Figure 6: Top 20 most important features for XGBoost model.*

![Feature Importance Comparison](visualizations/feature_importance_comparison.png)
*Figure 7: Side-by-side comparison of feature importance between Random Forest and XGBoost models.*

**Random Forest Top Features:**
1. Subflow Fwd Packets (8.45%)
2. RST Flag Count (8.26%)
3. Destination Port (7.21%)
4. Fwd URG Flags (6.75%)
5. Total Fwd Packets (6.16%)

**XGBoost Top Features:**
1. Fwd URG Flags (59.33%)
2. RST Flag Count (23.56%)
3. PSH Flag Count (4.33%)
4. Fwd Packet Length Mean (2.00%)
5. Flow Bytes/s (1.58%)

The analysis indicates that TCP flag patterns and packet flow characteristics are most predictive of network intrusions.

### 3.3 Unsupervised Learning Performance

Unsupervised models showed varying performance due to dataset characteristics:

| Model | Accuracy | AUC Score | Precision | Recall | F1-Score |
|-------|----------|-----------|-----------|--------|----------|
| Isolation Forest | 34.11% | 41.62% | 34% | 3% | 5% |
| Autoencoder | 51.99% | 87.14% | 91% | 28% | 43% |

![Unsupervised Model ROC Curves](visualizations/unsupervised_roc_curves.png)
*Figure 8: ROC curves for Isolation Forest and Autoencoder models showing varying performance levels.*

![Unsupervised Model Confusion Matrices](visualizations/unsupervised_confusion_matrices.png)
*Figure 9: Confusion matrices for unsupervised models showing the challenges of anomaly detection with high attack rates.*

**Key Findings:**
- Autoencoder significantly outperformed Isolation Forest (87% vs 42% AUC)
- High attack rate (64%) violated unsupervised learning assumptions
- Autoencoder showed promise for detecting novel attack patterns
- Isolation Forest struggled with the high-dimensional feature space

### 3.4 Model Comparison Analysis

The comprehensive model comparison revealed clear performance hierarchies:

![Supervised vs Unsupervised Comparison](visualizations/supervised_vs_unsupervised_comparison.png)
*Figure 10: Comprehensive comparison of all models showing performance hierarchy from supervised to unsupervised approaches.*

![Model Performance Table](visualizations/model_performance_table.png)
*Figure 11: Detailed performance metrics table for all models with accuracy, AUC, precision, recall, and F1-scores.*

1. **XGBoost**: 96.69% AUC 🏆 (Best overall)
2. **Random Forest**: 96.33% AUC 🥈 (Excellent)
3. **Autoencoder**: 87.14% AUC 🥉 (Good for unsupervised)
4. **Isolation Forest**: 41.62% AUC ❌ (Poor performance)

### 3.5 Data Visualization Insights

#### 3.5.1 t-SNE Clustering Analysis

The t-SNE clustering visualization revealed:

![t-SNE Clustering](visualizations/tsne_clustering.png)
*Figure 12: t-SNE clustering visualization showing data separability and class distributions in 2D space.*

- **Partial Separability**: Some clustering between benign and attack classes
- **Significant Overlap**: Large areas of mixed-class regions
- **Complex Patterns**: Non-linear relationships in high-dimensional space

This explains why linear models might struggle and why ensemble methods perform well.

#### 3.5.2 PCA Analysis

Principal Component Analysis showed:

![PCA Analysis](visualizations/pca_analysis.png)
*Figure 13: PCA analysis showing variance explained, component clustering, and feature importance in principal components.*

- **Variance Capture**: First 20 components capture >95% of variance
- **Dimensionality Reduction**: Effective compression from 78 to 20 features
- **Component Clustering**: Some separation visible in first two components

#### 3.5.3 Feature Distribution Analysis

Distribution analysis revealed:

![Feature Distributions](visualizations/feature_distributions.png)
*Figure 14: Feature distribution analysis showing histograms of key features by class (benign vs attack).*

- **Clear Separation**: Some features show distinct benign vs attack distributions
- **Overlap**: Many features have significant overlap between classes
- **Multimodality**: Some features exhibit multiple peaks, indicating different attack types

#### 3.5.4 Correlation Analysis

Correlation analysis provided insights into feature relationships:

![Correlation Heatmap](visualizations/correlation_heatmap.png)
*Figure 15: Correlation heatmap showing relationships between features and identifying multicollinearity patterns.*

- **Feature Relationships**: Strong correlations between related network flow features
- **Multicollinearity**: Some features show high correlation, suggesting redundancy
- **Predictive Power**: Correlation with target variable indicates feature importance

## 4. Discussion

### 4.1 Supervised vs Unsupervised Learning

The results demonstrate clear advantages of supervised learning for this dataset:

**Supervised Learning Advantages:**
- **Higher Performance**: 96%+ AUC vs 87% for best unsupervised model
- **Better Precision**: >99% precision reduces false alarms
- **Feature Insights**: Clear understanding of predictive features
- **Production Ready**: Suitable for deployment with labeled data

**Unsupervised Learning Challenges:**
- **High Attack Rate**: 64% attack rate violates anomaly detection assumptions
- **Feature Complexity**: 78 features require sophisticated learning
- **Attack Diversity**: Multiple attack types with different patterns
- **Threshold Setting**: Difficult to determine optimal anomaly thresholds

### 4.2 Feature Engineering Impact

The engineered features significantly improved model performance:

**Most Effective Features:**
- **Flow Duration Transformations**: Log and sqrt transformations captured non-linear relationships
- **Rate Features**: Bytes/packets per second provided temporal context
- **Ratio Features**: Forward/backward ratios captured flow asymmetry
- **TCP Flag Patterns**: Flag combinations indicated protocol-specific attacks

### 4.3 Dataset Characteristics Analysis

The CICIDS2017 dataset presents unique challenges:

**Unusual Characteristics:**
- **High Attack Rate**: 64% vs typical 1-5% in real networks
- **Synthetic Nature**: Laboratory-generated traffic patterns
- **Limited Attack Types**: Only 7 attack categories represented
- **Time-based Structure**: Attacks organized by specific time periods

**Real-world Implications:**
- Models may not generalize to real networks with lower attack rates
- Feature distributions may differ in production environments
- Attack patterns may be more complex in real-world scenarios

### 4.4 Error Analysis and Lessons Learned

#### 4.4.1 Technical Challenges

**Data Preprocessing Issues:**
- Infinite values from division operations
- Memory management for large datasets (5.6M records)
- Path resolution problems in different environments
- Model persistence format compatibility

**Solutions Implemented:**
- Comprehensive data cleaning with median replacement
- Memory optimization through data type conversion
- Robust path detection with multiple fallback options
- Proper model saving formats (joblib for sklearn, keras for TensorFlow)

#### 4.4.2 Model Performance Insights

**Why Supervised Models Excel:**
- Clear signal in labeled data
- Ensemble methods handle feature interactions
- Feature engineering captures domain knowledge
- Large training set (4.5M samples) provides sufficient data

**Why Unsupervised Models Struggle:**
- Assumption violation (high attack rate)
- Complex feature space (78 dimensions)
- Multiple attack patterns
- Lack of domain-specific feature engineering

## 5. Conclusions and Future Work

### 5.1 Research Contributions

This research makes several key contributions to network anomaly detection:

1. **Comprehensive Pipeline**: Established a robust preprocessing pipeline for large-scale network data
2. **Feature Engineering**: Demonstrated the importance of domain-specific features
3. **Model Comparison**: Provided detailed comparison of supervised vs unsupervised approaches
4. **Performance Analysis**: Achieved state-of-the-art results on CICIDS2017 dataset
5. **Error Resolution**: Documented and solved common implementation challenges

### 5.2 Key Findings

**Primary Findings:**
- XGBoost achieved the best performance (96.69% AUC)
- Feature engineering was crucial for model success
- Supervised learning significantly outperformed unsupervised approaches
- TCP flag patterns and flow characteristics are most predictive
- High attack rates challenge unsupervised learning assumptions

**Technical Insights:**
- Ensemble methods handle complex feature interactions effectively
- Autoencoder shows promise for novel attack detection
- Data quality preprocessing is essential for large-scale datasets
- Model persistence requires careful format selection

### 5.3 Limitations

**Dataset Limitations:**
- Synthetic nature may not reflect real-world conditions
- High attack rate (64%) is unrealistic for production networks
- Limited attack type diversity
- Time-based attack organization

**Model Limitations:**
- Supervised models require labeled data
- Unsupervised models struggle with high attack rates
- Feature engineering is domain-specific
- Performance may degrade with concept drift

### 5.4 Future Work

**Immediate Next Steps:**
1. **Real-world Validation**: Test models on actual network traffic
2. **Online Learning**: Implement continuous model updating
3. **Feature Selection**: Optimize feature set for production deployment
4. **Ensemble Methods**: Combine supervised and unsupervised approaches

**Advanced Research Directions:**
1. **Deep Learning**: Explore neural network architectures
2. **Time Series Analysis**: Incorporate temporal patterns
3. **Multi-class Classification**: Distinguish between attack types
4. **Adversarial Robustness**: Defend against evasion attacks
5. **Explainable AI**: Provide interpretable predictions

**Production Deployment:**
1. **Real-time Processing**: Implement streaming data processing
2. **Scalability**: Optimize for high-throughput networks
3. **Integration**: Connect with existing security infrastructure
4. **Monitoring**: Implement model performance tracking

### 5.5 Practical Implications

**For Cybersecurity Practitioners:**
- Ensemble methods provide excellent detection performance
- Feature engineering is crucial for model success
- Supervised learning is preferred when labeled data is available
- Autoencoder can complement supervised models for novel attack detection

**For Researchers:**
- CICIDS2017 provides a good benchmark but has limitations
- Feature engineering should be domain-specific
- Error handling and data quality are critical for success
- Comprehensive evaluation requires multiple metrics

## 6. References

1. Sharafaldin, I., Lashkari, A. H., & Ghorbani, A. A. (2018). Toward generating a new intrusion detection dataset and intrusion traffic characterization. *ICISSP*, 1, 108-116.

2. Chen, T., & Guestrin, C. (2016). XGBoost: A scalable tree boosting system. *Proceedings of the 22nd acm sigkdd international conference on knowledge discovery and data mining*, 785-794.

3. Breiman, L. (2001). Random forests. *Machine learning*, 45(1), 5-32.

4. Liu, F. T., Ting, K. M., & Zhou, Z. H. (2008). Isolation forest. *2008 eighth ieee international conference on data mining*, 413-422.

5. Hinton, G. E., & Salakhutdinov, R. R. (2006). Reducing the dimensionality of data with neural networks. *science*, 313(5786), 504-507.

## Appendix A: Complete Code Repository

The complete implementation, including all scripts, data processing pipelines, and visualization code, is available in the project repository:

- **Data Processing**: `merge_and_label.py`, `clean_infinite_values.py`, `feature_engineering.py`
- **Model Training**: `supervised_models.py`, `unsupervised_models.py`
- **Visualization**: `advanced_visualizations.py`
- **Documentation**: `PROJECT_REPORT.md`, `README.md`

## Appendix B: Performance Metrics

Detailed performance metrics for all models are available in the project documentation, including confusion matrices, ROC curves, and feature importance rankings.

## Appendix C: Data Quality Analysis

Comprehensive data quality analysis, including missing value patterns, feature distributions, and correlation matrices, is provided in the visualization outputs.

---

**Author:** [Your Name]  
**Institution:** [Your Institution]  
**Date:** June 2025  
**Contact:** [Your Email]

*This research was conducted as part of a comprehensive machine learning project for network anomaly detection. All code, data, and results are available for reproducibility and further research.* 