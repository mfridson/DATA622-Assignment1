#!/usr/bin/env python3
"""
Bank Marketing Dataset - Complete Analysis
==========================================
Comprehensive EDA, Algorithm Selection, and Preprocessing Strategy
Author: Marc Fridson
Date: 9/28/2025

Dataset: Portuguese Bank Marketing Campaign
Goal: Predict term deposit subscription
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
from scipy import stats
from sklearn.preprocessing import StandardScaler, LabelEncoder, OneHotEncoder
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.decomposition import PCA
from pathlib import Path

# Configuration
warnings.filterwarnings('ignore')
pd.set_option('display.max_columns', None)
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (12, 8)

# ==========================================
# SECTION 1: DATA LOADING & INITIAL REVIEW
# ==========================================

print("=" * 80)
print("BANK MARKETING DATASET - COMPLETE ANALYSIS")
print("=" * 80)

# Paths
SCRIPT_DIR = Path(__file__).resolve().parent
DATA_DIR = SCRIPT_DIR / 'bank+marketing' / 'bank-additional'
OUTPUTS_DIR = SCRIPT_DIR / 'outputs'
OUTPUTS_DIR.mkdir(exist_ok=True)

# Load datasets
df = pd.read_csv(DATA_DIR / 'bank-additional-full.csv', sep=';')
df_small = pd.read_csv(DATA_DIR / 'bank-additional.csv', sep=';')

print(f"\n📊 Dataset Overview:")
print(f"  Full dataset: {df.shape[0]:,} rows × {df.shape[1]} columns")
print(f"  Small dataset: {df_small.shape[0]:,} rows × {df_small.shape[1]} columns")

# Data types
numerical_cols = df.select_dtypes(include=['int64', 'float64']).columns.tolist()
categorical_cols = df.select_dtypes(include=['object']).columns.tolist()

print(f"\n  Numerical features ({len(numerical_cols)}): {', '.join(numerical_cols[:5])}...")
print(f"  Categorical features ({len(categorical_cols)}): {', '.join(categorical_cols[:5])}...")

# ==========================================
# SECTION 2: DATA QUALITY ASSESSMENT
# ==========================================

print("\n" + "=" * 40)
print("DATA QUALITY ANALYSIS")
print("=" * 40)

# Check for missing values
missing = df.isnull().sum()
print(f"\n✓ Traditional missing values: {missing.sum()} (No NaN values)")

# Check for 'unknown' values
print("\n⚠ 'Unknown' values (treated as missing):")
unknown_summary = {}
for col in categorical_cols:
    unknown_count = df[col].str.lower().eq('unknown').sum()
    if unknown_count > 0:
        pct = 100 * unknown_count / len(df)
        unknown_summary[col] = pct
        print(f"  {col:15s}: {pct:5.2f}%")

# Duplicates
duplicates = df.duplicated().sum()
print(f"\n📋 Duplicate rows: {duplicates}")

# Target variable analysis
target_dist = df['y'].value_counts()
target_pct = df['y'].value_counts(normalize=True) * 100
imbalance_ratio = target_dist['no'] / target_dist['yes']

print(f"\n🎯 Target Variable (Term Deposit Subscription):")
print(f"  No:  {target_dist['no']:,} ({target_pct['no']:.1f}%)")
print(f"  Yes: {target_dist['yes']:,} ({target_pct['yes']:.1f}%)")
print(f"  Imbalance Ratio: {imbalance_ratio:.2f}:1")

# ==========================================
# SECTION 3: STATISTICAL ANALYSIS
# ==========================================

print("\n" + "=" * 40)
print("STATISTICAL SUMMARY")
print("=" * 40)

# Basic statistics
desc_stats = df[numerical_cols].describe()
print("\nKey Statistics (selected features):")
key_features = ['age', 'campaign', 'previous', 'emp.var.rate']
print(desc_stats[key_features].round(2))

# Outlier detection
print("\n📍 Outlier Detection (IQR Method):")
outlier_counts = {}
for col in numerical_cols:
    Q1 = df[col].quantile(0.25)
    Q3 = df[col].quantile(0.75)
    IQR = Q3 - Q1
    outliers = df[(df[col] < Q1 - 1.5*IQR) | (df[col] > Q3 + 1.5*IQR)]
    if len(outliers) > 0:
        outlier_counts[col] = len(outliers)

for col, count in sorted(outlier_counts.items(), key=lambda x: x[1], reverse=True)[:5]:
    print(f"  {col:15s}: {count:,} outliers ({100*count/len(df):.1f}%)")

# ==========================================
# SECTION 4: CORRELATION ANALYSIS
# ==========================================

print("\n" + "=" * 40)
print("CORRELATION ANALYSIS")
print("=" * 40)

# Encode target for correlation
df_corr = df.copy()
df_corr['y_encoded'] = df_corr['y'].map({'no': 0, 'yes': 1})

# Calculate correlations
corr_matrix = df_corr[numerical_cols + ['y_encoded']].corr()

# Correlations with target
target_corr = corr_matrix['y_encoded'].sort_values(ascending=False)[1:]
print("\n🎯 Top Features Correlated with Target:")
for feat, corr in target_corr.head(5).items():
    direction = "↑" if corr > 0 else "↓"
    print(f"  {feat:15s}: {corr:+.3f} {direction}")

# High inter-feature correlations
print("\n🔗 Highly Correlated Feature Pairs (|r| > 0.7):")
high_corr_pairs = []
for i in range(len(corr_matrix.columns)):
    for j in range(i+1, len(corr_matrix.columns)):
        if abs(corr_matrix.iloc[i, j]) > 0.7:
            print(f"  {corr_matrix.columns[i]:15s} ↔ {corr_matrix.columns[j]:15s}: {corr_matrix.iloc[i, j]:.3f}")
            high_corr_pairs.append((corr_matrix.columns[i], corr_matrix.columns[j]))

# ==========================================
# SECTION 5: KEY PATTERNS & INSIGHTS
# ==========================================

print("\n" + "=" * 40)
print("KEY PATTERNS & INSIGHTS")
print("=" * 40)

# Previous campaign effectiveness
prev_success = df[df['poutcome'] == 'success']['y'].value_counts(normalize=True) * 100
prev_failure = df[df['poutcome'] == 'failure']['y'].value_counts(normalize=True) * 100
prev_none = df[df['poutcome'] == 'nonexistent']['y'].value_counts(normalize=True) * 100

print("\n📈 Previous Campaign Impact on Current Success:")
print(f"  Previous Success → Current Yes: {prev_success.get('yes', 0):.1f}%")
print(f"  Previous Failure → Current Yes: {prev_failure.get('yes', 0):.1f}%")
print(f"  No Previous Contact → Current Yes: {prev_none.get('yes', 0):.1f}%")

# Economic indicators by outcome
print("\n💰 Economic Conditions by Subscription:")
for feat in ['emp.var.rate', 'euribor3m', 'cons.conf.idx']:
    yes_mean = df[df['y'] == 'yes'][feat].mean()
    no_mean = df[df['y'] == 'no'][feat].mean()
    diff = ((yes_mean - no_mean) / abs(no_mean)) * 100
    print(f"  {feat:20s}: Yes={yes_mean:6.2f}, No={no_mean:6.2f}, Diff={diff:+.1f}%")

# Duration analysis (with warning)
duration_yes = df[df['y'] == 'yes']['duration'].mean()
duration_no = df[df['y'] == 'no']['duration'].mean()
print(f"\n⚠️ Duration Impact (NOT usable for prediction - post-call data):")
print(f"  Average duration - Yes: {duration_yes:.0f}s, No: {duration_no:.0f}s")
print(f"  Ratio: {duration_yes/duration_no:.2f}x longer for successful calls")

# ==========================================
# SECTION 6: VISUALIZATIONS
# ==========================================

print("\n" + "=" * 40)
print("GENERATING VISUALIZATIONS")
print("=" * 40)

# Create comprehensive visualization
fig = plt.figure(figsize=(16, 12))

# 1. Target Distribution
ax1 = plt.subplot(3, 3, 1)
target_dist.plot(kind='pie', autopct='%1.1f%%', colors=['#FF6B6B', '#4ECDC4'], ax=ax1)
ax1.set_title('Target Distribution', fontweight='bold')
ax1.set_ylabel('')

# 2. Age Distribution by Target
ax2 = plt.subplot(3, 3, 2)
df[df['y']=='yes']['age'].hist(alpha=0.6, label='Yes', bins=20, color='#4ECDC4', ax=ax2)
df[df['y']=='no']['age'].hist(alpha=0.6, label='No', bins=20, color='#FF6B6B', ax=ax2)
ax2.set_title('Age Distribution by Subscription', fontweight='bold')
ax2.set_xlabel('Age')
ax2.legend()

# 3. Previous Outcome Impact
ax3 = plt.subplot(3, 3, 3)
poutcome_impact = pd.crosstab(df['poutcome'], df['y'], normalize='index') * 100
poutcome_impact.plot(kind='bar', ax=ax3, color=['#FF6B6B', '#4ECDC4'])
ax3.set_title('Previous Campaign → Current', fontweight='bold')
ax3.set_xticklabels(ax3.get_xticklabels(), rotation=45)
ax3.legend(['No', 'Yes'])

# 4. Top Jobs
ax4 = plt.subplot(3, 3, 4)
df['job'].value_counts().head(8).plot(kind='barh', ax=ax4, color='#95E1D3')
ax4.set_title('Top 8 Job Categories', fontweight='bold')

# 5. Contact by Month
ax5 = plt.subplot(3, 3, 5)
month_order = ['jan', 'feb', 'mar', 'apr', 'may', 'jun', 'jul', 'aug', 'sep', 'oct', 'nov', 'dec']
month_counts = df['month'].value_counts().reindex(month_order, fill_value=0)
month_counts.plot(kind='bar', ax=ax5, color='#FFA07A')
ax5.set_title('Contacts by Month', fontweight='bold')
ax5.set_xticklabels(ax5.get_xticklabels(), rotation=45)

# 6. Campaign Contacts Distribution
ax6 = plt.subplot(3, 3, 6)
campaign_dist = df['campaign'].value_counts().head(10).sort_index()
campaign_dist.plot(kind='bar', ax=ax6, color='#98D8C8')
ax6.set_title('Number of Contacts per Customer', fontweight='bold')
ax6.set_xlabel('Number of Contacts')

# 7. Economic Indicators
ax7 = plt.subplot(3, 3, 7)
econ_cols = ['emp.var.rate', 'cons.price.idx', 'euribor3m']
df[econ_cols].boxplot(ax=ax7)
ax7.set_title('Economic Indicators Distribution', fontweight='bold')
ax7.set_xticklabels(ax7.get_xticklabels(), rotation=45)

# 8. Education Level
ax8 = plt.subplot(3, 3, 8)
edu_order = ['illiterate', 'basic.4y', 'basic.6y', 'basic.9y', 'high.school', 
             'professional.course', 'university.degree']
edu_counts = df['education'].value_counts().reindex(edu_order, fill_value=0)
edu_counts.plot(kind='bar', ax=ax8, color='#B19CD9')
ax8.set_title('Education Distribution', fontweight='bold')
ax8.set_xticklabels(ax8.get_xticklabels(), rotation=45, ha='right')

# 9. Correlation Heatmap (subset)
ax9 = plt.subplot(3, 3, 9)
corr_subset = df[['age', 'campaign', 'previous', 'emp.var.rate', 'euribor3m']].corr()
sns.heatmap(corr_subset, annot=True, fmt='.2f', cmap='coolwarm', center=0, ax=ax9, cbar_kws={'shrink': 0.8})
ax9.set_title('Feature Correlations', fontweight='bold')

plt.suptitle('Bank Marketing Dataset - Complete Analysis', fontsize=16, fontweight='bold')
plt.tight_layout()
plt.savefig(OUTPUTS_DIR / 'complete_analysis.png', dpi=300, bbox_inches='tight')
print(f"✓ Visualizations saved to {OUTPUTS_DIR / 'complete_analysis.png'}")

# ==========================================
# SECTION 7: ALGORITHM SELECTION
# ==========================================

print("\n" + "=" * 40)
print("ALGORITHM SELECTION")
print("=" * 40)

print("\n🎯 Problem Type: Binary Classification (Supervised Learning)")
print(f"  • Dataset size: {len(df):,} records (sufficient for complex models)")
print(f"  • Features: {len(df.columns)-1} (mixed types)")
print(f"  • Class imbalance: {imbalance_ratio:.1f}:1")
print(f"  • Labels: Present ('y' column)")

print("\n📊 Algorithm Recommendations:")

algorithms = [
    {
        'name': 'XGBoost',
        'score': 9.5,
        'pros': ['Handles imbalance via scale_pos_weight', 'Robust to outliers', 
                 'Feature importance', 'Built-in regularization'],
        'cons': ['Requires tuning', 'Less interpretable than linear models']
    },
    {
        'name': 'Random Forest',
        'score': 8.5,
        'pros': ['No scaling needed', 'Handles non-linear patterns', 'Parallel training'],
        'cons': ['Can overfit with imbalance', 'Memory intensive']
    },
    {
        'name': 'Logistic Regression',
        'score': 7.0,
        'pros': ['Highly interpretable', 'Fast training', 'Good baseline'],
        'cons': ['Assumes linearity', 'Requires scaling', 'Sensitive to outliers']
    }
]

for algo in algorithms:
    print(f"\n{algo['name']} (Score: {algo['score']}/10)")
    print(f"  ✓ Pros: {', '.join(algo['pros'])}")
    print(f"  ✗ Cons: {', '.join(algo['cons'])}")

print("\n🏆 PRIMARY RECOMMENDATION: XGBoost")
print("  Reasoning: Best handles imbalance, captures non-linear patterns, provides interpretability")

# ==========================================
# SECTION 8: PREPROCESSING PIPELINE
# ==========================================

print("\n" + "=" * 40)
print("PREPROCESSING STRATEGY")
print("=" * 40)

class BankPreprocessor:
    """Complete preprocessing pipeline for bank marketing data"""
    
    def __init__(self):
        self.numerical_features = ['age', 'campaign', 'previous', 'emp.var.rate', 
                                  'cons.price.idx', 'cons.conf.idx', 'euribor3m', 'nr.employed']
        self.categorical_features = ['job', 'marital', 'education', 'contact', 'month', 
                                    'day_of_week', 'poutcome', 'default', 'housing', 'loan']
    
    def clean_data(self, df):
        """Step 1: Data cleaning"""
        df_clean = df.copy()
        df_clean = df_clean.drop_duplicates()  # Remove 12 duplicates
        if 'duration' in df_clean.columns:
            df_clean = df_clean.drop('duration', axis=1)  # Remove data leakage
        return df_clean
    
    def engineer_features(self, df):
        """Step 2: Feature engineering"""
        df_eng = df.copy()
        
        # Customer engagement features
        df_eng['total_contacts'] = df_eng['campaign'] + df_eng['previous']
        df_eng['has_previous_success'] = (df_eng['poutcome'] == 'success').astype(int)
        df_eng['never_contacted'] = (df_eng['pdays'] == 999).astype(int)
        
        # Economic pressure index
        df_eng['economic_pressure'] = (df_eng['emp.var.rate'] * df_eng['euribor3m']) / (df_eng['cons.conf.idx'].abs() + 1)
        
        # Temporal features
        month_to_quarter = {'jan': 1, 'feb': 1, 'mar': 1, 'apr': 2, 'may': 2, 'jun': 2,
                          'jul': 3, 'aug': 3, 'sep': 3, 'oct': 4, 'nov': 4, 'dec': 4}
        df_eng['quarter'] = df_eng['month'].map(month_to_quarter)
        
        return df_eng
    
    def handle_outliers(self, df, strategy='cap'):
        """Step 3: Handle outliers"""
        df_out = df.copy()
        for col in self.numerical_features:
            if col in df_out.columns:
                Q1, Q3 = df_out[col].quantile([0.25, 0.75])
                IQR = Q3 - Q1
                if strategy == 'cap':
                    lower, upper = Q1 - 1.5*IQR, Q3 + 1.5*IQR
                    df_out[col] = df_out[col].clip(lower, upper)
        return df_out
    
    def create_preprocessing_pipeline(self):
        """Step 4: Create sklearn pipeline"""
        from sklearn.compose import ColumnTransformer
        from sklearn.pipeline import Pipeline
        from sklearn.preprocessing import StandardScaler, OneHotEncoder
        
        numeric_transformer = StandardScaler()
        categorical_transformer = OneHotEncoder(handle_unknown='ignore')
        
        preprocessor = ColumnTransformer(
            transformers=[
                ('num', numeric_transformer, self.numerical_features),
                ('cat', categorical_transformer, self.categorical_features)
            ])
        
        return preprocessor
    
    def transform(self, df):
        """Apply complete preprocessing"""
        df = self.clean_data(df)
        df = self.engineer_features(df)
        df = self.handle_outliers(df)
        return df

# Apply preprocessing
print("\n📋 Preprocessing Steps:")
print("  1. Data Cleaning:")
print("     • Remove 12 duplicate records")
print("     • Drop 'duration' column (data leakage)")
print("     • Keep 'unknown' as separate category")

print("\n  2. Feature Engineering:")
print("     • Create engagement score (total_contacts)")
print("     • Add has_previous_success flag")
print("     • Generate economic_pressure index")
print("     • Extract temporal features (quarter)")

print("\n  3. Handle Class Imbalance:")
print(f"     • XGBoost: scale_pos_weight = {imbalance_ratio:.2f}")
print("     • Alternative: SMOTE for synthetic samples")
print("     • Metrics: Use F1-score and PR-AUC, not accuracy")

print("\n  4. Handle Outliers:")
print("     • Strategy: Cap at 1.5 IQR boundaries")
print("     • Most affected: campaign, previous, duration")

# ==========================================
# SECTION 9: IMPLEMENTATION CODE
# ==========================================

print("\n" + "=" * 40)
print("IMPLEMENTATION READY CODE")
print("=" * 40)

implementation_code = """
# Complete Implementation Pipeline
from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import classification_report, roc_auc_score
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline as ImbPipeline

# Initialize preprocessor
preprocessor = BankPreprocessor()
df_processed = preprocessor.transform(df)

# Prepare for modeling
X = df_processed.drop('y', axis=1)
y = df_processed['y'].map({'no': 0, 'yes': 1})

# Split data
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# Create pipeline
pipeline = ImbPipeline([
    ('preprocessor', preprocessor.create_preprocessing_pipeline()),
    ('smote', SMOTE(random_state=42)),
    ('classifier', XGBClassifier(
        scale_pos_weight=7.88,
        n_estimators=100,
        max_depth=5,
        learning_rate=0.1,
        random_state=42
    ))
])

# Train and evaluate
pipeline.fit(X_train, y_train)
y_pred = pipeline.predict(X_test)

print(classification_report(y_test, y_pred))
print(f"ROC-AUC Score: {roc_auc_score(y_test, y_pred):.3f}")
"""

print("Sample Implementation Code:")
print(implementation_code)

# ==========================================
# SECTION 10: BUSINESS RECOMMENDATIONS
# ==========================================

print("\n" + "=" * 40)
print("BUSINESS INSIGHTS & RECOMMENDATIONS")
print("=" * 40)

insights = {
    "🎯 Targeting Strategy": 
        "Focus on customers with previous campaign success (65% conversion vs 9% baseline)",
    
    "📅 Timing Optimization": 
        "Campaign during economic uncertainty periods (employment rate changes correlate with success)",
    
    "📞 Contact Quality": 
        "Prioritize relationship building over contact volume - quality beats quantity",
    
    "👥 Customer Segmentation": 
        "Create distinct strategies for 'never contacted' vs 'previous contact' segments",
    
    "📊 Success Metrics": 
        "Track F1-score and Precision-Recall AUC, not accuracy (due to imbalance)",
    
    "🔄 Continuous Improvement": 
        "Implement A/B testing framework to validate model predictions",
    
    "💡 Quick Wins": 
        "Immediate 7x improvement possible by focusing on previous success customers",
    
    "⚠️ Risk Mitigation": 
        "Monitor model drift as economic conditions change"
}

for key, value in insights.items():
    print(f"\n{key}:")
    print(f"  → {value}")

# ==========================================
# FINAL SUMMARY
# ==========================================

print("\n" + "=" * 80)
print("ANALYSIS COMPLETE - KEY DELIVERABLES")
print("=" * 80)

print(f"""
✅ DATA INSIGHTS:
   • Dataset: {df.shape[0]:,} records with {df.shape[1]} features
   • Class Imbalance: {imbalance_ratio:.1f}:1 ratio
   • Key Pattern: Previous success → 65% conversion rate
   • Data Quality: {len(unknown_summary)} features with 'unknown' values

✅ ALGORITHM SELECTION:
   • Primary: XGBoost (handles imbalance, non-linear patterns)
   • Secondary: Random Forest (ensemble validation)
   • Baseline: Logistic Regression (interpretability)

✅ PREPROCESSING PIPELINE:
   • Remove duplicates and data leakage
   • Engineer 5 new predictive features
   • Handle outliers with IQR capping
   • Address imbalance with SMOTE + scale_pos_weight

✅ BUSINESS IMPACT:
   • Potential 7x improvement in targeting efficiency
   • Clear actionable insights for campaign timing
   • Data-driven customer segmentation strategy

📊 Visualization saved: complete_analysis.png
📝 Ready for production implementation
""")

print("GitHub Repository: [Add your GitHub URL here]")
print("=" * 80)
