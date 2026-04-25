"""
Comprehensive 4-Model Cricket Match Prediction Evaluation
Models: XGBoost (Pre-trained), Neural Network, Logistic Regression, Random Forest
Generates organized results in results/ directory with model-specific subfolders
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import joblib
import os
import shap
from pathlib import Path
import optuna
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.model_selection import GroupShuffleSplit
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import accuracy_score, confusion_matrix, ConfusionMatrixDisplay
from sklearn.neural_network import MLPClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.inspection import permutation_importance
import warnings
warnings.filterwarnings('ignore')

# ============================================================================
# 1. DIRECTORY SETUP
# ============================================================================
print("=" * 70)
print("STEP 1: Setting up directory structure")
print("=" * 70)

base_results_dir = Path("results")
base_results_dir.mkdir(exist_ok=True)

model_dirs = {
    'XGBoost': base_results_dir / 'XGBoost',
    'Neural_Network': base_results_dir / 'Neural_Network',
    'Logistic_Regression': base_results_dir / 'Logistic_Regression',
    'Random_Forest': base_results_dir / 'Random_Forest'
}

for model_name, model_dir in model_dirs.items():
    model_dir.mkdir(parents=True, exist_ok=True)
    print(f"✓ Created: {model_dir}")

# ============================================================================
# 2. DATA LOADING & PREPROCESSING
# ============================================================================
print("\n" + "=" * 70)
print("STEP 2: Loading and preprocessing data")
print("=" * 70)

df = pd.read_csv('/Users/vipulsharma/Documents/projects/cricket/code/csv/cricket_features_with_defense_final.csv')
df_base = df.drop(columns=['Date', 'Venue', 'Reference_Team', 'Opponent'])

le = LabelEncoder()
df_base['Result'] = le.fit_transform(df_base['Result'])

X = df_base.drop(columns=['Result'])
y = df_base['Result']
groups = X['Match_ID']

# Train/Test Split (By Match_ID)
gss = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
train_idx, test_idx = next(gss.split(X, y, groups))

X_train_full = X.iloc[train_idx]
y_train_full = y.iloc[train_idx]

X_test = X.iloc[test_idx].drop(columns=['Match_ID'])
y_test = y.iloc[test_idx]

X_train_clean = X_train_full.drop(columns=['Match_ID'])

print(f"Train set: {X_train_clean.shape}")
print(f"Test set: {X_test.shape}")
print(f"Features: {X_train_clean.shape[1]}")

# ============================================================================
# 3. HELPER FUNCTION: GENERATE PLOTS FOR EACH MODEL
# ============================================================================

def calculate_accuracy_vs_overs(X_test, y_test, preds):
    """Calculate accuracy with a wider 30-over bin at the end to prevent drop-off"""
    results_df = X_test.copy()
    results_df['Actual'] = y_test.values
    results_df['Predicted'] = preds
    results_df['Correct'] = (results_df['Actual'] == results_df['Predicted']).astype(int)
    
    # NEW FIX: Group the final 30 overs (0-30), then use 15-over bins for the rest
    bins = [0, 30] + list(np.arange(45, 451, 15))
    
    results_df['Over_Bin'] = pd.cut(results_df['Overs_Remaining'], bins=bins)
    acc_plot = results_df.groupby('Over_Bin', observed=False)['Correct'].mean() * 100
    
    x_labels = [f"{int(b.right)}-{int(b.left)}" for b in acc_plot.index]
    return acc_plot, x_labels, results_df

def plot_confusion_matrix(y_test, preds, output_path, model_name):
    """Generate confusion matrix plot"""
    fig, ax = plt.subplots(figsize=(8, 6), dpi=100)
    cm = confusion_matrix(y_test, preds)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=['Loss', 'Draw', 'Win'])
    disp.plot(ax=ax, cmap='Blues')
    ax.set_title(f'Confusion Matrix - {model_name}', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"✓ Saved: {output_path}")

def plot_feature_importance(X_train, y_train, model, output_path, model_name, method='auto'):
    """Generate feature importance plot"""
    fig, ax = plt.subplots(figsize=(10, 8), dpi=100)
    
    if method == 'feature_importances':
        importances = model.feature_importances_
    elif method == 'coef':
        importances = np.abs(model.coef_[0])
    elif method == 'permutation':
        perm_imp = permutation_importance(model, X_train, y_train, n_repeats=10, random_state=42)
        importances = perm_imp.importances_mean
    
    feat_importance = pd.DataFrame({
        'Feature': X_train.columns,
        'Importance': importances
    }).sort_values('Importance')
    
    ax.barh(feat_importance['Feature'], feat_importance['Importance'], color='steelblue')
    ax.set_xlabel('Importance Score', fontweight='bold')
    ax.set_title(f'Feature Importance - {model_name}', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"✓ Saved: {output_path}")

def plot_shap_summary(X_train, model, output_path, model_name):
    """Generate SHAP summary plot for 'Win' class"""
    try:
        # Use a sample for faster computation
        sample_size = min(100, X_train.shape[0])
        X_sample = X_train.sample(n=sample_size, random_state=42)

        if hasattr(model, 'predict_proba'):
            explainer = shap.TreeExplainer(model) if hasattr(model, 'tree_') else shap.KernelExplainer(model.predict_proba, X_sample)
            shap_values = explainer.shap_values(X_sample)

            # For multi-class, take the 'Win' class (index 2)
            if isinstance(shap_values, list):
                shap_vals = shap_values[2]  # Win class
            else:
                shap_vals = shap_values[:, :, 2] if shap_values.ndim == 3 else shap_values

            fig, ax = plt.subplots(figsize=(12, 8), dpi=100)
            shap.summary_plot(shap_vals, X_sample, plot_type="bar", max_display=15, show=False)
            plt.title(f'SHAP Summary Plot (Win Class) - {model_name}', fontsize=14, fontweight='bold')
            plt.tight_layout()
            plt.savefig(output_path, dpi=300, bbox_inches='tight')
            plt.close()
            print(f"✓ Saved: {output_path}")
        else:
            print("⚠️  SHAP plot skipped (model doesn't support predict_proba)")
    except Exception as e:
        print(f"⚠️  SHAP plot generation failed: {str(e)}")

def plot_in_play_dynamics(X_test, model, output_path, model_name):
    """Plot match dynamics for a single match"""
    try:
        # Get Match_ID column if it exists
        if 'Match_ID' in X_test.columns:
            unique_matches = X_test['Match_ID'].unique()
            selected_match_id = np.random.choice(unique_matches)
            match_data = X_test[X_test['Match_ID'] == selected_match_id].sort_values('Overs_Remaining', ascending=False)
        else:
            # If no Match_ID, just use first match sequence by Overs_Remaining
            match_data = X_test.sort_values('Overs_Remaining', ascending=False).iloc[:50]
            selected_match_id = "Sample"
        
        if len(match_data) < 2:
            print(f"⚠️  In-play dynamics skipped (insufficient data for match)")
            return
        
        # Get predictions with probabilities
        if hasattr(model, 'predict_proba'):
            probs = model.predict_proba(match_data.drop(columns=['Match_ID'], errors='ignore'))
            overs = match_data['Overs_Remaining'].values
            
            fig, ax = plt.subplots(figsize=(12, 6), dpi=100)
            ax.plot(overs, probs[:, 0], marker='o', label='Loss', linewidth=2, color='red')
            ax.plot(overs, probs[:, 1], marker='s', label='Draw', linewidth=2, color='orange')
            ax.plot(overs, probs[:, 2], marker='^', label='Win', linewidth=2, color='green')
            
            ax.set_xlabel('Overs Remaining', fontsize=12, fontweight='bold')
            ax.set_ylabel('Probability', fontsize=12, fontweight='bold')
            ax.set_title(f'In-Play Match Dynamics (Match ID: {selected_match_id}) - {model_name}', fontsize=14, fontweight='bold')
            ax.legend(fontsize=11)
            ax.grid(True, linestyle='--', alpha=0.7)
            ax.set_xlim(450, 0)
            
            plt.tight_layout()
            plt.savefig(output_path, dpi=300, bbox_inches='tight')
            plt.close()
            print(f"✓ Saved: {output_path}")
        else:
            print(f"⚠️  In-play dynamics skipped (model doesn't support predict_proba)")
    except Exception as e:
        print(f"⚠️  In-play dynamics generation failed: {str(e)}")

def plot_accuracy_vs_overs(acc_plot, x_labels, output_path, model_name, color='blue'):
    """Generate accuracy vs overs remaining plot"""
    fig, ax = plt.subplots(figsize=(14, 7), dpi=100)
    
    ax.plot(x_labels, acc_plot.values, marker='o', linestyle='-', color=color, 
            linewidth=2, markersize=6, label=f'{model_name}')
    
    ax.axhline(y=61.11, color='red', linestyle='--', linewidth=2, label='Baseline (61.11%)')
    ax.set_ylim(0, 105)
    ax.grid(True, linestyle='--', alpha=0.7)
    ax.invert_xaxis()
    
    ax.set_title(f'Accuracy vs. Match Progression - {model_name}', fontsize=14, fontweight='bold')
    ax.set_xlabel('Overs Remaining in the Match', fontsize=12, fontweight='bold')
    ax.set_ylabel('Prediction Accuracy (%)', fontsize=12, fontweight='bold')
    ax.set_xticks(range(len(x_labels)))
    ax.set_xticklabels(x_labels, rotation=45)
    ax.legend(fontsize=11)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"✓ Saved: {output_path}")


# ============================================================================
# OPTUNA TUNING HELPERS
# ============================================================================
try:
    import xgboost as xgb
    _HAS_XGB = True
except Exception:
    _HAS_XGB = False


def tune_model(X, y, model_key, n_trials=20, random_state=42):
    """Tune hyperparameters for a given model_key using Optuna and return best params."""
    skf = StratifiedKFold(n_splits=3, shuffle=True, random_state=random_state)

    def objective_nn(trial):
        hidden1 = trial.suggest_categorical('h1', [64, 128, 256])
        hidden2 = trial.suggest_categorical('h2', [32, 64, 128])
        alpha = trial.suggest_loguniform('alpha', 1e-6, 1e-1)
        lr = trial.suggest_loguniform('lr', 1e-4, 1e-1)
        model = MLPClassifier(hidden_layer_sizes=(hidden1, hidden2), alpha=alpha,
                              learning_rate_init=lr, max_iter=300, random_state=random_state)
        return cross_val_score(model, X, y, cv=skf, scoring='accuracy', n_jobs=-1).mean()

    def objective_lr(trial):
        C = trial.suggest_loguniform('C', 1e-4, 1e2)
        model = LogisticRegression(C=C, max_iter=500, random_state=random_state)
        return cross_val_score(model, X, y, cv=skf, scoring='accuracy', n_jobs=-1).mean()

    def objective_rf(trial):
        n_estimators = trial.suggest_int('n_estimators', 50, 200)
        max_depth = trial.suggest_int('max_depth', 3, 20)
        min_samples_split = trial.suggest_int('min_samples_split', 2, 10)
        model = RandomForestClassifier(n_estimators=n_estimators, max_depth=max_depth,
                                       min_samples_split=min_samples_split, n_jobs=-1, random_state=random_state)
        return cross_val_score(model, X, y, cv=skf, scoring='accuracy', n_jobs=-1).mean()

    def objective_xgb(trial):
        params = {
            'n_estimators': trial.suggest_int('n_estimators', 50, 200),
            'max_depth': trial.suggest_int('max_depth', 3, 12),
            'learning_rate': trial.suggest_loguniform('learning_rate', 1e-3, 0.3),
            'subsample': trial.suggest_uniform('subsample', 0.5, 1.0),
            'colsample_bytree': trial.suggest_uniform('colsample_bytree', 0.5, 1.0),
            'random_state': random_state
        }
        model = xgb.XGBClassifier(**params)
        return cross_val_score(model, X, y, cv=skf, scoring='accuracy', n_jobs=-1).mean()

    if model_key == 'Neural_Network':
        study = optuna.create_study(direction='maximize')
        study.optimize(objective_nn, n_trials=n_trials)
        return study.best_params
    if model_key == 'Logistic_Regression':
        study = optuna.create_study(direction='maximize')
        study.optimize(objective_lr, n_trials=n_trials)
        return study.best_params
    if model_key == 'Random_Forest':
        study = optuna.create_study(direction='maximize')
        study.optimize(objective_rf, n_trials=n_trials)
        return study.best_params
    if model_key == 'XGBoost' and _HAS_XGB:
        study = optuna.create_study(direction='maximize')
        study.optimize(objective_xgb, n_trials=n_trials)
        return study.best_params
    return {}

# ============================================================================
# 4. APPLY STANDARDSCALER TO NEW MODELS
# ============================================================================
print("\n" + "=" * 70)
print("STEP 3: Scaling features for new models")
print("=" * 70)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train_clean)
X_test_scaled = scaler.transform(X_test)

X_train_scaled_df = pd.DataFrame(X_train_scaled, columns=X_train_clean.columns, index=X_train_clean.index)
X_test_scaled_df = pd.DataFrame(X_test_scaled, columns=X_test.columns, index=X_test.index)

print(f"✓ Features scaled using StandardScaler")

# ============================================================================
# 5. MODEL 1: XGBOOST (PRE-TRAINED - NO RETRAINING)
# ============================================================================
print("\n" + "=" * 70)
print("STEP 4: Loading XGBoost Model (Pre-trained)")
print("=" * 70)

try:
    if _HAS_XGB:
        # Load the pre-trained model instead of training a new one
        model_path = '/Users/vipulsharma/Documents/projects/cricket/code/training /cricket_model_optuna4.pkl'
        xgb_model = joblib.load(model_path)
        print(f"✓ XGBoost model loaded successfully from: {model_path}")
    else:
        print("⚠️  XGBoost library not found. Skipping XGBoost evaluation.")
        raise FileNotFoundError("XGBoost library not installed")

    xgb_preds = xgb_model.predict(X_test)
    xgb_accuracy = accuracy_score(y_test, xgb_preds)
    print(f"✓ XGBoost Test Accuracy: {xgb_accuracy*100:.2f}%")
    
    # Generate plots for XGBoost
    print("\nGenerating XGBoost plots...")
    output_dir = model_dirs['XGBoost']
    
    plot_confusion_matrix(y_test, xgb_preds, output_dir / 'confusion_matrix.png', 'XGBoost')
    
    feat_imp = pd.DataFrame({
        'Feature': X_train_clean.columns,
        'Importance': xgb_model.feature_importances_
    }).sort_values('Importance')
    fig, ax = plt.subplots(figsize=(10, 8), dpi=100)
    ax.barh(feat_imp['Feature'], feat_imp['Importance'], color='teal')
    ax.set_xlabel('Importance Score', fontweight='bold')
    ax.set_title('Feature Importance - XGBoost', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(output_dir / 'feature_importance.png', dpi=300, bbox_inches='tight')
    plt.close()
    print(f"✓ Saved: {output_dir / 'feature_importance.png'}")
    
    # SHAP
    try:
        sample_size = min(100, X_train_clean.shape[0])
        X_sample = X_train_clean.sample(n=sample_size, random_state=42)
        explainer = shap.TreeExplainer(xgb_model)
        shap_values = explainer.shap_values(X_sample)
        shap_vals = shap_values[:, :, 2] if shap_values.ndim == 3 else shap_values
        
        fig, ax = plt.subplots(figsize=(12, 8), dpi=100)
        shap.summary_plot(shap_vals, X_sample, plot_type="bar", max_display=15, show=False)
        plt.title('SHAP Summary Plot (Win Class) - XGBoost', fontsize=14, fontweight='bold')
        plt.tight_layout()
        plt.savefig(output_dir / 'shap_summary.png', dpi=300, bbox_inches='tight')
        plt.close()
        print(f"✓ Saved: {output_dir / 'shap_summary.png'}")
    except Exception as e:
        print(f"⚠️  SHAP plot skipped: {str(e)}")
    
    # In-play dynamics
    plot_in_play_dynamics(X_test, xgb_model, output_dir / 'in_play_dynamics.png', 'XGBoost')
    
    # Accuracy vs Overs
    xgb_acc_plot, xgb_x_labels, xgb_results_df = calculate_accuracy_vs_overs(X_test, y_test, xgb_preds)
    plot_accuracy_vs_overs(xgb_acc_plot, xgb_x_labels, output_dir / 'accuracy_vs_overs.png', 'XGBoost', color='green')
    
except FileNotFoundError:
    print("⚠️  XGBoost model file not found. Skipping XGBoost evaluation.")
    xgb_accuracy = None
    xgb_acc_plot = None

# ============================================================================
# 6. MODEL 2: NEURAL NETWORK
# ============================================================================
print("\n" + "=" * 70)
print("STEP 5: Training Neural Network (MLPClassifier)")
print("=" * 70)

nn_model = MLPClassifier(hidden_layer_sizes=(128, 64, 32), early_stopping=True, 
                         max_iter=1000, random_state=42, verbose=0, n_iter_no_change=20)
# Tune Neural Network with Optuna
print("Tuning Neural Network with Optuna (fast trials)...")
nn_best = tune_model(X_train_scaled_df, y_train_full, 'Neural_Network', n_trials=20)
print(f"✓ Best NN params: {nn_best}")
nn_model = MLPClassifier(hidden_layer_sizes=(nn_best.get('h1',128), nn_best.get('h2',64)),
                         alpha=nn_best.get('alpha',0.0001), learning_rate_init=nn_best.get('lr',0.001),
                         max_iter=1000, random_state=42)
nn_model.fit(X_train_scaled, y_train_full)
nn_preds = nn_model.predict(X_test_scaled)
nn_accuracy = accuracy_score(y_test, nn_preds)
print(f"✓ Neural Network Test Accuracy: {nn_accuracy*100:.2f}%")

output_dir = model_dirs['Neural_Network']
print("\nGenerating Neural Network plots...")
plot_confusion_matrix(y_test, nn_preds, output_dir / 'confusion_matrix.png', 'Neural Network')

perm_imp = permutation_importance(nn_model, X_train_scaled_df, y_train_full, n_repeats=10, random_state=42)
feat_imp = pd.DataFrame({
    'Feature': X_train_clean.columns,
    'Importance': perm_imp.importances_mean
}).sort_values('Importance')
fig, ax = plt.subplots(figsize=(10, 8), dpi=100)
ax.barh(feat_imp['Feature'], feat_imp['Importance'], color='steelblue')
ax.set_xlabel('Importance Score', fontweight='bold')
ax.set_title('Feature Importance (Permutation) - Neural Network', fontsize=14, fontweight='bold')
plt.tight_layout()
plt.savefig(output_dir / 'feature_importance.png', dpi=300, bbox_inches='tight')
plt.close()
print(f"✓ Saved: {output_dir / 'feature_importance.png'}")

plot_in_play_dynamics(X_test_scaled_df, nn_model, output_dir / 'in_play_dynamics.png', 'Neural Network')

nn_acc_plot, nn_x_labels, nn_results_df = calculate_accuracy_vs_overs(X_test, y_test, nn_preds)
plot_accuracy_vs_overs(nn_acc_plot, nn_x_labels, output_dir / 'accuracy_vs_overs.png', 'Neural Network', color='blue')

# ============================================================================
# 7. MODEL 3: LOGISTIC REGRESSION
# ============================================================================
print("\n" + "=" * 70)
print("STEP 6: Training Logistic Regression")
print("=" * 70)

lr_model = LogisticRegression(max_iter=1000, random_state=42)
print("Tuning Logistic Regression with Optuna...")
lr_best = tune_model(X_train_scaled_df, y_train_full, 'Logistic_Regression', n_trials=20)
print(f"✓ Best LR params: {lr_best}")
lr_model = LogisticRegression(C=lr_best.get('C',1.0), max_iter=1000, random_state=42)
lr_model.fit(X_train_scaled, y_train_full)
lr_preds = lr_model.predict(X_test_scaled)
lr_accuracy = accuracy_score(y_test, lr_preds)
print(f"✓ Logistic Regression Test Accuracy: {lr_accuracy*100:.2f}%")

output_dir = model_dirs['Logistic_Regression']
print("\nGenerating Logistic Regression plots...")
plot_confusion_matrix(y_test, lr_preds, output_dir / 'confusion_matrix.png', 'Logistic Regression')

feat_imp = pd.DataFrame({
    'Feature': X_train_clean.columns,
    'Importance': np.abs(lr_model.coef_[0])
}).sort_values('Importance')
fig, ax = plt.subplots(figsize=(10, 8), dpi=100)
ax.barh(feat_imp['Feature'], feat_imp['Importance'], color='orange')
ax.set_xlabel('Importance Score', fontweight='bold')
ax.set_title('Feature Importance (Coef) - Logistic Regression', fontsize=14, fontweight='bold')
plt.tight_layout()
plt.savefig(output_dir / 'feature_importance.png', dpi=300, bbox_inches='tight')
plt.close()
print(f"✓ Saved: {output_dir / 'feature_importance.png'}")

plot_in_play_dynamics(X_test_scaled_df, lr_model, output_dir / 'in_play_dynamics.png', 'Logistic Regression')

lr_acc_plot, lr_x_labels, lr_results_df = calculate_accuracy_vs_overs(X_test, y_test, lr_preds)
plot_accuracy_vs_overs(lr_acc_plot, lr_x_labels, output_dir / 'accuracy_vs_overs.png', 'Logistic Regression', color='orange')

# ============================================================================
# 8. MODEL 4: RANDOM FOREST
# ============================================================================
print("\n" + "=" * 70)
print("STEP 7: Training Random Forest")
print("=" * 70)

rf_model = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1, verbose=0)
print("Tuning Random Forest with Optuna...")
rf_best = tune_model(X_train_scaled_df, y_train_full, 'Random_Forest', n_trials=20)
print(f"✓ Best RF params: {rf_best}")
rf_model = RandomForestClassifier(n_estimators=rf_best.get('n_estimators',100),
                                  max_depth=rf_best.get('max_depth',None),
                                  min_samples_split=rf_best.get('min_samples_split',2),
                                  random_state=42, n_jobs=-1)
rf_model.fit(X_train_scaled, y_train_full)
rf_preds = rf_model.predict(X_test_scaled)
rf_accuracy = accuracy_score(y_test, rf_preds)
print(f"✓ Random Forest Test Accuracy: {rf_accuracy*100:.2f}%")

output_dir = model_dirs['Random_Forest']
print("\nGenerating Random Forest plots...")
plot_confusion_matrix(y_test, rf_preds, output_dir / 'confusion_matrix.png', 'Random Forest')

feat_imp = pd.DataFrame({
    'Feature': X_train_clean.columns,
    'Importance': rf_model.feature_importances_
}).sort_values('Importance')
fig, ax = plt.subplots(figsize=(10, 8), dpi=100)
ax.barh(feat_imp['Feature'], feat_imp['Importance'], color='purple')
ax.set_xlabel('Importance Score', fontweight='bold')
ax.set_title('Feature Importance - Random Forest', fontsize=14, fontweight='bold')
plt.tight_layout()
plt.savefig(output_dir / 'feature_importance.png', dpi=300, bbox_inches='tight')
plt.close()
print(f"✓ Saved: {output_dir / 'feature_importance.png'}")

plot_in_play_dynamics(X_test_scaled_df, rf_model, output_dir / 'in_play_dynamics.png', 'Random Forest')

rf_acc_plot, rf_x_labels, rf_results_df = calculate_accuracy_vs_overs(X_test, y_test, rf_preds)
plot_accuracy_vs_overs(rf_acc_plot, rf_x_labels, output_dir / 'accuracy_vs_overs.png', 'Random Forest', color='purple')

# ============================================================================
# 9. COMBINED ACCURACY PLOT
# ============================================================================
print("\n" + "=" * 70)
print("STEP 9: Creating Combined Accuracy Plot")
print("=" * 70)

fig, ax = plt.subplots(figsize=(16, 8), dpi=100)

# Plot all models
if xgb_acc_plot is not None:
    ax.plot(xgb_x_labels, xgb_acc_plot.values, marker='o', linestyle='-', color='green', 
            linewidth=2.5, markersize=7, label=f'XGBoost ({xgb_accuracy*100:.2f}%)', alpha=0.8)

ax.plot(nn_x_labels, nn_acc_plot.values, marker='s', linestyle='-', color='blue', 
        linewidth=2.5, markersize=7, label=f'Neural Network ({nn_accuracy*100:.2f}%)', alpha=0.8)

ax.plot(lr_x_labels, lr_acc_plot.values, marker='^', linestyle='-', color='orange', 
        linewidth=2.5, markersize=7, label=f'Logistic Regression ({lr_accuracy*100:.2f}%)', alpha=0.8)

ax.plot(rf_x_labels, rf_acc_plot.values, marker='D', linestyle='-', color='purple', 
        linewidth=2.5, markersize=7, label=f'Random Forest ({rf_accuracy*100:.2f}%)', alpha=0.8)

# Baseline
ax.axhline(y=61.11, color='red', linestyle='--', linewidth=2.5, label='Baseline (61.11%)', alpha=0.7)

ax.set_ylim(0, 105)
ax.grid(True, linestyle='--', alpha=0.7)
ax.invert_xaxis()

ax.set_title('Accuracy Comparison: All 4 Models vs. Match Progression', fontsize=16, fontweight='bold')
ax.set_xlabel('Overs Remaining in the Match', fontsize=13, fontweight='bold')
ax.set_ylabel('Prediction Accuracy (%)', fontsize=13, fontweight='bold')
ax.set_xticks(range(len(nn_x_labels)))
ax.set_xticklabels(nn_x_labels, rotation=45)
ax.legend(fontsize=11, loc='best') 

plt.tight_layout()
plt.savefig(base_results_dir / 'combined_accuracy_comparison.png', dpi=300, bbox_inches='tight')
plt.close()
print(f"✓ Saved: {base_results_dir / 'combined_accuracy_comparison.png'}")

# ============================================================================
# 10. SUMMARY REPORT
# ============================================================================
print("\n" + "=" * 70)
print("EVALUATION SUMMARY")
print("=" * 70)

summary = pd.DataFrame({
    'Model': ['XGBoost', 'Neural Network', 'Logistic Regression', 'Random Forest'],
    'Test Accuracy': [
        f"{xgb_accuracy*100:.2f}%" if xgb_accuracy else "N/A",
        f"{nn_accuracy*100:.2f}%",
        f"{lr_accuracy*100:.2f}%",
        f"{rf_accuracy*100:.2f}%"
    ]
})

print("\n" + summary.to_string(index=False))

print("\n" + "=" * 70)
print("OUTPUT DIRECTORIES")
print("=" * 70)
for model_name, model_dir in model_dirs.items():
    print(f"✓ {model_name}: {model_dir}/")

print(f"\n✓ Combined Plot: {base_results_dir / 'combined_accuracy_comparison.png'}")
print("\n" + "=" * 70)
print("✓ Evaluation complete!")
print("=" * 70)
