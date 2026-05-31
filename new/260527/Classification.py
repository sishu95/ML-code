import numpy as np
import pandas as pd
import shap
import matplotlib.pyplot as plt
from sklearn.model_selection import GridSearchCV, StratifiedKFold, train_test_split
from sklearn.ensemble import RandomForestClassifier, StackingClassifier
from xgboost import XGBClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler, label_binarize
from sklearn.svm import SVC
from imblearn.over_sampling import SMOTE
from sklearn.multiclass import OneVsRestClassifier
from imblearn.pipeline import Pipeline as ImbPipeline
from sklearn.metrics import roc_curve, auc


print("=== Step 1: Loading Data ===")
data = pd.read_csv("/data/users/PVK/data.csv")
X = data.iloc[:, 0:13].values
Y = data.iloc[:, 14].values  
X_new = np.delete(X, [4, 5], axis=1)
X_n = list(data)[0:13]
X_m = [name for i, name in enumerate(X_n) if i not in [4, 5]]

X_train, X_test, y_train, y_test = train_test_split(
    X_new, Y,
    test_size=0.2,
    random_state=42,
    stratify=Y
)
smote = SMOTE(random_state=42)
X_train_res, y_train_res = smote.fit_resample(X_train, y_train)

def optimize_model(pipeline, params, X, y):
    grid = GridSearchCV(
        estimator=pipeline,
        param_grid=params,
        scoring='roc_auc_ovr', 
        cv=StratifiedKFold(5, shuffle=True, random_state=100),
        n_jobs=-1,
        error_score='raise'
    )
    grid.fit(X, y)
    return grid.best_estimator_, grid.best_params_

svc_pipe = ImbPipeline([
    ('sampler', SMOTE(random_state=84)),
    ('scaler', StandardScaler()),
    ('model', SVC(probability=True, decision_function_shape='ovr', random_state=84))
])
svc_params = {
    'model__C': [0.1, 1, 10],
    'model__kernel': ['linear', 'rbf'],
    'model__gamma': ['scale', 0.1, 1]
}
best_svc, svc_best_params = optimize_model(svc_pipe, svc_params, X_train, y_train)
print("\n=== SVC best ===")
print(svc_best_params)

rf_pipe = ImbPipeline([
    ('sampler', SMOTE(random_state=42)),
    ('model', RandomForestClassifier( random_state=42))
])
rf_params = {
    'model__n_estimators': [200, 400],
    'model__max_depth': [3, 5, None],
    'model__max_features': ['sqrt', 0.8]
}
best_rf, rf_best_params = optimize_model(rf_pipe, rf_params, X_train, y_train)

xgb_pipe = ImbPipeline([
    ('sampler', SMOTE(random_state=42)),
    ('model', XGBClassifier(
        objective='multi:softprob',
        eval_metric='mlogloss', 
        random_state=42))
])
xgb_params = {
    'model__learning_rate': [0.05, 0.1],
    'model__max_depth': [3, 5, 7],        
    'model__subsample': [0.8, 1.0],
    'model__colsample_bytree': [0.6, 0.8, 1.0], 
    'model__gamma': [0, 0.1, 1]            
}
best_xgb, xgb_best_params = optimize_model(xgb_pipe, xgb_params, X_train, y_train)

lr_pipe = ImbPipeline([
    ('sampler', SMOTE(random_state=55)),
    ('scaler', StandardScaler()),
    ('model', OneVsRestClassifier(LogisticRegression(max_iter=10000, random_state=55))) 
])

lr_params = {
    'model__estimator__C': [0.1, 1, 10],       
    'model__estimator__solver': ['lbfgs', 'saga']
}
best_lr, lr_best_params = optimize_model(lr_pipe, lr_params, X_train, y_train)

custom_weights = {0: 1, 1: 1, 2: 5}
stacking_clf = StackingClassifier(
    estimators=[
        ('svc', best_svc),
        ('rf', best_rf),
        ('xgb', best_xgb), 
        ('lr', best_lr)
    ],

    final_estimator=LogisticRegression(class_weight=custom_weights, max_iter=1000, random_state=42),  
    cv=StratifiedKFold(5, shuffle=True, random_state=42), 
    n_jobs=-1,
    passthrough=False 
)

stacking_clf.fit(X_train, y_train)
y_score = stacking_clf.predict_proba(X_test)

y_train_bin = label_binarize(y_train, classes=np.unique(y_train))
y_test_bin = label_binarize(y_test, classes=np.unique(y_train))

n_classes = y_test_bin.shape[1]
print("X_test shape:", X_test.shape)
print("y_score shape:", y_score.shape)
print("y_test_bin shape:", y_test_bin.shape)

plt.figure(figsize=(10,9))  
target_class = 2 
class_names = np.unique(y_train_res)
class_idx = np.where(class_names == target_class)[0][0]
model_list = [
    ('SVC', best_svc, 'dodgerblue', '--'),
    ('RF', best_rf, 'limegreen', '-.'),
    ('XGB', best_xgb, 'crimson', ':'),
    ('LR', best_lr, 'darkcyan', '--'),
    ('Ensemble', stacking_clf, 'purple', '-')
]
plt.plot([0, 1], [0, 1], color='gray', linestyle='--', 
        label='Random (AUC=0.50)', alpha=0.8)

for name, model, color, ls in model_list:
    try:
        y_score = model.predict_proba(X_test)
        
        fpr, tpr, _ = roc_curve(y_test_bin[:, class_idx], y_score[:, class_idx])
        roc_auc = auc(fpr, tpr)
        
        plt.plot(fpr, tpr, 
                color=color,
                linestyle='-',
                linewidth=2.5,
                alpha=0.9,
                label=f'{name} (AUC={roc_auc:.2f})')
        
    except Exception as e:
        print(f"{name} false: {str(e)}")
        continue

plt.xlim([-0.05, 1.05])
plt.ylim([-0.05, 1.05])
plt.xticks(np.arange(0, 1.1, 0.2), fontsize=20)
plt.yticks(np.arange(0, 1.1, 0.2), fontsize=20)

ax = plt.gca()
for spine in ax.spines.values():
    spine.set_linewidth(2.5)  
plt.xlabel('False Positive Rate', fontsize=32, labelpad=10) 
plt.ylabel('True Positive Rate', fontsize=32, labelpad=10)  
plt.title(f'ROC Curve Comparison (Class {target_class})', fontsize=20, pad=15, weight='bold')  
legend = plt.legend(
    loc='lower right',
    fontsize=22,  
    title='Models',
    title_fontsize=22,
    bbox_to_anchor=(0.98,0.01),  
    borderpad=0.4,                
    handlelength=2.5,             
    handletextpad=0.5,            
    labelspacing=0.8,             
    frameon=False                
)

ax.tick_params(axis='both', which='major', 
              labelsize=25) 

plt.savefig('roc_comparison.png', 
           dpi=300, 
           bbox_inches='tight',
           facecolor='white')
plt.show()

background_data = shap.sample(X_train, 100) 
explainer = shap.KernelExplainer(stacking_clf.predict_proba, background_data)
shap_values_all = explainer.shap_values(X_new)

target_class_idx = 2 
if isinstance(shap_values_all, list):
    shap_values_target = shap_values_all[target_class_idx]
else:
    shap_values_target = shap_values_all

columns = [f'feature{i+1}' for i in range(X_new.shape[1])]
X_df = pd.DataFrame(X_new, columns=columns)

shap_df = pd.DataFrame(shap_values_target, columns=X_m)
shap_df.to_csv('shap_values_class2.csv', index=False)

shap.initjs() 
plt.figure(figsize=(10, 8))
plt.rcParams['font.sans-serif'] = "Arial" 
plt.rcParams.update({'font.size': 14}) 

shap.summary_plot(shap_values_target, X_new, plot_type="bar", 
                feature_names=X_m, show=False)
plt.tight_layout()
plt.savefig('shap1_bar.png', dpi=300)
plt.close()  

plt.rcParams.update({
    'font.size': 16,           
    'font.weight': 'bold',      
    'axes.labelweight': 'bold', 
    'axes.titleweight': 'bold'  
})

plt.figure(figsize=(10, 8))

shap.summary_plot(shap_values_target, X_new, plot_type="dot",
                  feature_names=X_m, show=False)

ax = plt.gca()
plt.xticks(fontsize=14, weight='bold')
plt.yticks(fontsize=14, weight='bold')

if ax.get_xlabel():
    ax.set_xlabel(ax.get_xlabel(), fontsize=18, weight='bold')

plt.tight_layout()
plt.savefig('shap2_dot.png', dpi=300)
plt.close()

plt.rcdefaults()

X = pd.DataFrame(X_new, columns=X_m)

plt.rcParams.update({
    'font.family': 'Arial',
    'font.size': 22,           
    'font.weight': 'bold',     
    'axes.labelweight': 'bold', 
    'axes.linewidth': 3     
})

for feature in X_m:
    plt.figure(figsize=(8, 6))  
    
    shap.dependence_plot(
        feature,         
        shap_values_target,    
        X,       
        interaction_index=None,  
        show=False,
        dot_size=40 
    )
    
    ax = plt.gca()
    
    if ax.get_xlabel():
        ax.set_xlabel(ax.get_xlabel(), fontsize=22, fontweight='bold', fontname='Arial')
    if ax.get_ylabel():
        ax.set_ylabel(ax.get_ylabel(), fontsize=22, fontweight='bold', fontname='Arial')
        
    for item in (ax.get_xticklabels() + ax.get_yticklabels()):
        item.set_fontname('Arial') 
        item.set_fontsize(22)
        item.set_fontweight('bold')
    
    ax.tick_params(width=3, length=6)
        
    for spine_name in ['top', 'right', 'bottom', 'left']:
        ax.spines[spine_name].set_visible(True)  
        ax.spines[spine_name].set_linewidth(3)
        
    plt.tight_layout()
    plt.savefig(f'dependence_{feature}.png', dpi=300)  
    plt.close()  

plt.rcdefaults() 
print("All dependence plots have been saved.")

mol_data = pd.read_csv("/data/users/PVK/test.csv")  
X_mol_raw = mol_data.iloc[:, 0:13].values  
X_mol = np.delete(X_mol_raw, [4, 5], axis=1) 

mol_proba = stacking_clf.predict_proba(X_mol)
mol_preds = stacking_clf.predict(X_mol)
mol_proba_class2 = mol_proba[:, target_class]

try:
    mol_names = mol_data['Name'].values
except KeyError:
    mol_names = mol_data.iloc[:, -1].values

mol_results = pd.DataFrame({
    "SampleID": mol_names,  
    "Predicted_Label": mol_preds,
    "Class2_Probability": mol_proba_class2
})
mol_results_sorted = mol_results.sort_values(by="Class2_Probability", ascending=False)
mol_results_sorted.to_csv("1mol_class2_probabilities_named.csv", index=False)

explainer_proba = shap.KernelExplainer(stacking_clf.predict_proba, background_data)
shap_values_mol = explainer_proba.shap_values(X_mol)

for i in range(len(X_mol)):
    pred_class = mol_preds[i]
    class2_prob = mol_proba_class2[i]

    if isinstance(shap_values_mol, list):
        current_shap_value = shap_values_mol[pred_class][i]
        current_base_value = explainer_proba.expected_value[pred_class]
    else:
        current_shap_value = shap_values_mol[i]
        current_base_value = explainer_proba.expected_value
    
    force_plot_html = shap.force_plot(
        base_value=current_base_value,
        shap_values=current_shap_value,
        features=X_mol[i],
        feature_names=X_m,
        matplotlib=False,
        show=False
    )
    shap.save_html(f"MOL_force_plot_sample_{i}_class2_{class2_prob:.2f}.html", force_plot_html)

    plt.figure()
    shap.force_plot(
        base_value=current_base_value,
        shap_values=current_shap_value,
        features=X_mol[i],
        feature_names=X_m,
        matplotlib=True,
        show=False 
    )
    plt.title(f"Sample {i} | Pred: Class {pred_class} | Class 2 Prob: {class2_prob:.2f}", fontsize=12, pad=20)
    plt.tight_layout()
    plt.savefig(f"MOL_force_plot_sample_{i}_class2_{class2_prob:.2f}.png", dpi=300, bbox_inches='tight')
    plt.close()