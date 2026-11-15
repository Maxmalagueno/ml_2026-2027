import pandas as pd
import numpy as np
import os
from sklearn.model_selection import KFold
from sklearn.metrics import mean_squared_error
from sklearn.feature_extraction.text import CountVectorizer
import lightgbm as lgb

base_dir = os.path.dirname(os.path.abspath(__file__))
train = pd.read_csv(os.path.join(base_dir, 'train_data.csv'))
test = pd.read_csv(os.path.join(base_dir, 'test_data.csv'))
sub_path = os.path.join(base_dir, 'sample_submition.csv')

num_cols = [
    'Year Published', 'Min Players', 'Max Players', 'Play Time',
    'Min Age', 'Users Rated', 'BGG Rank', 'Complexity Average', 'Owned Users'
]

for df in [train, test]:
    for col in num_cols + ['Rating Average']:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col].astype(str).str.replace(',', '.'), errors='coerce')

target_col = 'Rating Average'
train = train.dropna(subset=[target_col]).reset_index(drop=True)

num_cols = [c for c in num_cols if c in train.columns and c in test.columns]
full_df = pd.concat([train[num_cols], test[num_cols]], axis=0, ignore_index=True)

if 'Year Published' in full_df.columns:
    full_df['Year Published'] = full_df['Year Published'].replace(0, np.nan)
    full_df['Game_Age'] = 2021 - full_df['Year Published']

if 'Users Rated' in full_df.columns and 'Owned Users' in full_df.columns:
    full_df['Rated_to_Owned_Ratio'] = full_df['Users Rated'] / (full_df['Owned Users'] + 1)

if 'Min Players' in full_df.columns and 'Max Players' in full_df.columns:
    full_df['Players_Range'] = full_df['Max Players'] - full_df['Min Players']

text_features = []
for col in ['Mechanics', 'Domains']:
    if col in train.columns and col in test.columns:
        combined_text = pd.concat([train[col], test[col]], axis=0, ignore_index=True).fillna('').astype(str)
        vec = CountVectorizer(tokenizer=lambda x: [i.strip() for i in x.split(',') if i.strip()], lowercase=False, token_pattern=None)
        mat = vec.fit_transform(combined_text)
        col_names = [f'{col}_{i}' for i in range(mat.shape[1])]
        text_df = pd.DataFrame(mat.toarray(), columns=col_names)
        text_features.append(text_df)
        full_df[f'{col}_Count'] = combined_text.apply(lambda x: len([i for i in x.split(',') if i.strip()]))

if text_features:
    full_df = pd.concat([full_df] + text_features, axis=1)

X = full_df.iloc[:len(train)].copy()
y = train[target_col].values
X_test = full_df.iloc[len(train):].copy()

kf = KFold(n_splits=5, shuffle=True, random_state=42)
oof_preds = np.zeros(len(train))
test_preds = np.zeros(len(test))

params = {
    'objective': 'regression',
    'metric': 'rmse',
    'learning_rate': 0.03,
    'num_leaves': 31,
    'subsample_for_bin': 50000,
    'colsample_bytree': 0.8,
    'subsample': 0.8,
    'random_state': 42,
    'n_estimators': 3000,
    'verbose': -1
}

for fold, (train_idx, val_idx) in enumerate(kf.split(X, y)):
    X_train, y_train = X.iloc[train_idx], y[train_idx]
    X_val, y_val = X.iloc[val_idx], y[val_idx]
    
    model = lgb.LGBMRegressor(**params)
    model.fit(
        X_train, y_train,
        eval_set=[(X_val, y_val)],
        callbacks=[lgb.early_stopping(stopping_rounds=50, verbose=False)]
    )
    
    val_pred = model.predict(X_val)
    oof_preds[val_idx] = val_pred
    print(f'Fold {fold + 1} MSE: {mean_squared_error(y_val, val_pred):.5f}')
    
    test_preds += model.predict(X_test) / kf.n_splits

print(f'OOF MSE: {mean_squared_error(y, oof_preds):.5f}')

predictions = np.round(test_preds, 2)
print('Массив ответов:')
print(predictions)

if os.path.exists(sub_path):
    submission = pd.read_csv(sub_path)
    id_col_name = submission.columns[0]
    target_col_name = submission.columns[1]
    if submission[id_col_name].duplicated().any():
        submission[id_col_name] = np.arange(len(submission))
    submission[target_col_name] = predictions[:len(submission)]
else:
    submission = pd.DataFrame({
        'ID': np.arange(len(predictions)),
        'Rating Average': predictions
    })

out_file = os.path.join(base_dir, 'submission.csv')
submission.to_csv(out_file, index=False)
print('Уникальных ID:', submission.iloc[:, 0].nunique(), 'из', len(submission))