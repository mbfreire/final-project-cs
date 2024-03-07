"""
Credit Card Fraud Detection System

This module implements a comprehensive system for detecting credit card fraud using a trained XGBoost model. 
It features an interactive Streamlit interface for model evaluation and instance-specific analysis, leveraging 
SHAP and LIME for interpretability. The system preprocesses the dataset with a custom StandardScalerTransform, 
evaluates model performance with various metrics, and uses visualizations like confusion matrices and SHAP summary 
plots to provide insights into the model's decision-making process. Additionally, it incorporates an Elliptic Envelope 
method for outlier detection, enhancing the model's ability to identify fraudulent transactions.
"""

#import libraries
import streamlit as st
import pandas as pd
import numpy as np
import pickle
import shap
from lime import lime_tabular
from source import StandardScalerTransform
from sklearn.metrics import classification_report, accuracy_score
import matplotlib.pyplot as plt
from sklearn.metrics import roc_auc_score, precision_recall_curve, auc, confusion_matrix
import seaborn as sns

st.title('Credit Card Fraud Detection')

# Load the Trained XGBoost Model
# This function loads a pretrained XGBoost model from a pickle file for further use in fraud detection analysis.
# It uses streamlit's `st.cache_resource` to cache the model and speed up subsequent loads.

# Load the trained XGBoost model
@st.cache_resource
def load_model():
    model_path = "xgboost_best_model_final.pkl"
    with open(model_path, 'rb') as file:
        model = pickle.load(file)
    return model

model = load_model()

# Load and Preprocess the Dataset
# Loads the credit card transactions dataset, applies feature scaling using the custom StandardScalerTransform,
# and returns the original DataFrame along with the features, scaled features, and labels.
# This function facilitates easy access to preprocessed data for model predictions and analysis.

# Load and preprocess the dataset
@st.cache_resource
def load_data():
    df = pd.read_csv('dataset/creditcard.csv')
    X = df.drop('Class', axis=1)
    y = df['Class'].values
    scaler = StandardScalerTransform()
    scaler.fit(X)
    X_scaled = scaler.transform(X)
    return df, X, X_scaled, y

df, X, X_scaled, y = load_data()

# Sidebar for User Inputs in Streamlit Interface
# Collects user input for selecting specific transactions to analyze. This allows for interactive, instance-specific
# analysis of the model's fraud detection performance on the Streamlit web interface.

# Sidebar for user inputs
st.sidebar.header('Credit Card Fraud Detection')

# Add a brief description or instructions
st.sidebar.markdown('Please enter the index of the transaction you want to analyze:')

# Display a loading message while the data is loading
with st.spinner('Loading data...'):
    instance_index = st.sidebar.number_input('Index of Instance', min_value=0, max_value=len(df)-1, value=0, key='instance_index_2')

# Create a two-column layout
col1, col2 = st.columns(2)

# Display Model Evaluation Metrics
# Calculates and displays key performance metrics of the fraud detection model, such as accuracy, recall, precision,
# F1 score, ROC AUC score, and precision-recall AUC score. This section provides a comprehensive overview of the model's
# effectiveness in identifying fraudulent transactions.

# Display actual vs predicted labels in the first column
with col1:
    st.markdown(f"### Analysis for Instance {instance_index}")
    st.markdown("---")
    actual_label = y[instance_index]
    predicted_label = model.predict(X_scaled[[instance_index]])[0]
    st.markdown(f"**Actual Label:** {'Fraud' if actual_label else 'Not Fraud'}")
    st.markdown(f"**Predicted Label:** {'Fraud' if predicted_label else 'Not Fraud'}")
    st.markdown("---")

# Display model evaluation metrics in the second column
with col2:
    st.markdown("### Performance Metrics")
    st.markdown("---")
    accuracy = accuracy_score(y, model.predict(X_scaled))
    st.markdown(f"**Accuracy:** {accuracy:.4f}")
    report = classification_report(y, model.predict(X_scaled), output_dict=True)
    recall = report['1']['recall']
    st.markdown(f"**Recall:** {recall:.4f}")
    f1_score = report['1']['f1-score']
    st.markdown(f"**F1 Score:** {f1_score:.4f}")
    precision = report['1']['precision']
    st.markdown(f"**Precision:** {precision:.4f}")
    roc_auc = roc_auc_score(y, model.predict_proba(X_scaled)[:, 1])
    st.markdown(f"**ROC AUC Score:** {roc_auc:.4f}")
    precision, recall, _ = precision_recall_curve(y, model.predict_proba(X_scaled)[:, 1])
    pr_auc = auc(recall, precision)
    st.markdown(f"**Precision-Recall AUC Score:** {pr_auc:.4f}")
    st.markdown("---")

# Confusion Matrix Visualization
# Generates and displays a heatmap visualization of the confusion matrix for the fraud detection model's predictions.
# This visualization aids in understanding the model's true positive, false positive, true negative, and false negative rates.

# Display the confusion matrix as a heatmap
st.markdown("### Confusion Matrix")
fig, ax = plt.subplots()
cm = confusion_matrix(y, model.predict(X_scaled))
cm_norm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
cm_df_norm = pd.DataFrame(cm_norm, index=['Not Fraud', 'Fraud'], columns=['Not Fraud', 'Fraud'])
cm_df = pd.DataFrame(cm, index=['Not Fraud', 'Fraud'], columns=['Not Fraud', 'Fraud'])
sns.heatmap(cm_df_norm, annot=cm_df.values, fmt='d', cmap='viridis', cbar=True, ax=ax)
st.pyplot(fig)

# Sorting and filtering the dataset
st.subheader('Dataset')
st.dataframe(df)

# Explainer Creation for SHAP and LIME Interpretability
# Functions to create SHAP and LIME explainers and generate explanations for selected instances.
# These explainers provide insights into the model's predictions, highlighting the features most influential to
# the decision-making process, thereby enhancing transparency and interpretability.

# Corrected caching for the explainer creation
def create_explainer(X_scaled):
    explainer = lime_tabular.LimeTabularExplainer(
        training_data=X_scaled,
        feature_names=X.columns.tolist(),
        class_names=['Not Fraud', 'Fraud'],
        mode='classification'
    )
    return explainer

explainer = create_explainer(X_scaled)

@st.cache_data(hash_funcs={"_main_.XGBClassifier": id, "builtins.dict": id, "numpy.ndarray": id})
def create_shap_explainer(_model, X_scaled):
    explainer_shap = shap.Explainer(_model, X_scaled)
    shap_values = explainer_shap.shap_values(X_scaled)
    return explainer_shap, shap_values

# Generate a LIME explanation for the selected instance
@st.cache_data(hash_funcs={"_main_.XGBClassifier": id, "builtins.dict": id, "numpy.ndarray": id})
def create_lime_explanation(_explainer, instance_index, _model, X_scaled):
    exp = _explainer.explain_instance(X_scaled[instance_index], _model.predict_proba, num_features=5)
    return exp

# Create a placeholder for the loading message
loading_message = st.empty()

# Display a loading message
loading_message.text('Loading SHAP explainer...')

# Load the SHAP explainer
explainer_shap, shap_values = create_shap_explainer(model, X_scaled)

# Update the loading message
loading_message.text('Loading LIME explanation...')

# Load the LIME explanation
exp = create_lime_explanation(explainer, instance_index, model, X_scaled)

# Clear the loading message
loading_message.empty()

# Create a placeholder for the loading message
loading_message = st.empty()

# Display a loading message
loading_message.text('Generating SHAP summary plots...')

# SHAP summary plots
st.markdown("### SHAP Summary Plots")

fig, ax = plt.subplots()
shap.summary_plot(shap_values, X, plot_type='bar', show=False)
st.pyplot(fig)
fig, ax = plt.subplots()
shap.summary_plot(shap_values, X, show=False)
st.pyplot(fig)

# Interpretation of SHAP Explanation
st.markdown(f"### SHAP Analysis for Transaction {instance_index}")
shap_values_instance = shap_values[instance_index]

# Create a DataFrame for easier manipulation and interpretation
shap_df = pd.DataFrame(list(zip(X.columns, shap_values_instance)), columns=['Feature', 'SHAP Value'])
shap_df['Absolute SHAP Value'] = shap_df['SHAP Value'].apply(abs)
shap_df.sort_values('Absolute SHAP Value', ascending=False, inplace=True)

st.markdown("Key factors influencing the fraud assessment for this transaction, ordered by their impact:")

for index, row in shap_df.head(5).iterrows():
    impact_type = "increasing" if row['SHAP Value'] > 0 else "decreasing"
    st.markdown(f"* **{row['Feature']}** ({row['SHAP Value']:.5f}): A {impact_type} influence on the fraud probability. This suggests that values {'above' if impact_type == 'increasing' else 'below'} the mean for this feature are significant for the model's fraud detection capability.")

st.markdown("This SHAP analysis reveals the transaction characteristics most responsible for the model's decision, assisting in identifying fraud patterns and refining detection algorithms.")


# Generating LIME Explanation
loading_message.text('Generating in-depth LIME analysis for this transaction...')

# Load the LIME explanation
exp = create_lime_explanation(explainer, instance_index, model, X_scaled)

# Display the LIME explanation
st.markdown(f"### LIME Analysis for Transaction {instance_index}")
lime_explanations = pd.DataFrame(exp.as_list(), columns=['Feature Range', 'Impact'])
st.table(lime_explanations)
fig = exp.as_pyplot_figure()
st.pyplot(fig)

# Interpretation of LIME Explanation
st.markdown(f"### Insightful LIME Interpretation for Transaction {instance_index}")
for feature_range, weight in exp.as_list():
    direction = "positively" if weight > 0 else "negatively"
    action = "enhancing" if weight > 0 else "reducing"
    st.markdown(f"""
    - **{feature_range}**: This condition {action} the transaction's fraud risk, with an impact score of {weight:.5f}. It indicates that being within this range {direction} affects the model's classification, providing crucial insights for targeted scrutiny and preventive action against fraud.
    """)

# Clear the loading message
loading_message.empty()


# Generate Final Interpretation Report
# Synthesizes the analyses provided by SHAP and LIME into a comprehensive report, detailing the model's predictive
# behavior on a selected transaction. This report underscores the critical features influencing the fraud prediction
# and offers recommended actions based on the analysis, facilitating informed decision-making in fraud management.

# Generate Final Interpretation Report
st.markdown(f"## Final Interpretation Report for Transaction {instance_index}")

alignment = "correctly" if actual_label == predicted_label else "incorrectly"
fraud_status = 'fraudulent' if predicted_label else 'not fraudulent'
actual_status = 'fraudulent' if actual_label else 'not fraudulent'

# Begin the report with a summary of the model's prediction vs the actual label
report_intro = f"""
The model has {alignment} classified Transaction {instance_index} as {fraud_status}, 
with the actual label being {actual_status}. 
This decision was influenced by a complex interplay of various transactional features, 
as detailed in the analyses provided by SHAP and LIME explanations.
"""
st.markdown(report_intro)

# Highlight the key features and their impacts from SHAP
top_shap_features = ", ".join([f"**{shap_df.iloc[i]['Feature']}**" for i in range(2)])
shap_insights = f"""
### Critical SHAP Insights
SHAP analysis has identified the most influential factors for this prediction. 
Features such as {top_shap_features}, and others play pivotal roles, either elevating or diminishing the likelihood of fraud. 
This granular understanding aids in pinpointing specific transaction characteristics 
that flag potential fraudulence.
"""
st.markdown(shap_insights)

# Provide insights from LIME analysis
top_lime_features = ", ".join([f"**{pair[0]}**" for pair in exp.as_list()[:2]])
lime_insights = f"""
### Key LIME Findings
LIME further contextualizes the model's decision-making by spotlighting 
how certain ranges of feature values impact the prediction. 
Features like {top_lime_features}, among others, 
demonstrate the nuanced influence of transaction data on fraud detection capabilities.
"""
st.markdown(lime_insights)

# Conclude with implications for fraud detection strategy
fraud_detection_implications = """
This case illustrates the model's capability to discern complex patterns indicative of fraud. 
By integrating insights from both SHAP and LIME, fraud analysts are equipped to not only trust 
the model's predictions but also understand the underlying reasons, fostering a proactive 
approach to refining fraud detection strategies. Enhanced focus on the identified key features 
and their impacts can streamline the detection process, allowing for more accurate and 
timely identification of fraudulent transactions.
"""
st.markdown("### Implications for Fraud Detection")
st.markdown(fraud_detection_implications)

# Offer action based on the analysis
recommended_action = f"""
Given the analysis, Transaction {instance_index} is {'recommended for further investigation' if predicted_label else 'considered legitimate, requiring no further action'}. 
Analysts are advised to particularly note the identified critical features for ongoing 
and future assessments to enhance the accuracy and efficiency of fraud detection mechanisms.
"""
st.markdown("### Recommended Action")
st.markdown(recommended_action)
