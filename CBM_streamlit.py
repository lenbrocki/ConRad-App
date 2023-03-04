import streamlit as st
import numpy as np
import pickle
from ConRad.conceptModelFinetune import conceptModelFinetune
import torch
import pytorch_lightning as pl

device = torch.device("cpu")

# name of concepts and  range of values
concepts = [
    ["subtlety",[1.0,5.0]],
    ["calcification",[1.0,6.0]],
    ["margin",[1.0,5.0]],
    ["lobulation",[1.0,5.0]],
    ["spiculation",[1.0,5.0]],
    ["diameter",[2.0,38.0]],
    ["texture",[1.0,5.0]],
    ["sphericity",[1.0,5.0]]
]
st.title('ConRad - Interpretable Lung Nodule Classification')
col1, col2, col3, col4 = st.columns(4, gap="large")

mal_dict= {1: "malignant", 0: "benign"}


# # load the concept regression model
@st.cache_resource
def load_model():
    concept_model = conceptModelFinetune().to(device)
    concept_model.load_state_dict(torch.load("ConRad/weights/concept_finetune_4.pt"))
    concept_model.eval()
    return concept_model
concept_model = load_model()

@st.cache_data(max_entries=100)
def load_data():
    with open("example_data/samples_fold_4.pkl", "rb") as f:
        samples = pickle.load(f)
    with open("example_data/scaler_fold_4.pkl", "rb") as f:
        scaler = pickle.load(f)
    with open("example_data/concept_labels_fold_4.pkl", "rb") as f:
        concept_labels = pickle.load(f)
    with open("example_data/target_labels_fold_4.pkl", "rb") as f:
        target_labels = pickle.load(f)
    with open("example_data/svm_linear_fold_4.pkl", "rb") as f:
        clf = pickle.load(f)
    return samples, scaler, concept_labels, target_labels, clf
samples, scaler, concept_labels, target_labels, clf = load_data()

@st.cache_data
def process_samples(_samples):
    samples_p = []
    for s in _samples:
        sample = np.array(s)
        sample = sample - sample.min()
        sample = sample/sample.max()
        samples_p.append(sample[0])
    return samples_p

# for plotting we use the processed samples
samples_p = process_samples(samples)

concept_names = [i[0] for i in concepts]
# weights = dict(zip(concept_names, clf.coef_[0]))

with col1:
    # st.title("Samples")
    # hack to insert vertical space
    for i in range(5):
        st.text("")
    choice = st.selectbox("Examples", np.arange(0,20))
    sample_p = samples_p[choice]
    sample = samples[choice]
    target_label = target_labels[choice]
    concept_label = concept_labels[choice]
    st.image(sample_p)


pred = concept_model(sample.to(device).unsqueeze(0)).cpu().detach().numpy()
pred_scaled = scaler.inverse_transform(pred)



# st.title("Biomarkers")
with col2:
    concepts_dict = dict(zip(concept_names,pred_scaled[0]))
    concepts_edit = []
    # print(concept_dict)
    for c in concepts[:4]:
        name = c[0]
        val_range = c[1]
        val = st.slider(name, value=concepts_dict[name].item(), min_value=val_range[0], max_value=val_range[1], step=0.5)
        # st.write(weights[name])
        concepts_edit.append(val)

with col3:
    for c in concepts[4:]:
        name = c[0]
        val_range = c[1]
        val = st.slider(name, value=concepts_dict[name].item(), min_value=val_range[0], max_value=val_range[1], step=0.5)
        # st.write(weights[name])
        concepts_edit.append(val)

with col4:
    # st.title("Malignancy")
    for i in range(7):
        st.text("")
    st.text("Classification:")
    st.write(mal_dict[clf.predict(scaler.transform([concepts_edit]))[0]])
    st.text("Label:")
    st.write(mal_dict[np.array(target_label).item()])

with st.expander("See details"):
    st.write("A CNN regression model predicts biomarkers which are passed to a linear SVM for malignancy classification.")
    st.write("Implementation details can be found in the accompanying GitHub repository https://github.com/lenbrocki/ConRad.")
    st.write("The meaning of the biomarker values is explaind here: https://pylidc.github.io/annotation.html")