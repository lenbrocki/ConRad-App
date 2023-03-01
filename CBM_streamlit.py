import streamlit as st
import numpy as np
import pickle
from ConRad.conceptModelFinetune import conceptModelFinetune
import torch

device = torch.device("cpu")

# name of concepts and  range of values
# concepts = {
#     "subtlety":[1.0,5.0],
#     "calcification":[1.0,6.0],
#     "margin":[1.0,5.0],
#     "lobulation":[1.0,5.0],
#     "spiculation":[1.0,5.0],
#     "diameter":[2.0,38.0],
#     "texture":[1.0,5.0],
#     "sphericity":[1.0,5.0]
# }
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


def process_sample(sample):
    sample = np.array(sample)
    sample = sample - sample.min()
    sample = sample/sample.max()
    return sample[0]

# # load the concept regression model
concept_model = conceptModelFinetune().to(device)

concept_model.load_state_dict(torch.load("ConRad/weights/concept_finetune_4.pt"))
concept_model.eval()


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

concept_names = [i[0] for i in concepts]
weights = dict(zip(concept_names, clf.coef_[0]))

with col1:
    # st.title("Samples")
    # hack to insert vertical space
    for i in range(5):
        st.text("")
    choice = st.selectbox("Examples", np.arange(0,20))
    sample = samples[choice]
    target_label = target_labels[choice]
    concept_label = concept_labels[choice]
    st.image(process_sample(sample))

pred = concept_model(sample.to(device).unsqueeze(0)).cpu().detach().numpy()
pred_scaled = scaler.inverse_transform(pred)


# st.title("Biomarkers")
with col2:
    concepts_dict = dict(zip(concept_names,pred_scaled[0]))
    concepts_edit = []
    # print(concept_dict)
    for c in concepts[:4]:
        name = c[0]
        print(name)
        val_range = c[1]
        val = st.slider(name, value=concepts_dict[name].item(), min_value=val_range[0], max_value=val_range[1], step=0.5)
        # st.write(weights[name])
        concepts_edit.append(val)

with col3:
    # st.title("Biomarkers")
    # concepts_dict = dict(zip(concepts,pred_scaled[0]))
    # concepts_edit = []
    # print(concept_dict)
    for c in concepts[4:]:
        name = c[0]
        val_range = c[1]
        val = st.slider(name, value=concepts_dict[name].item(), min_value=val_range[0], max_value=val_range[1], step=0.5)
        # st.write(weights[name])
        concepts_edit.append(val)

print(dict(zip(concept_names, concepts_edit)))
with col4:
    # st.title("Malignancy")
    for i in range(7):
        st.text("")
    st.text("Classification:")
    st.write(mal_dict[clf.predict(scaler.transform([concepts_edit]))[0]])
    st.text("Label:")
    st.write(mal_dict[np.array(target_label).item()])