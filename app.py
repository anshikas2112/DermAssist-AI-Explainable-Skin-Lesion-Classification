import streamlit as st
from PIL import Image
from model_load import load_model, HAM_CLASSES
from gradcam import predict_with_gradcam
from report_gen import save_report_pdf

st.title("DermAssist AI — Skin Lesion Diagnosis")

model, device = load_model("best_model.pth")

uploaded = st.file_uploader("Upload Image", type=['jpg','jpeg','png'])

if uploaded:
    img = Image.open(uploaded).convert("RGB")
    st.image(img)
    labels, confs, overlay, cam = predict_with_gradcam(img, model, device, HAM_CLASSES)
    st.subheader("Prediction")
    st.write(f"{labels[0]} — {confs[0]*100:.2f}%")
    st.image(overlay, caption="Grad-CAM Overlay")

    if st.button("Generate PDF Report"):
        out="output/report.pdf"
        save_report_pdf("patient_001", labels, confs, overlay, out)
        st.success("Report Generated!")
        with open(out,"rb") as f:
            st.download_button("Download Report", f, file_name="report.pdf", mime="application/pdf")
