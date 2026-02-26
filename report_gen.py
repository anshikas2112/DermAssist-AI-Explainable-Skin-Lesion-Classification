from fpdf import FPDF
from PIL import Image
import numpy as np
import io, os, tempfile

def save_report_pdf(pid, labels, confs, overlay, out_path):
    pdf=FPDF(); pdf.add_page(); pdf.set_font("Arial",size=12)
    pdf.cell(0,8,"DermAssist AI — Diagnosis Report",ln=1)
    pdf.cell(0,6,f"Patient ID: {pid}",ln=1)
    pdf.cell(0,6,f"Prediction: {labels[0]} ({confs[0]*100:.2f}%)",ln=1)
    pdf.ln(4)
    pdf.multi_cell(0,6,"Grad-CAM explanation. Consult dermatologist for evaluation.")
    os.makedirs(os.path.dirname(out_path), exist_ok=True)

    fd,tmp=tempfile.mkstemp(suffix='.png'); os.close(fd)
    try:
        arr=overlay if overlay.dtype==np.uint8 else (np.clip(overlay,0,1)*255).astype(np.uint8)
        Image.fromarray(arr).save(tmp,'PNG')
        pdf.image(tmp, x=10, y=pdf.get_y()+5, w=120)
        pdf.output(out_path)
    finally:
        try: os.remove(tmp)
        except: pass
