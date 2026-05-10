import os
import torch
import gradio as gr
import numpy as np
from PIL import Image
from torchvision import transforms, models
from huggingface_hub import hf_hub_download
from groq import Groq

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def load_model(filename, num_classes):
    path  = hf_hub_download(repo_id="Lucifer37273/mediscan-ai", filename=filename)
    model = models.efficientnet_b0(weights=None)
    model.classifier[1] = torch.nn.Linear(model.classifier[1].in_features, num_classes)
    model.load_state_dict(torch.load(path, map_location=device))
    model.to(device).eval()
    return model

print("Loading models...")
chest_model = load_model("chest_best.pth", 2)
eye_model   = load_model("eye_best.pth",   4)
skin_model  = load_model("skin_best.pth",  7)
print("All models loaded!")

client = Groq(api_key=os.environ.get("GROQ_API_KEY"))

SPECIALIST_MAP = {
    "Melanoma":             "oncologist dermatologist",
    "Basal Cell Carcinoma": "dermatologist",
    "Actinic Keratoses":    "dermatologist",
    "Benign Keratosis":     "dermatologist",
    "Dermatofibroma":       "dermatologist",
    "Melanocytic Nevi":     "dermatologist",
    "Vascular Lesions":     "dermatologist vascular surgeon",
    "Pneumonia":            "pulmonologist",
    "Normal":               "general physician",
    "Cataract":             "ophthalmologist",
    "Diabetic Retinopathy": "ophthalmologist retina specialist",
    "Glaucoma":             "ophthalmologist",
}

def find_nearby_doctors(disease, city="Coimbatore"):
    specialist = SPECIALIST_MAP.get(disease, "general physician")
    query      = f"{specialist} near {city}".replace(" ", "+")
    maps_url   = f"https://www.google.com/maps/search/{query}"
    return {"specialist": specialist, "maps_url": maps_url}

SYSTEM_PROMPT = """You are MediScan AI, a compassionate medical assistant.
A CNN model has analyzed the patient medical image and produced a diagnosis.
Your job is to:
1. Explain the diagnosis in simple, non-scary language (2-3 sentences)
2. Ask 2-3 targeted follow-up symptom questions (one at a time)
3. After the patient answers, give practical precautions and home care tips
4. Recommend the right specialist to consult

Rules:
- Never diagnose yourself, always say the CNN detected this
- Always remind the user you are an AI and they should see a real doctor
- Be warm, clear, and concise
- Do not dump everything at once, have a natural conversation

Current diagnosis from CNN: {diagnosis}
Nearby specialist search link: {maps_url}
When giving your final recommendation include the maps link naturally.
"""

class MediScanAgent:
    def __init__(self, diagnosis, city="Coimbatore"):
        disease      = diagnosis.split(" detected")[0]
        doctors      = find_nearby_doctors(disease, city)
        self.history = []
        self.system  = SYSTEM_PROMPT.format(
            diagnosis = diagnosis,
            maps_url  = doctors["maps_url"],
        )

    def chat(self, user_message):
        self.history.append({"role": "user", "content": user_message})
        response = client.chat.completions.create(
            model       = "llama-3.3-70b-versatile",
            messages    = [{"role": "system", "content": self.system}] + self.history,
            max_tokens  = 512,
            temperature = 0.7,
        )
        reply = response.choices[0].message.content
        self.history.append({"role": "assistant", "content": reply})
        return reply

    def start(self):
        return self.chat("Hello, I just got my scan results. Can you explain what was found?")

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225])
])

def run_cnn(image, scan_type):
    if isinstance(image, np.ndarray):
        img = Image.fromarray(image).convert("RGB")
    else:
        img = image.convert("RGB")
    tensor    = transform(img).unsqueeze(0).to(device)
    model_map = {
        "Chest X-Ray": (chest_model, ["Normal","Pneumonia"]),
        "Eye Scan":    (eye_model,   ["Cataract","Diabetic Retinopathy","Glaucoma","Normal"]),
        "Skin Image":  (skin_model,  ["Actinic Keratoses","Basal Cell Carcinoma","Benign Keratosis",
                                      "Dermatofibroma","Melanocytic Nevi","Melanoma","Vascular Lesions"]),
    }
    model, classes = model_map[scan_type]
    with torch.no_grad():
        probs     = torch.softmax(model(tensor), dim=1)
        conf, idx = probs.max(dim=1)
    return classes[idx.item()], round(conf.item() * 100, 2)

current_agent = None

def analyze(image, scan_type, city):
    global current_agent
    if image is None:
        return "⚠️ Please upload a medical image first.", [], ""
    city          = city.strip() or "Coimbatore"
    disease, conf = run_cnn(image, scan_type)
    diagnosis     = f"{disease} detected with {conf}% confidence ({scan_type})"
    current_agent = MediScanAgent(diagnosis, city)
    opening       = current_agent.start()
    doctors       = find_nearby_doctors(disease, city)
    maps_link     = f"📍 [Find a {doctors['specialist']} near {city}]({doctors['maps_url']})"
    return diagnosis, [{"role": "assistant", "content": opening}], maps_link
    
def chat(message, history):
    global current_agent
    if current_agent is None:
        history.append({"role": "assistant", "content": "⚠️ Please upload and analyze an image first."})
        return "", history
    reply = current_agent.chat(message)
    history.append({"role": "user", "content": message})
    history.append({"role": "assistant", "content": reply})
    return "", history

with gr.Blocks(title="MediScan AI") as demo:
    gr.Markdown("# 🏥 MediScan AI\n### Medical Image Diagnosis + AI Consultation")
    with gr.Row():
        with gr.Column(scale=1):
            gr.Markdown("### 📷 Upload Medical Image")
            image_input = gr.Image(label="Upload Image", type="numpy")
            scan_type   = gr.Radio(
                choices=["Chest X-Ray","Eye Scan","Skin Image"],
                label="Select Scan Type", value="Chest X-Ray"
            )
            city_input  = gr.Textbox(label="Your City", value="Coimbatore")
            analyze_btn = gr.Button("🔬 Analyze Image", variant="primary")
            result_box  = gr.Textbox(label="CNN Diagnosis", interactive=False)
            doctors_box = gr.Markdown("")
        with gr.Column(scale=1):
            gr.Markdown("### 💬 AI Medical Consultation")
            chatbot = gr.Chatbot(label="MediScan AI Agent", height=400)
            msg_input = gr.Textbox(label="Your message", placeholder="Describe your symptoms...")
            send_btn  = gr.Button("Send", variant="primary")
    analyze_btn.click(
        fn=analyze,
        inputs=[image_input, scan_type, city_input],
        outputs=[result_box, chatbot, doctors_box]
    )
    send_btn.click(
        fn=chat,
        inputs=[msg_input, chatbot],
        outputs=[msg_input, chatbot]
    )
    gr.Markdown("---\n⚠️ *MediScan AI is not a substitute for professional medical advice.*")

demo.launch()