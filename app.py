import streamlit as st
import torch # type: ignore
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

# Page configuration
st.set_page_config(
    page_title="Supermarket Assistant",
    page_icon="🛍️",  # Changed icon to a shopping bag
    layout="wide",
    initial_sidebar_state="expanded",
)

# Custom styling
st.markdown(
    """
    <style>
    body {
        background-color: #2c3e50; /* Interactive dark blue-gray background */
        color: #ecf0f1; /* Light text for contrast */
    }
    .main-content {
        background-color: #34495e; /* Slightly lighter blue-gray for content background */
        padding: 20px;
        border-radius: 12px;
        box-shadow: 0 4px 15px rgba(0, 0, 0, 0.5); /* Enhanced shadow for depth */
    }
    .title {
        text-align: center;
        font-size: 2.5em;
        color: #1abc9c; /* Vibrant teal for the title */
        margin-bottom: 20px;
    }

    .message {
        border-radius: 12px;
        padding: 12px;
        margin: 5px 0;
        font-size: 1em;
    }
    .user-message {
        background-color: #3498db; /* Interactive blue for user messages */
        color: #ffffff; /* White text for readability */
        text-align: left;
        margin-right: auto;
    }
    .ai-message {
        background-color: #e67e22; /* Warm orange for AI messages */
        color: #ffffff; /* White text for readability */
        text-align: left;
        margin-left: auto;
    }
    .button-container {
        margin-top: 10px;
        text-align: center;
    }
    .history-title {
        text-align: center;
        font-size: 1.5em;
        color: #e74c3c; /* Bright red for history title */
        margin-bottom: 10px;
    }
    .greet-box {
        background-color: #1abc9c; /* Teal background for greeting */
        color: #ffffff; /* White text for contrast */
        padding: 10px;
        border-radius: 8px;
        margin-bottom: 15px;
        text-align: center;
        font-weight: bold;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# Initialize chat history if not already done
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

if "greeted" not in st.session_state:
    st.session_state.greeted = False

# Layout: Left for History, Right for Chat Interface
col1, col2 = st.columns([1, 3])

# History Section (Left)
with col1:
    st.markdown('<div class="history-box">', unsafe_allow_html=True)
    st.markdown('<h3 class="history-title">Chat History</h3>', unsafe_allow_html=True)

    # Display the full chat history in the box
    if st.session_state.chat_history:
        for message in reversed(st.session_state.chat_history):
            if "User:" in message:
                st.markdown(f'<div class="message user-message">{message.replace("User: ", "")}</div>', unsafe_allow_html=True)
            elif "AI:" in message:
                st.markdown(f'<div class="message ai-message">{message.replace("AI: ", "")}</div>', unsafe_allow_html=True)
    else:
        st.markdown("<p style='color: gray; text-align: center;'>No chats yet</p>", unsafe_allow_html=True)
    st.markdown('</div>', unsafe_allow_html=True)

# Chat Section (Right)
with col2:
    st.markdown('<div class="main-content">', unsafe_allow_html=True)
    st.markdown('<h1 class="title">🛍️ Supermarket Assistant</h1>', unsafe_allow_html=True)

    # Input Fields
    name = st.text_input("Your name:", placeholder="Type your name here")

    # Greet the user
    if name and not st.session_state.greeted:
        greeting_message = f"Hello, {name}! How can I assist you today?"
        st.markdown(f'<div class="greet-box">{greeting_message}</div>', unsafe_allow_html=True)
        st.session_state.chat_history.insert(0, f"AI: {greeting_message}")
        st.session_state.greeted = True

    category = st.text_input("Product category:", placeholder="E.g., Beverages, Snacks")
    product = st.text_input("Product name:", placeholder="E.g., Coke, Chips")

    # Chat submission
    if name and category and product:
        st.session_state.chat_history.insert(0, f"User: My name is {name}. I'm looking for {product} in the {category} category.")

        # Model selection dropdown
        model_option = st.selectbox(
            "Choose the prediction model:",
            [
                "Fine-Tuned BART (Predict Units)",
                "Fine-Tuned FLAN-T5 (Predict Price)",
                "Fine-Tuned T5 (Predict Location)",
            ],
        )

        # Prediction button
        if st.button("Get Prediction"):
            try:
                # Load selected model
                if model_option == "Fine-Tuned BART (Predict Units)":
                    model_path = "saved_model_bartbase(unit left)"
                elif model_option == "Fine-Tuned FLAN-T5 (Predict Price)":
                    model_path = "saved_model_flant5(cost)"
                elif model_option == "Fine-Tuned T5 (Predict Location)":
                    model_path = "./custom_t5_supermarket_model"

                tokenizer = AutoTokenizer.from_pretrained(model_path)
                model = AutoModelForSeq2SeqLM.from_pretrained(model_path)

                # Device configuration
                device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
                model = model.to(device)

                # Prepare input text
                input_text = f"Predict location for Product: {product} | Category: {category}" if model_option == "Fine-Tuned T5 (Predict Location)" else f"Customer: Hello, my name is {name}. I am looking for {product} in the {category} category."
                inputs = tokenizer(input_text, return_tensors="pt", padding=True, truncation=True, max_length=128)
                inputs = {key: value.to(device) for key, value in inputs.items()}

                # Generate response
                with torch.no_grad():
                    outputs = model.generate(**inputs, max_length=128, num_beams=5, early_stopping=True)
                response = tokenizer.decode(outputs[0], skip_special_tokens=True)

                # Append AI response to chat history
                st.session_state.chat_history.insert(0, f"AI: {response}")

            except Exception as e:
                st.error(f"An error occurred: {str(e)}")

    # Display Chat Messages
    st.markdown('<div class="chat-container">', unsafe_allow_html=True)
    if st.session_state.chat_history:
        for message in st.session_state.chat_history:
            if "User:" in message:
                st.markdown(
                    f'<div class="message user-message">{message.replace("User: ", "")}</div>',
                    unsafe_allow_html=True,
                )
            elif "AI:" in message:
                st.markdown(
                    f'<div class="message ai-message">{message.replace("AI: ", "")}</div>',
                    unsafe_allow_html=True,
                )
    else:
        st.markdown("<p style='color: gray; text-align: center;'>No chat messages yet</p>", unsafe_allow_html=True)
    st.markdown('</div>', unsafe_allow_html=True)

    st.markdown('</div>', unsafe_allow_html=True)
