import sys
import os
import yaml
import torch
import wandb
import streamlit as st
from transformers import AutoTokenizer, AutoModelForQuestionAnswering

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from data.dataset import SquadDataset
from metrics.metrics import compute_exact_match, compute_f1

def load_config(config_path: str):
    """Load configuration from YAML file"""
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)

@st.cache_resource
def load_model_and_tokenizer():
    """
    Load model and tokenizer from wandb (cached by Streamlit)
    """
    # Load configuration
    config_path = os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
        "config",
        "config.yaml"
    )
    config = load_config(config_path)
    
    # Setup device
    if torch.cuda.is_available():
        device = torch.device("cuda")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")
    
    # # Initialize wandb and download model
    # wandb.login(key=config['wandb']['API_TOKEN'])
    # run = wandb.init(
    #     project=config["wandb"]["project"],
    #     entity=config["wandb"]["entity"]
    # )
    
    # artifact = run.use_artifact('anadea/squad_v2_qa/model-k5u8rjhw:v15', type='model') # distill bert
    # #run.use_artifact('anadea/squad_v2_qa/model-vrh8tbn4:v1', type='model') # bert
    # #run.use_artifact('anadea/squad_v2_qa/model-k5u8rjhw:v15', type='model') # distill bert
    # model_dir = artifact.download()
    
    # # Load tokenizer and model
    # tokenizer = AutoTokenizer.from_pretrained(model_dir)
    # model = AutoModelForQuestionAnswering.from_pretrained(model_dir).to(device)
    
    # wandb.finish()

    repo_name = config["inference"]["model_name"]  
    tokenizer = AutoTokenizer.from_pretrained(repo_name)
    model = AutoModelForQuestionAnswering.from_pretrained(repo_name).to(device)
    
    return model, tokenizer, device

def get_answer(model, tokenizer, question: str, context: str, device: torch.device) -> str:
    """
    Get answer for a question given the context
    """
    # Tokenize input
    inputs = tokenizer(
        question,
        context,
        max_length=384,
        truncation="only_second",
        return_tensors="pt"
    )
    
    # Move inputs to device
    inputs = {k: v.to(device) for k, v in inputs.items()}
    print(inputs)
    

    # Get predictions
    with torch.no_grad():
        outputs = model(**inputs)
    
    # Get answer span
    answer_start = torch.argmax(outputs.start_logits)
    answer_end = torch.argmax(outputs.end_logits)

    
    # Convert to answer text
    answer_tokens = inputs["input_ids"][0][answer_start:answer_end + 1]
    answer = tokenizer.decode(answer_tokens, skip_special_tokens=True)
    
    return answer

def main():
    st.title("Question Answering System")
    
    # Load model and tokenizer
    with st.spinner("Loading model..."):
        model, tokenizer, device = load_model_and_tokenizer()
    
    # Input areas
    st.subheader("Context")
    context = st.text_area(
        "Enter the context paragraph:",
        height=200,
        help="Paste the text passage that contains the answer to your question"
    )
    
    st.subheader("Question")
    question = st.text_input(
        "Enter your question:",
        help="Ask a question about the context above"
    )
    
    # Add expected answer input and unanswerable toggle in columns
    col1, col2 = st.columns([3, 1])
    with col1:
        expected_answer = st.text_input(
            "Expected Answer (optional):",
            help="If you know the correct answer, enter it here to see how well the model performs"
        )
    with col2:
        is_unanswerable = st.checkbox(
            "Unanswerable Question",
            help="Check this if the question cannot be answered from the given context"
        )
    
    # Get answer when both inputs are provided
    if st.button("Get Answer") and context and question:
        with st.spinner("Finding answer..."):
            answer = get_answer(model, tokenizer, question, context, device)
            
            st.subheader("Answer")
            st.success(answer)
            
            # If expected answer is provided or question is marked as unanswerable
            if expected_answer or is_unanswerable:
                # Use empty string as expected answer if question is unanswerable
                true_answer = "" if is_unanswerable else expected_answer
                exact_match = compute_exact_match(answer, true_answer)
                f1_score = compute_f1(answer, true_answer)
                
                st.subheader("Evaluation Metrics")
                col1, col2 = st.columns(2)
                with col1:
                    st.metric("Exact Match", f"{100 * exact_match:.1f}%")
                with col2:
                    st.metric("F1 Score", f"{100 * f1_score:.1f}%")
            
            # Display confidence scores
            st.info(
                "Note: This is the model's best guess at the answer based on the "
                "provided context. The answer is extracted directly from the context."
            )
    
    # Example usage
    with st.expander("See example from validation set"):
        st.markdown("""
        **Example Context:**
        The Normans (Norman: Nourmands; French: Normands; Latin: Normanni) were the people who in the 10th and 11th centuries gave their name to Normandy, a region in France. They were descended from Norse ("Norman" comes from "Norseman") raiders and pirates from Denmark, Iceland and Norway who, under their leader Rollo, agreed to swear fealty to King Charles III of West Francia. Through generations of assimilation and mixing with the native Frankish and Roman-Gaulish populations, their descendants would gradually merge with the Carolingian-based cultures of West Francia. The distinct cultural and ethnic identity of the Normans emerged initially in the first half of the 10th century, and it continued to evolve over the succeeding centuries.
        
        **Example Question:**
        When were the Normans in Normandy?
        
        **Expected Answer:**
        10th and 11th centuries
        """)

if __name__ == "__main__":
    main() 