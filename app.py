import re

import streamlit as st
import torch
from supabase import create_client
from transformers import AutoTokenizer, AutoModelForCausalLM


# ==================================================
# 1. PAGE CONFIGURATION
# ==================================================

st.set_page_config(
    page_title="CBSE AI Assessment",
    page_icon="📚",
    layout="centered",
)

st.title("CBSE AI Assessment")
st.write(
    "Retrieve existing questions and generate new "
    "subject-specific questions using AI."
)


# ==================================================
# 2. SUPABASE CONNECTION
# ==================================================

@st.cache_resource
def get_supabase_client(url, key):
    return create_client(url, key)


try:
    supabase = get_supabase_client(
        st.secrets["SUPABASE_URL"],
        st.secrets["SUPABASE_KEY"],
    )
except Exception as e:
    st.error(f"Supabase connection failed: {e}")
    st.stop()


# ==================================================
# 3. SUBJECT LABELS
# ==================================================

subject_labels = {
    "maths": "Mathematics",
    "science": "Science",
    "social_science": "Social Science",
    "computer_application": "Computer Applications",
    "home_science": "Home Science",
}


# ==================================================
# 4. MATHEMATICAL FORMATTING
# ==================================================

def format_math_text(text):
    """Convert common LaTeX delimiters to Markdown math."""

    if not isinstance(text, str):
        return str(text or "")

    # Convert \( expression \) into $expression$
    text = re.sub(
        r"\\\((.*?)\\\)",
        lambda match: "$" + match.group(1) + "$",
        text,
        flags=re.DOTALL,
    )

    # Convert \[ expression \] into $$expression$$
    text = re.sub(
        r"\\\[(.*?)\\\]",
        lambda match: "$$" + match.group(1) + "$$",
        text,
        flags=re.DOTALL,
    )

    return text


def display_question_text(text):
    """Display text with mathematical formatting."""

    st.markdown(format_math_text(text))


def display_options(options_text):
    """Display options separated by the | character."""

    if not options_text:
        return

    options = str(options_text).split("|")

    for option in options:
        option = option.strip()

        # Remove common PDF page-number information.
        option = re.sub(
            r"\s+(?:\d+\s+)?Page\s+\d+\s+of\s+\d+.*$",
            "",
            option,
            flags=re.IGNORECASE,
        ).strip()

        if option:
            display_question_text(option)


# ==================================================
# 5. DETECT QUESTION TYPE
# ==================================================

def detect_question_type(question_text, requested_type):
    """Identify the generated question type."""

    valid_types = [
        "Assertion-Reason",
        "Case-Based",
        "MCQ",
        "Short/Long Answer",
    ]

    if requested_type in valid_types:
        return requested_type

    text = question_text.lower()

    match = re.search(
        r"question\s*type\s*:\s*([^\n]+)",
        question_text,
        flags=re.IGNORECASE,
    )

    if match:
        label = match.group(1).strip().lower()

        if "assertion" in label and "reason" in label:
            return "Assertion-Reason"

        if "case" in label and "based" in label:
            return "Case-Based"

        if "mcq" in label or "multiple choice" in label:
            return "MCQ"

        if "short" in label or "long" in label:
            return "Short/Long Answer"

    if "assertion-reason" in text or (
        "assertion" in text and "reason" in text
    ):
        return "Assertion-Reason"

    if "case-based" in text or "case based" in text:
        return "Case-Based"

    if "multiple choice" in text or "mcq" in text:
        return "MCQ"

    return "Short/Long Answer"


# ==================================================
# 6. CLASS AND QUESTION SELECTION
# ==================================================

st.subheader("Choose Your Questions")

selected_class = st.selectbox(
    "Select your class",
    ["class9", "class10"],
    format_func=lambda value: value.replace("class", "Class "),
)

if selected_class == "class9":
    subjects = [
        "maths",
        "science",
        "social_science",
    ]
else:
    subjects = [
        "computer_application",
        "home_science",
        "maths",
        "social_science",
    ]

selected_subject = st.selectbox(
    "Select your subject",
    subjects,
    format_func=lambda value: subject_labels.get(
        value,
        value.replace("_", " ").title(),
    ),
)

question_type = st.selectbox(
    "Select question type",
    [
        "All types",
        "Assertion-Reason",
        "Case-Based",
        "MCQ",
        "Short/Long Answer",
    ],
)

number_of_questions = st.selectbox(
    "How many existing questions do you want?",
    [5, 10, 15, 20],
)


# ==================================================
# 7. LOAD EXISTING QUESTIONS FROM SUPABASE
# ==================================================

st.subheader("Existing Question Bank")

if st.button(
    "Load Sample Questions",
    use_container_width=True,
):
    try:
        query = (
            supabase.table("question_bank")
            .select(
                "question_id, class, subject, "
                "question_type, question_text_clean, options"
            )
            .eq("class", selected_class)
            .eq("subject", selected_subject)
        )

        if question_type != "All types":
            query = query.eq("question_type", question_type)

        response = query.limit(number_of_questions).execute()
        questions = response.data or []

        if questions:
            st.success(f"Loaded {len(questions)} questions!")

            for i, question in enumerate(questions, start=1):
                st.markdown(f"### Question {i}")

                display_question_text(
                    question.get("question_text_clean", "")
                )

                display_options(
                    question.get("options")
                )

                st.caption(
                    f"Type: "
                    f"{question.get('question_type') or 'Unknown'} "
                    f"| ID: "
                    f"{question.get('question_id', 'Unknown')}"
                )

                st.divider()

        else:
            st.info(
                "No questions found. Try another class, "
                "subject, or question type."
            )

    except Exception as e:
        st.error(f"Could not load questions: {e}")


# ==================================================
# 8. LOAD QWEN LANGUAGE MODEL
# ==================================================

@st.cache_resource(show_spinner=False)
def load_llm():
    """Load and cache the Qwen model."""

    model_name = "Qwen/Qwen2.5-1.5B-Instruct"

    device = "cuda" if torch.cuda.is_available() else "cpu"

    dtype = (
        torch.float16
        if device == "cuda"
        else torch.float32
    )

    tokenizer = AutoTokenizer.from_pretrained(model_name)

    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=dtype,
        low_cpu_mem_usage=True,
    )

    model.to(device)
    model.eval()

    return tokenizer, model, device


# ==================================================
# 9. RUN THE LANGUAGE MODEL
# ==================================================

def run_model(
    tokenizer,
    model,
    device,
    prompt,
    max_new_tokens=350,
):
    """Generate text using Qwen."""

    messages = [
        {
            "role": "user",
            "content": prompt,
        }
    ]

    formatted_prompt = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
    )

    inputs = tokenizer(
        formatted_prompt,
        return_tensors="pt",
    ).to(device)

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            pad_token_id=tokenizer.eos_token_id,
        )

    input_length = inputs["input_ids"].shape[1]
    new_tokens = outputs[0][input_length:]

    return tokenizer.decode(
        new_tokens,
        skip_special_tokens=True,
    ).strip()


# ==================================================
# 10. GENERATE A QUESTION
# ==================================================

def generate_question(
    class_name,
    subject,
    selected_type,
):
    """Generate one question using the Qwen model."""

    tokenizer, model, device = load_llm()

    subject_display = subject_labels.get(
        subject,
        subject.replace("_", " ").title(),
    )

    class_display = {
        "class9": "Class 9",
        "class10": "Class 10",
    }.get(class_name, class_name)

    if selected_type == "All types":
        type_instruction = """
Choose one suitable question type:
MCQ, Assertion-Reason, Case-Based, or Short/Long Answer.

Clearly state the chosen question type.
"""
    else:
        type_instruction = (
            f"Generate a {selected_type} question only. "
            "Clearly state the question type."
        )

    generation_prompt = f"""
You are an educational question setter creating original
practice material for CBSE students.

Generate exactly ONE question.

Class: {class_display}
Subject: {subject_display}

{type_instruction}

SUBJECT RELEVANCE RULES:

1. The question must directly test {subject_display}.
2. Do not generate questions from unrelated subjects.
3. For Home Science, use appropriate topics such as food,
   nutrition, health, textiles, human development,
   family resources, or household management.
4. For Mathematics, use appropriate mathematical concepts.
5. For Science, use appropriate scientific concepts.
6. For Social Science, use relevant history, geography,
   civics, or economics topics.
7. For Computer Applications, use relevant computing topics.
8. Match the selected class and use age-appropriate language.
9. Do not claim this is an official CBSE question.

ANSWER RULES:

1. For MCQs, provide four options labelled A, B, C, D.
2. Work out the answer before choosing the options.
3. Check calculations, facts, and units carefully.
4. Ensure the correct answer appears among the options.
5. Make sure the explanation agrees with the answer.
6. Do not choose the closest incorrect option.
7. Use $...$ around inline mathematical formulas.
8. Use $$...$$ around display mathematical formulas.

For MCQs, use this format:

Question Type: MCQ

Question: [Question text]

(A) [Option A]
(B) [Option B]
(C) [Option C]
(D) [Option D]

Correct Option: [Letter and answer]

Explanation: [Brief explanation]

For Assertion-Reason questions, include an assertion,
a reason, answer choices, and the correct answer.

For Case-Based questions, include a relevant passage,
questions, and answers.

For Short/Long Answer questions, include the question
and a model answer.

Return ONLY the complete question and answer.
Do not include a checklist, review instructions, or commentary.
"""

    draft = run_model(
        tokenizer,
        model,
        device,
        generation_prompt,
        max_new_tokens=350,
    )

    if not draft:
        raise ValueError("The model returned an empty question.")

    # Return the generated question directly.
    # The second AI review call has been removed.
    return draft


# ==================================================
# 11. GENERATE A NEW QUESTION
# ==================================================

st.divider()
st.subheader("Generate a New Question with AI")

st.write(
    f"**Class:** {selected_class.replace('class', 'Class ')}"
)

st.write(
    f"**Subject:** "
    f"{subject_labels.get(selected_subject, selected_subject)}"
)

st.write(f"**Question type:** {question_type}")

if st.button(
    "Generate New Question",
    use_container_width=True,
):
    try:
        with st.spinner(
            "Loading the AI model and generating a question. "
            "This may take a while on a CPU..."
        ):
            generated_question = generate_question(
                selected_class,
                selected_subject,
                question_type,
            )

        if generated_question:
            st.session_state["generated_question"] = (
                generated_question
            )

            st.session_state["generated_class"] = selected_class
            st.session_state["generated_subject"] = selected_subject
            st.session_state["generated_type_selection"] = question_type

            st.session_state["generated_type"] = (
                detect_question_type(
                    generated_question,
                    question_type,
                )
            )

            st.session_state["question_saved"] = False

            st.success("Question generated successfully!")

        else:
            st.error("The model returned an empty response.")

    except Exception as e:
        st.error(f"Question generation failed: {e}")


# ==================================================
# 12. DISPLAY AND SAVE GENERATED QUESTION
# ==================================================

if st.session_state.get("generated_question"):

    generated_question = st.session_state[
        "generated_question"
    ]

    generated_class = st.session_state.get(
        "generated_class",
        selected_class,
    )

    generated_subject = st.session_state.get(
        "generated_subject",
        selected_subject,
    )

    generated_type_selection = st.session_state.get(
        "generated_type_selection",
        question_type,
    )

    generated_type = st.session_state.get("generated_type")

    if not generated_type:
        generated_type = detect_question_type(
            generated_question,
            generated_type_selection,
        )

        st.session_state["generated_type"] = generated_type

    st.divider()
    st.subheader("Generated Question")

    display_question_text(generated_question)

    st.caption(
        f"Class: {generated_class.replace('class', 'Class ')} | "
        f"Subject: "
        f"{subject_labels.get(generated_subject, generated_subject)} | "
        f"Type: {generated_type}"
    )

    st.warning(
        "Check the question and answer before saving. "
        "AI-generated answers may contain mistakes."
    )

    if not st.session_state.get("question_saved", False):

        if st.button("Save Generated Question to Supabase"):
            try:
                (
                    supabase.table("generated_questions")
                    .insert(
                        {
                            "class": generated_class,
                            "subject": generated_subject,
                            "question_type": generated_type,
                            "question_text": generated_question,
                        },
                        returning="minimal",
                    )
                    .execute()
                )

                st.session_state["question_saved"] = True

                st.success(
                    "Generated question saved successfully!"
                )

            except Exception as e:
                st.error(f"Could not save question: {e}")

    else:
        st.success(
            "This generated question has already been saved."
        )


# ==================================================
# 13. VIEW SAVED GENERATED QUESTIONS
# ==================================================

st.divider()
st.subheader("Saved Generated Questions")

if st.button(
    "View Saved Questions",
    use_container_width=True,
):
    try:
        response = (
            supabase.table("generated_questions")
            .select(
                "id, class, subject, question_type, "
                "question_text, created_at"
            )
            .order("created_at", desc=True)
            .limit(50)
            .execute()
        )

        saved_questions = response.data or []

        if saved_questions:
            st.success(
                f"Found {len(saved_questions)} saved questions."
            )

            for saved in saved_questions:
                st.markdown(
                    f"### Saved Question ID: "
                    f"{saved.get('id', 'Unknown')}"
                )

                saved_class = saved.get("class", "Unknown")
                class_label = saved_class.replace("class", "Class ")

                saved_subject = saved.get("subject", "Unknown")

                st.caption(
                    f"Class: {class_label} | "
                    f"Subject: "
                    f"{subject_labels.get(saved_subject, saved_subject)} | "
                    f"Type: {saved.get('question_type', 'Unknown')}"
                )

                display_question_text(
                    saved.get("question_text", "")
                )

                created_at = saved.get("created_at")

                if created_at:
                    st.caption(f"Created at: {created_at}")

                st.divider()

        else:
            st.info("No generated questions have been saved yet.")

    except Exception as e:
        st.error(f"Could not retrieve saved questions: {e}")