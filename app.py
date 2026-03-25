import time
from datetime import datetime

import numpy as np
import pandas as pd
import streamlit as st
import torch
import torch.optim as optim

from model.evolving_nn import EvolvingNet


st.set_page_config(page_title="Self-Evolving Neural Network", layout="wide")

st.markdown(
    """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Manrope:wght@400;500;600;700;800&display=swap');

    html, body, [class*="css"] {
        font-family: 'Manrope', sans-serif;
    }

    #MainMenu, footer, header {
        visibility: hidden;
    }

    [data-testid="stToolbar"],
    [data-testid="stDecoration"],
    [data-testid="collapsedControl"],
    [data-testid="stStatusWidget"] {
        display: none !important;
    }

    .stApp {
        background:
            radial-gradient(circle at top left, rgba(43, 95, 255, 0.14), transparent 28%),
            radial-gradient(circle at top right, rgba(16, 185, 129, 0.12), transparent 24%),
            linear-gradient(180deg, #071019 0%, #0b1320 100%);
        color: #e7edf7;
    }

    .block-container {
        max-width: 1380px !important;
        padding-top: 1.4rem !important;
        padding-bottom: 1rem !important;
    }

    .hero-title {
        font-size: 2.2rem;
        font-weight: 800;
        color: #f8fbff;
        margin-bottom: 0.2rem;
        letter-spacing: -0.03em;
    }

    .hero-sub {
        color: #9db0c6;
        font-size: 1rem;
        margin-bottom: 0.8rem;
        max-width: 820px;
        line-height: 1.6;
    }

    .panel-title {
        font-size: 1rem;
        font-weight: 800;
        color: #f5f8ff;
        margin-bottom: 0.45rem;
    }

    .panel-copy {
        color: #9db0c6;
        font-size: 0.93rem;
        line-height: 1.6;
        margin-bottom: 0;
    }

    .note-box {
        background: rgba(255, 255, 255, 0.04);
        border: 1px solid rgba(255, 255, 255, 0.08);
        border-radius: 16px;
        padding: 14px 16px;
        margin-bottom: 0.8rem;
    }

    .section-label {
        color: #a8bad1;
        font-size: 0.78rem;
        font-weight: 700;
        text-transform: uppercase;
        letter-spacing: 0.08em;
        margin-bottom: 0.3rem;
    }

    div.stButton > button:first-child {
        width: 100%;
        border-radius: 12px;
        border: 1px solid rgba(92, 131, 255, 0.28);
        background: linear-gradient(135deg, #3b82f6, #2563eb);
        color: white;
        font-weight: 700;
        padding: 0.7rem 1rem;
        box-shadow: 0 10px 24px rgba(37, 99, 235, 0.18);
    }

    div.stButton > button:hover {
        filter: brightness(1.06);
    }

    div[data-baseweb="input"] > div,
    div[data-baseweb="select"] > div {
        border-radius: 12px !important;
        border: 1px solid rgba(255, 255, 255, 0.10) !important;
        background: rgba(255, 255, 255, 0.05) !important;
    }

    .stSlider label,
    .stNumberInput label,
    .stSelectbox label {
        color: #edf3fb !important;
        font-weight: 700 !important;
    }

    .footer-text {
        color: #8ea2bb;
        text-align: center;
        margin-top: 0.8rem;
        font-size: 0.9rem;
    }
    </style>
    """,
    unsafe_allow_html=True,
)


def build_dataset(n_samples=320, input_size=10, seed=42):
    torch.manual_seed(seed)
    x = torch.randn(n_samples, input_size)

    target = (
        0.55 * torch.sin(x[:, 0:1] * 2.4)
        + 0.35 * torch.cos(x[:, 1:2] * 1.8)
        + 0.25 * x[:, 2:3] * x[:, 3:4]
        - 0.15 * x[:, 4:5] ** 2
        + 0.2 * torch.tanh(x[:, 5:6] + x[:, 6:7])
    )

    noise = 0.03 * torch.randn_like(target)
    y = target + noise

    split_idx = int(0.8 * n_samples)
    return x[:split_idx], y[:split_idx], x[split_idx:], y[split_idx:]


def reset_state():
    st.session_state.model = EvolvingNet()
    st.session_state.history = []
    st.session_state.best_val_loss = float("inf")


if "model" not in st.session_state:
    reset_state()

if "history" not in st.session_state:
    st.session_state.history = []

if "best_val_loss" not in st.session_state:
    st.session_state.best_val_loss = float("inf")


st.markdown('<div class="hero-title">Self-Evolving Neural Network Lab</div>', unsafe_allow_html=True)
st.markdown(
    """
    <div class="hero-sub">
    This demo trains a small neural network on a nonlinear regression task and lets the model expand
    its own architecture when progress stalls. The goal is to show a simple, believable version of
    architecture growth without turning the interface into a flashy dashboard.
    </div>
    """,
    unsafe_allow_html=True,
)

top_left, top_right = st.columns([2.2, 1.2], gap="large")

with top_left:
    info_cols = st.columns(3, gap="medium")
    with info_cols[0]:
        st.markdown(
            """
            <div class="note-box">
                <div class="section-label">Idea</div>
                <div class="panel-title">Train and expand</div>
                <div class="panel-copy">When validation loss stops improving, the network can widen or deepen itself.</div>
            </div>
            """,
            unsafe_allow_html=True,
        )
    with info_cols[1]:
        st.markdown(
            """
            <div class="note-box">
                <div class="section-label">Mutation</div>
                <div class="panel-title">Structure changes</div>
                <div class="panel-copy">Width adds neurons. Depth adds a new hidden layer with identity-style initialization.</div>
            </div>
            """,
            unsafe_allow_html=True,
        )
    with info_cols[2]:
        st.markdown(
            """
            <div class="note-box">
                <div class="section-label">Task</div>
                <div class="panel-title">Synthetic regression</div>
                <div class="panel-copy">The network learns a nonlinear target built from sine, cosine, interaction, and noise.</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

with top_right:
    st.markdown(
        """
        <div class="note-box">
            <div class="section-label">Current model</div>
            <div class="panel-title">Architecture snapshot</div>
            <div class="panel-copy">Use reset if you want to start again from the base architecture.</div>
        </div>
        """,
        unsafe_allow_html=True,
    )
    if st.button("Reset Model"):
        reset_state()
        st.rerun()

st.write("")

controls_col, metrics_col = st.columns([1.2, 2.3], gap="large")

with controls_col:
    st.markdown("### Training Controls")

    learning_rate = st.number_input("Learning Rate", min_value=0.0001, max_value=0.1, value=0.01, step=0.0005, format="%.4f")
    epochs = st.slider("Epochs", min_value=100, max_value=1200, value=450, step=50)
    mutation_threshold = st.slider("Mutation Trigger (Val Loss)", min_value=0.005, max_value=0.300, value=0.050, step=0.005)
    patience_window = st.slider("Patience Window", min_value=15, max_value=120, value=40, step=5)
    width_step = st.select_slider("Width Step", options=[4, 8, 16, 32], value=8)
    max_params = st.number_input("Max Parameter Budget", min_value=1000, max_value=200000, value=25000, step=1000)
    mutation_mode = st.selectbox("Mutation Strategy", ["Auto", "Prefer Width", "Prefer Depth"])

    st.write("")
    train_button = st.button("Start Training")

with metrics_col:
    metric_cols = st.columns(4, gap="medium")
    metric_train = metric_cols[0].empty()
    metric_val = metric_cols[1].empty()
    metric_params = metric_cols[2].empty()
    metric_arch = metric_cols[3].empty()

    st.write("")
    chart_cols = st.columns([2, 1], gap="large")
    with chart_cols[0]:
        st.markdown("### Loss Curves")
        loss_chart = st.empty()
    with chart_cols[1]:
        st.markdown("### Model Size")
        params_chart = st.empty()

    st.write("")
    log_cols = st.columns([1.25, 1], gap="large")
    with log_cols[0]:
        st.markdown("### Mutation Log")
        log_box = st.empty()
    with log_cols[1]:
        st.markdown("### Recent Training State")
        status_box = st.empty()


def refresh_dashboard(model, history):
    if not history:
        metric_train.metric("Train Loss", "0.0000")
        metric_val.metric("Validation Loss", "0.0000")
        metric_params.metric("Parameters", f"{model.parameter_count():,}")
        metric_arch.metric("Hidden Dims", " x ".join(map(str, model.hidden_dims)))
        log_box.code("No mutations yet.")
        status_box.code("Model is idle.")
        return

    df = pd.DataFrame(history)
    metric_train.metric("Train Loss", f"{df['train_loss'].iloc[-1]:.4f}")
    metric_val.metric("Validation Loss", f"{df['val_loss'].iloc[-1]:.4f}")
    metric_params.metric("Parameters", f"{int(df['params'].iloc[-1]):,}")
    metric_arch.metric("Hidden Dims", " x ".join(map(str, model.hidden_dims)))

    loss_chart.line_chart(df[["train_loss", "val_loss"]], height=280)
    params_chart.area_chart(df["params"], height=280)

    recent_logs = model.mutation_log[-8:] if model.mutation_log else ["No mutations yet."]
    log_box.code("\n".join(recent_logs))

    last_row = df.iloc[-1]
    status_box.code(
        "\n".join(
            [
                f"epoch: {int(last_row['epoch'])}",
                f"architecture: {model.architecture_text()}",
                f"train_loss: {last_row['train_loss']:.6f}",
                f"val_loss: {last_row['val_loss']:.6f}",
                f"params: {int(last_row['params'])}",
            ]
        )
    )


refresh_dashboard(st.session_state.model, st.session_state.history)

if train_button:
    model = st.session_state.model
    history = st.session_state.history

    x_train, y_train, x_val, y_val = build_dataset()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    criterion = torch.nn.MSELoss()

    best_val_loss = st.session_state.best_val_loss
    patience_counter = 0

    progress = st.progress(0)

    for epoch in range(1, epochs + 1):
        model.train()
        optimizer.zero_grad()

        train_pred = model(x_train)
        train_loss = criterion(train_pred, y_train)
        train_loss.backward()
        optimizer.step()

        model.eval()
        with torch.no_grad():
            val_pred = model(x_val)
            val_loss = criterion(val_pred, y_val)

        train_loss_value = float(train_loss.item())
        val_loss_value = float(val_loss.item())
        param_count = model.parameter_count()

        history.append(
            {
                "epoch": epoch,
                "train_loss": train_loss_value,
                "val_loss": val_loss_value,
                "params": param_count,
            }
        )

        refresh_dashboard(model, history)
        progress.progress(epoch / epochs)

        if val_loss_value < best_val_loss - 1e-4:
            best_val_loss = val_loss_value
            patience_counter = 0
        else:
            patience_counter += 1

        should_mutate = (
            epoch >= patience_window
            and patience_counter >= patience_window
            and val_loss_value > mutation_threshold
            and param_count < max_params
        )

        if should_mutate:
            mutated = False

            if mutation_mode == "Prefer Width":
                mutated = model.mutate_width(step_size=width_step, layer_index=0)
            elif mutation_mode == "Prefer Depth":
                mutated = model.mutate_depth()
            else:
                if np.random.rand() > 0.5:
                    mutated = model.mutate_width(step_size=width_step, layer_index=0)
                else:
                    mutated = model.mutate_depth()

            if mutated:
                if model.parameter_count() <= max_params:
                    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
                    patience_counter = 0
                else:
                    model.mutation_log.append(
                        f"[{datetime.now().strftime('%H:%M:%S')}] Mutation exceeded parameter budget"
                    )

        time.sleep(0.01)

    st.session_state.model = model
    st.session_state.history = history
    st.session_state.best_val_loss = best_val_loss
    progress.empty()

st.markdown(
    '<div class="footer-text">A simpler demo of adaptive architecture growth with cleaner structure and less UI noise.</div>',
    unsafe_allow_html=True,
)
