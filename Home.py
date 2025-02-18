import streamlit as st
from transformer_lens import HookedTransformer, utils
from io import StringIO
import sys
import torch
from functools import partial
import plotly.offline as pyo
import plotly.graph_objs as go
import numpy as np
import plotly.express as px
import circuitsvis as cv
from transformer_lens import HookedTransformer, HookedTransformerConfig, FactoredMatrix, ActivationCache
# Little bit of front end for model selector

# Radio buttons
#model_name = st.sidebar.radio("Model", [
# Backend code
import transformer_lens.utils as utils
cfg = HookedTransformerConfig(
    n_layers = 8,
    d_model = 512,
    d_head = 64,
    n_heads = 8,
    d_mlp = 2048,
    d_vocab = 61,
    n_ctx = 59,
    act_fn="gelu",
    normalization_type="LNPre"
)
model = HookedTransformer(cfg)
sd = utils.download_file_from_hf("NeelNanda/Othello-GPT-Transformer-Lens", "synthetic_model.pth")
# champion_ship_sd = utils.download_file_from_hf("NeelNanda/Othello-GPT-Transformer-Lens", "championship_model.pth")
model.load_state_dict(sd)


def predict_next_token(prompt):
    logits = model(prompt)[0,-1]
    answer= logits.argmax()
    answer = f"<b>|{answer}|</b>"#TODO make this a nice looking square board 
    return answer

def compute_residual_stream_patch(clean_prompt=None, answer=None, corrupt_prompt=None, corrupt_answer=None, layers=None):
    model.reset_hooks()
    clean_answer_index = answer[0]#TODO make the selector 1 token
    corrupt_answer_index = corrupt_answer[0]
    clean_tokens = clean_prompt
    _, corrupt_cache = model.run_with_cache(corrupt_prompt)
    # Patching function
    def patch_residual_stream(activations, hook, layer="blocks.6.hook_resid_post", pos=5):
        activations[:, pos, :] = corrupt_cache[layer][:, pos, :]
        return activations
    # Compute logit diffs
    n_layers = len(layers)
    n_pos = len(clean_tokens)
    patching_effect = torch.zeros(n_layers, n_pos)
    for l, layer in enumerate(layers):
        for pos in range(n_pos):
            fwd_hooks = [(layer, partial(patch_residual_stream, layer=layer, pos=pos))]
            prediction_logits = model.run_with_hooks(clean_prompt, fwd_hooks=fwd_hooks)[0, -1]
            patching_effect[l, pos] = prediction_logits[clean_answer_index] - prediction_logits[corrupt_answer_index]
    return patching_effect


def compute_attn_patch(clean_prompt=None, answer=None, corrupt_prompt=None, corrupt_answer=None):
    use_attn_result_prev = model.cfg.use_attn_result
    model.cfg.use_attn_result = True
    clean_answer_index = answer[0]
    corrupt_answer_index =corrupt_answer[0]
    _, corrupt_cache = model.run_with_cache(corrupt_prompt)
    # Patching function
    def patch_head_result(activations, hook, head=None, pos=None):
        activations[:, pos, head, :] = corrupt_cache[hook.name][:, pos, head, :]
        return activations

    n_layers = model.cfg.n_layers
    n_heads = model.cfg.n_heads
    n_pos = len(clean_prompt)
    patching_effect = torch.zeros(n_layers*n_heads, n_pos)
    for layer in range(n_layers):
        for head in range(n_heads):
          for pos in range(n_pos):
              fwd_hooks = [(f"blocks.{layer}.attn.hook_result", partial(patch_head_result, head=head, pos=pos))]
              prediction_logits = model.run_with_hooks(clean_prompt, fwd_hooks=fwd_hooks)[0, -1]
              patching_effect[n_heads*layer+head, pos] = prediction_logits[clean_answer_index] - prediction_logits[corrupt_answer_index]
    model.cfg.use_attn_result = use_attn_result_prev
    return patching_effect

def imshow(tensor, xlabel="X", ylabel="Y", zlabel=None, xticks=None, yticks=None, c_midpoint=0.0, c_scale="RdBu", **kwargs):
    tensor = utils.to_numpy(tensor)
    xticks = [str(x) for x in xticks]
    yticks = [str(y) for y in yticks]
    labels = {"x": xlabel, "y": ylabel}
    if zlabel is not None:
        labels["color"] = zlabel
    fig = px.imshow(tensor, x=xticks, y=yticks, labels=labels, color_continuous_midpoint=c_midpoint,
                    color_continuous_scale=c_scale, **kwargs)
    return fig

def plot_residual_stream_patch(clean_prompt=None, answer=None, corrupt_prompt=None, corrupt_answer=None):
    layers = ["blocks.0.hook_resid_pre", *[f"blocks.{i}.hook_resid_post" for i in range(model.cfg.n_layers)]]
    token_labels = clean_prompt
    patching_effect = compute_residual_stream_patch(clean_prompt=clean_prompt, answer=answer, corrupt_prompt=corrupt_prompt, corrupt_answer=corrupt_answer, layers=layers)
    fig = imshow(patching_effect, xticks=token_labels, yticks=layers, xlabel="Position", ylabel="Layer",
       zlabel="Logit Difference", title="Patching residual stream at specific layer and position")
    return fig

def plot_attn_patch(clean_prompt=None, answer=None, corrupt_prompt=None, corrupt_answer=None):

    n_layers = model.cfg.n_layers
    n_heads = model.cfg.n_heads
    layerhead_labels = [f"{l}.{h}" for l in range(n_layers) for h in range(n_heads)]
    token_labels = [f"(pos {i:2}) {t}" for i, t in enumerate(clean_prompt)]
    patching_effect = compute_attn_patch(clean_prompt=clean_prompt, answer=answer, corrupt_prompt=corrupt_prompt, corrupt_answer=corrupt_answer)
    return imshow(patching_effect, xticks=token_labels, yticks=layerhead_labels, xlabel="Position", ylabel="Layer.Head",
           zlabel="Logit Difference", title=f"Patching attention outputs for specific layer, head, and position", width=600, height=300+200*n_layers)

# Frontend code
st.title("Simple Orthello Mech Int")
st.subheader("Transformer Mechanistic Interpretability")
st.markdown("Powered by [TransformerLens](https://github.com/neelnanda-io/TransformerLens/)")
st.markdown("For _what_ these plots are, and _why_, see this [tutorial](https://docs.google.com/document/d/1e6cs8d9QNretWvOLsv_KaMp6kSPWpJEW0GWc0nwjqxo/).")

# Predict next token
st.header("Predict the next token")
st.markdown("Just a simple test UI, enter a series of moves and model will predict te next")

numbers = list(range(1, 61))


def button_grid():
    grid_size = 8
    skip_rows = [grid_size//2-1, grid_size//2]
    skip_cols = [grid_size//2-1, grid_size//2]
    st.markdown(""" <style> button[kind="secondary"] { padding:0px; margin:0px; width:50px; height:50px; } </style>""", unsafe_allow_html=True)
    
    if "selected_numbers" not in st.session_state:
        st.session_state.selected_numbers = []

    def add_to_list(num):
        st.session_state.selected_numbers.append(num)

    def update_list(selected_nums):
        st.session_state.selected_numbers = selected_nums
    
    grid = st.container()
    
    with grid:
        num=0
        cols = st.columns(grid_size)
        for j in range(grid_size):
            for i in range(grid_size):
                col_idx = i % grid_size
                
                if i in skip_rows and j in skip_cols:
                    
                    cols[col_idx].button("",key=f'blank{i}{j}')
                    
                else:
                    num=num+1
                    cols[col_idx].button(str(num), on_click=lambda num=num: add_to_list(num) )#add_to_list(num))        
    selected_nums = st.multiselect('Selected tokens:', numbers, default=st.session_state.selected_numbers, on_change=lambda num=num: update_list(selected_nums))
    st.session_state.selected_numbers = selected_nums
button_grid()

if "prompt_simple_output" not in st.session_state:
    st.session_state.prompt_simple_output = None

if st.button("Run model", key="key_button_prompt_simple",type="primary"):
    prompt_simple=torch.tensor(st.session_state.selected_numbers)
    res = predict_next_token(prompt_simple)
    st.session_state.prompt_simple_output = res

if st.session_state.prompt_simple_output:
    st.markdown(st.session_state.prompt_simple_output, unsafe_allow_html=True)


#Predict next token button




# Residual stream patching

st.header("Residual stream patching")
st.markdown("Enter a clean prompt, correct answer, corrupt prompt and corrupt answer, the model will compute the patching effect")


clean_prompt   = torch.tensor(st.multiselect("Clean Prompt:"   ,numbers,[20]))
clean_answer   = torch.tensor(st.multiselect("Correct Answer:" ,numbers,[20]))
corrupt_prompt = torch.tensor(st.multiselect("Corrupt Prompt:" ,numbers,[20]))
corrupt_answer = torch.tensor(st.multiselect("Corrupt Answer:" ,numbers,[20]))

if "residual_stream_patch_out" not in st.session_state:
    st.session_state.residual_stream_patch_out = None

if st.button("Run model", key="key_button_residual_stream_patch",type="primary"):
    
    fig = plot_residual_stream_patch(clean_prompt=clean_prompt, answer=clean_answer, corrupt_prompt=corrupt_prompt, corrupt_answer=corrupt_answer)
    st.session_state.residual_stream_patch_out = fig

if st.session_state.residual_stream_patch_out:
    st.plotly_chart(st.session_state.residual_stream_patch_out)


# Attention head output

st.header("Attention head output patching")
st.markdown("Enter a clean prompt, correct answer, corrupt prompt and corrupt answer, the model will compute the patching effect")


clean_prompt_attn   = torch.tensor(st.multiselect("Clean Prompt:"   ,numbers,[20],key="key2_clean_prompt_attn"))
clean_answer_attn   = torch.tensor(st.multiselect("Correct Answer:" ,numbers,[20],key="key2_clean_answer_attn"))
corrupt_prompt_attn = torch.tensor(st.multiselect("Corrupt Prompt:" ,numbers,[20], key="key2_corrupt_prompt_attn"))
corrupt_answer_attn = torch.tensor(st.multiselect("Corrupt Answer:" ,numbers,[20], key="key2_corrupt_answer_attn"))


if "attn_head_patch_out" not in st.session_state:
    st.session_state.attn_head_patch_out = None

if st.button("Run model", key="key_button_attn_head_patch",type="primary"):
    fig = plot_attn_patch(clean_prompt=clean_prompt_attn, answer=clean_answer_attn, corrupt_prompt=corrupt_prompt_attn, corrupt_answer=corrupt_answer_attn)
    st.session_state.attn_head_patch_out = fig

if st.session_state.attn_head_patch_out:
    st.plotly_chart(st.session_state.attn_head_patch_out)


# Attention Head Visualization

st.header("Attention Pattern Visualization")
st.markdown("Powered by [CircuitsVis](https://github.com/alan-cooney/CircuitsVis)")
st.markdown("Enter a prompt, show attention patterns")

prompt_attn= torch.tensor(st.multiselect("Prompt:",numbers,[20]))


if "attn_html" not in st.session_state:
    st.session_state.attn_html = None

if st.button("Run model", key="key_button_attention_head",type="primary"):
    _, cache = model.run_with_cache(prompt_attn)
    st.session_state.attn_html = []
    for layer in range(model.cfg.n_layers):

        html = cv.attention.attention_patterns(tokens=[str(i) for i in prompt_attn.numpy()],
                                attention=cache[f'blocks.{layer}.attn.hook_pattern'][0])
        st.session_state.attn_html.append(html.show_code())

if st.session_state.attn_html:
    for layer in range(len(st.session_state.attn_html)):
        st.write(f"Attention patterns Layer {layer}:")
        st.components.v1.html(st.session_state.attn_html[layer], height=500)

