import streamlit as st

from transformer_lens import HookedTransformer, utils
from io import StringIO
import sys
import torch
from functools import partial
import plotly.offline as pyo
import plotly.graph_objs as go
import numpy as np

model = HookedTransformer.from_pretrained("gelu-2l")

def the_model(prompt):
    logits = model(prompt)[0,-1]
    answer_index = logits.argmax()
    answer = model.tokenizer.decode(answer_index)
    print(answer)
    return str(answer)


clean_prompt_input = st.text_input("Clean Prompt", key="clean_prompt2")

st.text_input("Correct Answer", key="correct_answer2")

st.write("nothing")

def run_button():
    res = the_model(clean_prompt_input)
    st.write(res)

st.button("Run in gelu-2l", key="run_gelu_2l2", on_click=run_button)

st.text_input("Clean Prompt", key="clean_prompt")

st.text_input("Correct Answer", key="correct_answer")

st.button("Run test_propmpt", key="run_test_prompt")

# common plotly demo
# import plotly.graph_objects as go
# import numpy as np

import plotly.express as px

fig = px.imshow(np.random.randn(10, 10))

st.plotly_chart(fig)



