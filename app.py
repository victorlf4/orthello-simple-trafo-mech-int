from flask import Flask, render_template, request, make_response
from transformer_lens import HookedTransformer, utils
from io import StringIO
import sys
import torch
from functools import partial
import plotly.offline as pyo
import plotly.graph_objs as go

model = HookedTransformer.from_pretrained("gelu-2l")

app = Flask(__name__)




@app.route("/")
def index():
    return render_template("index.html")

def the_model(prompt, answer):
    logits = model(prompt)[0,-1]
    answer_index = logits.argmax()
    answer = model.tokenizer.decode(answer_index)
    return str(answer)

def test_prompt(prompt, answer):
    output = StringIO()
    sys.stdout = output
    utils.test_prompt(prompt, answer, model)
    output = output.getvalue()
    output = output.replace("\n", "<br>")
    return make_response(output)

def patch_stream(clean_prompt=None, answer=None, corrupt_prompt=None, corrupt_answer=None):
    clean_answer_index = model.tokenizer.encode(answer)[0]
    corrupt_answer_index = model.tokenizer.encode(corrupt_answer)[0]
    _, corrupt_cache = model.run_with_cache(corrupt_prompt)
    clean_tokens = model.str_to_tokens(clean_prompt)
    def patch_residual_stream(activations, hook, layer="blocks.6.hook_resid_post", pos=5):
        activations[:, pos, :] = corrupt_cache[layer][:, pos, :]
        return activations
    layers = ["blocks.0.hook_resid_pre", *[f"blocks.{i}.hook_resid_post" for i in range(model.cfg.n_layers)]]
    n_layers = len(layers)
    n_pos = len(clean_tokens)
    patching_effect = torch.zeros(n_layers, n_pos)
    for l, layer in enumerate(layers):
        for pos in range(n_pos):
            fwd_hooks = [(layer, partial(patch_residual_stream, layer=layer, pos=pos))]
            prediction_logits = model.run_with_hooks(clean_prompt, fwd_hooks=fwd_hooks)[0, -1]
            patching_effect[l, pos] = prediction_logits[clean_answer_index] - prediction_logits[corrupt_answer_index]



@app.route("/run_the_model", methods=["POST"])
def run_the_model():
    param1 = request.form["param1"]
    param2 = request.form["param2"]
    # Run the Python code here
    result = the_model(param1,param2)
    # Return the result to the user
    return result

@app.route("/run_test_prompt", methods=["POST"])
def run_test_prompt():
    param1 = request.form["param1"]
    param2 = request.form["param2"]
    # Run the Python code here
    result = test_prompt(param1,param2)
    # Return the result to the user
    return result

@app.route("/run_stream_patch", methods=["POST"])
def run_stream_patch():
    param1 = request.form["param1"]
    param2 = request.form["param2"]
    trace = go.Scatter(x=[1, 2, 3], y=[4, 5, 6])
    data = [trace]
    layout = go.Layout(title="My Plot")
    fig = go.Figure(data=data, layout=layout)
    plot_div = pyo.plot(fig, output_type='div', include_plotlyjs=False)
    return plot_div

if __name__ == "__main__":
    app.run(debug=False)

