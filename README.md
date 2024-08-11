import gradio as gr
from sklearn import tree

# Training data
features=[[10,20,30],[5,20,30],[10,20,30],[5,11,15],[5,0,15],[5,11,15]]
labels=[0,0,0,1,1,1]


# Training the model
myClassifier = tree.DecisionTreeClassifier()
myModel = myClassifier.fit(features, labels)

# Create Gradio interface
inputs = [
    gr.Slider(0, 1, step=1, label="Victim (0: No, 1: Yes)"),
    gr.Slider(0, 1, step=1, label="Money Involved (0: No, 1: Yes)"),
    gr.Slider(0, 1, step=1, label="Major Tools Used (0: No, 1: Yes)")
]
outputs = gr.Textbox(label="Predicted Crime Type")

interface = gr.Interface(
    fn=lambda victim, money_involved, major_tools_used: myModel.predict([[victim, money_involved, major_tools_used]])[0],
    inputs=inputs,
    outputs=outputs,
    title="Crime Type Predictor"
)

interface.launch()
