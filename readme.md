# Steps to Initialize the Project

I recommend creating a virtual environment using Python 3.10 for this project, but you can use another version if you prefer.

To create a virtual environment in Python, use the following command:
`python -m venv .venv`

Or, to specify a Python version:
`py -3.10 -m venv .venv` (you can also use 3.11)

To activate the virtual environment:
- On Windows: `.\.venv\Scripts\activate`
- On Linux/macOS: `source .venv/bin/activate`

1. First, install the requirements using the following command:
`pip install -r requirements.txt`
2. If you are not sure whether you have an NVIDIA GPU (CUDA), run the following command:
`python ./tests/test.py` If the output is `True`, you can use CUDA with your NVIDIA GPU to train the model faster.
3. Run the application:
`python app.py`



I've committed the model I trained locally with 50 epochs. It can still be improved.
You can find it in the `./weights/last.pt` file.