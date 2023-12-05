# Configuration
virtual environemnt.
```
python -m venv env
```
Install necessary dependencies.

```bash
pip install -r requirements.txt
```
One can download and install git lfs 
and run in cmd to setup lfs
```
git lfs install 
```
Clone the repository containing LaMini-Flan-T5-248M which is the LLM we're using.
```
git clone https://huggingface.co/MBZUAI/LaMini-Flan-T5-248M
```
The LaMini-Flan-T5-248M and project should be in the same folder.
## Running
Run the app with below code :
```
streamlit run app.py
```
Upload a pdf file and see the summarization :)
![pdf]()
![summary]()
