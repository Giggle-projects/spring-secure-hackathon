import torch
import torch.nn as nn
from fastapi import FastAPI,Form
import ast
import numpy as np
from fastapi.middleware.cors import CORSMiddleware

app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:1234"],
    allow_methods=["*"],
    allow_headers=["*"],
)

class LSTMModel(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, output_size):
        super(LSTMModel, self).__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)
    
    def forward(self, x):
        out, _ = self.lstm(x)
        out = self.fc(out)
        
        return out

input_size = 13
hidden_size = 64
num_layers = 2
output_size = 4
model = LSTMModel(input_size, hidden_size, num_layers, output_size)


def predict_single_input(input_data):
    model.eval()
    with torch.no_grad():
        input_tensor = torch.tensor(input_data, dtype=torch.float32).view(1, -1, input_size)
        output = model(input_tensor)
        _, prediction = torch.max(output, dim=2)  # Get the class indices
    return prediction.item()  # Get the class index from the tensor


@app.post("/predict/")
async def predict(input_data: str = Form(...),type_num: int = Form(...)):


    # 모델에 대한 타입 지정을 num 에 해줘야함
    # 모델의 타입은 직렬 정보에 따라서 1 = 2 = 3 = 4 = 5 = 6 = 




    model.load_state_dict(torch.load('lstm_model_'+str(type_num)+'.pth'))
    model.eval()

    input_data = ast.literal_eval(input_data)
    filtered_values = [value for value in input_data if value != -1]
    mean_value = np.mean(filtered_values)
    
    input_data = [mean_value if value == -1 else value for value in input_data]
    input_data.append(sum(input_data))
    
    print(f"model input data is {input_data}")

    result = predict_single_input(input_data)
    return {"prediction": result}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
