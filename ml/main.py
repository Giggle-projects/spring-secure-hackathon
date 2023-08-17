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
        self.softmax = nn.Softmax()  # Apply softmax along the third dimension


    def forward(self, x):
        x = x.unsqueeze(0)
        print(x.size(),x)
        out, _ = self.lstm(x)
        out = self.fc(out)  # Use the last time step's output for classification
        out = self.softmax(out)  # Apply softmax to get class probabilities

        return out
input_size = 12

model = LSTMModel(input_size=input_size,hidden_size=64,num_layers=3,output_size=2)


def predict_single_input(input_data):
    model.eval()
    with torch.no_grad():
        input_tensor  = torch.tensor(input_data, dtype=torch.float32)
        output = model(input_tensor)
        print('model output',output)

        max_value = output[0][0]


    return max_value




@app.post("/predict/")
async def predict(input_data: str = Form(...),type_num: int = Form(...)):


    # 모델에 대한 타입 지정을 num 에 해줘야함
    # 모델의 타입은 직렬 정보에 따라서 1 = 2 = 3 = 4 = 5 = 6 =
    model.load_state_dict(torch.load('model_'+str(type_num)+'.pth'))
    model.eval()

    input_data = ast.literal_eval(input_data)
    # 모두 0이면 리스트에 0 추가

    new_data=[]
    for i in input_data:
        if i ==0:
            new_data.append(5)
        else:
            new_data.append(i)

    print(f"model input data is {new_data}")

    max_value = predict_single_input(new_data)
    print(max_value.item())

    return {"prediction":max_value.item()*100}


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
