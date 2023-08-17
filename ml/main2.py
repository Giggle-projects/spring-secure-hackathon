import torch
import torch.nn as nn
from fastapi import FastAPI,Form
import ast
import numpy as np
from fastapi.middleware.cors import CORSMiddleware
# Define your neural network architecture

app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)




class DNNModel(nn.Module):
    def __init__(self, input_size, hidden_size,second_hidden, output_size):
        super(DNNModel, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_size,second_hidden)
        self.fc3 = nn.Linear(second_hidden, output_size)

    def forward(self, x):
        out = self.fc1(x)
        out = self.relu(out)
        out = self.fc2(out)
        out = self.fc3(out)
        return out



type_num=1
hidden_size = 32
output_size = 2
# Create an instance of your model
model = DNNModel(input_size=12, hidden_size=hidden_size,second_hidden=16, output_size=output_size)


def predict_single_input(input_data):
    model.eval()
    with torch.no_grad():
        input_tensor  = torch.tensor(input_data, dtype=torch.float32)
        output = model(input_tensor)
        predicted_prob = torch.softmax(output, dim=0)  # Apply softmax for probabilities
        print('model output',predicted_prob)

        max_value = predicted_prob[1]

    return max_value



@app.post("/predict")
async def predict(input_data: str = Form(...),type_num: int = Form(...)):
    print(model)

    # 모델에 대한 타입 지정을 num 에 해줘야함
    # 모델의 타입은 직렬 정보에 따라서 1 = 2 = 3 = 4 = 5 = 6 =
    model.load_state_dict(torch.load('model2_'+str(type_num)+'.pth'))
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
