
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import pandas as pd

# Set random seed for reproducibility
np.random.seed(42)

df=pd.read_csv('./train_data.csv')
df=df.sort_values('member_id')
applicationType_list=df['applicationType'].unique().tolist()
applicationType_list

df=df.drop('id',axis=1)


def data_preprocess(df):
    member_id_list=df['member_id']
    member_id_list

    pre=0
    new_applicationType_list=[]
    for i in member_id_list:
        if i!=pre:
            new_applicationType_list.append(applicationType_list[i%len(applicationType_list)])
        else:
            pre=i
            new_applicationType_list.append(applicationType_list[i%len(applicationType_list)])


    df['applicationType']=new_applicationType_list
    return df
df=data_preprocess(df)


type_1_df=df[df['applicationType']=='경찰직공무원(남)']
type_2_df=df[df['applicationType']=='경찰직공무원(여)']
type_3_df=df[df['applicationType']=='소방직공무원(남)']
type_4_df=df[df['applicationType']=='소방직공무원(여)']
type_5_df=df[df['applicationType']=='경호직공무원(남)']
type_6_df=df[df['applicationType']=='경호직공무원(여)']

type_1_df=type_1_df.drop('applicationType',axis=1)
type_2_df=type_2_df.drop('applicationType',axis=1)
type_3_df=type_3_df.drop('applicationType',axis=1)
type_4_df=type_4_df.drop('applicationType',axis=1)
type_5_df=type_5_df.drop('applicationType',axis=1)
type_6_df=type_6_df.drop('applicationType',axis=1)

print(type_3_df)
def convert(df):

    data = {
        'member_id':df['member_id'] ,
        'score': df['score'],
        'month': df['month']
    }
    df = pd.DataFrame(data)

    # 피벗 테이블 생성
    pivot_table = df.groupby(['member_id', 'month'])['score'].sum().unstack()

    return pivot_table

type_1_df=convert(type_1_df)
type_2_df=convert(type_2_df)
type_3_df=convert(type_3_df)
type_4_df=convert(type_4_df)
type_5_df=convert(type_5_df)
type_6_df=convert(type_6_df)


type_1_df=type_1_df.fillna(5)
type_2_df=type_2_df.fillna(5)
type_3_df=type_3_df.fillna(5)
type_4_df=type_4_df.fillna(5)
type_5_df=type_5_df.fillna(5)
type_6_df=type_6_df.fillna(5)


def set_label(df):
    # 각 회원의 월별 점수 총합 계산
    df["total_score"] = df.sum(axis=1)

    # 총합 점수의 백분위수 계산
    percentile_93 = df["total_score"].quantile(0.93)  # 상위 7% 기준

    # 레이블 생성
    df["label"] = df["total_score"].apply(
        lambda x: 1 if x >= percentile_93 else 0
    )
    return df

# 각 데이터프레임에 레이블 할당
type_1_df = set_label(type_1_df)
type_2_df = set_label(type_2_df)
type_3_df = set_label(type_3_df)
type_4_df = set_label(type_4_df)
type_5_df = set_label(type_5_df)
type_6_df = set_label(type_6_df)



# Define your neural network architecture
class LSTMModel(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, output_size):
        super(LSTMModel, self).__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size,output_size)
        self.softmax = nn.Softmax()  # Apply softmax along the third dimension


    def forward(self, x):
        out, _ = self.lstm(x)
        out = self.fc(out)  # Use the last time step's output for classification
        out = self.softmax(out)  # Apply softmax to get class probabilities

        return out
type_list=[type_1_df,type_2_df,type_3_df,type_4_df,type_5_df,type_6_df]

num=1
for i in type_list:
    # Convert pandas DataFrames to numpy arrays
    X = i.drop(['label', 'total_score'], axis=1).to_numpy()
    y = i['label'].to_numpy()

    # Convert numpy arrays to PyTorch tensors
    X = torch.tensor(X, dtype=torch.float32)
    y = torch.tensor(y, dtype=torch.float32)

    hidden_size = 64
    num_layers=3
    output_size=2
    # Create an instance of your model
    input_size = X.shape[1]
    model = LSTMModel(input_size=input_size,hidden_size=hidden_size,num_layers=num_layers,output_size=output_size)
    print(sum(p.numel() for p in  model.parameters()))
    # Define a loss function and an optimizer
    criterion = nn.BCELoss()  # Binary Cross-Entropy loss
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    # Training loop
    num_epochs = 100
    for epoch in range(num_epochs):
        optimizer.zero_grad()
        outputs = model(X)
        # Assuming outputs shape is [1389, 2]
        predicted_probs = outputs[:, 0]


        loss = criterion(predicted_probs, y)  # Compare predicted probabilities with binary labels
        loss.backward()
        optimizer.step()


        if (epoch + 1) % 10 == 0:
            print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.4f}')

    # After training, you can use this model for prediction
    with torch.no_grad():
        predicted_probs = model(X)
        predicted_labels = (predicted_probs >= 0.5).squeeze().int().numpy()

    # Now 'predicted_labels' contains the predicted labels for each sample in the data.
    torch.save(model.state_dict(), "./model_"+str(num)+".pth")
    num+=1

