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


type_1_df=type_1_df.fillna(type_1_df.mean())
type_2_df=type_2_df.fillna(type_1_df.mean())
type_3_df=type_3_df.fillna(type_1_df.mean())
type_4_df=type_4_df.fillna(type_1_df.mean())
type_5_df=type_5_df.fillna(type_1_df.mean())
type_6_df=type_6_df.fillna(type_1_df.mean())


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


type_1_df=type_1_df.drop("total_score",axis=1)
type_2_df=type_2_df.drop("total_score",axis=1)
type_3_df=type_3_df.drop("total_score",axis=1)
type_4_df=type_4_df.drop("total_score",axis=1)
type_5_df=type_5_df.drop("total_score",axis=1)
type_6_df=type_6_df.drop("total_score",axis=1)




class LSTMModel(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, output_size):
        super(LSTMModel, self).__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        out,_= self.lstm(x)
        out = self.fc(out)  # Remove this line
        out = torch.sigmoid(out)

        return out

df_list=[type_1_df,type_2_df,type_3_df,type_4_df,type_5_df,type_6_df]
num=1
for i in df_list:

    # Convert the DataFrame to PyTorch tensors
    data_tensor = torch.tensor(i.drop(columns=['label']).values, dtype=torch.float32)
    label_tensor = torch.tensor(i['label'].values, dtype=torch.float32)  # Change dtype to long



    # Split the data into training and testing sets
    train_size = int(0.8 * len(data_tensor))
    train_data = data_tensor[:train_size]
    train_labels = label_tensor[:train_size]
    test_data = data_tensor[train_size:]
    test_labels = label_tensor[train_size:]

    # Create an instance of the LSTMModel
    input_size = 12
    hidden_size = 64
    num_layers = 2
    output_size = 1

    model = LSTMModel(input_size, hidden_size, num_layers, output_size)

    # Define loss function and optimizer
    criterion = nn.BCEWithLogitsLoss()  # Change to CrossEntropyLoss
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    num_epochs = 100
    for epoch in range(num_epochs):
        model.train()
        optimizer.zero_grad()

        outputs = model(train_data)
        loss = criterion(outputs.squeeze(), train_labels)  # Remove the dimension from outputs

        loss.backward()
        optimizer.step()

        if (epoch + 1) % 10 == 0:
            print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.4f}')

    # ... 이하 코드는 이전 코드와 동일하게 유지합니다 ...

    # Test accuracy calculation
    model.eval()
    with torch.no_grad():
        test_outputs = model(test_data)
        _, test_preds = torch.max(test_outputs, dim=1)  # Get the class predictions

    accuracy = (test_preds == test_labels).float().mean()
    print(f'Test Accuracy: {accuracy.item():.4f}')

    torch.save(model.state_dict(), 'lstm_model_'+str(num)+'.pth')
    num+=1




