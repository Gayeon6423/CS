# DP(Data Parallel)

from torch import nn
from torch.optim import AdamW
from torch.utils.data import DataLoader
from transformers import BertForSequenceClassification, BertTokenizer
from datasets import load_dataset


def data_parallel(module,inputs,labels,device_ids,output_device):
    # 입력 데이터를 device_ids들에 scatter
    inputs = nn.parallel.scatter(inputs,device_ids)
    # 모델을 device_ids들에 복제
    replicas = nn.parallel.replicate(module,device_ids)
    # 각 device에 복제된 모델이 각 device의 데이터를 forward 연산
    logit = nn.parallel.parallel_apply(replicas,inputs)
    # 모델의 logit을 output_device(하나의 device)로 모아줌
    logits = nn.parallel.gather(logit,output_device)
    return logits


# 1. Create Dataset
datasets = load_dataset('multi_nli').data['train']
datasets = [
    {
        "premise": str(p),
        "hypothesis": str(h),
        "labels": l.as_py(),
    }
    for p, h, l in zip(datasets[2], datasets[5], datasets[9])
]
data_loader = DataLoader(datasets,batch_size=128,num_workers=4)

# 2. Create model and tokenizer
model_name = 'bert-base-uncased'
tokenizer = BertTokenizer.from_pretrained(model_name)
model = BertForSequenceClassification.from_pretrained(model_name,num_labels=3).cuda()

# 3. make data parallel module
## device ids : 사용할 디바이스 리스트 / output_divice : 출력값을 모을 디바이스
model = nn.DataParallel(model,device_ids=[0,1,2,3],output_device=0)

# 4. Create optimizer and loss function
optimizer = AdamW(model.parameters(),lr=3e-5)
loss_fn = nn.CrossEntropyLoss(reduction='mean')

# 5. start trainig
for i,data in enumerate(data_loader):
    optimizer.zero_grad()
    tokens = tokenizer(
        data['premise'],
        data['hypothesis'],
        padding=True,
        truncation=True,
        max_length=512,
        return_tensors='pt',
    )

    loss = model(
        input_ids = tokens.input_ids.cuda(),
        attention_mask = tokens.attention_mask.cuda(),
        labels = data['labels']
    ).loss

    loss = loss.mean()
    loss.backward()
    optimizer.step()

    if i%10 == 0:
        print(f"step:{i}, loss:{loss}")
    if i == 100:
        break



