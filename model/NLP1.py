"""
NLP
解题思路：
思路1：TF-IDF提取特征 + SVM分类
思路2：训练FastText词向量并分类
思路4：训练BERT词向量并分类
思路5：BERT分类 + 统计特征的树模型
"""
# 赛题以新闻数据为赛题数据，数据集报名后可见并可下载。赛题数据为新闻文本，
# 并按照字符级别进行匿名处理。整合划分出14个候选分类类别：财经、彩票、房产、股票、家居、教育、科技、社会、时尚、时政、体育、星座、游戏、娱乐的文本数据。
# 赛题数据由以下几个部分构成：训练集20w条样本，测试集A包括5w条样本，测试集B包括5w条样本。为了预防选手人工标注测试集的情况，
# 我们将比赛数据的文本按照字符级别进行了匿名处理。处理后的赛题训练数据如下：
import numpy as np
import pandas as pd
from sklearn.model_selection import KFold
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import LinearSVC  # 可以使用其它机器学习模型
from sklearn.metrics import f1_score
train_df = pd.read_csv('./train_set.csv', sep='\t', nrows=8000)
test_df = pd.read_csv('./test_a.csv', sep='\t')
tfidf = TfidfVectorizer(
    sublinear_tf=True,
    strip_accents='unicode',
    analyzer='word',
    token_pattern=r'\w{1,}',
    stop_words='english',
    ngram_range=(1,1),
    max_features=1000)

tfidf.fit(pd.concat([train_df['text'], test_df['text']]))
train_word_features = tfidf.transform(train_df['text'])
test_word_features = tfidf.transform(test_df['text'])

X_train = train_word_features
y_train = train_df['label']
X_test = test_word_features

KF = KFold(n_splits=5, random_state=7)
clf = LinearSVC()
# 存储测试集预测结果 行数：len(X_test) ,列数：1列
test_pred = np.zeros((X_test.shape[0], 1), int)
for KF_index, (train_index,valid_index) in enumerate(KF.split(X_train)):
    print('第', KF_index+1, '折交叉验证开始...')
    # 训练集划分
    x_train_, x_valid_ = X_train[train_index], X_train[valid_index]
    y_train_, y_valid_ = y_train[train_index], y_train[valid_index]
    # 模型构建
    clf.fit(x_train_, y_train_)
    # 模型预测
    val_pred = clf.predict(x_valid_)
    print("LinearSVC准确率为：",f1_score(y_valid_, val_pred, average='macro'))
    # 保存测试集预测结果
    test_pred = np.column_stack((test_pred, clf.predict(X_test)))  # 将矩阵按列合并
# 多数投票
preds = []
for i, test_list in enumerate(test_pred):
    preds.append(np.argmax(np.bincount(test_list)))
preds = np.array(preds)

# FastText在文本分类任务上是优于TF-IDF的：
#
# FastText用单词的Embedding叠加获得的文档向量，将相似的句子分为一类；
# FastText学习到的Embedding空间维度比较低，可以快速进行训练
import numpy as np
import pandas as pd
import fasttext
from sklearn.model_selection import KFold
from sklearn.metrics import f1_score
train_df = pd.read_csv('./train_set.csv', sep='\t', nrows=8000)
test_df = pd.read_csv('./test_a.csv', sep='\t')
# 转换为FastText需要的格式
train_df['label_ft'] = '__label__' + train_df['label'].astype(str)


# 各个类别性能度量的函数
def category_performance_measure(labels_right, labels_pred):
    text_labels = list(set(labels_right))
    text_pred_labels = list(set(labels_pred))

    TP = dict.fromkeys(text_labels, 0)  # 预测正确的各个类的数目
    TP_FP = dict.fromkeys(text_labels, 0)  # 测试数据集中各个类的数目
    TP_FN = dict.fromkeys(text_labels, 0)  # 预测结果中各个类的数目

    # 计算TP等数量
    for i in range(0, len(labels_right)):
        TP_FP[labels_right[i]] += 1
        TP_FN[labels_pred[i]] += 1
        if labels_right[i] == labels_pred[i]:
            TP[labels_right[i]] += 1
    # 计算准确率P，召回率R，F1值
    for key in TP_FP:
        P = float(TP[key]) / float(TP_FP[key] + 1)
        R = float(TP[key]) / float(TP_FN[key] + 1)
        F1 = P * R * 2 / (P + R) if (P + R) != 0 else 0
        print("%s:\t P:%f\t R:%f\t F1:%f" % (key, P, R, F1))
#FastText是一种融合深度学习和机器学习各自优点的文本分类模型，速度非常快，但是模型结构简单，效果还算中上游。由于其使用词袋思想，语义信息获取有限。
X_train = train_df['text']
y_train = train_df['label']
X_test = test_df['text']
KF = KFold(n_splits=5, random_state=7, shuffle=True)
test_pred = np.zeros((X_test.shape[0], 1), int)  # 存储测试集预测结果 行数：len(X_test) ,列数：1列
for KF_index, (train_index, valid_index) in enumerate(KF.split(X_train)):
    print('第', KF_index + 1, '折交叉验证开始...')
    # 转换为FastText需要的格式
    train_df[['text', 'label_ft']].iloc[train_index].to_csv('train_df.csv', header=None, index=False, sep='\t')
    # 模型构建
    model = fasttext.train_supervised('train_df.csv', lr=0.1, epoch=27, wordNgrams=5,
                                      verbose=2, minCount=1, loss='hs')
    # 模型预测
    val_pred = [int(model.predict(x)[0][0].split('__')[-1]) for x in X_train.iloc[valid_index]]
    print('Fasttext准确率为：', f1_score(list(y_train.iloc[valid_index]), val_pred, average='macro'))
    category_performance_measure(list(y_train.iloc[valid_index]), val_pred)

    # 保存测试集预测结果
    test_pred_ = [int(model.predict(x)[0][0].split('__')[-1]) for x in X_test]
    test_pred = np.column_stack((test_pred, test_pred_))  # 将矩阵按列合并
# 取测试集中预测数量最多的数
preds = []
for i, test_list in enumerate(test_pred):
    preds.append(np.argmax(np.bincount(test_list)))
preds = np.array(preds)
#BERT
import numpy as np
import pandas as pd
from sklearn import metrics
import transformers
import time
import torch
from tqdm import tqdm
from torch.utils.data import Dataset, DataLoader, RandomSampler, SequentialSampler
from transformers import BertTokenizer, BertModel, BertConfig

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
train_df = pd.read_csv('./train_set.csv', sep='\t')
test_df = pd.read_csv('./test_a.csv', sep='\t')
test_df['label'] = 0

tokenizer = BertTokenizer.from_pretrained('./emb/bert-mini/vocab.txt')
tokenizer.encode_plus("2967 6758 339 2021 1854",
        add_special_tokens=True,
        max_length=20,
        truncation=True)
# token_type_ids 通常第一个句子全部标记为0，第二个句子全部标记为1。
# attention_mask padding的地方为0，未padding的地方为1。
class CustomDataset(Dataset):

    def __init__(self, dataframe, tokenizer, max_len):
        self.tokenizer = tokenizer
        self.data = dataframe
        self.comment_text = self.data.text
        self.targets = self.data.label
        self.max_len = max_len

    def __len__(self):
        return len(self.comment_text)

    def __getitem__(self, index):
        comment_text = self.comment_text[index]

        inputs = self.tokenizer.encode_plus(
            comment_text,
            truncation=True,
            add_special_tokens=True,
            max_length=self.max_len,
            pad_to_max_length=True,
            return_token_type_ids=True
        )
        ids = inputs['input_ids']
        mask = inputs['attention_mask']
        token_type_ids = inputs["token_type_ids"]

        return {
            'ids': torch.tensor(ids, dtype=torch.long),
            'mask': torch.tensor(mask, dtype=torch.long),
            'token_type_ids': torch.tensor(token_type_ids, dtype=torch.long),
            'targets': torch.tensor(self.targets[index], dtype=torch.float)
        }

# Creating the dataset and dataloader for the neural network
MAX_LEN = 256
train_size = 0.8
train_dataset = train_df.sample(frac=train_size,random_state=7)
valid_dataset = train_df.drop(train_dataset.index).reset_index(drop=True)
train_dataset = train_dataset.reset_index(drop=True)


print("FULL Dataset: {}".format(train_df.shape))
print("TRAIN Dataset: {}".format(train_dataset.shape))
print("VALID Dataset: {}".format(valid_dataset.shape))
print("TEST Dataset: {}".format(test_df.shape))

train_set = CustomDataset(train_dataset, tokenizer, MAX_LEN)
valid_set = CustomDataset(valid_dataset, tokenizer, MAX_LEN)
test_set = CustomDataset(test_df, tokenizer, MAX_LEN)
TRAIN_BATCH_SIZE = 32
VALID_BATCH_SIZE = 16
TEST_BATCH_SIZE = 16
train_params = {'batch_size': TRAIN_BATCH_SIZE,
                'shuffle': True}

valid_params = {'batch_size': VALID_BATCH_SIZE,
                'shuffle': True}

test_params = {'batch_size': TEST_BATCH_SIZE,
                'shuffle': False}

train_loader = DataLoader(train_set, **train_params)
valid_loader = DataLoader(valid_set, **valid_params)
test_loader = DataLoader(test_set, **test_params)


# Creating the customized model, by adding a drop out and a dense layer on top of distil bert to get the final output for the model.
class BERTClass(torch.nn.Module):
    def __init__(self):
        super(BERTClass, self).__init__()
        self.config = BertConfig.from_pretrained('./emb/bert-mini/bert_config.json', output_hidden_states=True)
        self.l1 = BertModel.from_pretrained('./emb/bert-mini/pytorch_model.bin', config=self.config)
        self.bilstm1 = torch.nn.LSTM(512, 64, 1, bidirectional=True)
        self.l2 = torch.nn.Linear(128, 64)
        self.a1 = torch.nn.ReLU()
        self.l3 = torch.nn.Dropout(0.3)
        self.l4 = torch.nn.Linear(64, 14)

    def forward(self, ids, mask, token_type_ids):
        sequence_output, pooler_output, hidden_states = self.l1(ids, attention_mask=mask, token_type_ids=token_type_ids)
        # [bs, 200, 256]  [bs,256]
        bs = len(sequence_output)
        h12 = hidden_states[-1][:, 0].view(1, bs, 256)
        h11 = hidden_states[-2][:, 0].view(1, bs, 256)
        concat_hidden = torch.cat((h12, h11), 2)
        x, _ = self.bilstm1(concat_hidden)
        x = self.l2(x.view(bs, 128))
        x = self.a1(x)
        x = self.l3(x)
        output = self.l4(x)
        return output


net = BERTClass()
net.to(device)
# 超参数设置
lr, num_epochs = 1e-5, 30
criterion = torch.nn.CrossEntropyLoss()  # 选择损失函数
optimizer = torch.optim.Adam(net.parameters(), lr=lr)  # 选择优化器
def evaluate_accuracy(data_iter, net, device=torch.device('cpu')):
    """Evaluate accuracy of a model on the given data set."""
    acc_sum, n = torch.tensor([0], dtype=torch.float32,device=device), 0
    y_pred_, y_true_ = [], []
    for data in tqdm(data_iter):
        # If device is the GPU, copy the data to the GPU.
        ids = data['ids'].to(device, dtype = torch.long)
        mask = data['mask'].to(device, dtype = torch.long)
        token_type_ids = data['token_type_ids'].to(device, dtype = torch.long)
        targets = data['targets'].to(device, dtype = torch.float)
        net.eval()
        y_hat_ = net(ids, mask, token_type_ids)
        with torch.no_grad():
            targets = targets.long()
            # [[0.2 ,0.4 ,0.5 ,0.6 ,0.8] ,[ 0.1,0.2 ,0.4 ,0.3 ,0.1]] => [ 4 , 2 ]
            acc_sum += torch.sum((torch.argmax(y_hat_, dim=1) == targets))
            y_pred_.extend(torch.argmax(y_hat_, dim=1).cpu().numpy().tolist())
            y_true_.extend(targets.cpu().numpy().tolist())
            n += targets.shape[0]
    valid_f1 = metrics.f1_score(y_true_, y_pred_, average='macro')
    return acc_sum.item()/n, valid_f1


def train(epoch, train_iter, test_iter, criterion, num_epochs, optimizer, device):
    print('training on', device)
    net.to(device)
    best_test_f1 = 0
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.5)  # 设置学习率下降策略
    #     scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=5, eta_min=2e-06)  # 余弦退火
    for epoch in range(num_epochs):
        train_l_sum = torch.tensor([0.0], dtype=torch.float32, device=device)
        train_acc_sum = torch.tensor([0.0], dtype=torch.float32, device=device)
        n, start = 0, time.time()
        y_pred, y_true = [], []
        for data in tqdm(train_iter):
            net.train()
            optimizer.zero_grad()
            ids = data['ids'].to(device, dtype=torch.long)
            mask = data['mask'].to(device, dtype=torch.long)
            token_type_ids = data['token_type_ids'].to(device, dtype=torch.long)
            targets = data['targets'].to(device, dtype=torch.float)
            y_hat = net(ids, mask, token_type_ids)
            loss = criterion(y_hat, targets.long())
            loss.backward()
            optimizer.step()

            with torch.no_grad():
                targets = targets.long()
                train_l_sum += loss.float()
                train_acc_sum += (torch.sum((torch.argmax(y_hat, dim=1) == targets))).float()
                y_pred.extend(torch.argmax(y_hat, dim=1).cpu().numpy().tolist())
                y_true.extend(targets.cpu().numpy().tolist())
                n += targets.shape[0]
        valid_acc, valid_f1 = evaluate_accuracy(test_iter, net, device)
        train_acc = train_acc_sum / n
        train_f1 = metrics.f1_score(y_true, y_pred, average='macro')
        print('epoch %d, loss %.4f, train acc %.3f, valid acc %.3f, '
              'train f1 %.3f, valid f1 %.3f, time %.1f sec'
              % (epoch + 1, train_l_sum / n, train_acc, valid_acc,
                 train_f1, valid_f1, time.time() - start))
        if valid_f1 > best_test_f1:
            print('find best! save at model/best.pth')
            best_test_f1 = valid_f1
            torch.save(net.state_dict(), 'model/best.pth')
        scheduler.step()  # 更新学习率
train(net,train_loader, valid_loader, criterion, num_epochs, optimizer, device)

def model_predict(net, test_iter):
    # 预测模型
    preds_list = []
    print('加载最优模型')
    net.load_state_dict(torch.load('model/best.pth'))
    net = net.to(device)
    print('inference测试集')
    with torch.no_grad():
        for data in tqdm(test_iter):
            ids = data['ids'].to(device, dtype=torch.long)
            mask = data['mask'].to(device, dtype=torch.long)
            token_type_ids = data['token_type_ids'].to(device, dtype=torch.long)
            batch_preds = list(net(ids, mask, token_type_ids).argmax(dim=1).cpu().numpy())
            for preds in batch_preds:
                preds_list.append(preds)
    return preds_list
preds_list = model_predict(net, test_loader)


