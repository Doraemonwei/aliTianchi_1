import torch
from data_process.data_process import my_loader
from model.bert_Model import MBertModel
from tqdm import tqdm


def bert_model_train():
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print('模型训练使用的是：{}'.format(device))
    loader = my_loader(r"IMCS-DAC/IMCS-DAC_train.json", batch_size=64, is_train=True)
    model = MBertModel().to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    criterion = torch.nn.CrossEntropyLoss()
    model.train()
    Epochs = 4
    for epoch in range(Epochs):
        print('目前正在跑第{:.0f}/{:.0f}轮'.format(epoch + 1, Epochs + 1))
        for i, (input_ids, attention_mask, token_type_ids, labels) in tqdm(enumerate(loader), total=len(loader)):
            input_ids = input_ids.to(device)
            attention_mask = attention_mask.to(device)
            token_type_ids = token_type_ids.to(device)
            labels = labels.to(device)
            out = model(input_ids, attention_mask, token_type_ids)
            loss = criterion(out, labels)
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
            # if i == 2: break

        # 一轮epoch训练完毕，检测在验证集上的准确率
        with torch.no_grad():
            dev_dataloader = my_loader(r"IMCS-DAC/IMCS-DAC_dev.json", is_train=False)
            # 计算每一个batch上的准确率然后除以batch的数目
            print("正在验证测试集准确率......")
            correct_sum = 0
            for i, (input_ids, attention_mask, token_type_ids, labels) in tqdm(enumerate(dev_dataloader),
                                                                               total=len(dev_dataloader)):
                input_ids = input_ids.to(device)
                attention_mask = attention_mask.to(device)
                token_type_ids = token_type_ids.to(device)
                labels = labels.to(device)
                out = model(input_ids, attention_mask, token_type_ids).argmax(dim=1)
                correct_sum += (out == labels).sum().item()

            this_epoch_dev_acc = (correct_sum / (i + 1))
            print('第{}/{}轮Epoch之后再验证集上的准确率为{:.2f}'.format(epoch + 1, Epochs + 1, this_epoch_dev_acc))
        # 保存模型
        torch.save(model, 'trainedModel/Bert_{}_{:.2f}'.format(epoch + 1, this_epoch_dev_acc))


if __name__ == '__main__':
    bert_model_train()
