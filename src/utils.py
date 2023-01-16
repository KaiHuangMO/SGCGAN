import sys
import os
import torch
import random
import math

from sklearn.utils import shuffle
from sklearn.metrics import f1_score
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
from sklearn.neighbors import NearestNeighbors
import torch.nn as nn
import numpy as np
import torch.nn.functional as F
from sklearn.preprocessing import label_binarize
import copy
from sklearn.metrics import roc_auc_score

def evaluate(dataCenter, ds, graphSage, classification, device, val_macro, val_micro,val_auc, val_roc_auc, test_macro,test_micro,test_auc, test_roc_auc, name, cur_epoch, labels_roc_auc, mulclass = False):
    # todo evaluate 记得修改回来
    test_nodes = getattr(dataCenter, ds + '_test')
    val_nodes = getattr(dataCenter, ds + '_val')
    labels = getattr(dataCenter, ds + '_labels')
    models = [graphSage, classification]
    params = []
    for model in models:
        for param in model.parameters():
            if param.requires_grad:
                param.requires_grad = False
                params.append(param)
    embs, _, _, _, _ = graphSage(val_nodes)
    logists = classification(embs)

    #_, predicts = torch.max(logists, 1)
    if mulclass:
        logists[logists >= .5] = 1
        logists[logists < .5] = 0
        predicts = logists
    else:
        _, predicts = torch.max(logists, 1)


    labels_val = labels[val_nodes]
    assert len(labels_val) == len(predicts)
    comps = zip(labels_val, predicts.data)

    vali_f1_micro = f1_score(labels_val, predicts.cpu().data, average="micro")  # macro  accuracy_score
    #vali_f1_micro = accuracy_score(labels_val, predicts.cpu().data)
    vali_f1_macro = f1_score(labels_val, predicts.cpu().data, average="macro")
    vali_auc = accuracy_score(labels_val, predicts.cpu().data)

    #vali_roc_auc = roc_auc_score(labels_val, F.softmax(logists.cpu().detach(), dim=-1).detach(), average='macro',
    #                              multi_class='ovr')  ####!!!!

    z = 1
    #labels_val_roc_auc = label_binarize(labels_val, classes=labels_roc_auc)
    #labels_predict_val_roc_auc = label_binarize(predicts.cpu().data, classes=labels_roc_auc)
    #vali_roc_auc = roc_auc_score(labels_val_roc_auc, labels_predict_val_roc_auc, average='macro', multi_class='ovr')


    if vali_f1_micro > val_micro:
        # if 1:
        val_micro = vali_f1_micro
        embs, _, _, _, _ = graphSage(test_nodes)
        logists = classification(embs)
        #_, predicts = torch.max(logists, 1)
        #zero = torch.zeros_like(logists)
        #one = torch.ones_like(logists)
        #logists = torch.where(logists < 0.5, zero, logists)
        #logists = torch.where(logists >= 0.5, one, logists)

        if mulclass:
            logists[logists >= .5] = 1
            logists[logists < .5] = 0
            logists = torch.where(torch.isnan(logists), torch.full_like(logists, 0), logists)
            predicts = logists
        else:
            _, predicts = torch.max(logists, 1)


        labels_test = labels[test_nodes]
        assert len(labels_test) == len(predicts)
        comps = zip(labels_test, predicts.data)
        pre = predicts.cpu().data.numpy()

        test_micro = f1_score(labels_test, pre, average="micro")
        #test_micro = accuracy_score(labels_test, predicts.cpu().data)

    print('cur_epoch:', cur_epoch)
    # print("Validation micro F1:", vali_f1)
    # print("Validation macro F1:", vali_f1)
    # print("Validation confusion_matrix:", confusion_matrix(labels_val, predicts.cpu().data))
    if vali_f1_macro > val_macro:
        # if 1:
        val_macro = vali_f1_macro
        embs, _, _, _, _ = graphSage(test_nodes)
        logists = classification(embs)
        #_, predicts = torch.max(logists, 1)
        if mulclass:
            logists[logists >= .5] = 1
            logists[logists < .5] = 0
            logists = torch.where(torch.isnan(logists), torch.full_like(logists, 0), logists)
            predicts = logists
        else:
            _, predicts = torch.max(logists, 1)

        labels_test = labels[test_nodes]
        assert len(labels_test) == len(predicts)
        comps = zip(labels_test, predicts.data)
        pre = predicts.cpu().data

        test_macro = f1_score(labels_test, pre, average="macro")

    if vali_auc > val_auc:
        # if 1:
        val_auc = vali_auc
        embs, _, _, _, _  = graphSage(test_nodes)
        logists = classification(embs)
        #_, predicts = torch.max(logists, 1)
        if mulclass:
            logists[logists >= .5] = 1
            logists[logists < .5] = 0
            logists = torch.where(torch.isnan(logists), torch.full_like(logists, 0), logists)
            predicts = logists
        else:
            _, predicts = torch.max(logists, 1)

        labels_test = labels[test_nodes]
        assert len(labels_test) == len(predicts)
        comps = zip(labels_test, predicts.data)
        pre = predicts.cpu().data

        test_auc = accuracy_score(labels_test, pre)
    val_roc_auc = .0
    test_roc_auc = .0
    if 0:
        # if 1:
        val_roc_auc = vali_roc_auc
        embs, _, _, _, _  = graphSage(test_nodes)
        logists = classification(embs)
        _, predicts = torch.max(logists, 1)
        labels_test = labels[test_nodes]
        assert len(labels_test) == len(predicts)
        comps = zip(labels_test, predicts.data)

        labels_test_roc_auc = label_binarize(labels_test, classes=labels_roc_auc)
        labels_predict_test_roc_auc = label_binarize(predicts.cpu().data, classes=labels_roc_auc)
        test_roc_auc = roc_auc_score(labels_test_roc_auc, labels_predict_test_roc_auc, average='macro', multi_class='ovr')
        test_roc_auc = roc_auc_score(labels_test, F.softmax(logists.cpu().detach(), dim=-1).detach(), average='macro',
                                      multi_class='ovr')


    for param in params:
        param.requires_grad = True
    return val_macro, val_micro,val_auc, val_roc_auc, test_macro,test_micro, test_auc, test_roc_auc

def evaluateEnhanced(dataCenter, ds, graphSageEnhanced, classification, device,val_macro, val_micro,val_auc, val_roc_auc, test_macro,test_micro,test_auc, test_roc_auc, name, cur_epoch,labels_roc_auc):
    test_nodes = getattr(dataCenter, ds + '_test')
    val_nodes = getattr(dataCenter, ds + '_val')
    labels = getattr(dataCenter, ds + '_labels')
    models = [graphSageEnhanced, classification]
    params = []
    for model in models:
        for param in model.parameters():
            if param.requires_grad:
                param.requires_grad = False
                params.append(param)
    embs, _, _, _ = graphSageEnhanced(val_nodes)
    logists = classification(embs)
    _, predicts = torch.max(logists, 1)
    labels_val = labels[val_nodes]
    assert len(labels_val) == len(predicts)
    comps = zip(labels_val, predicts.data)
    vali_f1_micro = f1_score(labels_val, predicts.cpu().data, average="micro")  # macro
    vali_f1_macro = f1_score(labels_val, predicts.cpu().data, average="macro")
    vali_auc = accuracy_score(labels_val, predicts.cpu().data)


    labels_val_roc_auc = label_binarize(labels_val, classes=labels_roc_auc)
    labels_predict_val_roc_auc = label_binarize(predicts.cpu().data, classes=labels_roc_auc)
    vali_roc_auc = roc_auc_score(labels_val_roc_auc, labels_predict_val_roc_auc, average='macro', multi_class='ovr') #average='macro', multi_class='ovr')

    vali_roc_auc = roc_auc_score(labels_val, F.softmax(logists.cpu().detach(), dim=-1).detach(), average='macro',
                                  multi_class='ovr')  ####!!!!

    if vali_f1_micro > val_micro:
        # if 1:
        val_micro = vali_f1_micro
        embs, _, _, _ = graphSageEnhanced(test_nodes)
        logists = classification(embs)
        _, predicts = torch.max(logists, 1)
        labels_test = labels[test_nodes]
        assert len(labels_test) == len(predicts)
        comps = zip(labels_test, predicts.data)

        test_micro = f1_score(labels_test, predicts.cpu().data, average="micro")

    print('cur_epoch:', cur_epoch)
    #print("Validation micro F1:", vali_f1)
    #print("Validation macro F1:", vali_f1)
    #print("Validation confusion_matrix:", confusion_matrix(labels_val, predicts.cpu().data))
    if vali_f1_macro > val_macro:
    #if 1:
        val_macro = vali_f1_macro
        embs, _, _, _ = graphSageEnhanced(test_nodes)
        logists = classification(embs)
        _, predicts = torch.max(logists, 1)
        labels_test = labels[test_nodes]
        assert len(labels_test) == len(predicts)
        comps = zip(labels_test, predicts.data)

        '''
        test_f1 = f1_score(labels_test, predicts.cpu().data, average="micro")
        print("Test micro F1:", test_f1)
        test_f1 = f1_score(labels_test, predicts.cpu().data, average="macro")
        print("Test macro F1:", test_f1)

        print("Test confusion_matrix:", confusion_matrix(labels_test, predicts.cpu().data))
        '''
        test_macro = f1_score(labels_test, predicts.cpu().data, average="macro")

    if vali_auc > val_auc:
        # if 1:
        val_auc = vali_auc
        embs, _, _, _ = graphSageEnhanced(test_nodes)
        logists = classification(embs)
        _, predicts = torch.max(logists, 1)
        labels_test = labels[test_nodes]
        assert len(labels_test) == len(predicts)
        comps = zip(labels_test, predicts.data)

        test_auc = accuracy_score(labels_test, predicts.cpu().data)

    if vali_roc_auc > val_roc_auc:
        # if 1:
        val_roc_auc = vali_roc_auc
        embs, _, _, _ = graphSageEnhanced(test_nodes)
        logists = classification(embs)
        _, predicts = torch.max(logists, 1)
        labels_test = labels[test_nodes]
        assert len(labels_test) == len(predicts)
        comps = zip(labels_test, predicts.data)

        labels_test_roc_auc = label_binarize(labels_test, classes=labels_roc_auc)
        labels_predict_test_roc_auc = label_binarize(predicts.cpu().data, classes=labels_roc_auc)
        test_roc_auc = roc_auc_score(labels_test_roc_auc, labels_predict_test_roc_auc, average='macro', multi_class='ovr')
        test_roc_auc = roc_auc_score(labels_test, F.softmax(logists.cpu().detach(), dim=-1).detach(), average='macro',
                                     multi_class='ovr')

    for param in params:
        param.requires_grad = True
    return val_macro, val_micro,val_auc, val_roc_auc, test_macro,test_micro, test_auc, test_roc_auc



def mulevaluateEnhanced(dataCenter, ds, graphSageEnhanced, classification, device,val_macro, val_micro,val_auc, val_roc_auc, test_macro,test_micro,test_auc, test_roc_auc, name, cur_epoch,labels_roc_auc):
    test_nodes = getattr(dataCenter, ds + '_test')
    val_nodes = getattr(dataCenter, ds + '_val')
    labels = getattr(dataCenter, ds + '_labels')
    models = [graphSageEnhanced, classification]
    params = []
    for model in models:
        for param in model.parameters():
            if param.requires_grad:
                param.requires_grad = False
                params.append(param)
    embs, _, _, _ = graphSageEnhanced(val_nodes, needIndx=False)
    logists = classification(embs)

    logists[logists >= .5] = 1
    logists[logists < .5] = 0
    logists = torch.where(torch.isnan(logists), torch.full_like(logists, 0), logists)

    predicts = logists

    labels_val = labels[val_nodes]
    assert len(labels_val) == len(predicts)
    comps = zip(labels_val, predicts.data)
    vali_f1_micro = f1_score(labels_val, predicts.cpu().data, average="micro")  # macro
    vali_f1_macro = f1_score(labels_val, predicts.cpu().data, average="macro")
    vali_auc = accuracy_score(labels_val, predicts.cpu().data)

    embs, _, _, _ = graphSageEnhanced(test_nodes, needIndx=False)
    logists = classification(embs)

    logists[logists >= .5] = 1
    logists[logists < .5] = 0
    logists = torch.where(torch.isnan(logists), torch.full_like(logists, 0), logists)

    if vali_f1_micro > val_micro:
        # if 1:
        val_micro = vali_f1_micro
        predicts = logists

        labels_test = labels[test_nodes]
        assert len(labels_test) == len(predicts)

        test_micro = f1_score(labels_test, predicts.cpu().data, average="micro")
        print ('val_micro ' + str(val_micro) + '; test_micro' + str(test_micro))

    print('cur_epoch:', cur_epoch)

    if vali_f1_macro > val_macro:
    #if 1:
        val_macro = vali_f1_macro
        predicts = logists

        labels_test = labels[test_nodes]
        assert len(labels_test) == len(predicts)

        test_macro = f1_score(labels_test, predicts.cpu().data, average="macro")
        print('val_macro ' + str(val_micro) + '; test_macro' + str(test_macro))

    if vali_auc > val_auc:
        # if 1:
        val_auc = vali_auc
        predicts = logists

        labels_test = labels[test_nodes]
        assert len(labels_test) == len(predicts)

        test_auc = accuracy_score(labels_test, predicts.cpu().data)
        print (predicts.cpu().data)
        print('val_auc ' + str(val_auc) + '; test_auc' + str(test_auc))

    test_roc_auc = 0.
    if 0:
        # if 1:
        val_roc_auc = vali_roc_auc
        embs, _, _, _ = graphSageEnhanced(test_nodes)
        logists = classification(embs)
        _, predicts = torch.max(logists, 1)
        labels_test = labels[test_nodes]
        assert len(labels_test) == len(predicts)
        comps = zip(labels_test, predicts.data)

        labels_test_roc_auc = label_binarize(labels_test, classes=labels_roc_auc)
        labels_predict_test_roc_auc = label_binarize(predicts.cpu().data, classes=labels_roc_auc)
        test_roc_auc = roc_auc_score(labels_test_roc_auc, labels_predict_test_roc_auc, average='macro', multi_class='ovr')
        test_roc_auc = roc_auc_score(labels_test, F.softmax(logists.cpu().detach(), dim=-1).detach(), average='macro',
                                     multi_class='ovr')

    for param in params:
        param.requires_grad = True
    return val_macro, val_micro,val_auc, val_roc_auc, test_macro,test_micro, test_auc, test_roc_auc






def get_gnn_embeddings(graphsa, dataCenter, ds):
    print('Loading embeddings from trained GraphSAGE model.')
    features = np.zeros((len(getattr(dataCenter, ds + '_labels')), graphsa.out_size))
    nodes = np.arange(len(getattr(dataCenter, ds + '_labels'))).tolist()  # 这里主要靠 dataCenter 然后进入 graphSage 里面搞
    # todo 这里 nodes 有错吗？

    b_sz = 500
    batches = math.ceil(len(nodes) / b_sz)
    embs = []
    for index in range(batches):
        nodes_batch = nodes[index * b_sz:(index + 1) * b_sz]  # 获得 node
        embs_batch, _, _, _,_ = graphsa(nodes_batch)
        assert len(embs_batch) == len(nodes_batch)
        embs.append(embs_batch)
    # if ((index+1)*b_sz) % 10000 == 0:
    #     print(f'Dealed Nodes [{(index+1)*b_sz}/{len(nodes)}]')

    assert len(embs) == batches
    embs = torch.cat(embs, 0)
    assert len(embs) == len(nodes)
    print('Embeddings loaded.')
    return embs.detach()


def get_gnn_enhanced_embeddings(graphSage, minNodes, minEnhanced):
    # todo 1. 这里有一个问题 是到底聚合多少？
    # todo 2. 先返回一个
    # 聚合完 self_feats =
    # 用 SageLayer 就好 self + agg 一层的话 无所谓了
    # aggregate_feats = torch.mean(minEnhanced, dim=1)

    for minE in minEnhanced:
        aggregate_feats = torch.mean(minE, dim=1)
        # 如果为1的haul 是 row_feature
        with torch.no_grad():
            cur_hidden_embs = graphSage.sage_layer1(self_feats=minNodes,
                                                    aggregate_feats=aggregate_feats)

    return cur_hidden_embs


def train_classification(dataCenter, graphSage, classification, ds, device, max_vali_f1, name, epochs=8):
    print('Training Classification ...')
    c_optimizer = torch.optim.SGD(classification.parameters(), lr=0.5)
    # train classification, detached from the current graph
    # classification.init_params()
    b_sz = 50
    train_nodes = getattr(dataCenter, ds + '_train')
    labels = getattr(dataCenter, ds + '_labels')
    features = get_gnn_embeddings(graphSage, dataCenter, ds)  # 这里 graphSage 就是 gnn todo
    # 这玩意知识get emb
    for epoch in range(epochs):
        train_nodes = shuffle(train_nodes)
        batches = math.ceil(len(train_nodes) / b_sz)
        visited_nodes = set()
        for index in range(batches):
            nodes_batch = train_nodes[index * b_sz:(index + 1) * b_sz]
            visited_nodes |= set(nodes_batch)
            labels_batch = labels[nodes_batch]
            embs_batch = features[nodes_batch]

            logists = classification(embs_batch)
            loss = -torch.sum(logists[range(logists.size(0)), labels_batch], 0)
            loss /= len(nodes_batch)
            # print('Epoch [{}/{}], Step [{}/{}], Loss: {:.4f}, Dealed Nodes [{}/{}] '.format(epoch+1, epochs, index, batches, loss.item(), len(visited_nodes), len(train_nodes)))

            loss.backward()

            nn.utils.clip_grad_norm_(classification.parameters(), 5)
            c_optimizer.step()
            c_optimizer.zero_grad()

        max_vali_f1 = evaluate(dataCenter, ds, graphSage, classification, device, max_vali_f1, name, epoch)
    return classification, max_vali_f1


def train_classification_smote_write(dataCenter, graphSage, classification, ds, device, max_vali_f1, name, epochs=8):
    print('Training Classification ...')
    c_optimizer = torch.optim.SGD(classification.parameters(), lr=0.5)
    # train classification, detached from the current graph
    # classification.init_params()
    b_sz = 50
    train_nodes = getattr(dataCenter, ds + '_train')
    labels = getattr(dataCenter, ds + '_labels')
    features = get_gnn_embeddings(graphSage, dataCenter, ds)  # 这里 graphSage 就是 gnn todo
    # 这玩意知识get emb
    writeFile = open('eSMOTE.txt', 'w')
    for epoch in range(epochs):
        train_nodes = shuffle(train_nodes)
        batches = math.ceil(len(train_nodes) / b_sz)
        visited_nodes = set()
        for index in range(batches):
            nodes_batch = train_nodes[index * b_sz:(index + 1) * b_sz]
            visited_nodes |= set(nodes_batch)
            labels_batch = labels[nodes_batch]
            embs_batch = features[nodes_batch]

            logists = classification(embs_batch)
            loss = -torch.sum(logists[range(logists.size(0)), labels_batch], 0)
            loss /= len(nodes_batch)
            # print('Epoch [{}/{}], Step [{}/{}], Loss: {:.4f}, Dealed Nodes [{}/{}] '.format(epoch+1, epochs, index, batches, loss.item(), len(visited_nodes), len(train_nodes)))

            loss.backward()

            nn.utils.clip_grad_norm_(classification.parameters(), 5)
            c_optimizer.step()
            c_optimizer.zero_grad()
            if index == 1:
                z = 1
                clone_detach_embeds = embs_batch.clone()
                write_embeds = clone_detach_embeds.cpu().detach().numpy()
                for i in range(0, len(labels_batch)):
                    ls = str(labels_batch[i]) + '\t'
                    if nodes_batch[i] in dataCenter.smote_idx:  # smote 加入的
                        ls = '7\t'
                    thisEmbd = write_embeds[i]
                    for j in thisEmbd:
                        ls = ls + str(j) + '\t'
                    writeFile.write(ls.strip() + '\n')
                    writeFile.flush()


        max_vali_f1 = evaluate(dataCenter, ds, graphSage, classification, device, max_vali_f1, name, epoch)
    writeFile.close()
    return classification, max_vali_f1

def train_classification_enhanced(dataCenter, graphSage, classification, ds, device, max_vali_f1, name, minNodes,
                                  minEnhanced, targetLabel, epochs=8):
    '''
    多的东西 应该就是 节点 和 节点 邻居 以方便进行编码
    :param dataCenter:
    :param graphSage:
    :param classification:
    :param ds:
    :param device:
    :param max_vali_f1:
    :param name:
    :param minIdex:
    :param minEnhanced:
    :param epochs:
    :return:
    '''
    print('Training Classification ...')
    c_optimizer = torch.optim.SGD(classification.parameters(), lr=0.5)

    # train classification, detached from the current graph
    # classification.init_params()
    b_sz = 50
    train_nodes = getattr(dataCenter, ds + '_train')
    labels = getattr(dataCenter, ds + '_labels')
    features = get_gnn_embeddings(graphSage, dataCenter, ds)

    enhanced_feature = get_gnn_enhanced_embeddings(graphSage, minNodes, minEnhanced)  # # todo 这里进行多的编码
    random.shuffle(enhanced_feature)
    enhanced_batch = math.ceil(enhanced_feature.size(0) / b_sz)  # 每个batch 加入多少
    enhanced_label_batch = []
    for i in range(0, enhanced_batch):
        enhanced_label_batch.append(targetLabel)
    # enhanced_label_batch = torch.LongTensor(enhanced_label_batch).cuda()

    for epoch in range(epochs):
        train_nodes = shuffle(train_nodes)
        batches = math.ceil(len(train_nodes) / b_sz)
        visited_nodes = set()
        for index in range(batches):
            nodes_batch = train_nodes[index * b_sz:(index + 1) * b_sz]
            visited_nodes |= set(nodes_batch)
            labels_batch = list(labels[nodes_batch])
            embs_batch = features[nodes_batch]
            # 加入增强 特征
            t = enhanced_feature[index * enhanced_batch:(index + 1) * enhanced_batch]
            embs_batch = torch.cat([embs_batch, t], dim=0)  # todo 主要还是这个的问题
            labels_batch.extend(enhanced_label_batch)

            logists = classification(embs_batch)
            loss = -torch.sum(logists[range(logists.size(0)), labels_batch], 0)
            loss /= len(embs_batch)
            # print('Epoch [{}/{}], Step [{}/{}], Loss: {:.4f}, Dealed Nodes [{}/{}] '.format(epoch+1, epochs, index, batches, loss.item(), len(visited_nodes), len(train_nodes)))

            loss.backward()

            nn.utils.clip_grad_norm_(classification.parameters(), 5)
            c_optimizer.step()
            c_optimizer.zero_grad()

        max_vali_f1 = evaluate(dataCenter, ds, graphSage, classification, device, max_vali_f1, name, epoch)
    return classification, max_vali_f1




def train_classification_smote(dataCenter, graphSage, classification, ds, device, max_vali_f1, name, enhancedEmb,
                                   targetLabel, epochs=8):
    '''

    :param dataCenter:
    :param graphSage:
    :param classification:
    :param ds:
    :param device:
    :param max_vali_f1:
    :param name:
    :param enhancedEmb:  生成的Emb
    :param targetLabel:  生成的label
    :param epochs:
    :return:
    '''
    print('Training Classification ...')
    c_optimizer = torch.optim.SGD(classification.parameters(), lr=0.5)

    # train classification, detached from the current graph
    # classification.init_params()
    b_sz = 50
    train_nodes = getattr(dataCenter, ds + '_train')
    labels = getattr(dataCenter, ds + '_labels')
    features = get_gnn_embeddings(graphSage, dataCenter, ds)

    enhanced_feature = torch.tensor( enhancedEmb ).to(device)  # # todo 这里进行多的编码
    random.shuffle(enhanced_feature)

    batchesNum = math.ceil(len(train_nodes) / b_sz)

    enhanced_batch = math.ceil(enhanced_feature.size(0) / batchesNum)  # 每个batch 加入多少
    enhanced_label_batch = []
    for i in range(0, enhanced_batch):
        enhanced_label_batch.append(targetLabel)
    # enhanced_label_batch = torch.LongTensor(enhanced_label_batch).cuda()
    writeFile = open('embSMOTE_wiki.txt', 'w')
    for epoch in range(1):
        train_nodes = shuffle(train_nodes)
        batches = math.ceil(len(train_nodes) / b_sz)
        print (batches)


        visited_nodes = set()
        for index in range(batches):
            print (index)
            if index == 49:
                z = 1
            nodes_batch = train_nodes[index * b_sz:(index + 1) * b_sz]
            visited_nodes |= set(nodes_batch)
            labels_batch = list(labels[nodes_batch])
            embs_batch = features[nodes_batch]
            # 加入增强 特征
            t = enhanced_feature[index * enhanced_batch:(index + 1) * enhanced_batch]

            if t.size(0) == len(enhanced_label_batch):
                embs_batch = torch.cat([embs_batch, t], dim=0)  # todo 主要还是这个的问题
                labels_batch.extend(enhanced_label_batch)

            logists = classification(embs_batch)
            loss = -torch.sum(logists[range(logists.size(0)), labels_batch], 0)
            loss /= len(embs_batch)
            # print('Epoch [{}/{}], Step [{}/{}], Loss: {:.4f}, Dealed Nodes [{}/{}] '.format(epoch+1, epochs, index, batches, loss.item(), len(visited_nodes), len(train_nodes)))

            loss.backward()

            nn.utils.clip_grad_norm_(classification.parameters(), 5)
            c_optimizer.step()
            c_optimizer.zero_grad()
            #print (len(labels_batch))
            #print (len(nodes_batch))
            #exit()
            if 1:
                z = 1
                clone_detach_embeds = embs_batch.clone()
                write_embeds = clone_detach_embeds.cpu().detach().numpy()
                for i in range(0, len(labels_batch)):
                    ls = str(labels_batch[i]) + '\t'
                    if i > len(nodes_batch) - 1:
                        ls = '100\t'
                    thisEmbd = write_embeds[i]
                    for j in thisEmbd:
                        ls = ls + str(j) + '\t'
                    writeFile.write(ls.strip() + '\n')
                    writeFile.flush()
    writeFile.close()
        #max_vali_f1 = evaluate(dataCenter, ds, graphSage, classification, device, max_vali_f1, name, epoch)
    return classification




def Multrain_classification_smote(dataCenter, graphSage, classification, ds, device, max_vali_f1, name, enhancedEmb,
                                   targetLabel, recordLabel, unsupervised_loss, epochs=8, writeFileFlag = False):
    '''

    :param dataCenter:
    :param graphSage:
    :param classification:
    :param ds:
    :param device:
    :param max_vali_f1:
    :param name:
    :param enhancedEmb:  生成的Emb
    :param targetLabel:  生成的label
    :param epochs:
    :return:
    '''
    print('Training Classification ...')
    c_optimizer = torch.optim.SGD(classification.parameters(), lr=0.5)

    # train classification, detached from the current graph
    # classification.init_params()
    b_sz = 50
    train_nodes = getattr(dataCenter, ds + '_train')
    labels = getattr(dataCenter, ds + '_labels')
    #features = get_gnn_embeddings(graphSage, dataCenter, ds)

    shuffleRecord = []
    for ii in range(0, len(enhancedEmb)):
        shuffleRecord.append(ii)
    random.shuffle(shuffleRecord)
    enhancedEmb1 = [enhancedEmb[i] for i in shuffleRecord]
    enhancedLabel1 = [recordLabel[i] for i in shuffleRecord]

    enhanced_feature = torch.tensor( enhancedEmb1 ).to(device)  # # todo 这里进行多的编码

    batchesNum = math.ceil(len(train_nodes) / b_sz)

    enhanced_batch = math.ceil(enhanced_feature.size(0) / batchesNum)  # 每个batch 加入多少

    enhanced_label = torch.FloatTensor(enhancedLabel1).cuda()
    writeFile = open('embSMOTE_ppi.txt', 'w')

    # enhanced_label_batch = torch.LongTensor(enhanced_label_batch).cuda()
    for epoch in range(1):
        train_nodes = shuffle(train_nodes)
        batches = math.ceil(len(train_nodes) / b_sz)
        print (batches)


        visited_nodes = set()
        for index in range(batches):
            print (index)
            if index == 49:
                z = 1
            nodes_batch = train_nodes[index * b_sz:(index + 1) * b_sz]
            #nodes_batch = np.asarray(list(unsupervised_loss.extend_nodes(nodes_batch, num_neg=100)))

            visited_nodes |= set(nodes_batch)
            labels_batch = list(labels[nodes_batch])
            labels_batch = torch.FloatTensor(labels_batch).cuda()
            #embs_batch = features[nodes_batch]
            embs_batch, _, _, _, _ = graphSage(nodes_batch)  # todo 这里也要加强 还有 labels_batch

            # 加入增强 特征
            t = enhanced_feature[index * enhanced_batch:(index + 1) * enhanced_batch]
            t2 = enhanced_label[index * enhanced_batch:(index + 1) * enhanced_batch]

            if 1:
                embs_batch = torch.cat([embs_batch, t], dim=0)  # todo 主要还是这个的问题
                enhanced_label_batch = torch.cat([labels_batch, t2], dim=0)

            logists = classification(embs_batch)

            criterion = nn.BCELoss()

            loss = criterion(logists, enhanced_label_batch)

            # loss_sup = -torch.sum(logists[range(logists.size(0)), labels_batch], 0) # todo
            # loss_sup /= len(nodes_batch)
            # unsuperivsed learning

            # print('Epoch [{}/{}], Step [{}/{}], Loss: {:.4f}, Dealed Nodes [{}/{}] '.format(epoch+1, epochs, index, batches, loss.item(), len(visited_nodes), len(train_nodes)))

            loss.backward()

            nn.utils.clip_grad_norm_(classification.parameters(), 5)
            c_optimizer.step()
            c_optimizer.zero_grad()
            #print (len(labels_batch))
            #print (len(nodes_batch))
            #exit()
            if writeFileFlag:
                z = 1
                clone_detach_embeds = embs_batch.clone()
                write_embeds = clone_detach_embeds.cpu().detach().numpy()
                for i in range(0, len(enhanced_label_batch)):
                    writeLabel = 0
                    if enhanced_label_batch[i][86] == 0:
                        writeLabel = 1
                    if i > len(nodes_batch) - 1:
                        writeLabel = 2
                    ls = str(writeLabel) + '\t'
                    thisEmbd = write_embeds[i]
                    for j in thisEmbd:
                        ls = ls + str(j) + '\t'
                    writeFile.write(ls.strip() + '\n')
                    writeFile.flush()
        #max_vali_f1 = evaluate(dataCenter, ds, graphSage, classification, device, max_vali_f1, name, epoch)
    writeFile.close()

    return classification


def train_classification_smote_multi(dataCenter, graphSage, classification, ds, device, max_vali_f1, name, enhancedEmb,
                                   targetLabel, epochs=8):
    '''

    :param dataCenter:
    :param graphSage:
    :param classification:
    :param ds:
    :param device:
    :param max_vali_f1:
    :param name:
    :param enhancedEmb:  生成的Emb
    :param targetLabel:  生成的label
    :param epochs:
    :return:
    '''
    print('Training Classification ...')
    c_optimizer = torch.optim.SGD(classification.parameters(), lr=0.5)

    # train classification, detached from the current graph
    # classification.init_params()
    b_sz = 50
    train_nodes = getattr(dataCenter, ds + '_train')
    labels = getattr(dataCenter, ds + '_labels')
    features = get_gnn_embeddings(graphSage, dataCenter, ds)

    enhanced_feature = torch.tensor( enhancedEmb ).to(device)  # # todo 这里进行多的编码
    #random.shuffle(enhanced_feature)

    batchesNum = math.ceil(len(train_nodes) / b_sz)

    enhanced_batch = math.ceil(enhanced_feature.size(0) / batchesNum)  # 每个batch 加入多少
    #enhanced_label_batch = []
    #for i in range(0, enhanced_batch):
    #    enhanced_label_batch.append(targetLabel)
    # enhanced_label_batch = torch.LongTensor(enhanced_label_batch).cuda()
    enhanced_label = targetLabel

    writeFile = open('embSMOTE_pubmed.txt', 'w')
    for epoch in range(1):
        train_nodes = shuffle(train_nodes)
        batches = math.ceil(len(train_nodes) / b_sz)
        print (batches)


        visited_nodes = set()
        for index in range(batches):
            print (index)
            if index == 49:
                z = 1
            nodes_batch = train_nodes[index * b_sz:(index + 1) * b_sz]

            visited_nodes |= set(nodes_batch)
            labels_batch = list(labels[nodes_batch])
            embs_batch = features[nodes_batch]
            # 加入增强 特征
            t = enhanced_feature[index * enhanced_batch:(index + 1) * enhanced_batch]
            # 加入增强 label
            t1 = enhanced_label[index * enhanced_batch:(index + 1) * enhanced_batch]
            embs_batch = torch.cat([embs_batch, t], dim=0)  # todo 主要还是这个的问题
            labels_batch.extend(t1)

            logists = classification(embs_batch)
            loss = -torch.sum(logists[range(logists.size(0)), labels_batch], 0)
            loss /= len(embs_batch)
            # print('Epoch [{}/{}], Step [{}/{}], Loss: {:.4f}, Dealed Nodes [{}/{}] '.format(epoch+1, epochs, index, batches, loss.item(), len(visited_nodes), len(train_nodes)))

            loss.backward()

            nn.utils.clip_grad_norm_(classification.parameters(), 5)
            c_optimizer.step()
            c_optimizer.zero_grad()
            #print (len(labels_batch))
            #print (len(nodes_batch))
            #exit()
            if 1:
                z = 1
                clone_detach_embeds = embs_batch.clone()
                write_embeds = clone_detach_embeds.cpu().detach().numpy()
                for i in range(0, len(labels_batch)):

                    ls = str(labels_batch[i]) + '\t'
                    if i > len(nodes_batch) - 1:
                        ls = '100\t'  # CORA 7  WIKI 10  ppi 3 blog 100
                    thisEmbd = write_embeds[i]
                    for j in thisEmbd:
                        ls = ls + str(j) + '\t'
                    writeFile.write(ls.strip() + '\n')
                    writeFile.flush()
    writeFile.close()
        #max_vali_f1 = evaluate(dataCenter, ds, graphSage, classification, device, max_vali_f1, name, epoch)
    return classification


def adj_mse_loss(adj_rec, adj_tgt, adj_mask = None):
    edge_num = adj_tgt.nonzero().shape[0]
    total_num = adj_tgt.shape[0]**2

    neg_weight = edge_num / (total_num-edge_num)

    weight_matrix = adj_rec.new(adj_tgt.shape).fill_(1.0)
    weight_matrix[adj_tgt==0] = neg_weight

    loss = torch.sum(weight_matrix * (adj_rec - adj_tgt) ** 2)

    return loss

def train_classification_edge(dataCenter, graphSage, edgeDecoder,classification, ds, device, max_vali_f1, name, enhancedEmb,
                                   targetLabel, epochs=8):
    print('Training Classification ...')
    c_optimizer = torch.optim.SGD(classification.parameters(), lr=0.5)

    # train classification, detached from the current graph
    # classification.init_params()
    b_sz = 50
    train_nodes = getattr(dataCenter, ds + '_train')
    labels = getattr(dataCenter, ds + '_labels')
    features = get_gnn_embeddings(graphSage, dataCenter, ds)

    enhanced_feature = torch.tensor(enhancedEmb).to(device)  # # todo 这里进行多的编码
    random.shuffle(enhanced_feature)
    enhanced_batch = math.ceil(enhanced_feature.size(0) / b_sz)  # 每个batch 加入多少
    enhanced_label_batch = []
    for i in range(0, enhanced_batch):
        enhanced_label_batch.append(targetLabel)


    return classification, max_vali_f1




def apply_model(dataCenter, ds, graphSage, classification, unsupervised_loss, b_sz, unsup_loss, device, learn_method, epoch, maxEpoch, smoteUse = False):
    '''

    :param dataCenter:  数据
    :param ds: 数据形式
    :param graphSage:  模型
    :param classification: 分类
    :param unsupervised_loss:
    :param b_sz: batch size
    :param unsup_loss:
    :param device:
    :param learn_method:  非监督啥的
    :return:
    '''
    test_nodes = getattr(dataCenter, ds + '_test')
    val_nodes = getattr(dataCenter, ds + '_val')
    train_nodes = getattr(dataCenter, ds + '_train')
    labels = getattr(dataCenter, ds + '_labels')
    labels = np.array(labels)
    if unsup_loss == 'margin':
        num_neg = 6
    elif unsup_loss == 'normal':
        num_neg = 100
    else:
        print("unsup_loss can be only 'margin' or 'normal'.")
        sys.exit(1)

    train_nodes = shuffle(train_nodes)

    models = [graphSage, classification]
    params = []
    for model in models:
        for param in model.parameters():
            if param.requires_grad:
                params.append(param)

    optimizer = torch.optim.Adam(params, lr=0.001) # 0.001
    optimizer.zero_grad()
    for model in models:
        model.zero_grad()

    batches = math.ceil(len(train_nodes) / b_sz)

    visited_nodes = set()
    writeFile = open('beforeEmb.txt','w')
    if smoteUse == True:
        writeFileSmote = open('eSMOTE.txt', 'w')

    for index in range(batches):
        nodes_batch = train_nodes[index * b_sz:(index + 1) * b_sz]  # 看这个 nodes batch就行 这个不打乱， 可以用

        # extend nodes batch for unspervised learning
        # no conflicts with supervised learning

        #nodes_batch = np.asarray(list(unsupervised_loss.extend_nodes(nodes_batch, num_neg=num_neg)))
        #visited_nodes |= set(nodes_batch)

        # get ground-truth for the nodes batch
        labels_batch = labels[nodes_batch]

        # feed nodes batch to the graphSAGE
        # returning the nodes embeddings
        #print (nodes_batch[353])
        embs_batch, _, _, _, _ = graphSage(nodes_batch) # todo 这里也要加强 还有 labels_batch
        # 写Embedding 文件
        if epoch == maxEpoch - 1 and index == 1:
            z = 1
            clone_detach_embeds = embs_batch.clone()
            write_embeds = clone_detach_embeds.cpu().detach().numpy()
            for i in range(0, len(labels_batch)):
                ls = str(labels_batch[i]) + '\t'
                thisEmbd = write_embeds[i]
                for j in thisEmbd:
                    ls = ls + str(j) + '\t'
                writeFile.write(ls.strip() + '\n')
                writeFile.flush()

        if learn_method == 'sup':
            # superivsed learning
            logists = classification(embs_batch)

            #logists = torch.where(torch.isnan(logists), torch.full_like(logists, 0), logists) # todo 物理操作

            '''
            write_embeds0 = embs_batch.cpu().detach().numpy()
            for w in write_embeds0:
                print (w)
            print ('####')
            write_embeds = logists.cpu().detach().numpy()
            for w in write_embeds:
                print (w)
            print ('*******')
            '''
            #exit()

            #tmp11 = logists[range(logists.size(0)), labels_batch]
            criterion = nn.CrossEntropyLoss()
            labels_batch = torch.LongTensor(labels_batch).cuda(0)
            loss_sup = criterion(logists, labels_batch)

            loss_sup = -torch.sum(logists[range(logists.size(0)), labels_batch], 0)
            loss_sup /= len(nodes_batch)
            loss = loss_sup

            if smoteUse == True:
                if epoch == maxEpoch - 1 and index == 1:
                    z = 1
                    clone_detach_embeds = embs_batch.clone()
                    write_embeds = clone_detach_embeds.cpu().detach().numpy()
                    for i in range(0, len(labels_batch)):
                        ls = str(labels_batch[i]) + '\t'
                        if nodes_batch[i] in dataCenter.smote_idx:  # smote 加入的
                            ls = '7\t'
                        thisEmbd = write_embeds[i]
                        for j in thisEmbd:
                            ls = ls + str(j) + '\t'
                        writeFileSmote.write(ls.strip() + '\n')
                        writeFileSmote.flush()

        elif learn_method == 'mul_sup':
            # superivsed learning
            out = classification(embs_batch)
            criterion = nn.BCELoss()
            tmp = np.array(labels_batch)
            bm = tmp.max()  # 最大值

            labels_batch = torch.FloatTensor(labels_batch).cuda(0)

            #print (torch.max(labels_batch, dim=1))

            loss_sup = criterion(out, labels_batch)

            #loss_sup = -torch.sum(logists[range(logists.size(0)), labels_batch], 0) # todo
            #loss_sup /= len(nodes_batch)
            # unsuperivsed learning

            loss = loss_sup


        if index % 1 == 0:
            print('Step [{}/{}], Loss: {:.4f}, Dealed Nodes [{}/{}] '.format(index + 1, batches, loss.item(),
                                                              len(visited_nodes), len(train_nodes)))

        loss.backward()
        for model in models:
            nn.utils.clip_grad_norm_(model.parameters(), 5)
        #assert torch.isnan(model.parameters).sum() == 0, print(model.parameters)

        optimizer.step()

        #assert torch.isnan(model.parameters).sum() == 0, print(model.parameters)
        #assert torch.isnan(model.parameters.grad).sum() == 0, print(model.parameters.grad)

        optimizer.zero_grad()
        for model in models:
            model.zero_grad()

        #break

    writeFile.close()
    if smoteUse == True:
        writeFileSmote.close()
    return graphSage, classification



def apply_model_enhanced(dataCenter, ds, graphSageEnhanced, classificationEnhanced, unsupervised_loss, b_sz, unsup_loss, device, learn_method,
                         minEnhanced, minNodeSelfFeature, enhancedLabel, epoch, maxEpoch):
    '''

    :param dataCenter:  数据
    :param ds: 数据形式
    :param graphSage:  模型
    :param classification: 分类
    :param unsupervised_loss:
    :param b_sz: batch size
    :param unsup_loss:
    :param device:
    :param learn_method:  非监督啥的
    minNodeSelfFeature 邻居特征
    minEnhanced 自身特征
    targetLabel 增强标签
    simpleEnhance 增强数量
    :return:
    '''
    test_nodes = getattr(dataCenter, ds + '_test')
    val_nodes = getattr(dataCenter, ds + '_val')
    train_nodes = getattr(dataCenter, ds + '_train')
    labels = getattr(dataCenter, ds + '_labels')

    if unsup_loss == 'margin':
        num_neg = 6
    elif unsup_loss == 'normal':
        num_neg = 100
    else:
        print("unsup_loss can be only 'margin' or 'normal'.")
        sys.exit(1)

    train_nodes = shuffle(train_nodes)

    models = [graphSageEnhanced, classificationEnhanced]
    params = []
    for model in models:
        for param in model.parameters():
            if param.requires_grad:
                params.append(param)

    optimizer = torch.optim.Adam(params, lr=0.001)
    optimizer.zero_grad()
    for model in models:
        model.zero_grad()

    batches = math.ceil(len(train_nodes) / b_sz)
    writeFile = open('afterEmb_pubmed.txt','w')
    print ('minEnhanced ' + str(len(minEnhanced[0])))
    print ('batches ' + str(batches))


    indexLen = max( int( len(minEnhanced[0]) / batches), 1) # b_sz
    print ('indexLen ' + str(indexLen))
    visited_nodes = set()
    for index in range(batches):
        nodes_batch = train_nodes[index * b_sz:(index + 1) * b_sz]  # 看这个 nodes batch就行 这个不打乱， 可以用

        # extend nodes batch for unspervised learning
        # no conflicts with supervised learning
        #nodes_batch = np.asarray(list(unsupervised_loss.extend_nodes(nodes_batch, num_neg=num_neg)))

        #nodes_batch = train_nodes[index * b_sz:(index + 1) * b_sz]

        visited_nodes |= set(nodes_batch)

        # get ground-truth for the nodes batch
        indexAdd = random.sample(range(enhancedLabel.shape[0]), indexLen)
        batchEnhancedLabel = enhancedLabel[indexAdd]

        indexAdd = torch.LongTensor(indexAdd).cuda()


        labels_batch_ori = labels[nodes_batch]
        labels_batch = np.concatenate((labels_batch_ori, batchEnhancedLabel))

        batchNodeSelfFeature = torch.index_select(minNodeSelfFeature, 0, indexAdd)
        batchEnhanced = [torch.index_select(minEnhanced[0], 0, indexAdd)]



        embs_batch, _, _, _ = graphSageEnhanced(nodes_batch, batchEnhanced, batchNodeSelfFeature) # todo 这里也要加强 还有 labels_batch
        # 写Embedding 文件

        if epoch == maxEpoch - 1 : # and index == 1:
            print ('**WRITE**')
            z = 1
            clone_detach_embeds = embs_batch.clone()
            write_embeds = clone_detach_embeds.cpu().detach().numpy()
            print('labels_batch ' + str(len(labels_batch)) + ' nodes_batch' + str(len(nodes_batch)))


            for i in range(0, len(labels_batch)):
                ls = str(labels_batch[i]) + '\t'
                if i < len(nodes_batch):

                    thisEmbd = write_embeds[i]
                    for j in thisEmbd:
                        ls = ls + str(j) + '\t'
                    writeFile.write(ls.strip() + '\n')
                    writeFile.flush()

                else:
                    ls = '100\t'  # cora 7 wiki 10 blog 100

                    thisEmbd = write_embeds[i]
                    for j in thisEmbd:
                        ls = ls + str(j) + '\t'
                    writeFile.write(ls.strip() + '\n')
                    writeFile.flush()

        if learn_method == 'sup':
            # superivsed learning
            logists = classificationEnhanced(embs_batch)
            loss_sup = -torch.sum(logists[range(logists.size(0)), labels_batch], 0)
            loss_sup /= (len(nodes_batch) + len(enhancedLabel))
            loss = loss_sup
        elif learn_method == 'plus_unsup':
            # superivsed learning
            logists = classificationEnhanced(embs_batch)
            loss_sup = -torch.sum(logists[range(logists.size(0)), labels_batch], 0)
            loss_sup /= len(nodes_batch)
            # unsuperivsed learning
            if unsup_loss == 'margin':
                loss_net = unsupervised_loss.get_loss_margin(embs_batch, nodes_batch)
            elif unsup_loss == 'normal':
                loss_net = unsupervised_loss.get_loss_sage(embs_batch, nodes_batch)
            loss = loss_sup + loss_net
        else:
            '''
            非监督
            '''
            if unsup_loss == 'margin':
                loss_net = unsupervised_loss.get_loss_margin(embs_batch, nodes_batch)
            elif unsup_loss == 'normal':
                loss_net = unsupervised_loss.get_loss_sage(embs_batch, nodes_batch)
            loss = loss_net
        if index % 1 ==0:
            print('Step [{}/{}], Loss: {:.4f}, Dealed Nodes [{}/{}] '.format(index + 1, batches, loss.item(),
                                                                             len(visited_nodes), len(train_nodes)))
        loss.backward(retain_graph=True) #todo why?
        for model in models:
            nn.utils.clip_grad_norm_(model.parameters(), 5)
        optimizer.step()

        optimizer.zero_grad()
        for model in models:
            model.zero_grad()
    writeFile.close()
    return graphSageEnhanced, classificationEnhanced

def mul_apply_model_enhanced(dataCenter, ds, graphSageEnhanced, classificationEnhanced, unsupervised_loss, b_sz, unsup_loss, device, learn_method,
                         minEnhanced, minNodeSelfFeature, minNodeSelfLabels, epoch, maxEpoch, writeFileFlag = False):
    '''

    :param dataCenter:  数据
    :param ds: 数据形式
    :param graphSage:  模型
    :param classification: 分类
    :param unsupervised_loss:
    :param b_sz: batch size
    :param unsup_loss:
    :param device:
    :param learn_method:  非监督啥的
    minNodeSelfFeature 邻居特征
    minEnhanced 自身特征
    targetLabel 增强标签
    simpleEnhance 增强数量
    :return:
    '''
    test_nodes = getattr(dataCenter, ds + '_test')
    val_nodes = getattr(dataCenter, ds + '_val')
    train_nodes = getattr(dataCenter, ds + '_train')
    labels = getattr(dataCenter, ds + '_labels')

    if unsup_loss == 'margin':
        num_neg = 6
    elif unsup_loss == 'normal':
        num_neg = 100
    else:
        print("unsup_loss can be only 'margin' or 'normal'.")
        sys.exit(1)

    train_nodes = shuffle(train_nodes)

    models = [graphSageEnhanced, classificationEnhanced]
    params = []
    for model in models:
        for param in model.parameters():
            if param.requires_grad:
                params.append(param)

    optimizer = torch.optim.Adam(params, lr=0.001)
    optimizer.zero_grad()
    for model in models:
        model.zero_grad()

    batches = math.ceil(len(train_nodes) / b_sz)
    writeFile = open('SGSMOTE_ppi.txt', 'w')

    visited_nodes = set()
    for index in range(batches):
        nodes_batch = train_nodes[index * b_sz:(index + 1) * b_sz]  # 看这个 nodes batch就行 这个不打乱， 可以用

        # extend nodes batch for unspervised learning
        # no conflicts with supervised learning
        #nodes_batch = np.asarray(list(unsupervised_loss.extend_nodes(nodes_batch, num_neg=num_neg)))
        #visited_nodes |= set(nodes_batch)  # todo 在GraphSage也是错的？

        # get ground-truth for the nodes batch
        labels_batch_ori = labels[nodes_batch]
        labels_batch_ori = torch.FloatTensor(labels_batch_ori).cuda()

        #indexAdd = torch.LongTensor(random.sample(range(minNodeSelfLabels.shape[0]), 4)).cuda()

        indexLen = max(int(labels_batch_ori.shape[0] * 0.01),1)

        indexAdd = random.sample(range(minNodeSelfLabels.shape[0]), indexLen)
        indexAdd = torch.LongTensor(indexAdd).cuda()


        batchEnhanced = [torch.index_select(minEnhanced[0], 0, indexAdd)]
        #batchEnhanced = [batchEnhanced]

        #indexAdd = torch.LongTensor(indexAdd).cuda()

        batchNodeSelfFeature = torch.index_select(minNodeSelfFeature, 0, indexAdd)

        batch_index = torch.index_select(minNodeSelfLabels, 0, indexAdd)

        labels_batch = torch.cat([labels_batch_ori, batch_index], dim=0)  # todo 这太多了

        #exit()
        embs_batch, _, _, _ = graphSageEnhanced(nodes_batch, batchEnhanced, batchNodeSelfFeature) # todo 这里也要加强 还有 labels_batch
        # 写Embedding 文件

        if learn_method == 'sup':
            # superivsed learning
            logists = classificationEnhanced(embs_batch)
            loss_sup = -torch.sum(logists[range(logists.size(0)), labels_batch], 0)
            loss_sup /= (len(nodes_batch) + len(minNodeSelfLabels))
            loss = loss_sup
        elif learn_method == 'mul_sup':
            # superivsed learning
            out = classificationEnhanced(embs_batch)
            criterion = nn.BCELoss()
            #labels_batch = torch.FloatTensor(labels_batch).cuda(0)
            loss_sup = criterion(out, labels_batch)

            loss = loss_sup
        if index % 20 ==0:
            print('Step [{}/{}], Loss: {:.4f}, Dealed Nodes [{}/{}] '.format(index + 1, batches, loss.item(),
                                                                             len(visited_nodes), len(train_nodes)))
        loss.backward(retain_graph=True) #todo why?
        for model in models:
            nn.utils.clip_grad_norm_(model.parameters(), 5)
        optimizer.step()

        optimizer.zero_grad()
        for model in models:
            model.zero_grad()
        #break

        if writeFileFlag:
            z = 1
            clone_detach_embeds = embs_batch.clone()
            write_embeds = clone_detach_embeds.cpu().detach().numpy()
            for i in range(0, len(labels_batch)):
                writeLabel = 0
                if labels_batch[i][86] == 1:
                    writeLabel = 1
                if i > len(labels_batch_ori) - 1:
                    writeLabel = 2
                ls = str(writeLabel) + '\t'
                thisEmbd = write_embeds[i]
                for j in thisEmbd:
                    ls = ls + str(j) + '\t'
                writeFile.write(ls.strip() + '\n')
                writeFile.flush()
    writeFile.close()

    return graphSageEnhanced, classificationEnhanced




def mul_apply_model_enhanced_2layer(dataCenter, ds, graphSageEnhanced, classificationEnhanced, unsupervised_loss, b_sz, unsup_loss, device, learn_method,
                         minEnhanced, minNodeSelfFeature, minNodeSelfLabels, epoch, maxEpoch):
    '''

    :param dataCenter:  数据
    :param ds: 数据形式
    :param graphSage:  模型
    :param classification: 分类
    :param unsupervised_loss:
    :param b_sz: batch size
    :param unsup_loss:
    :param device:
    :param learn_method:  非监督啥的
    minNodeSelfFeature 邻居特征
    minEnhanced 自身特征
    targetLabel 增强标签
    simpleEnhance 增强数量
    :return:
    '''
    test_nodes = getattr(dataCenter, ds + '_test')
    val_nodes = getattr(dataCenter, ds + '_val')
    train_nodes = getattr(dataCenter, ds + '_train')
    labels = getattr(dataCenter, ds + '_labels')

    if unsup_loss == 'margin':
        num_neg = 6
    elif unsup_loss == 'normal':
        num_neg = 100
    else:
        print("unsup_loss can be only 'margin' or 'normal'.")
        sys.exit(1)

    train_nodes = shuffle(train_nodes)

    models = [graphSageEnhanced, classificationEnhanced]
    params = []
    for model in models:
        for param in model.parameters():
            if param.requires_grad:
                params.append(param)

    optimizer = torch.optim.Adam(params, lr=0.001)
    optimizer.zero_grad()
    for model in models:
        model.zero_grad()

    batches = math.ceil(len(train_nodes) / b_sz)

    visited_nodes = set()
    for index in range(batches):
        nodes_batch = train_nodes[index * b_sz:(index + 1) * b_sz]  # 看这个 nodes batch就行 这个不打乱， 可以用

        # extend nodes batch for unspervised learning
        # no conflicts with supervised learning
        #nodes_batch = np.asarray(list(unsupervised_loss.extend_nodes(nodes_batch, num_neg=num_neg)))
        #visited_nodes |= set(nodes_batch)

        # get ground-truth for the nodes batch
        labels_batch_ori = labels[nodes_batch]
        labels_batch_ori = torch.FloatTensor(labels_batch_ori).cuda()
        labels_batch = labels_batch_ori
        # feed nodes batch to the graphSAGE
        # returning the nodes embeddings
        #print (minNodeSelfFeature.shape)
        #print(nodes_batch.shape)
        #print(len(minEnhanced))
        #print (minEnhanced[0].shape)
        #exit()
        embs_batch, _, _, _, enhance_idx = graphSageEnhanced(nodes_batch, minEnhanced, minNodeSelfFeature) # todo 这里也要加强 还有 labels_batch
        #embs_batch, _, _, _, enhance_idx = graphSageEnhanced(nodes_batch, [], []) # todo 这里也要加强 还有 labels_batch

        # 写Embedding 文件
        if len(enhance_idx) > 0:
            minNodeSelfLabels = labels[enhance_idx]
            minNodeSelfLabels = torch.FloatTensor(minNodeSelfLabels).cuda()

            labels_batch = torch.cat([labels_batch_ori, minNodeSelfLabels], dim=0)



        if learn_method == 'sup':
            # superivsed learning
            logists = classificationEnhanced(embs_batch)
            loss_sup = -torch.sum(logists[range(logists.size(0)), labels_batch], 0)
            loss_sup /= (len(nodes_batch) + len(minNodeSelfLabels))
            loss = loss_sup
        elif learn_method == 'mul_sup':
            # superivsed learning
            out = classificationEnhanced(embs_batch)
            criterion = nn.BCELoss()
            #labels_batch = torch.FloatTensor(labels_batch).cuda(0)
            loss_sup = criterion(out, labels_batch)

            loss = loss_sup
        if index % 20 ==0:
            print('Step [{}/{}], Loss: {:.4f}, Dealed Nodes [{}/{}] '.format(index + 1, batches, loss.item(),
                                                                             len(visited_nodes), len(train_nodes)))
        loss.backward() #todo why?
        for model in models:
            nn.utils.clip_grad_norm_(model.parameters(), 5)
        optimizer.step()

        optimizer.zero_grad()
        for model in models:
            model.zero_grad()
        #break

    return graphSageEnhanced, classificationEnhanced


def apply_model_enhanced_multi(dataCenter, ds, graphSageEnhanced, classificationEnhanced, unsupervised_loss, b_sz, unsup_loss, device, learn_method,
                         minEnhanced, minRecord, targetLabel, epoch, maxEpoch, needIndx = False):
    '''

    :param dataCenter:  数据
    :param ds: 数据形式
    :param graphSage:  模型
    :param classification: 分类
    :param unsupervised_loss:
    :param b_sz: batch size
    :param unsup_loss:
    :param device:
    :param learn_method:  非监督啥的
    minNodeSelfFeature 邻居特征
    minEnhanced 自身特征
    targetLabel 增强标签
    simpleEnhance 增强数量
    :return:
    '''
    test_nodes = getattr(dataCenter, ds + '_test')
    val_nodes = getattr(dataCenter, ds + '_val')
    train_nodes = getattr(dataCenter, ds + '_train')
    labels = getattr(dataCenter, ds + '_labels')

    if unsup_loss == 'margin':
        num_neg = 6
    elif unsup_loss == 'normal':
        num_neg = 100
    else:
        print("unsup_loss can be only 'margin' or 'normal'.")
        sys.exit(1)

    train_nodes = shuffle(train_nodes)

    models = [graphSageEnhanced, classificationEnhanced]
    params = []
    for model in models:
        for param in model.parameters():
            if param.requires_grad:
                params.append(param)

    optimizer = torch.optim.Adam(params, lr=0.001)
    optimizer.zero_grad()
    for model in models:
        model.zero_grad()

    batches = math.ceil(len(train_nodes) / b_sz)
    #writeFile = open('afterEmb2.txt','w')

    visited_nodes = set()
    for index in range(batches):
        nodes_batch = train_nodes[index * b_sz:(index + 1) * b_sz]  # 看这个 nodes batch就行 这个不打乱， 可以用

        # extend nodes batch for unspervised learning
        # no conflicts with supervised learning
        nodes_batch = np.asarray(list(unsupervised_loss.extend_nodes(nodes_batch, num_neg=num_neg)))
        visited_nodes |= set(nodes_batch)

        # get ground-truth for the nodes batch
        labels_batch_ori = labels[nodes_batch]
        #labels_batch = np.concatenate((labels_batch_ori, enhancedLabel))

        # feed nodes batch to the graphSAGE
        # returning the nodes embeddings
        embs_batch, _, _, _ = graphSageEnhanced(nodes_batch, minEnhanced, minRecord) # todo 这里也要加强 还有 labels_batch
        enhancedLabel = []
        # 写Embedding 文件 todo 这里要对 label 操作一下
        if len(embs_batch) > len(nodes_batch):

            for j in range(len(nodes_batch), len(embs_batch)):
                enhancedLabel.append(targetLabel)
            enhancedLabel = np.array(enhancedLabel)

            labels_batch = np.concatenate((labels_batch_ori, enhancedLabel))
        else:
            labels_batch = labels_batch_ori

        #if epoch == maxEpoch - 1 and index == 1:
        if 0:
            z = 1
            clone_detach_embeds = embs_batch.clone()
            write_embeds = clone_detach_embeds.cpu().detach().numpy()
            for i in range(0, len(labels_batch)):
                ls = str(labels_batch[i]) + '\t'
                if i > len(nodes_batch):
                    ls = '7\t'
                thisEmbd = write_embeds[i]
                for j in thisEmbd:
                    ls = ls + str(j) + '\t'
                writeFile.write(ls.strip() + '\n')
                writeFile.flush()

        if learn_method == 'sup':
            # superivsed learning
            logists = classificationEnhanced(embs_batch)
            loss_sup = -torch.sum(logists[range(logists.size(0)), labels_batch], 0)
            loss_sup /= (len(nodes_batch) + len(enhancedLabel))
            loss = loss_sup
        elif learn_method == 'plus_unsup':
            # superivsed learning
            logists = classificationEnhanced(embs_batch)
            loss_sup = -torch.sum(logists[range(logists.size(0)), labels_batch], 0)
            loss_sup /= len(nodes_batch)
            # unsuperivsed learning
            if unsup_loss == 'margin':
                loss_net = unsupervised_loss.get_loss_margin(embs_batch, nodes_batch)
            elif unsup_loss == 'normal':
                loss_net = unsupervised_loss.get_loss_sage(embs_batch, nodes_batch)
            loss = loss_sup + loss_net
        else:
            '''
            非监督
            '''
            if unsup_loss == 'margin':
                loss_net = unsupervised_loss.get_loss_margin(embs_batch, nodes_batch)
            elif unsup_loss == 'normal':
                loss_net = unsupervised_loss.get_loss_sage(embs_batch, nodes_batch)
            loss = loss_net

        if index % 1 == 0:
            print('Step [{}/{}], Loss: {:.4f}, Dealed Nodes [{}/{}] '.format(index + 1, batches, loss.item(),
                                                                             len(visited_nodes), len(train_nodes)))
        loss.backward(retain_graph=True) #todo why?
        for model in models:
            nn.utils.clip_grad_norm_(model.parameters(), 5)
        optimizer.step()

        optimizer.zero_grad()
        for model in models:
            model.zero_grad()
    #writeFile.close()
    return graphSageEnhanced, classificationEnhanced



def apply_model_enhanced_multi_multiclass(dataCenter, ds, graphSageEnhanced, classificationEnhanced, unsupervised_loss, b_sz, unsup_loss, device, learn_method,
                         minEnhanced, minRecord, targetLabel, epoch, maxEpoch, needIndx = False):
    '''

    :param dataCenter:  数据
    :param ds: 数据形式
    :param graphSage:  模型
    :param classification: 分类
    :param unsupervised_loss:
    :param b_sz: batch size
    :param unsup_loss:
    :param device:
    :param learn_method:  非监督啥的
    minNodeSelfFeature 邻居特征
    minEnhanced 自身特征
    targetLabel 增强标签
    simpleEnhance 增强数量
    :return:
    '''
    test_nodes = getattr(dataCenter, ds + '_test')
    val_nodes = getattr(dataCenter, ds + '_val')
    train_nodes = getattr(dataCenter, ds + '_train')
    labels = getattr(dataCenter, ds + '_labels')

    if unsup_loss == 'margin':
        num_neg = 6
    elif unsup_loss == 'normal':
        num_neg = 100
    else:
        print("unsup_loss can be only 'margin' or 'normal'.")
        sys.exit(1)

    train_nodes = shuffle(train_nodes)

    models = [graphSageEnhanced, classificationEnhanced]
    params = []
    for model in models:
        for param in model.parameters():
            if param.requires_grad:
                params.append(param)

    optimizer = torch.optim.Adam(params, lr=0.001)
    optimizer.zero_grad()
    for model in models:
        model.zero_grad()

    batches = math.ceil(len(train_nodes) / b_sz)
    writeFile = open('afterEmb2.txt','w')

    visited_nodes = set()
    for index in range(batches):
        nodes_batch = train_nodes[index * b_sz:(index + 1) * b_sz]  # 看这个 nodes batch就行 这个不打乱， 可以用

        # extend nodes batch for unspervised learning
        # no conflicts with supervised learning
        nodes_batch = np.asarray(list(unsupervised_loss.extend_nodes(nodes_batch, num_neg=num_neg)))
        visited_nodes |= set(nodes_batch)

        # get ground-truth for the nodes batch
        labels_batch_ori = labels[nodes_batch]
        #labels_batch = np.concatenate((labels_batch_ori, enhancedLabel))

        # feed nodes batch to the graphSAGE
        # returning the nodes embeddings
        embs_batch, _, _, _, enhance_idx = graphSageEnhanced(nodes_batch, minEnhanced, minRecord, needIndx=True) # todo 这里也要加强 还有 labels_batch
        enhancedLabel = []
        # 写Embedding 文件 todo 这里要对 label 操作一下
        if len(embs_batch) > len(nodes_batch):

            for j in range(len(nodes_batch), len(embs_batch)):
                enhancedLabel.append(targetLabel[enhance_idx[ j - len(nodes_batch) ]])
            enhancedLabel = np.array(enhancedLabel)

            labels_batch = np.concatenate((labels_batch_ori, enhancedLabel))
        else:
            labels_batch = labels_batch_ori

        #if epoch == maxEpoch - 1 and index == 1:
        if 0:
            z = 1
            clone_detach_embeds = embs_batch.clone()
            write_embeds = clone_detach_embeds.cpu().detach().numpy()
            for i in range(0, len(labels_batch)):
                ls = str(labels_batch[i]) + '\t'
                if i > len(nodes_batch):
                    ls = '7\t'
                thisEmbd = write_embeds[i]
                for j in thisEmbd:
                    ls = ls + str(j) + '\t'
                writeFile.write(ls.strip() + '\n')
                writeFile.flush()

        if learn_method == 'sup':
            # superivsed learning
            logists = classificationEnhanced(embs_batch)
            loss_sup = -torch.sum(logists[range(logists.size(0)), labels_batch], 0)
            loss_sup /= (len(nodes_batch) + len(enhancedLabel))
            loss = loss_sup
        elif learn_method == 'plus_unsup':
            # superivsed learning
            logists = classificationEnhanced(embs_batch)
            loss_sup = -torch.sum(logists[range(logists.size(0)), labels_batch], 0)
            loss_sup /= len(nodes_batch)
            # unsuperivsed learning
            if unsup_loss == 'margin':
                loss_net = unsupervised_loss.get_loss_margin(embs_batch, nodes_batch)
            elif unsup_loss == 'normal':
                loss_net = unsupervised_loss.get_loss_sage(embs_batch, nodes_batch)
            loss = loss_sup + loss_net
        else:
            '''
            非监督
            '''
            if unsup_loss == 'margin':
                loss_net = unsupervised_loss.get_loss_margin(embs_batch, nodes_batch)
            elif unsup_loss == 'normal':
                loss_net = unsupervised_loss.get_loss_sage(embs_batch, nodes_batch)
            loss = loss_net

        print('Step [{}/{}], Loss: {:.4f}, Dealed Nodes [{}/{}] '.format(index + 1, batches, loss.item(),
                                                                         len(visited_nodes), len(train_nodes)))
        loss.backward(retain_graph=True) #todo why?
        for model in models:
            nn.utils.clip_grad_norm_(model.parameters(), 5)
        optimizer.step()

        optimizer.zero_grad()
        for model in models:
            model.zero_grad()
    writeFile.close()
    return graphSageEnhanced, classificationEnhanced

def apply_model_edge(dataCenter, ds, edgeDecoder, graphSage, classification, unsupervised_loss, b_sz, unsup_loss, device, learn_method, epoch, maxEpoch):
    '''

    :param dataCenter:  数据
    :param ds: 数据形式
    :param graphSage:  模型
    :param classification: 分类
    :param unsupervised_loss:
    :param b_sz: batch size
    :param unsup_loss:
    :param device:
    :param learn_method:  非监督啥的
    :return:
    '''
    test_nodes = getattr(dataCenter, ds + '_test')
    val_nodes = getattr(dataCenter, ds + '_val')
    train_nodes = getattr(dataCenter, ds + '_train')
    labels = getattr(dataCenter, ds + '_labels')
    features = getattr(dataCenter, ds + '_feats')

    if unsup_loss == 'margin':
        num_neg = 6
    elif unsup_loss == 'normal':
        num_neg = 100
    else:
        print("unsup_loss can be only 'margin' or 'normal'.")
        sys.exit(1)

    #train_nodes = shuffle(train_nodes)
    '''
    models = [graphSage, classification]
    params = []
    for model in models:
        for param in model.parameters():
            if param.requires_grad:
                params.append(param)

    optimizer = torch.optim.Adam(params, lr=0.001)
    optimizer.zero_grad()
    for model in models:
        model.zero_grad()
    '''
    # 边
    #e_optimizer = torch.optim.SGD(classification.parameters(), lr=0.5)
    e_optimizer= torch.optim.Adam(edgeDecoder.parameters(),
                 lr=0.001, weight_decay=5e-4)

    visited_nodes = set()
    nodes_batch = train_nodes
    nodes_batch = np.hstack((train_nodes, val_nodes, test_nodes))

    #nodes_batch = np.asarray(list(unsupervised_loss.extend_nodes(nodes_batch, num_neg=num_neg)))
    #visited_nodes |= set(nodes_batch)

    # get ground-truth for the nodes batch
    #labels_batch = labels[nodes_batch]

    # feed nodes batch to the graphSAGE
    # returning the nodes embeddings
    embs_batch, _, _, _ = graphSage(nodes_batch)  # todo 这里也要加强 还有 labels_batch
    generated_G = edgeDecoder(embs_batch)
    loss_rec = adj_mse_loss(generated_G.double(), torch.tensor( dataCenter.adj_dense).double().to(device))
    loss_rec/= len(nodes_batch)
    loss = loss_rec
    loss.backward()
    e_optimizer.step()
    e_optimizer.zero_grad()

    return graphSage, edgeDecoder, classification, loss


def apply_model_edge_finetuning(dataCenter, ds, edgeDecoder, graphSage, classificationGS, unsupervised_loss, b_sz, unsup_loss,
                     device, learn_method, epoch, maxEpoch):
    '''

    :param dataCenter:  数据
    :param ds: 数据形式
    :param graphSage:  模型
    :param classification: 分类
    :param unsupervised_loss:
    :param b_sz: batch size
    :param unsup_loss:
    :param device:
    :param learn_method:  非监督啥的
    :return:
    '''
    test_nodes = getattr(dataCenter, ds + '_test')
    val_nodes = getattr(dataCenter, ds + '_val')
    train_nodes = getattr(dataCenter, ds + '_train')
    labels = getattr(dataCenter, ds + '_labels')
    features = getattr(dataCenter, ds + '_feats')

    if unsup_loss == 'margin':
        num_neg = 6
    elif unsup_loss == 'normal':
        num_neg = 100
    else:
        print("unsup_loss can be only 'margin' or 'normal'.")
        sys.exit(1)

    # train_nodes = shuffle(train_nodes)

    models = [graphSage, classificationGS]
    params = []
    for model in models:
        for param in model.parameters():
            if param.requires_grad:
                params.append(param)

    optimizer = torch.optim.Adam(params, lr=0.001)
    #optimizer = torch.optim.SGD(params, lr=0.5)


    optimizer.zero_grad()
    for model in models:
        model.zero_grad()

    # 边
    e_optimizer= torch.optim.Adam(edgeDecoder.parameters(),
                 lr=0.001, weight_decay=5e-4)
    visited_nodes = set()
    nodes_batch = np.hstack((train_nodes, val_nodes, test_nodes))

    # nodes_batch = np.asarray(list(unsupervised_loss.extend_nodes(nodes_batch, num_neg=num_neg)))
    # visited_nodes |= set(nodes_batch)

    # get ground-truth for the nodes batch
    # labels_batch = labels[nodes_batch]

    # feed nodes batch to the graphSAGE
    # returning the nodes embeddings
    embs_batch, _, _, _ = graphSage(nodes_batch)  # todo 这里也要加强 还有 labels_batch
    generated_G = edgeDecoder(embs_batch)
    loss_rec = adj_mse_loss(generated_G.double(), torch.tensor(dataCenter.adj_dense).double().to(device))
    loss_rec /= len(nodes_batch)
    labels = getattr(dataCenter, ds + '_labels')
    idx_train =  list( getattr(dataCenter, ds + '_train') )

    embs_batch, _, _, _ = graphSage(nodes_batch)  # todo 这里不能用 batch
    output = classificationGS(embs_batch, torch.tensor(dataCenter.adj_dense, dtype=torch.float32).to(device))  # dtype=torch.float32

    tmp3 = output[torch.tensor(idx_train , dtype=torch.long )]
    tmp4 = torch.tensor(labels[torch.tensor(idx_train , dtype=torch.long ) ]).to(device)

    #loss_train = F.cross_entropy(tmp3, tmp4)

    loss_train = -torch.sum(tmp3[range(tmp3.size(0)), tmp4], 0)
    loss_train /= len(nodes_batch)

    #loss = loss_train +  0.5 * loss_rec.float()
    loss = loss_train
    loss.backward()
    e_optimizer.step()
    for model in models:
        nn.utils.clip_grad_norm_(model.parameters(), 5)
    optimizer.step()

    e_optimizer.zero_grad()

    # loss.backward()

    optimizer.zero_grad()
    for model in models:
        model.zero_grad()



    return graphSage, edgeDecoder, classificationGS, loss_train, loss_rec

def checkUnSunEmb(dataCenter, ds, graphSage, b_sz, targetLabel):
    train_nodes = getattr(dataCenter, ds + '_train')
    labels = getattr(dataCenter, ds + '_labels')

    batches = math.ceil(len(train_nodes) / b_sz)
    visited_nodes = set()
    import copy
    targetEmb = []  # 记录隐层
    targetIndex = []  # 记录 index
    targetSamp_neighs = []
    targetMask = []
    maxLen = 0
    dim = 0
    # todo 记录 Emb信息 先record起来吧
    Xemb = []
    Yemb = []
    for index in range(batches):
        nodes_batch = train_nodes[index * b_sz:(index + 1) * b_sz]  # 看这个 nodes batch就行 这个不打乱， 可以用

        # extend nodes batch for unspervised learning
        # no conflicts with supervised learning
        # print (nodes_batch)
        # nodes_batch = np.asarray(list(unsupervised_loss.extend_nodes(nodes_batch, num_neg=num_neg)))
        visited_nodes |= set(nodes_batch)

        # get ground-truth for the nodes batch
        labels_batch = labels[nodes_batch]

        # feed nodes batch to the graphSAGE
        # returning the nodes embeddings
        # pre_hidden_embs, pre_feature, samp_neighs
        embs_batch, pre_feature, samp_neighs, mask = graphSage(nodes_batch)  #
        # 通过 embs_batch 返回 embedding
        # 通过 samp_neighs 和 mask 返回 邻居信息 进行解码
        # labels_batch2 = copy.deepcopy(labels_batch)
        # embs_batch2 = copy.deepcopy(embs_batch)

        labels_batch_data = labels_batch
        embs_batch = embs_batch

        for i in range(0, len(labels_batch_data)):
            emb_cpu = embs_batch.detach().cpu().numpy()
            Xemb.extend(emb_cpu)
            Yemb.extend(nodes_batch)
            if labels_batch_data[i] == targetLabel:  # 如果是目标
                targetEmb.append(emb_cpu[i])  # right
                targetIndex.append(nodes_batch[i])

                sampleMask = mask[i]  # 当前节点的邻居信息
                neighbourIndex = (sampleMask > 0).nonzero().squeeze().cuda()
                # print (neighbourIndex)
                try:  # todo 为了防止没有邻居
                    tLen = neighbourIndex.size(-1)
                    if tLen > maxLen:
                        maxLen = tLen
                except:
                    print(neighbourIndex)
                targetNeighbourFeature = torch.index_select(pre_feature, 0, neighbourIndex)
                dim = targetNeighbourFeature.size(-1)
                targetSamp_neighs.append(targetNeighbourFeature.detach().cpu().numpy())

    return targetEmb, targetIndex, targetSamp_neighs, maxLen, dim




def checkUnSunEmbAdaSyn(dataCenter, ds, graphSage, b_sz, targetLabel):
    train_nodes = getattr(dataCenter, ds + '_train')
    labels = getattr(dataCenter, ds + '_labels')

    batches = math.ceil(len(train_nodes) / b_sz)
    visited_nodes = set()
    import copy
    targetEmb = []  # 记录隐层
    targetIndex = []  # 记录 index
    targetSamp_neighs = []
    targetMask = []
    maxLen = 0
    dim = 0
    # todo 记录 Emb信息 先record起来吧
    Xemb = []
    Yemb = []
    count = -1
    minClass = []

    for index in range(batches):
        nodes_batch = train_nodes[index * b_sz:(index + 1) * b_sz]  # 看这个 nodes batch就行 这个不打乱， 可以用

        # extend nodes batch for unspervised learning
        # no conflicts with supervised learning
        # print (nodes_batch)
        # nodes_batch = np.asarray(list(unsupervised_loss.extend_nodes(nodes_batch, num_neg=num_neg)))
        visited_nodes |= set(nodes_batch)

        # get ground-truth for the nodes batch
        labels_batch = labels[nodes_batch]

        # feed nodes batch to the graphSAGE
        # returning the nodes embeddings
        # pre_hidden_embs, pre_feature, samp_neighs
        embs_batch, pre_feature, samp_neighs, mask = graphSage(nodes_batch)  #
        # 通过 embs_batch 返回 embedding
        # 通过 samp_neighs 和 mask 返回 邻居信息 进行解码
        # labels_batch2 = copy.deepcopy(labels_batch)
        # embs_batch2 = copy.deepcopy(embs_batch)

        labels_batch_data = labels_batch
        embs_batch = embs_batch

        emb_cpu = embs_batch.detach().cpu().numpy()
        Xemb.extend(emb_cpu)
        Yemb.extend(labels_batch)

        for i in range(0, len(labels_batch_data)):
            count += 1
            if labels_batch_data[i] == targetLabel:  # 如果是目标
                minClass.append(count)
                targetEmb.append(emb_cpu[i])  # right
                targetIndex.append(nodes_batch[i])

                sampleMask = mask[i]  # 当前节点的邻居信息
                neighbourIndex = (sampleMask > 0).nonzero().squeeze().cuda()
                # print (neighbourIndex)
                try:  # todo 为了防止没有邻居
                    tLen = neighbourIndex.size(-1)
                    if tLen > maxLen:
                        maxLen = tLen
                except:
                    print(neighbourIndex)
                targetNeighbourFeature = torch.index_select(pre_feature, 0, neighbourIndex)
                dim = targetNeighbourFeature.size(-1)
                targetSamp_neighs.append(targetNeighbourFeature.detach().cpu().numpy())


    from sklearn.neighbors import NearestNeighbors
    N_NEIGHBOURS = 75
    nbrs = NearestNeighbors(n_neighbors=N_NEIGHBOURS, algorithm='auto').fit(Xemb)

    min_count = 0
    count = 0
    ratioSum = 0.0
    ratioDict = {}
    for i in minClass:
        distances, indices = nbrs.kneighbors(Xemb[i].reshape(1, -1))
        # 进行固定维度的排序 （例如第一维度）
        i_index = []
        n_count = 0
        j_count = 0
        k_count = 0

        for j in indices[0]:
            if j == i: continue  # 不包含本身
            j_count += 1
            if Yemb[j] != targetLabel:
                if j_count < N_NEIGHBOURS:
                    k_count += 1
                continue  # 不包含 少类
            i_index.append(Xemb[j])  # 直接输入 数据
            # 写一下 X——train 进行记录

            n_count += 1
            if n_count == N_NEIGHBOURS: break
        # fwrite.write('\n')
        # fwrite.flush()
        ratio = float(k_count) / float(j_count)
        #print (ratio)
        ratioSum += ratio
        ratioDict[min_count] = ratio
        min_count += 1
    #exit()
    rationNorDict = {}
    recordEpoch = []
    G = 30
    for k, v in ratioDict.items():
        z = v / ratioSum
        print (z)
        rationNorDict[k] = (int(z * G))  # 计算生成比率
        recordEpoch.append(int(z * G))

    print ('MAX' + str(np.max(recordEpoch)))

    return targetEmb, targetIndex, targetSamp_neighs, maxLen, dim, recordEpoch



def checkUnSunEmbAdaSyn2(dataCenter, ds, graphSage, b_sz, targetLabel, portion = .5):
    train_nodes = getattr(dataCenter, ds + '_train')
    labels = getattr(dataCenter, ds + '_labels')

    batches = math.ceil(len(train_nodes) / b_sz)
    visited_nodes = set()
    import copy
    targetEmb = []  # 记录隐层
    targetIndex = []  # 记录 index
    targetSamp_neighs = []
    targetMask = []
    maxLen = 0
    dim = 0
    # todo 记录 Emb信息 先record起来吧
    Xemb = []
    Yemb = []
    count = -1
    minClass = []
    lenInfo = []
    from collections import Counter

    for index in range(batches):
        nodes_batch = train_nodes[index * b_sz:(index + 1) * b_sz]  # 看这个 nodes batch就行 这个不打乱， 可以用
        visited_nodes |= set(nodes_batch)
        # get ground-truth for the nodes batch
        labels_batch = labels[nodes_batch]
        # feed nodes batch to the graphSAGE
        # returning the nodes embeddings
        # pre_hidden_embs, pre_feature, samp_neighs
        embs_batch, pre_feature, samp_neighs, mask, _ = graphSage(nodes_batch)  #
        # 通过 embs_batch 返回 embedding
        # 通过 samp_neighs 和 mask 返回 邻居信息 进行解码
        # labels_batch2 = copy.deepcopy(labels_batch)
        # embs_batch2 = copy.deepcopy(embs_batch)

        labels_batch_data = labels_batch
        embs_batch = embs_batch

        emb_cpu = embs_batch.detach().cpu().numpy()
        Xemb.extend(emb_cpu)
        #Yemb.extend(labels_batch)

        for i in range(0, len(labels_batch_data)):
            count += 1
            if labels_batch_data[i] == targetLabel:  # 如果是目标
                Yemb.append(1)
                minClass.append(count)
                targetEmb.append(emb_cpu[i])  # right
                targetIndex.append(nodes_batch[i])

                sampleMask = mask[i]  # 当前节点的邻居信息
                neighbourIndex = (sampleMask > 0).nonzero().squeeze().cuda()
                # print (neighbourIndex)
                try:
                    tLen = neighbourIndex.size(-1)
                    lenInfo.append(tLen)
                    if tLen > maxLen:
                        maxLen = tLen
                except:
                    print(neighbourIndex)
                targetNeighbourFeature = torch.index_select(pre_feature, 0, neighbourIndex)
                dim = targetNeighbourFeature.size(-1)
                targetSamp_neighs.append(targetNeighbourFeature.detach().cpu().numpy())
            else:
                Yemb.append(0)
    print(Counter(Yemb))
    maxLen = int(np.mean(lenInfo))  # todo 多少 padding
    print ('mean Len: ' + str(maxLen))
    # 这里要做的 是 造出SMOTE最近的点
    from imblearn.over_sampling import SMOTE
    from imblearn.over_sampling import ADASYN
    import smote_variants as sv
    #oversampler = sv.Borderline_SMOTE2(n_neighbors = 50, k_neighbors=50)
    oversampler = sv.MWMOTE(k1 = 50, k2 = 50, k3 = 50) # 距离多数类样本越近密度越低越容易抽到 https://blog.csdn.net/weixin_40118768/article/details/80282378
    # blog 500 其他 50
    N_NEIGHBOURS = len(Xemb)  # Blog

    nbrs = NearestNeighbors(n_neighbors=N_NEIGHBOURS, algorithm='auto').fit(Xemb)

    Xsmote, Ysmote = oversampler.fit_resample(np.array(Xemb), np.array(Yemb))
    print(Counter(Ysmote))


    recordEpoch = []  # 哪一个点
    recordEmb = []  # Embedding
    print ('Xemb: ' + str( len(Xemb) ) + ' ; Xsmote: ' + str( len(Xsmote) ))

    Xshuffle = Xsmote[len(Xemb): ]
    #random.shuffle(Xshuffle)
    n_samples = int(float(len(minClass)) * portion)
    print ('n_samples: ' + str(n_samples))
    for i in range(0, min( n_samples, len(Xshuffle))):
        tmp = np.array(Xshuffle[i]).reshape(1, -1)
        distances, indices = nbrs.kneighbors( tmp )
        for j in indices[0]:
            if Yemb[j] == 1: # 属于哪一个小类
                recordEpoch.append( j )
                break
        recordEmb.append(Xshuffle[i])
    assert len(recordEpoch) == len(recordEmb)
    print ('recordEpoch ' + str(len(recordEpoch)))

    print ('recordEmb ' + str(len(recordEmb)))
    #exit()

    return targetEmb, targetIndex, targetSamp_neighs, maxLen, dim, recordEpoch, recordEmb



def MulcheckUnSunEmbAdaSyn2(dataCenter, ds, graphSage, b_sz, targetLabel, portion = .5):
    train_nodes = getattr(dataCenter, ds + '_train')
    labels = getattr(dataCenter, ds + '_labels')

    batches = math.ceil(len(train_nodes) / b_sz)
    visited_nodes = set()
    import copy
    targetEmb = []  # 记录隐层
    targetIndex = []  # 记录 index
    targetSamp_neighs = []
    targetMask = []
    maxLen = 0
    dim = 0
    # todo 记录 Emb信息 先record起来吧 ppi的
    Xemb = []
    Yemb = []
    count = -1
    minClass = []
    lenInfo = []
    Yemb2 = []
    Xembmin = []
    mintoAll = {}

    for index in range(batches):
        nodes_batch = train_nodes[index * b_sz:(index + 1) * b_sz]  # 看这个 nodes batch就行 这个不打乱， 可以用
        visited_nodes |= set(nodes_batch)
        # get ground-truth for the nodes batch
        labels_batch = labels[nodes_batch]
        # feed nodes batch to the graphSAGE
        # returning the nodes embeddings
        # pre_hidden_embs, pre_feature, samp_neighs
        embs_batch, pre_feature, samp_neighs, mask, _ = graphSage(nodes_batch)  #
        # 通过 embs_batch 返回 embedding
        # 通过 samp_neighs 和 mask 返回 邻居信息 进行解码
        # labels_batch2 = copy.deepcopy(labels_batch)
        # embs_batch2 = copy.deepcopy(embs_batch)

        labels_batch_data = labels_batch
        embs_batch = embs_batch

        emb_cpu = embs_batch.detach().cpu().numpy()
        Xemb.extend(emb_cpu)
        #Yemb.extend(labels_batch)

        for i in range(0, len(labels_batch_data)):
            count += 1
            if  labels_batch_data[i][targetLabel] == 1:  # 如果是目标
                mintoAll[len(Xembmin)] = len(Yemb)

                Yemb.append(1)
                Yemb2.append(labels_batch_data[i])
                Xembmin.append(emb_cpu[i])
                minClass.append(count)
                targetEmb.append(emb_cpu[i])  # right
                targetIndex.append(nodes_batch[i])

                sampleMask = mask[i]  # 当前节点的邻居信息
                neighbourIndex = (sampleMask > 0).nonzero().squeeze().cuda()
                # print (neighbourIndex)
                try:
                    tLen = neighbourIndex.size(-1)
                    lenInfo.append(tLen)
                    if tLen > maxLen:
                        maxLen = tLen
                except:
                    print(neighbourIndex)
                targetNeighbourFeature = torch.index_select(pre_feature, 0, neighbourIndex)
                dim = targetNeighbourFeature.size(-1)
                targetSamp_neighs.append(targetNeighbourFeature.detach().cpu().numpy())
            else:
                Yemb.append(0)
    maxLen = int(np.mean(lenInfo))  # todo 多少 padding
    print ('mean Len: ' + str(maxLen))
    # 这里要做的 是 造出SMOTE最近的点
    from imblearn.over_sampling import SMOTE
    from imblearn.under_sampling import RandomUnderSampler
    import smote_variants as sv
    oversampler = sv.SMOTE()
    #oversampler = sv.MWMOTE() # 距离多数类样本越近密度越低越容易抽到 https://blog.csdn.net/weixin_40118768/article/details/80282378
    #oversampler = sv.Borderline_SMOTE2(n_neighbors = 50, k_neighbors=50)
    us = RandomUnderSampler(sampling_strategy = .1)
    X_resampled = np.array(Xemb)
    y_resampled = np.array(Yemb)

    #X_resampled, y_resampled = us.fit_resample(np.array(Xemb), np.array(Yemb))

    oversampler = sv.MWMOTE(k1 = 5, k2 = 5, k3 = 5, M=5) # 距离多数类样本越近密度越低越容易抽到 https://blog.csdn.net/weixin_40118768/article/details/80282378
    # blog 500 其他 50
    #oversample = ADASYN(n_neighbors=40)
    #oversample = SMOTE()
    N_NEIGHBOURS = 40
    #N_NEIGHBOURS = len(Xemb)  # Blog

    nbrs = NearestNeighbors(n_neighbors=N_NEIGHBOURS, algorithm='auto').fit(Xembmin)

    Xsmote, Ysmote = oversampler.fit_resample(X_resampled, y_resampled)
    recordEpoch = []  # 哪一个点
    recordEmb = []  # Embedding
    print ('Xemb: ' + str( len(Xemb) ) + ' ; Xsmote: ' + str( len(Xsmote) ))

    Xshuffle = Xsmote[len(X_resampled): ]
    #random.shuffle(Xshuffle)
    n_samples = int(float(len(minClass)) * portion)
    print ('n_samples: ' + str(n_samples))
    for i in range(0, min( n_samples, len(Xshuffle))):
        tmp = np.array(Xshuffle[i]).reshape(1, -1)
        distances, indices = nbrs.kneighbors( tmp )
        t = indices[0][0]
        recordEpoch.append(mintoAll[t])

        recordEmb.append(Xshuffle[i])
    assert len(recordEpoch) == len(recordEmb)

    #exit()

    return targetEmb, targetIndex, targetSamp_neighs, maxLen, dim, recordEpoch, recordEmb



def checkUnSunEmbAdaSynMulti(dataCenter, ds, graphSage, b_sz, targetLabel, portion = .5, dataDim = 1433, padding = 10):
    train_nodes = getattr(dataCenter, ds + '_train')
    labels = getattr(dataCenter, ds + '_labels')

    batches = math.ceil(len(train_nodes) / b_sz)
    visited_nodes = set()
    import copy
    targetEmb = []  # 记录隐层
    targetIndex = []  # 记录 index
    targetSamp_neighs = []
    targetMask = []
    maxLen = 0
    dim = 0
    maxLen = dataDim * 5  # todo cora 是5 publemd 是7  cora 2layer 的
    targetIndex = []
    targetIndex2 = []
    targetFeature = []
    targetFirstFeature = []
    Xemb = []
    Yemb = []
    minClass = 0
    Lens = []
    for index in range(batches):
        nodes_batch = train_nodes[index * b_sz:(index + 1) * b_sz]  # 看这个 nodes batch就行 这个不打乱， 可以用

        lower_layer_nodes = list(nodes_batch)
        nodes_batch_layers = [(lower_layer_nodes,)]  # 第0个 是 近的，后面的是远的
        # self.dc.logger.info('get_unique_neighs.')
        for i in range(2):
            lower_samp_neighs, lower_layer_nodes_dict, lower_layer_nodes = graphSage._get_unique_neighs_list(
                lower_layer_nodes)
            nodes_batch_layers.insert(0, (
            lower_layer_nodes, lower_samp_neighs, lower_layer_nodes_dict))  # nodes_batch_layer 是所有的信息
        # 注意！这个insert是插入最前 所以 是从后往前搞
        visited_nodes |= set(nodes_batch)

        # get ground-truth for the nodes batch
        labels_batch = labels[nodes_batch]

        embs_batch, pre_feature, cur_hidden_emb, mask, sage_pre_feature = graphSage(nodes_batch, need_previous = True)  # 好像用这玩意就可以找到哈哈哈哈！

        embs_batch = embs_batch[-1]
        pre_feature = pre_feature
        #samp_neighs = samp_neighs[-1]

        labels_batch_data = labels_batch
        embs_batch = embs_batch

        for i in range(0, len(labels_batch_data)):
            Xemb.append(embs_batch[i].detach().cpu().numpy())
            Yemb.append(labels_batch_data[i])
            if labels_batch_data[i] == targetLabel:  # 如果是目标
                minClass += 1
                lower_layer_nodes = list(nodes_batch)
                targetFeature.append(sage_pre_feature[1][i])  # 最近节点特征
                targetIndex.append(nodes_batch[i])

                ns = []
                # 在layer=2层进行追溯
                firstNeigh = mask[1][i]
                t = (firstNeigh > 0).nonzero()
                targetEmb.append(embs_batch[i].detach().cpu().numpy()) # 最里层embedding
                feature = []
                firstNeighbour = []
                if t.size(0) > 1:
                    neighbourIndex1 = (firstNeigh > 0).nonzero().squeeze().cuda()
                    # print (neighbourIndex1) # if not neighbour
                    # 这里有一个bug
                    ns.append(sage_pre_feature[0][neighbourIndex1])

                    for j in neighbourIndex1:
                        secondNeigh = mask[0][j]
                        neighbourIndex = (secondNeigh > 0).nonzero().squeeze().cuda()
                        #targetEmb.append(cur_hidden_emb[j].detach().cpu().numpy())
                        targetNeighbourFeature2 = torch.index_select(pre_feature[0], 0, neighbourIndex)
                        targetNeighbourFeature3 = torch.flatten(targetNeighbourFeature2)
                        # 拉平之后 补0
                        targetNeighbourFeature = torch.zeros(dataDim + maxLen)
                        targetNeighbourFeature[0: dataDim] = sage_pre_feature[0][j]
                        if (targetNeighbourFeature3.size(0)) <= maxLen:
                            targetNeighbourFeature[dataDim : (dataDim + targetNeighbourFeature3.size(0))] = targetNeighbourFeature3
                        else :
                            targetNeighbourFeature[dataDim: (dataDim + maxLen)] = targetNeighbourFeature3[:maxLen]

                        dim = targetNeighbourFeature.size(-1)
                        # 打平 连接到 ns 后面
                        feature.append(targetNeighbourFeature.detach().cpu().numpy())

                        try:
                            tLen = neighbourIndex.size(-1)
                            Lens.append(tLen)
                            if tLen > maxLen:
                                maxLen = tLen
                        except:
                            print(neighbourIndex)
                else:  # 只有一个邻居 todo 以后要搞没有邻居
                    neighbourIndex1 = (firstNeigh > 0).nonzero().squeeze().cuda()
                    j = neighbourIndex1
                    secondNeigh = mask[0][j]
                    neighbourIndex = (secondNeigh > 0).nonzero().squeeze().cuda()
                    targetNeighbourFeature2 = torch.index_select(pre_feature[0], 0, neighbourIndex)
                    targetNeighbourFeature3 = torch.flatten(targetNeighbourFeature2)
                    # 拉平之后 补0
                    targetNeighbourFeature = torch.zeros(dataDim + maxLen)
                    targetNeighbourFeature[0: dataDim] = sage_pre_feature[0][j]
                    #if (targetNeighbourFeature3.size(0)) < maxLen:
                    targetNeighbourFeature[dataDim: (targetNeighbourFeature3.size(0) + dataDim)] = targetNeighbourFeature3[0: min(maxLen, targetNeighbourFeature3.size(0))]

                    dim = targetNeighbourFeature.size(-1)
                    feature.append(targetNeighbourFeature.detach().cpu().numpy())
                    ns.append(sage_pre_feature[0][neighbourIndex1])

                    try:
                        tLen = neighbourIndex.size(-1)
                        if tLen > maxLen:
                            maxLen = tLen
                    except:
                        print(neighbourIndex)

                # todo
                # targetEmb.append(embs_batch[i].detach().cpu().numpy()) # right
                # targetIndex.append(nodes_batch[i])
                targetFirstFeature.append(ns)
                targetSamp_neighs.append(feature)
            # print (neighbourIndex)
        # 这里要做的 是 造出SMOTE最近的点
    from imblearn.over_sampling import SMOTE
    from imblearn.over_sampling import ADASYN
    import smote_variants as sv
    oversampler = sv.MSYN()
    oversampler = sv.MWMOTE()  # 距离多数类样本越近密度越低越容易抽到 https://blog.csdn.net/weixin_40118768/article/details/80282378
    # oversampler = sv.Borderline_SMOTE2(n_neighbors = 50, k_neighbors=50)
    oversampler = sv.MWMOTE(k1=5, k2=5,
                            k3=5)  # 距离多数类样本越近密度越低越容易抽到 https://blog.csdn.net/weixin_40118768/article/details/80282378

    # oversample = ADASYN(n_neighbors=40)
    # oversample = SMOTE()
    N_NEIGHBOURS = 40
    N_NEIGHBOURS = len(Xemb)  # Blog

    nbrs = NearestNeighbors(n_neighbors=N_NEIGHBOURS, algorithm='auto').fit(Xemb)

    Xsmote, Ysmote = oversampler.fit_resample(np.array(Xemb), np.array(Yemb))
    recordEpoch = []  # 哪一个点
    recordEmb = []  # Embedding
    print('Xemb: ' + str(len(Xemb)) + ' ; Xsmote: ' + str(len(Xsmote)))

    Xshuffle = Xsmote[len(Xemb):]
    # random.shuffle(Xshuffle)
    n_samples = int(float((minClass)) * portion)
    #n_samples = 1
    print('n_samples: ' + str(n_samples))
    for i in range(0, min(n_samples, len(Xshuffle))):
        tmp = np.array(Xshuffle[i]).reshape(1, -1)
        distances, indices = nbrs.kneighbors(tmp)
        for j in indices[0]:
            if Yemb[j] == 1:  # 属于哪一个小类
                recordEpoch.append(j)
                break
        recordEmb.append(Xshuffle[i])
    assert len(recordEpoch) == len(recordEmb)
    maxLen = int(np.mean(Lens)) # todo 长度
    print ('maxLen : ' + str(maxLen))
    return targetEmb, targetIndex, targetSamp_neighs, maxLen, dim, \
           recordEpoch, recordEmb



def MulcheckUnSunEmbAdaSynMultiSyn2(dataCenter, ds, graphSage, b_sz, targetLabel, portion = .5, dataDim = 1433, padding = 10):
    train_nodes = getattr(dataCenter, ds + '_train')
    labels = getattr(dataCenter, ds + '_labels')

    batches = math.ceil(len(train_nodes) / b_sz)
    visited_nodes = set()
    import copy
    targetEmb = []  # 记录隐层
    targetIndex = []  # 记录 index
    targetSamp_neighs = []
    targetMask = []
    maxLen = 0
    dim = 0
    maxLen = dataDim * 5  # todo cora 是5 publemd 是7  ppi 2layer 的 来自 2 layer的改动
    targetIndex = []
    targetIndex2 = []
    targetFeature = []
    targetFirstFeature = []
    Xemb = []
    Yemb = []
    minClass = []
    Lens = []
    Yemb2 = []
    Xembmin = []
    mintoAll = {}
    count = -1

    for index in range(batches):
        nodes_batch = train_nodes[index * b_sz:(index + 1) * b_sz]  # 看这个 nodes batch就行 这个不打乱， 可以用

        lower_layer_nodes = list(nodes_batch)
        nodes_batch_layers = [(lower_layer_nodes,)]  # 第0个 是 近的，后面的是远的
        # self.dc.logger.info('get_unique_neighs.')
        for i in range(2):
            lower_samp_neighs, lower_layer_nodes_dict, lower_layer_nodes = graphSage._get_unique_neighs_list(
                lower_layer_nodes)
            nodes_batch_layers.insert(0, (
            lower_layer_nodes, lower_samp_neighs, lower_layer_nodes_dict))  # nodes_batch_layer 是所有的信息
        # 注意！这个insert是插入最前 所以 是从后往前搞
        visited_nodes |= set(nodes_batch)

        # get ground-truth for the nodes batch
        labels_batch = labels[nodes_batch]

        embs_batch, pre_feature, cur_hidden_emb, mask, sage_pre_feature = graphSage(nodes_batch, need_previous = True)  # 好像用这玩意就可以找到哈哈哈哈！

        embs_batch = embs_batch[-1]
        pre_feature = pre_feature
        #samp_neighs = samp_neighs[-1]

        labels_batch_data = labels_batch
        embs_batch = embs_batch

        emb_cpu = embs_batch.detach().cpu().numpy()
        Xemb.extend(emb_cpu)


        for i in range(0, len(labels_batch_data)):
            #Xemb.append(embs_batch[i].detach().cpu().numpy())
            #Yemb.append(labels_batch_data[i])
            count += 1

            if labels_batch_data[i][targetLabel] == 1:  # 如果是目标

                mintoAll[len(targetEmb)] = len(Yemb)

                Yemb.append(1)
                Yemb2.append(labels_batch_data[i])
                minClass.append(count)
                targetEmb.append(emb_cpu[i])  # right
                targetIndex.append(nodes_batch[i])

                targetFeature.append(sage_pre_feature[1][i])  # 最近节点特征
                targetIndex.append(nodes_batch[i])

                ns = []
                # 在layer=2层进行追溯
                firstNeigh = mask[1][i]
                t = (firstNeigh > 0).nonzero()
                feature = []
                firstNeighbour = []
                if t.size(0) > 1:
                    neighbourIndex1 = (firstNeigh > 0).nonzero().squeeze().cuda()
                    # print (neighbourIndex1) # if not neighbour
                    # 这里有一个bug
                    ns.append(sage_pre_feature[0][neighbourIndex1])

                    for j in neighbourIndex1:
                        secondNeigh = mask[0][j]
                        neighbourIndex = (secondNeigh > 0).nonzero().squeeze().cuda()
                        #targetEmb.append(cur_hidden_emb[j].detach().cpu().numpy())
                        targetNeighbourFeature2 = torch.index_select(pre_feature[0], 0, neighbourIndex)
                        targetNeighbourFeature3 = torch.flatten(targetNeighbourFeature2)
                        # 拉平之后 补0
                        targetNeighbourFeature = torch.zeros(dataDim + maxLen)
                        targetNeighbourFeature[0: dataDim] = sage_pre_feature[0][j]
                        if (targetNeighbourFeature3.size(0)) <= maxLen:
                            targetNeighbourFeature[dataDim : (dataDim + targetNeighbourFeature3.size(0))] = targetNeighbourFeature3
                        else :
                            targetNeighbourFeature[dataDim: (dataDim + maxLen)] = targetNeighbourFeature3[:maxLen]

                        dim = targetNeighbourFeature.size(-1)
                        # 打平 连接到 ns 后面
                        feature.append(targetNeighbourFeature.detach().cpu().numpy())

                        try:
                            tLen = neighbourIndex.size(-1)
                            Lens.append(tLen)
                            if tLen > maxLen:
                                maxLen = tLen
                        except:
                            print(neighbourIndex)
                else:  # 只有一个邻居 todo 以后要搞没有邻居
                    neighbourIndex1 = (firstNeigh > 0).nonzero().squeeze().cuda()
                    j = neighbourIndex1
                    secondNeigh = mask[0][j]
                    neighbourIndex = (secondNeigh > 0).nonzero().squeeze().cuda()
                    targetNeighbourFeature2 = torch.index_select(pre_feature[0], 0, neighbourIndex)
                    targetNeighbourFeature3 = torch.flatten(targetNeighbourFeature2)
                    # 拉平之后 补0
                    targetNeighbourFeature = torch.zeros(dataDim + maxLen)
                    targetNeighbourFeature[0: dataDim] = sage_pre_feature[0][j]
                    #if (targetNeighbourFeature3.size(0)) < maxLen:
                    targetNeighbourFeature[dataDim: (targetNeighbourFeature3.size(0) + dataDim)] = targetNeighbourFeature3[0: min(maxLen, targetNeighbourFeature3.size(0))]

                    dim = targetNeighbourFeature.size(-1)
                    feature.append(targetNeighbourFeature.detach().cpu().numpy())
                    ns.append(sage_pre_feature[0][neighbourIndex1])

                    try:
                        tLen = neighbourIndex.size(-1)
                        if tLen > maxLen:
                            maxLen = tLen
                    except:
                        print(neighbourIndex)

                # todo
                # targetEmb.append(embs_batch[i].detach().cpu().numpy()) # right
                # targetIndex.append(nodes_batch[i])
                targetFirstFeature.append(ns)
                targetSamp_neighs.append(feature)
            else:
                Yemb.append(0)
            # print (neighbourIndex)
        # 这里要做的 是 造出SMOTE最近的点
    from imblearn.over_sampling import SMOTE
    from imblearn.over_sampling import ADASYN
    import smote_variants as sv
    oversampler = sv.MSYN()
    oversampler = sv.MWMOTE()  # 距离多数类样本越近密度越低越容易抽到 https://blog.csdn.net/weixin_40118768/article/details/80282378
    # oversampler = sv.Borderline_SMOTE2(n_neighbors = 50, k_neighbors=50)
    oversampler = sv.MWMOTE(k1=5, k2=5,
                            k3=5, M=5)  # 距离多数类样本越近密度越低越容易抽到 https://blog.csdn.net/weixin_40118768/article/details/80282378
    #oversampler = sv.SMOTE()
    # oversample = ADASYN(n_neighbors=40)
    # oversample = SMOTE()
    N_NEIGHBOURS = 40
    #N_NEIGHBOURS = len(Xemb)  # Blog


    nbrs = NearestNeighbors(n_neighbors=N_NEIGHBOURS, algorithm='auto').fit(targetEmb)

    Xsmote, Ysmote = oversampler.fit_resample(np.array(Xemb), np.array(Yemb))
    recordEpoch = []  # 哪一个点
    recordEmb = []  # Embedding
    print ('Xemb: ' + str( len(Xemb) ) + ' ; Xsmote: ' + str( len(Xsmote) ))

    Xshuffle = Xsmote[len(Xemb): ]
    #random.shuffle(Xshuffle)
    n_samples = int(float(len(minClass)) * portion)
    print ('n_samples: ' + str(n_samples))
    for i in range(0, min( n_samples, len(Xshuffle))):
        tmp = np.array(Xshuffle[i]).reshape(1, -1)
        distances, indices = nbrs.kneighbors( tmp )
        t = indices[0][0]
        recordEpoch.append(mintoAll[t])

        recordEmb.append(Xshuffle[i])
    assert len(recordEpoch) == len(recordEmb)

    maxLen = int(np.mean(Lens)) # todo 长度
    print ('maxLen : ' + str(maxLen))
    return targetEmb, targetIndex, targetSamp_neighs, maxLen, dim, \
           recordEpoch, recordEmb


def buildMultiLayerEmb(embs_batch, mask, num_layer):
    Xemb = {}
    Mask = {}
    indexEmb = []
    for i in range(0, num_layer):
        emb_cpu = embs_batch[i].detach().cpu().numpy()
        mask_cpu = mask[i].detach().cpu().numpy()
        if i not in Xemb.keys():
            Xemb[i] = []
            Xemb[i].extend(emb_cpu)
            Mask[i] = []
            Mask[i].extend(mask_cpu)
        else:
            Xemb[i].extend(emb_cpu)
            Mask[i].extend(mask_cpu)
    # 构建隐层特征
    Xfeature = []
    neighbourRecord = set()
    indexNeighour = []
    XembRecordNeighS = []
    for i in range(num_layer - 1, num_layer - 2, -1):
        # todo 先搞一层吧 不卷了
        for j in range(0, len(Mask[i])):
            feature = (Xemb[i][j]) # 当前特征
            previous_idxs = np.where(Mask[i][j] == 1)[0]
            for previous_idx in previous_idxs:
                feature_copy = list(copy.deepcopy(feature))
                neighbourRecord.add(previous_idx)
                Xemb_previous = Xemb[0][previous_idx]
                #feature_copy.extend(Xemb_previous) # 特征连接
                c = feature_copy + Xemb_previous # 特征相加
                Xfeature.append(c) # todo 2022 03-30
                indexEmb.append(j)
                indexNeighour.append(previous_idx)
                XembRecordNeighS.append(Xemb_previous)
    return Xfeature, indexEmb, indexNeighour, XembRecordNeighS


def checkUnSunEmbAdaSynMultiLayer(dataCenter, ds, graphSage, b_sz, targetLabel, minRecord_0, portion = .5, num_layer = 2):
    '''

    :param dataCenter:
    :param ds:
    :param graphSage:
    :param b_sz:
    :param targetLabel:
    :param minRecord_0:  最里面那层 的 ID
    :param portion:
    :param num_layer:
    :return:
    '''
    train_nodes = getattr(dataCenter, ds + '_train')
    labels = getattr(dataCenter, ds + '_labels')

    batches = math.ceil(len(train_nodes) / b_sz)
    visited_nodes = set()
    targetEmb = []  # 多层 使用 dict 进行记录 记录隐层
    targetIndex = []  # 记录 index
    targetSamp_neighs = []
    maxLen = 0
    dim = 0
    # todo 记录 Emb信息 先record起来吧
    Xemb = []
    Yemb = []
    count = -1
    minClass = []
    #print (minRecord_0)
    extendNode_neighourFeature = {}

    for index in range(batches):
        nodes_batch = train_nodes[index * b_sz:(index + 1) * b_sz]  # 看这个 nodes batch就行 这个不打乱， 可以用
        visited_nodes |= set(nodes_batch)
        # get ground-truth for the nodes batch
        labels_batch = labels[nodes_batch]
        # feed nodes batch to the graphSAGE
        # returning the nodes embeddings
        # pre_hidden_embs, pre_feature, samp_neighs
        embs_batch, pre_feature, samp_neighs, mask, _ = graphSage(nodes_batch, need_previous = True)  # 多层拿的是 List
        labels_batch_data = labels_batch # label 不用dict

        embs_batch = embs_batch
        Xfeature, indexEmb, indexNeighour, XembRecordNeighS = buildMultiLayerEmb(embs_batch, mask, num_layer)
        Xemb.extend(Xfeature)
        for i in range(0, len(indexEmb)):
            count += 1
            if labels_batch[indexEmb[i]] == targetLabel:
                Yemb.append(1)
                minClass.append(count)
                targetEmb.append(Xfeature[i])
                targetIndex.append([nodes_batch[indexEmb[i]], indexEmb[i]])  # todo 这个应该不需要？ 需要知道生成节点 和 那个 路线相近
                sampleMask = mask[0][indexNeighour[i]]
                neighbourIndex = (sampleMask > 0).nonzero().squeeze().cuda()
                try:
                    tLen = neighbourIndex.size(-1)
                    if tLen > maxLen:
                        maxLen = tLen
                except:
                    print(neighbourIndex)
                targetNeighbourFeature = torch.index_select(pre_feature[0], 0, neighbourIndex)
                dim = targetNeighbourFeature.size(-1)
                targetSamp_neighs.append(targetNeighbourFeature.detach().cpu().numpy())
                mR = nodes_batch[indexEmb[i]]
                #
                if nodes_batch[indexEmb[i]] in minRecord_0:
                    #print(mR)
                    if nodes_batch[indexEmb[i]] not in extendNode_neighourFeature:
                        extendNode_neighourFeature[nodes_batch[indexEmb[i]]] = []

                    extendNode_neighourFeature[nodes_batch[indexEmb[i]]].append(XembRecordNeighS[i]) # 一堆多 所以会少于 minRecord_0

            else:
                Yemb.append(0)
    #exit()
    '''
    # 这里要做的 是 造出SMOTE最近的点
    from imblearn.over_sampling import SMOTE
    from imblearn.over_sampling import ADASYN
    import smote_variants as sv
    #oversampler = sv.MSYN()
    #oversampler = sv.MWMOTE()
    #oversampler = sv.Borderline_SMOTE2(n_neighbors = 50, k_neighbors=50)
    oversampler = sv.MWMOTE(k1 = 50, k2 = 50, k3 = 50)
    # 距离多数类样本越近密度越低越容易抽到 https://blog.csdn.net/weixin_40118768/article/details/80282378

    #oversample = ADASYN(n_neighbors=40)
    #oversample = SMOTE()
    #N_NEIGHBOURS = 40
    N_NEIGHBOURS = len(Xemb)  # Blog

    nbrs = NearestNeighbors(n_neighbors=N_NEIGHBOURS, algorithm='auto').fit(Xemb)

    Xsmote, Ysmote = oversampler.fit_resample(np.array(Xemb), np.array(Yemb))
    recordEpoch = []  # 哪一个点
    recordEmb = []  # Embedding
    print ('Xemb: ' + str( len(Xemb) ) + ' ; Xsmote: ' + str( len(Xsmote) ))

    Xshuffle = Xsmote[len(Xemb): ]
    #random.shuffle(Xshuffle)
    n_samples = int(float(len(minClass)) * portion)
    print ('n_samples: ' + str(n_samples))
    for i in range(0, min( n_samples, len(Xshuffle))):
        tmp = np.array(Xshuffle[i]).reshape(1, -1)
        distances, indices = nbrs.kneighbors( tmp )
        for j in indices[0]:
            if Yemb[j] == 1: # 属于哪一个小类
                recordEpoch.append( j )
                break
        recordEmb.append(Xshuffle[i])
    assert len(recordEpoch) == len(recordEmb)
    '''
    #exit()
    return targetEmb, targetIndex, targetSamp_neighs, maxLen, dim, None, None, extendNode_neighourFeature

def checkUnSunEmbAdaSynMultiLayerCenter(dataCenter, ds, graphSage, b_sz, targetLabel, portion = .5, num_layer = 2):
    '''
    多层 中心 节点 做 Wsmb
    :param dataCenter:
    :param ds:
    :param graphSage:
    :param b_sz:
    :param targetLabel:
    :param portion:
    :param num_layer:
    :return:
    '''
    train_nodes = getattr(dataCenter, ds + '_train')
    labels = getattr(dataCenter, ds + '_labels')

    batches = math.ceil(len(train_nodes) / b_sz)
    visited_nodes = set()
    targetEmb = []  # 多层 使用 dict 进行记录 记录隐层
    targetIndex = []  # 记录 index
    targetSamp_neighs = []
    maxLen = 0
    dim = 0
    # todo 记录 Emb信息 先record起来吧
    Xemb = []
    Yemb = []
    count = -1
    minClass = []

    for index in range(batches):
        nodes_batch = train_nodes[index * b_sz:(index + 1) * b_sz]  # 看这个 nodes batch就行 这个不打乱， 可以用
        visited_nodes |= set(nodes_batch)
        # get ground-truth for the nodes batch
        labels_batch = labels[nodes_batch]
        # feed nodes batch to the graphSAGE
        # returning the nodes embeddings
        # pre_hidden_embs, pre_feature, samp_neighs
        embs_batch, pre_feature, samp_neighs, mask, _ = graphSage(nodes_batch, need_previous = True)  # 多层拿的是 List
        embs_batch = embs_batch[-1]
        pre_feature = pre_feature[-1]
        samp_neighs = samp_neighs[-1]
        mask = mask[-1]
        # 通过 embs_batch 返回 embedding
        # 通过 samp_neighs 和 mask 返回 邻居信息 进行解码
        # labels_batch2 = copy.deepcopy(labels_batch)
        # embs_batch2 = copy.deepcopy(embs_batch)

        labels_batch_data = labels_batch
        embs_batch = embs_batch

        emb_cpu = embs_batch.detach().cpu().numpy()
        Xemb.extend(emb_cpu)
        # Yemb.extend(labels_batch)

        for i in range(0, len(labels_batch_data)):
            count += 1
            targetIndex.append(nodes_batch[i])
            if labels_batch_data[i] == targetLabel:  # 如果是目标
                Yemb.append(1)
                minClass.append(count)
                targetEmb.append(emb_cpu[i])  # right


                sampleMask = mask[i]  # 当前节点的邻居信息
                neighbourIndex = (sampleMask > 0).nonzero().squeeze().cuda()
                # print (neighbourIndex)
                try:
                    tLen = neighbourIndex.size(-1)
                    if tLen > maxLen:
                        maxLen = tLen
                except:
                    print(neighbourIndex)
                targetNeighbourFeature = torch.index_select(pre_feature, 0, neighbourIndex)
                dim = targetNeighbourFeature.size(-1)
                targetSamp_neighs.append(targetNeighbourFeature.detach().cpu().numpy())
            else:
                Yemb.append(0)

    # 这里要做的 是 造出SMOTE最近的点
    from imblearn.over_sampling import SMOTE
    from imblearn.over_sampling import ADASYN
    import smote_variants as sv
    oversampler = sv.MSYN()
    oversampler = sv.MWMOTE()  # 距离多数类样本越近密度越低越容易抽到 https://blog.csdn.net/weixin_40118768/article/details/80282378
    # oversampler = sv.Borderline_SMOTE2(n_neighbors = 50, k_neighbors=50)
    oversampler = sv.MWMOTE(k1=50, k2=50,
                            k3=50)  # 距离多数类样本越近密度越低越容易抽到 https://blog.csdn.net/weixin_40118768/article/details/80282378

    # oversample = ADASYN(n_neighbors=40)
    # oversample = SMOTE()
    N_NEIGHBOURS = 40
    N_NEIGHBOURS = len(Xemb)  # Blog

    nbrs = NearestNeighbors(n_neighbors=N_NEIGHBOURS, algorithm='auto').fit(Xemb)

    Xsmote, Ysmote = oversampler.fit_resample(np.array(Xemb), np.array(Yemb))
    recordEpoch = []  # 哪一个点
    recordEmb = []  # Embedding
    print('Xemb: ' + str(len(Xemb)) + ' ; Xsmote: ' + str(len(Xsmote)))

    Xshuffle = Xsmote[len(Xemb):]
    # random.shuffle(Xshuffle)
    n_samples = int(float(len(minClass)) * portion)
    print('n_samples: ' + str(n_samples))
    for i in range(0, min(n_samples, len(Xshuffle))):
        tmp = np.array(Xshuffle[i]).reshape(1, -1)
        distances, indices = nbrs.kneighbors(tmp)
        for j in indices[0]:
            if Yemb[j] == 1:  # 属于哪一个小类
                recordEpoch.append(targetIndex[j]) # todo 这个只是X里面哪一个点而已 和 原始的 nodeBatch 无关系
                break
        recordEmb.append(Xshuffle[i])
    assert len(recordEpoch) == len(recordEmb)

    # exit()

    return targetEmb, targetIndex, targetSamp_neighs, maxLen, dim, recordEpoch, recordEmb




def checkUnSunEmbSMOTE(dataCenter, ds, graphSage, b_sz, targetLabel, portion):
    train_nodes = getattr(dataCenter, ds + '_train')
    labels = getattr(dataCenter, ds + '_labels')


    batches = math.ceil(len(train_nodes) / b_sz)
    visited_nodes = set()
    import copy
    targetEmb = []  # 记录隐层
    targetIndex = []  # 记录 index
    targetSamp_neighs = []
    targetMask = []
    maxLen = 0
    dim = 0
    # todo 记录 Emb信息 先record起来吧
    Xemb = []
    Yemb = []
    count = -1
    minClass = []

    for index in range(batches):
        nodes_batch = train_nodes[index * b_sz:(index + 1) * b_sz]  # 看这个 nodes batch就行 这个不打乱， 可以用

        # extend nodes batch for unspervised learning
        # no conflicts with supervised learning
        # print (nodes_batch)
        # nodes_batch = np.asarray(list(unsupervised_loss.extend_nodes(nodes_batch, num_neg=num_neg)))
        visited_nodes |= set(nodes_batch)

        # get ground-truth for the nodes batch
        labels_batch = labels[nodes_batch]

        # feed nodes batch to the graphSAGE
        # returning the nodes embeddings
        # pre_hidden_embs, pre_feature, samp_neighs
        embs_batch, pre_feature, samp_neighs, mask, _ = graphSage(nodes_batch)  #
        # 通过 embs_batch 返回 embedding
        # 通过 samp_neighs 和 mask 返回 邻居信息 进行解码
        # labels_batch2 = copy.deepcopy(labels_batch)
        # embs_batch2 = copy.deepcopy(embs_batch)

        labels_batch_data = labels_batch
        embs_batch = embs_batch

        emb_cpu = embs_batch.detach().cpu().numpy()
        Xemb.extend(emb_cpu)
        #Yemb.extend(labels_batch)

        for i in range(0, len(labels_batch_data)):
            count += 1
            if labels_batch_data[i] == targetLabel:  # 如果是目标
                Yemb.append(1)
                minClass.append(count)
                targetEmb.append(emb_cpu[i])  # right
                targetIndex.append(nodes_batch[i])

                sampleMask = mask[i]  # 当前节点的邻居信息
                neighbourIndex = (sampleMask > 0).nonzero().squeeze().cuda()
                # print (neighbourIndex)
                try:
                    tLen = neighbourIndex.size(-1)
                    if tLen > maxLen:
                        maxLen = tLen
                except:
                    print(neighbourIndex)
                targetNeighbourFeature = torch.index_select(pre_feature, 0, neighbourIndex)
                dim = targetNeighbourFeature.size(-1)
                targetSamp_neighs.append(targetNeighbourFeature.detach().cpu().numpy())
            else:
                Yemb.append(0)

    # 这里要做的 是 造出SMOTE最近的点
    from imblearn.over_sampling import SMOTE
    #from imblearn.over_sampling import SMOTE
    oversample = SMOTE()
    from imblearn.over_sampling import ADASYN
    import smote_variants as sv
    #oversampler = sv.MSYN()
    #oversampler = sv.MWMOTE()  # 距离多数类样本越近密度越低越容易抽到 https://blog.csdn.net/weixin_40118768/article/details/80282378
    # oversampler = sv.Borderline_SMOTE2(n_neighbors = 50, k_neighbors=50)
    oversampler = sv.MWMOTE(k1=50, k2=50,
                            k3=50)  # 距离多数类样本越近密度越低越容易抽到 https://blog.csdn.net/weixin_40118768/article/details/80282378


    Xsmote, Ysmote = oversample.fit_resample(Xemb, Yemb)
    recordEmb = []
    over_nnumber = int(float(len(minClass)) * portion)
    print (over_nnumber)

    for i in range(len(Xemb), min(len(Xemb) + over_nnumber, len(Xsmote))): # SMOTE  采样
        recordEmb.append(Xsmote[i])

    #exit()  recordEpoch 这个不需要

    return targetEmb, targetIndex, targetSamp_neighs, maxLen, dim, recordEmb


def MulcheckUnSunEmbSMOTE(dataCenter, ds, graphSage, b_sz, targetLabel, portion):
    train_nodes = getattr(dataCenter, ds + '_train')
    labels = getattr(dataCenter, ds + '_labels')


    batches = math.ceil(len(train_nodes) / b_sz)
    visited_nodes = set()
    import copy
    targetEmb = []  # 记录隐层
    targetIndex = []  # 记录 index
    targetSamp_neighs = []
    targetMask = []
    maxLen = 0
    dim = 0
    # todo 记录 Emb信息 先record起来吧
    Xemb = []
    Yemb = []
    count = -1
    minClass = []
    Yemb2 = []
    Xembmin = []

    for index in range(batches):
        nodes_batch = train_nodes[index * b_sz:(index + 1) * b_sz]  # 看这个 nodes batch就行 这个不打乱， 可以用

        # extend nodes batch for unspervised learning
        # no conflicts with supervised learning
        # print (nodes_batch)
        # nodes_batch = np.asarray(list(unsupervised_loss.extend_nodes(nodes_batch, num_neg=num_neg)))
        visited_nodes |= set(nodes_batch)

        # get ground-truth for the nodes batch
        labels_batch = labels[nodes_batch]

        # feed nodes batch to the graphSAGE
        # returning the nodes embeddings
        # pre_hidden_embs, pre_feature, samp_neighs
        embs_batch, pre_feature, samp_neighs, mask, _ = graphSage(nodes_batch)  #
        # 通过 embs_batch 返回 embedding
        # 通过 samp_neighs 和 mask 返回 邻居信息 进行解码
        # labels_batch2 = copy.deepcopy(labels_batch)
        # embs_batch2 = copy.deepcopy(embs_batch)

        labels_batch_data = labels_batch
        embs_batch = embs_batch

        emb_cpu = embs_batch.detach().cpu().numpy()
        Xemb.extend(emb_cpu)
        #Yemb.extend(labels_batch)

        for i in range(0, len(labels_batch_data)):
            count += 1
            if  labels_batch_data[i][targetLabel] == 1:  # 如果是目标
                Yemb.append(1)
                Yemb2.append(labels_batch_data[i])
                Xembmin.append(emb_cpu[i])

                minClass.append(count)
                targetEmb.append(emb_cpu[i])  # right
                targetIndex.append(nodes_batch[i])

                sampleMask = mask[i]  # 当前节点的邻居信息
                neighbourIndex = (sampleMask > 0).nonzero().squeeze().cuda()
                # print (neighbourIndex)
                try:
                    tLen = neighbourIndex.size(-1)
                    if tLen > maxLen:
                        maxLen = tLen
                except:
                    print(neighbourIndex)
                targetNeighbourFeature = torch.index_select(pre_feature, 0, neighbourIndex)
                dim = targetNeighbourFeature.size(-1)
                targetSamp_neighs.append(targetNeighbourFeature.detach().cpu().numpy())
            else:
                Yemb.append(0)

    # 这里要做的 是 造出SMOTE最近的点
    from imblearn.over_sampling import SMOTE
    #from imblearn.over_sampling import SMOTE
    oversample = SMOTE()
    #from imblearn.over_sampling import ADASYN
    #import smote_variants as sv
    #oversampler = sv.MSYN()
    #oversampler = sv.MWMOTE()  # 距离多数类样本越近密度越低越容易抽到 https://blog.csdn.net/weixin_40118768/article/details/80282378
    # oversampler = sv.Borderline_SMOTE2(n_neighbors = 50, k_neighbors=50)
    #oversampler = sv.MWMOTE(k1=50, k2=50,
    #                        k3=50)  # 距离多数类样本越近密度越低越容易抽到 https://blog.csdn.net/weixin_40118768/article/details/80282378


    Xsmote, Ysmote = oversample.fit_resample(Xemb, Yemb)
    recordEmb = []
    over_nnumber = int(float(len(minClass)) * portion)
    print (over_nnumber)

    from sklearn.neighbors import NearestNeighbors
    N_NEIGHBOURS = 1
    nbrs = NearestNeighbors(n_neighbors=N_NEIGHBOURS, algorithm='auto').fit(Xembmin)
    recordLabel = []
    for i in range(len(Xemb), min(len(Xemb) + over_nnumber, len(Xsmote))): # SMOTE  采样
        recordEmb.append(Xsmote[i])
        distances, indices = nbrs.kneighbors(np.array(Xsmote[i]).reshape(1, -1))
        t = indices[0][0]
        recordLabel.append(Yemb2[t])

    #exit()  recordEpoch 这个不需要

    return targetEmb, targetIndex, targetSamp_neighs, maxLen, dim, recordEmb, recordLabel


def checkUnSunGSSMOTE(dataCenter, ds, graphSage, b_sz, targetLabel, over_nnumber):
    train_nodes = getattr(dataCenter, ds + '_train')
    labels = getattr(dataCenter, ds + '_labels')
    val_nodes = getattr(dataCenter, ds + '_val')
    test_nodes = getattr(dataCenter, ds + '_test')

    batches = math.ceil(len(train_nodes) / b_sz)
    visited_nodes = set()
    import copy
    targetEmb = []  # 记录隐层
    targetIndex = []  # 记录 index
    targetSamp_neighs = []
    targetMask = []
    maxLen = 0
    dim = 0
    # todo 记录 Emb信息 先record起来吧
    Xemb = []
    Yemb = []
    Xmin = []
    count = -1
    minClass = []

    for index in range(batches):
        nodes_batch = train_nodes[index * b_sz:(index + 1) * b_sz]  # 看这个 nodes batch就行 这个不打乱， 可以用

        # extend nodes batch for unspervised learning
        # no conflicts with supervised learning
        # print (nodes_batch)
        # nodes_batch = np.asarray(list(unsupervised_loss.extend_nodes(nodes_batch, num_neg=num_neg)))
        visited_nodes |= set(nodes_batch)

        # get ground-truth for the nodes batch
        labels_batch = labels[nodes_batch]

        # feed nodes batch to the graphSAGE
        # returning the nodes embeddings
        # pre_hidden_embs, pre_feature, samp_neighs
        embs_batch, pre_feature, samp_neighs, mask, _ = graphSage(nodes_batch)  #
        # 通过 embs_batch 返回 embedding
        # 通过 samp_neighs 和 mask 返回 邻居信息 进行解码
        # labels_batch2 = copy.deepcopy(labels_batch)
        # embs_batch2 = copy.deepcopy(embs_batch)

        labels_batch_data = labels_batch
        embs_batch = embs_batch

        emb_cpu = embs_batch.detach().cpu().numpy()
        Xemb.extend(emb_cpu)

        # Yemb.extend(labels_batch)

        for i in range(0, len(labels_batch_data)):
            count += 1
            if labels_batch_data[i] == targetLabel:  # 如果是目标
                Xmin.append(emb_cpu[i])
                Yemb.append(1)
                minClass.append(count)
                targetEmb.append(emb_cpu[i])  # right
                targetIndex.append(nodes_batch[i])

                sampleMask = mask[i]  # 当前节点的邻居信息
                neighbourIndex = (sampleMask > 0).nonzero().squeeze().cuda()
                # print (neighbourIndex)
                try:
                    tLen = neighbourIndex.size(-1)
                    if tLen > maxLen:
                        maxLen = tLen
                except:
                    #print(neighbourIndex)
                    z = 1
                targetNeighbourFeature = torch.index_select(pre_feature, 0, neighbourIndex)
                dim = targetNeighbourFeature.size(-1)
                targetSamp_neighs.append(targetNeighbourFeature.detach().cpu().numpy())
            else:
                Yemb.append(0)

    nodes_batch = np.hstack((train_nodes, val_nodes, test_nodes))
    emb, _, _, _, _ = graphSage(nodes_batch)
    Xallemb = list(emb.detach().cpu().numpy())

    # 这里要做的 是 造出SMOTE最近的点
    from imblearn.over_sampling import SMOTE
    # from imblearn.over_sampling import SMOTE
    oversample = SMOTE()
    Xsmote, Ysmote = oversample.fit_resample(Xemb, Yemb)
    recordEmb = []
    XallEmb = Xemb
    idx_train_new =  list( getattr(dataCenter, ds + '_train') )
    labels_new = list(labels)
    # todo 做个邻居查找而已 正宗的SMOTE 要做插值
    N_NEIGHBOURS = len(Xemb)
    nbrs = NearestNeighbors(n_neighbors=N_NEIGHBOURS, algorithm='auto').fit(Xemb)
    #nbrs_min = NearestNeighbors(n_neighbors=80, algorithm='auto').fit(Xmin)

    newLabel = []
    for i in range(len(Xemb), min(len(Xemb) + over_nnumber, len(Xsmote))):  # SMOTE  采样
        recordEmb.append(Xsmote[i])
        Xallemb.append(Xsmote[i])
        idx_train_new.append(len(Xallemb) - 1)
        labels_new.append(targetLabel)

        tmp = np.array(Xsmote[i]).reshape(1, -1)

        distances, indices = nbrs.kneighbors(tmp)
        fs = []
        for j in indices[0]:
            if Yemb[j] == 1:  # 属于小类
                fs.append(j)
                if len(fs) == 2:
                    newLabel.append(fs)
                    break
        '''
        fs = []
        distances, indices = nbrs_min.kneighbors(tmp)
        for j in indices[0]:
            fs.append(j)
            if len(fs) == 2:
                newLabel.append(fs)
                break
        '''

    print (newLabel)

    # exit()  recordEpoch 这个不需要

    return targetEmb, targetIndex, targetSamp_neighs, maxLen, dim, recordEmb, Xallemb, idx_train_new, torch.tensor(labels_new), newLabel
