import os
import torch
from torch import nn
import numpy as np

'''

'''

class EFGenerator(nn.Module):
    def __init__(self, input_dim, output_dim, hidden_dim = 32):
        super(EFGenerator, self).__init__()

        self.input_dim = input_dim
        self.output_dim = output_dim

        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc1.reset_parameters()

        self.fc2 = nn.Linear(hidden_dim, output_dim)
        self.fc2.reset_parameters()

        self.se1 = nn.Sequential(
            #nn.BatchNorm1d(input_dim, affine=True),
            self.fc1,
            nn.Tanh()
        )

        self.se2 = nn.Sequential(
            #nn.BatchNorm1d(hidden_dim, affine=True),
            self.fc2,
            nn.Tanh() #
        )

    def forward(self, Xembedding):
        x = self.se1(Xembedding)
        x = self.se2(x)
        return x


class EFDiscriminator(nn.Module):
    def __init__(self, input_dim_e, input_dim_f, output_dim, hidden_dim_1 = 32, hidden_dim_2 = 16):
        super(EFDiscriminator, self).__init__()
        self.input_dim_e = input_dim_e
        self.input_dim_f = input_dim_f
        self.output_dim = output_dim
        self.hidden_dim_1 = hidden_dim_1

        self.fce = nn.Linear(input_dim_e, hidden_dim_1)
        self.fce.reset_parameters()

        self.see = nn.Sequential(
            #nn.BatchNorm1d(input_dim_e, affine=True),
            self.fce,
        )


        self.fcf = nn.Linear(input_dim_f, hidden_dim_1)
        self.fcf.reset_parameters()

        self.sef = nn.Sequential(
            #nn.BatchNorm1d(input_dim_f, affine=True),
            self.fcf,
        )

        self.fc3 = nn.Linear(hidden_dim_1 * 2, hidden_dim_2)
        self.fc3.reset_parameters()

        self.fc4 = nn.Linear(hidden_dim_2, 2)
        self.fc4.reset_parameters()

        self.se3 = nn.Sequential(
            #nn.BatchNorm1d(hidden_dim_1 * 2, affine=True),
            self.fc3,
            #nn.BatchNorm1d(hidden_dim_2, affine=True),
            self.fc4,
            nn.Softmax(dim=1)
        )

    def forward(self, xe, xf):
        xe = self.see(xe)
        xf = self.sef(xf)
        x = torch.cat([xe, xf], 1)
        y = self.se3(x)
        return y



class pixGan():
    def __init__(self, minEmb, minIdex, minNeighs, maxLen, dataDim):
        self.minEmb = torch.Tensor(minEmb).cuda()
        self.minIdex = minIdex # 没用到
        self.minNeighs = minNeighs
        self.maxLen = maxLen # 邻居最长长度 对特征做处理
        self.dataDim = dataDim
        # self.first
        z = 1
        paddingNeighs = []
        self.embDim = self.minEmb[0].size(-1)
        for sampleT in self.minNeighs:
            # 进行填充
            if len(sampleT) > 1:
                if len(sampleT) < maxLen:
                    a = np.zeros((maxLen - len(sampleT), dataDim)) # padding 的值
                    b = np.concatenate((sampleT, a))
                elif len(sampleT) >= maxLen:
                    b = np.array(sampleT[:maxLen])


            else: # 只有一个
                a = np.zeros((maxLen - 1, dataDim))
                b = np.concatenate((sampleT, a))
            b = b.flatten()
            self.dataDim = len(b)
            paddingNeighs.append(np.array(b))

        self.paddingNeighs = torch.Tensor(paddingNeighs).cuda()

        efG = EFGenerator(self.embDim, self.dataDim)
        efD = EFDiscriminator(self.embDim, self.dataDim, 2)

        self.efG = efG.cuda() # 生成器
        self.efD = efD.cuda() # 判别器
        self.batchSize = self.minEmb.size(0) # 由于是少数据 都放进去
        z = 1

    def train(self):
        '''
        进行训练
        :return:
        '''
        criterion = nn.CrossEntropyLoss()

        optimizer_g = torch.optim.Adam(self.efG.parameters(),
                                       lr=1e-4)
        optimizer_d = torch.optim.Adam(self.efD.parameters(),
                                       lr=1e-4)
        epochs = 1001

        false_label = np.zeros(self.batchSize)
        false_label = torch.LongTensor(false_label)
        true_label = np.ones(self.batchSize)
        true_label = torch.LongTensor(true_label)

        false_label = false_label.cuda()
        true_label = true_label.cuda()

        for epoch in range(epochs):
            batch_xe = self.minEmb

            batch_xe2 = (0.1 ** 0.5) * torch.randn(self.batchSize, self.embDim).cuda()
            # todo 如果加噪音在理论上是不对的。。
            batch_xe += batch_xe2
            batch_xf = self.paddingNeighs

            fake_A = self.efG(batch_xe)
            ######################
            # (1) Update D network
            ######################
            optimizer_d.zero_grad()

            pred_fake = self.efD.forward(batch_xe, fake_A.detach())
            loss_d_fake = criterion(pred_fake, false_label)

            pred_real = self.efD.forward(batch_xe, batch_xf)
            loss_d_real = criterion(pred_real, true_label)

            loss_d = (loss_d_fake + loss_d_real) * 0.5

            loss_d.backward()

            optimizer_d.step()

            ######################
            # (2) Update G network
            ######################

            optimizer_g.zero_grad()

            pred_fake = self.efD.forward(batch_xe, fake_A)

            loss_g_gan = criterion(pred_fake, true_label)

            loss_g = loss_g_gan

            loss_g.backward()

            optimizer_g.step()
            if epoch % 10 == 0:
                print("===> Epoch:{}, Loss_D: {:.4f} Loss_G: {:.4f}".format(
                    epoch, loss_d.item(), loss_g.item()))

    def save(self):
        torch.save(self.efD, 'efD.pth')
        torch.save(self.efG, 'efG.pth')

    def load(self):
        self.efD = torch.load('efD.pth')
        self.efG = torch.load('efG.pth')

    def generate(self, simpleEnhance = 1, num_sample=10):
        '''
        生成一些数据
        :param simpleEnhance: 简单的提升生成 数量
        :param num_sample: 邻居数量
        :return:
        '''
        writeFile = open('targetGen.txt', 'w')
        generate_result = []
        for i in range(0, simpleEnhance):
            gene = self.efG(self.minEmb)
            gene2 = torch.reshape(gene, (gene.size(0), num_sample, -1))
            #gene = gene.reshape(-1, num_sample, -1)
            generate_result.append( gene2 )
            if 1:
                clone_detach_x = gene2.clone().detach()
                clone_detach_x2 = clone_detach_x.cpu().numpy()

                for ii in clone_detach_x2:
                    ii = list(ii)
                    for jj in ii:
                        jj = list(jj)
                        threshold = 0

                        for kk in jj:
                            if kk >= .5:
                                threshold += 1
                        #print (threshold)
                        ls = str(jj) + '\n'
                        writeFile.write(ls)
                        writeFile.flush()
        writeFile.close()

        z = 1

        return generate_result


    def generateAdaSyn(self, recordEpoch = [], num_sample=10):
        '''
        生成一些数据
        :param simpleEnhance: 简单的提升生成 数量
        :return:
        '''
        writeFile = open('targetGen.txt', 'w')
        generate_result1 = []
        generate_result2 = []
        maxRun = np.max(recordEpoch)
        count = 0
        genRecord = {}
        minRecord = []
        for i in range(0, maxRun):
            gene = self.efG(self.minEmb)
            gene2 = torch.reshape(gene, (gene.size(0), num_sample, -1))
            #gene = gene.reshape(-1, num_sample, -1)
            generate_result1.append( gene2 )
            if 1:
                clone_detach_x = gene2.clone().detach()
                # cora 数据集
                threshold = 0.5
                clone_detach_x[clone_detach_x < threshold] = 0.0
                clone_detach_x[clone_detach_x >= threshold] = 1.0


                clone_detach_x2 = clone_detach_x.cpu().numpy()

                for j in range(0, len(clone_detach_x2)):
                    ii = clone_detach_x2[j]
                    if j not in genRecord:
                        genRecord[j] = 0
                    if genRecord[j] < recordEpoch[j]:
                        minRecord.append(j) # 对应第几个 minNode
                        generate_result2.append(gene2[j])
                        genRecord[j] += 1


                        ii = list(ii)
                        for jj in ii:
                            jj = list(jj)
                            threshold = 0

                            for kk in jj:
                                if kk >= .5:
                                    threshold += 1
                            print (threshold)
                            # 测试

                            ls = str(jj) + '\n'
                            writeFile.write(ls)
                            writeFile.flush()
        writeFile.close()
        #concateResult = generate_result2[0]
        #for i in range(1, len(generate_result2)):
        #    concateResult =
        z = 1
        tmp = torch.stack(generate_result2,dim=-1)
        tmp2 = tmp.permute(2, 0, 1)

        return [tmp2], minRecord


    def generateAdaSyn2(self,recordEmb, num_sample=10):
        '''
        :param simpleEnhance:
        num_sample :
        :return:
        '''
        generate_result = []
        count = 0
        print ('pixel recordEmb' + str(len(recordEmb)))

        minEmb = torch.Tensor(recordEmb).cuda()

        generate_result = []
        for i in range(0, 1):
            gene = self.efG(minEmb)

            gene2 = gene.clone().detach()

            #gene2[gene2 < 0.] = 0.0
            #gene2[gene2 >= 1.] = 1.0

            gene2 = torch.reshape(gene2, (gene.size(0), self.maxLen, -1))
            # gene = gene.reshape(-1, num_sample, -1)
            generate_result.append(gene2)

        print ('pixel generate_result' + str(len(generate_result)))

        z = 1

        return generate_result



    def generateAdaSynCora(self,recordEmb, num_sample=10):
        '''
        生成一些数据
        :param simpleEnhance: 简单的提升生成 数量
        num_sample : 这个是数量
        :return:
        '''
        writeFile = open('targetGen.txt', 'w')
        generate_result = []
        count = 0
        minEmb = torch.Tensor(recordEmb).cuda()
        generate_result = []
        threshold = .5
        for i in range(0, 1):
            gene = self.efG(minEmb)
            gene2 = gene.clone().detach()
            gene2[gene2 < threshold] = 0.0
            gene2[gene2 >= threshold] = 1.0

            gene2 = torch.reshape(gene2, (gene.size(0),  self.maxLen, -1))
            #gene = gene.reshape(-1, num_sample, -1)
            generate_result.append( gene2 )
            if 1:
                clone_detach_x = gene2.clone().detach()

                clone_detach_x2 = clone_detach_x.cpu().numpy()
                for ii in clone_detach_x2:
                    ii = list(ii)
                    for jj in ii:
                        jj = list(jj)
                        threshold = 0

                        for kk in jj:
                            if kk >= .5:
                                threshold += 1
                        #print (threshold)
                        ls = str(jj) + '\n'
                        writeFile.write(ls)
                        writeFile.flush()
        writeFile.close()

        z = 1

        return generate_result






