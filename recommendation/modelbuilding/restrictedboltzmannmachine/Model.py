"""
Created on Fri Feb 14 12:29:49 2020

@author: Avinash.Kumar
"""
import os
import torch
import numpy as np

'''
Restricted boltzmann machine(RBM) model.
'''

if torch.cuda.is_available():
    dev = "cuda:0"
else:
    dev = "cpu"

class Model():
    def __init__(self, numberVisible, numberHidden, step, batchSize, epoch, learningRate):
        '''
        Parameters
        ----------
        numberVisible : Number of Visible nodes for the model which is equal to the number of courses available.
        numberHidden : Number of Hidden nodes for the model which is equal to the number features the needs to be learnt.
        step : Number of steps till which the model is learning.
        batchSize : Number of batches at which the data is sent for training.
        epoch : Number of epochs for which the model is running (Early Stopping has been applied).
        learningRate : Rate at which the model is learning.

        Returns
        -------
        None.
        '''       
        print('Building the Model')
        self.numberVisible = numberVisible
        self.numberHidden = numberHidden
        self.step = step
        self.batchSize = batchSize
        self.epoch = epoch
        self.learningRate = learningRate
        torch.manual_seed(0)
        self.W = torch.randn(numberHidden, numberVisible)
        self.W = self.W.to(torch.device(dev))
        self.a = torch.randn(1, numberHidden)
        self.a = self.a.to(torch.device(dev))
        self.b = torch.randn(1, numberVisible)
        self.b = self.b.to(torch.device(dev))
       
    def sampleHidden(self, x):
        '''
        Parameters
        ----------
        x : Input data from visible nodes.

        Returns
        -------
        Probability of the hidden nodes given visible nodes and value hidden nodes given the visible nodes.
        '''
        x = x.to(torch.device(dev))
        wx = torch.mm(x, self.W.t())
        activation = wx + self.a.expand_as(wx)
        # Sigmoid is the activation function used
        p_h_given_v = torch.sigmoid(activation)
        return p_h_given_v, torch.bernoulli(p_h_given_v)
   
    def sampleVisible(self, y):
        '''
        Parameters
        ----------
        y : Input data from hidden nodes.

        Returns
        -------
        Probability of the visible nodes given the hidden node and value of visible nodes given the hidden nodes.
        '''
        y =y.to(torch.device(dev))
        wy = torch.mm(y, self.W)
        activation = wy + self.b.expand_as(wy)
        # Sigmoid is the activation function used
        p_v_given_h = torch.sigmoid(activation)
        return p_v_given_h, torch.bernoulli(p_v_given_h)
    
    def fit(self, trainData,testData,numberUsers):
        '''
        Parameters
        ----------
        trainData : Data for training.
        testData : Data for testing.
        numberUsers : Number of user in dataset.

        Returns
        -------
        epochList : List of all epochs value.
        trainLossList : List of all training mean square error value for every epoch.
        testLossList : List of all test mean square error value for every epoch.
        minValueLoss : Minimum test loss for the dataset.
        bestEpoch : Epoch at which the minimum value of test loss is obtained.
        predictedValue : Course predictes.
        probabilityOfValue : Probability of the predicted course.
        '''
        epochList = []
        trainLossList = []
        testLossList = []
        minValueLoss = np.Inf
        epochsNoImprove = 0
        '''
        Patience for early stopping. The model will stop training their is no improvement in the 
        test loss for 15 epochs.
        '''
        bestEpoch = 0
        patience = 15
        for value in range(1, self.epoch+ 1):
            trainLossMAE = 0
            trainLossRMSE = 0
            testLoss = 0
            s = 0.
            for idUser in range(0, numberUsers - self.batchSize, self.batchSize):
                vk = trainData[idUser:idUser+self.batchSize]
                v0 = trainData[idUser:idUser+self.batchSize]
                v0 = v0.to(torch.device(dev))
                ph0,_ = self.sampleHidden(v0)
                for k in range(self.step):
                    _,hk = self.sampleHidden(vk)
                    _,vk = self.sampleVisible(hk)
                    vk[v0<0] = v0[v0<0]
                phk, _ = self.sampleHidden(vk)
                self.W += self.learningRate * (torch.mm(ph0.t(),v0) - torch.mm(phk.t(), vk))
                self.b += torch.sum((v0 - vk), 0)
                self.a += torch.sum((ph0 - phk), 0)
                v0 = v0.cpu()
                vk = vk.cpu()
                trainLossMAE += torch.mean(torch.abs(v0[v0>=0] - vk[v0>=0]))
                trainLossRMSE += torch.sqrt(torch.mean((v0[v0>=0] - vk[v0>=0])**2))
                s +=1.
                ph0 = ph0.cpu()
                phk = phk.cpu()
            predictedValue, probabilityOfValue, testLoss = self.test(trainData,testData,numberUsers)
            epochList.append(value)
            trainLossList.append(trainLossRMSE/s)
            testLossList.append(testLoss)
            print('epoch: '+str(value)+' Training loss : RMSE '+ str(trainLossRMSE/s) + ' Test loss : RMSE '+ str(testLoss))
            torch.cuda.empty_cache()
            if testLoss >= minValueLoss:
                epochsNoImprove += 1
            else:
                bestEpoch = value
                epochsNoImprove = 0
                minValueLoss = testLoss
            # Check early stopping condition
            if epochsNoImprove == patience:
                print('Early Stopping!')
                print('The best number of epochs to run: '+ str(bestEpoch))
                print('The best value of error: '+ str(minValueLoss))
                self.W = self.W.cpu()
                self.a = self.a.cpu()
                self.b = self.b.cpu()
                break
        return epochList, trainLossList, testLossList, minValueLoss, bestEpoch, predictedValue, probabilityOfValue
   
    def train(self, trainData, numberUsers):
        '''
        Parameters
        ----------
        trainData : Training dataset.
        numberUsers : Number of Users.
        
        Returns
        -------
        None.

        '''   
        for value in range(1, self.epoch+ 1):
            trainLossMAE = 0
            trainLossRMSE = 0
            s = 0.
            for idUser in range(0, numberUsers - self.batchSize, self.batchSize):
                vk = trainData[idUser:idUser+self.batchSize]
                v0 = trainData[idUser:idUser+self.batchSize]
                v0 = v0.to(torch.device(dev))
                ph0,_ = self.sampleHidden(v0)
                for k in range(self.step):
                    _,hk = self.sampleHidden(vk)
                    _,vk = self.sampleVisible(hk)
                    vk[v0<0] = v0[v0<0]
                phk, _ = self.sampleHidden(vk)
                self.W += self.learningRate * (torch.mm(ph0.t(),v0) - torch.mm(phk.t(), vk))
                self.b += torch.sum((v0 - vk), 0)
                self.a += torch.sum((ph0 - phk), 0)
                v0 = v0.cpu()
                vk = vk.cpu()
                trainLossMAE += torch.mean(torch.abs(v0[v0>=0] - vk[v0>=0]))
                trainLossRMSE += torch.sqrt(torch.mean((v0[v0>=0] - vk[v0>=0])**2))
                s +=1.
                ph0 = ph0.cpu()
                phk = phk.cpu()
            print('epoch: '+str(value)+' loss : MAE '+ str(trainLossMAE/s) + ' loss : RMSE '+ str(trainLossRMSE/s))
        '''
        Saving the value after training
        W = Weights learnt by the model.
        a = Bias in the hidden nodes.
        b = Bias in the visible nodes.
        '''
        print('Saving weights and bias')
        torch.save(self.W, os.getcwd()+'/recommendation/model/learning.pt')
        torch.save(self.a, os.getcwd()+'/recommendation/model/hidden_bias.pt')
        torch.save(self.b, os.getcwd()+'/recommendation/model/visible_bias.pt')
        
    def test(self, trainData, testData,numberUsers):
        '''
        Parameters
        ----------
        trainData : Training data.
        testData : Test data.
        numberUsers : Number of users.

        Returns
        -------
        predictedValue : Course predictes.
        probabilityOfValue : Probability of the predicted course.
        testLossRMSE : Root mean square value of the test loss.
        '''       
        testLossMAE = 0
        testLossRMSE = 0
        s = 0.
        predictedValue = []
        probabilityOfValue = []
        for idUser in range(numberUsers+1):
            v = trainData[idUser:idUser+1]
            vt = testData[idUser:idUser+1]
            if len(vt[vt>=0])>0:
                ph,h = self.sampleHidden(v)
                pv,v = self.sampleVisible(h)
                ph = ph.cpu()
                h = h.cpu()
                pv =pv.cpu()
                v = v.cpu()
                predictedValue.append(v)
                probabilityOfValue.append(pv)
                testLossMAE += torch.mean(torch.abs(vt[vt>=0] - v[vt>=0]))
                testLossRMSE += torch.sqrt(torch.mean((vt[vt>=0] - v[vt>=0])**2))
                s +=1.
        print('test loss : MAE '+ str(testLossMAE/s) + ' RMSE: '+ str(testLossRMSE/s))
        return predictedValue, probabilityOfValue, testLossRMSE/s
       
    def predict(self, x):
        '''
        Parameters
        ----------
        x : Input data.

        Returns
        -------
        pv : Probability of Visible node.
        v : Value of the visible node.
        '''
        ph, h = self.sampleHidden(x)
        pv, v = self.sampleVisible(h)
        return pv,v
    
    def predict_topk(self, userId, numberOfRecommendation, data, labelPerson, labelObject):
        '''
        Parameters
        ----------
        userId : User ID for whom the recommendations are needed.
        numberOfRecommendation : Number of recommendations required.
        data : Input data.
        labelPerson : Label encoding of the course ID.
        labelObject : Label encoding of the object ID.

        Returns
        -------
        courseId : Course IDs returned.
        '''
        x = np.array([userId])
        x = labelPerson.transform(x)
        prob, predict = self.predict(data[x])
        probability,courseId = torch.topk(prob, k=numberOfRecommendation, dim=-1, largest=True)
        courseId = courseId.numpy()
        for i in range(0,np.size(courseId,0)):
            courseId[i] = labelObject.inverse_transform(courseId[i])
        return courseId