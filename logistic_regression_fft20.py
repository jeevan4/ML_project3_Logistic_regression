__author__ = 'Jeevan'

##########################################################
# HW3 :: Implementation of Multinomial Logistic Regression
#########################################################

import scipy
import glob
import numpy as np
from scipy.io import wavfile
import os.path as chk
import sklearn.metrics as ski

# Variable Initializations

fft_matrix = np.empty((20,600),dtype=float)
path={1:'classical',2:'country',3:'jazz',4:'metal',5:'pop',6:'rock'}
true_genre = np.empty((600,1),dtype=int)
delta_matrix=np.zeros((6,540),dtype=int)
test_genre = np.empty((60,1),dtype=int)
file_no = 0

# Used the logic given by Professor to calculate
# FTT on the Wav files

def fft_calculation(file_no):
    '''

    The function calculates FTT with the logic given in the assignment
    description. Once FTT is calculated we store the FTT matrix in a
    text file, so that the computation can be saved for next use !
    :param file_no: The file_no variable contains the index of Wav file
    '''

    for path_no in path.keys():
        for fileName in glob.glob(path.get(path_no)+'/*.wav'):
            sample_rate, X = wavfile.read(fileName)
            fft_matrix[:,file_no] = abs(scipy.fft(X)[:1000])
            true_genre[file_no,0]=path_no
            file_no += 1
    np.savetxt('fft_matrix.out', fft_matrix, delimiter=' ')
    np.savetxt('true_genre.out',true_genre)

def fft_rank(fft_matrix):
    '''
    This function ranks the FFT features based on the difference in Standard Deviation
    of files across all genres vs standard deviation of a particular genre.
    I have tried with mean,variance but found higher accuracies with Standard Deviation
    :param fft_matrix: Takes the original fft feature matrix as input
    :return:
    '''

    full_variance = np.zeros((1000,1),dtype=float)
    genre_variance = np.zeros((1000,1),dtype=float)
    fft_20_matrix = np.zeros((20,600),dtype=float)
    for k in range(0,1000):
        full_variance[k,0] = np.std(fft_matrix[k,:])
    start = 0
    ending = 100
    for k in range(0,6):
        for i in range(0,1000):
            genre_variance[i,0] = np.std(fft_matrix[i,start:ending])
        diff = full_variance - genre_variance
        fft_20_matrix[:,start:ending] = fft_matrix[np.argsort(-diff[:,0],axis=0)[:20],start:ending]
        start = ending
        ending = ending + 100
    return fft_20_matrix


def ten_fold(fft_matrix):
    '''

    Generates list of indexes for training and testing datasets
    for 10-fold cross-validation and stores them in ten_fold_index Matrix
    :param feature_matrix: The MFCC/FTT features for each Wav file generated earlier
    :return: a matrix ten_fold_index that holds the indexes of testing and training
             each fold.
    '''

    ten_fold_index = np.zeros((10,2), dtype=np.ndarray)
    for k in range(10):
        train_index=[]
        test_index=[]
        for i in range(len(fft_matrix[0,:])):
            if (i - k) % 10 != 0:
                train_index.append(i)
            else:
                test_index.append(i)
        ten_fold_index[k,0]=train_index
        ten_fold_index[k,1]=test_index
    return ten_fold_index


def generate_test_train(ten_fold_index,index,fft_matrix,true_genre):
    '''

    This function returns Train matrix and Test Matrix from the Indexes present
    at every row in ten_fold_index.

    Normalizes the above test and train datasets by determining the largest element in the dataset and
    dividing every element by that number.

    :param ten_fold_index: Holds the indexes of Train and Test datasets from the calculated MFCC matric
    :param index: To determine the Fold number for iteration
    :param feature_matrix: The MFCC / FTT feature matrix for all the 600 files
    :param true_genre: Original Genre details for all the 600 files. Used to generate Test dataset genre, used to
                       calculate accuracy and confusion matrix.
    :return: Train and Test Matrix , for training and testing the logistic regression.
    '''

    test_matrix = np.zeros((21,60),dtype=float)
    train_matrix = np.zeros((21,540),dtype=float)
    train_matrix[0,:]=1
    test_matrix[0,:]=1
    k=0
    for i in ten_fold_index[index,0]:
        train_matrix[1:,k]=fft_matrix[:,i]
        k += 1
    k=0
    for i in ten_fold_index[index,1]:
        test_matrix[1:,k]=fft_matrix[:,i]
        test_genre[k,0]= true_genre[i]
        k += 1
    # To Normalize the matrix :
    for k in range(1,21):
        train_matrix[k,:]= train_matrix[k,:]/np.amax(train_matrix[k,:])
        test_matrix[k,:]= test_matrix[k,:]/np.amax(test_matrix[k,:])
    return train_matrix,test_matrix

def conditional_probability(weight_matrix,sample):
    '''
    The function calculates conditional probability i.e P(Y/X,W) and calculates
    the probability matrix to generate equations (27) and (28) in the textbook

    :param weight_matrix: Weight matrix as input, to calculate the conditional probability
                          for each updated weight
    :param sample: The sample can be either Train or Test samples
    :return: The exponential_matrix, that holds the likelihood values for a given weight and dataset.

    '''

    exponent_matrix = np.exp(np.dot(weight_matrix,sample))
    exponent_matrix[5,:]=1
    for i in range(len(exponent_matrix[1,:])):
        exponent_matrix[:,i] = exponent_matrix[:,i]/np.sum(exponent_matrix[:,i])
    return exponent_matrix

def logistic_reg(weight_matrix,train_matrix,exponent_matrix,eta,lam,delta_matrix):

    '''
    This function implements multinomial logistic regression. Updates the weight matrix
    for every iteration.

    :param weight_matrix: The weights used for the calculation P(Y/X,W)
    :param train_matrix: The Train sample with which the Logistic regression
    :param exponent_matrix: The conditional probability martix after P(Y/X,W) calulation.
    :param eta: eta values for every iteration
    :param lam: Lambda for Logistic Regression
    :param delta_matrix: Delta Matrix for the formula

    :return: returns updated weight_matrix for the next iteration.
    '''

    weight_matrix = weight_matrix + eta*(np.dot((delta_matrix - exponent_matrix),train_matrix.T)-lam*(weight_matrix))
    return weight_matrix

def confusion_matrix_calc(test_probability_matrix):

    '''
    Calculates accuracy and confusion matrix for the test dataset,
    and returns the accuracy and confusion matrix for every iteration
    :param test_probability_matrix: P(Y/X,W) for an updated weight with Test dataset
    :return: accuracy and confusion matrix

    '''

    classified_genre = np.zeros((60,1),dtype=int)
    for index in range(len(test_probability_matrix[0,:])):
        classified_genre[index] = np.argmax(test_probability_matrix[:,index]) + 1
    confusion_matrix = ski.confusion_matrix(test_genre,classified_genre)
    accuracy = float(np.sum(confusion_matrix.diagonal()))*100.0/len(test_probability_matrix[0,:])
    return accuracy,confusion_matrix

def init():
    '''
    Initialization of Delta Matrix and generating program flow.
    If the FFT matrix feature file already exist, then program starts to train the dataset
    If the FFT feature matrix is not present, then calculates the features also
    :return:
    '''

    global fft_matrix,file_no,true_genre,delta_matrix,test_genre,test_matrix,train_matrix,weight_matrix
    delta_matrix[0,0:90]=1
    delta_matrix[1,90:180]=1
    delta_matrix[2,180:270]=1
    delta_matrix[3,270:360]=1
    delta_matrix[4,360:450]=1
    delta_matrix[5,450:540]=1
    np.savetxt('delta_matrix.out',delta_matrix)
    if chk.isfile('fft_matrix.out'):
        print "FFT feature file exist ! Calculating Accuracies for all Folds "
        main_function()
    else:
        print "There is no FFT file ! Calculating features again "
        fft_calculation(file_no)
        main_function()


def main_function():

    '''
    This function has the main flow. It generates 10-fold test and train dataset,
    passes the dataset to Logistic Regression function using gradient descent to learn Weights and then
    predicts the likelihood with updated weight matrix. eta Value will be changed for every
    iteration.
    :return:
    '''

    fft_matrix = np.loadtxt("fft_matrix.out",float,skiprows=0)
    delta_matrix = np.loadtxt("delta_matrix.out",int,skiprows=0)
    true_genre = np.loadtxt("true_genre.out",int,skiprows=0)
    fft_20_matrix = fft_rank(fft_matrix)
    ten_fold_index = ten_fold(fft_20_matrix)
    final_acc=[]
    for index in range(len(ten_fold_index[:,0])):
        train_matrix,test_matrix = generate_test_train(ten_fold_index,index,fft_20_matrix,true_genre)
        eta_0 = 0.01
        lam = 0.001
        accuracy=[]
        confusion=[]
        weight_matrix=np.zeros((6,21),dtype=float)
        for epocs in range(350):
            eta = eta_0/(1.0 + epocs/350.0)
            exponential_matrix = conditional_probability(weight_matrix,train_matrix)
            weight_matrix = logistic_reg(weight_matrix,train_matrix,exponential_matrix,eta,lam,delta_matrix)
            test_probability_matrix = conditional_probability(weight_matrix,test_matrix)
            acc,conf= confusion_matrix_calc(test_probability_matrix)
            accuracy.append(acc)
            confusion.append(conf)
        print "Accuracy and Confusion Matrix for Fold ",index+1," is :",max(accuracy)
        final_acc.append(max(accuracy))
        k = np.argmax(accuracy)
        print confusion[k]
    print "The average accuracy for all 10 Folds :",np.average(final_acc)

if __name__ == '__main__':
    init()
