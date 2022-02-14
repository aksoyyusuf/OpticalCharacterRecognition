# Importing the libraries
import time
import numpy as np
import pandas as pd
from PIL import Image
#import msvcrt as m
from sklearn import preprocessing, datasets, svm, metrics
from sklearn.tree import DecisionTreeRegressor
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestRegressor
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import GridSearchCV
from matplotlib import pyplot as plt
import seaborn as sns; sns.set()
import warnings
plt.style.use('ggplot')

warnings.filterwarnings('ignore')   # To ignore scaling warning
le = preprocessing.LabelEncoder()   # Label encoder for letters


class Operations:

    def readDataset(self, path):
        #dataset = pd.read_table(path, sep=None, header=None, engine='python')
        dataset = pd.read_csv(path, sep=None, header=None, engine='python')

        # Label columns 
        dataset.columns = ["id","letter","next_id","word_id","position","fold",
                           "p_0_0","p_0_1","p_0_2","p_0_3","p_0_4","p_0_5","p_0_6","p_0_7",
                           "p_1_0","p_1_1","p_1_2","p_1_3","p_1_4","p_1_5","p_1_6","p_1_7",
                           "p_2_0","p_2_1","p_2_2","p_2_3","p_2_4","p_2_5","p_2_6","p_2_7",
                           "p_3_0","p_3_1","p_3_2","p_3_3","p_3_4","p_3_5","p_3_6","p_3_7",
                           "p_4_0","p_4_1","p_4_2","p_4_3","p_4_4","p_4_5","p_4_6","p_4_7",
                           "p_5_0","p_5_1","p_5_2","p_5_3","p_5_4","p_5_5","p_5_6","p_5_7",
                           "p_6_0","p_6_1","p_6_2","p_6_3","p_6_4","p_6_5","p_6_6","p_6_7",
                           "p_7_0","p_7_1","p_7_2","p_7_3","p_7_4","p_7_5","p_7_6","p_7_7",
                           "p_8_0","p_8_1","p_8_2","p_8_3","p_8_4","p_8_5","p_8_6","p_8_7",
                           "p_9_0","p_9_1","p_9_2","p_9_3","p_9_4","p_9_5","p_9_6","p_9_7",
                           "p_10_0","p_10_1","p_10_2","p_10_3","p_10_4","p_10_5","p_10_6","p_10_7",
                           "p_11_0","p_11_1","p_11_2","p_11_3","p_11_4","p_11_5","p_11_6","p_11_7",
                           "p_12_0","p_12_1","p_12_2","p_12_3","p_12_4","p_12_5","p_12_6","p_12_7",
                           "p_13_0","p_13_1","p_13_2","p_13_3","p_13_4","p_13_5","p_13_6","p_13_7",
                           "p_14_0","p_14_1","p_14_2","p_14_3","p_14_4","p_14_5","p_14_6","p_14_7",
                           "p_15_0","p_15_1","p_15_2","p_15_3","p_15_4","p_15_5","p_15_6","p_15_7", 
                           "spare"]

        return dataset

    def extractData(self, dataset, trainFold):

        pixelStartIndex = 'p_0_0'
        pixelEndIndex = 'p_15_7'
        trainFrames_X = []
        trainFrames_Y = []


        dataset = dataset.groupby("fold")
        

        for i in range (0,10):
            foldSet = dataset.get_group(i)
            target = foldSet.loc[:,'letter']
            pixels = foldSet.loc[:,pixelStartIndex:pixelEndIndex]

            print('Fold Number: {} , Number of letters : {} '.format(i, len(pixels.index)))

            if(i == trainFold):
                X_test = pixels
                y_test = target
            else:
                trainFrames_X.append(pixels)
                trainFrames_Y.append(target)


        X_train = pd.concat(trainFrames_X)
        y_train = pd.concat(trainFrames_Y)

        return X_train, y_train, X_test, y_test


    def plotImageFromDataset(self, dataset, imageNum):
        
        # define image sizes
        columnSize = 8
        rowSize = 16
        offset = 6 # pixel offset in dataset

        img = Image.new('1', (columnSize,rowSize), "black" ) # create grayscale image
        pixelMap = img.load() #create the pixel map


        for row in range(img.size[1]):    # for every row:
            for column in range(img.size[0]):    # For every column
                pixelMap[column, row] = int(dataset.iloc[imageNum, offset + row*img.size[0] + column]) # set the colour
        img.show() 

        # Save image (optional)
        #img.save('out.bmp')


    def plotImage(self, pixels, letters):
        
        # define image sizes
        columnSize = 8
        rowSize = 16

        img = Image.new('1', (columnSize,rowSize), "black" ) # create grayscale image
        pixelMap = img.load() #create the pixel map

        for row in range(img.size[1]):    # for every row:
            for column in range(img.size[0]):    # For every column
                pixelMap[column, row] = int(pixels[row*img.size[0] + column]) # set the colour
        img.show() 
        print("Image on the screen is : ")
        print(letters)

        # Save image (optional)
        #img.save('out.bmp')


    def kNN(self, X_train, X_test, y_train, y_test):

        # If it is 'True' it plots test and training results
        # for varying number of nearest neighbors.
        knnGridSearch = False

        operation = Operations() #instance


        if(knnGridSearch == False):
            # Setup a knn classifier with k neighbors
            # Best number of nearest neighbors parameter found as 3
            knn=KNeighborsClassifier(n_neighbors=3,algorithm="kd_tree",n_jobs=-1)
        
            # Fit the model
            knn.fit(X_train,y_train.ravel())
        
            # Compute accuracy on the training set
            train_accuracy = knn.score(X_train, y_train.ravel())
        
            #Compute accuracy on the test set
            test_accuracy = knn.score(X_test, y_test.ravel())

            return test_accuracy, train_accuracy

        else:
            # Find best accurate number of nearest neighbors (Optional)
            neighbours = np.arange(1,5)
            train_accuracy = np.empty(len(neighbours))
            test_accuracy = np.empty(len(neighbours))
            
            for i,k in enumerate(neighbours):
            
                #Setup a knn classifier with k neighbors
                knn=KNeighborsClassifier(n_neighbors=k,algorithm="kd_tree",n_jobs=-1)
            
                #Fit the model
                knn.fit(X_train,y_train.ravel())
            
                #Compute accuracy on the training set
                train_accuracy[i] = knn.score(X_train, y_train.ravel())
            
                #Compute accuracy on the test set
                test_accuracy[i] = knn.score(X_test, y_test.ravel())
            
            
            #Generate plot
            plt.title('k-NN Varying number of neighbors')
            plt.plot(neighbours, test_accuracy, label='Testing Accuracy(Overall: {:.4f})'.format(np.mean(test_accuracy)))
            plt.plot(neighbours, train_accuracy, label='Training accuracy(Overall: {:.4f})'.format(np.mean(train_accuracy)))
            plt.legend()
            plt.xlabel('Number of neighbors')
            plt.ylabel('Accuracy')
            plt.show()

            # Return zero becuase this is for only optimization
            return 0, 0


        

    # SVM parameter optimizer
    def svm_param_selection(self, X_train, y_train, n_folds = 3):
        Cs = [1, 10, 100]
        gammas = [0.001, 0.01, 0.1]
        param_grid = {'C': Cs, 'gamma' : gammas}
        grid_search = GridSearchCV(svm.SVC(kernel='rbf'), param_grid, cv=n_folds)
        grid_search.fit(X_train, y_train)
        print("Best Params: ", grid_search.best_params_)
        print("Score: ", grid_search.best_score_)
        means = grid_search.cv_results_['mean_test_score']
        stds = grid_search.cv_results_['std_test_score']

        # Plot search results (optional)
        #to_vary = ('C', 'gamma')
        #to_keep = {'mean_test_score', 'std_test_score'}
        #plot.grid_search(grid_search.grid_scores_, to_vary, to_keep)
        #plt.show

        return grid_search


    def svm(self, X_train, X_test, y_train, y_test, fold):

        operation = Operations() #instance


        # If it is 'True', searchs best classifier parameters 
        # and trains with it for each fold.
        # Else trains with default parameters.
        svmGridSearch = False


        # Set classifier
        if(svmGridSearch == True):
            classifier = operation.svm_param_selection(X_train, y_train)
        else:
            classifier = svm.SVC(gamma=0.01, C = 10, kernel = 'rbf')

        # Train
        classifier.fit(X_train, y_train)

        # Predict the response for test dataset
        y_pred_test = classifier.predict(X_test)
        y_pred_train = classifier.predict(X_train)

        test_accuracy = metrics.accuracy_score(y_test, y_pred_test)
        train_accuracy = metrics.accuracy_score(y_train, y_pred_train)

        # Model Accuracy
        print("SVM Accuracy:", test_accuracy)

        return test_accuracy, train_accuracy




def main(): 

    # Select training method
    isSVM = True
    isKNN = False

    operation = Operations() #instance

    datasetPath = "letter.data"

    # Read dataset
    dataset = operation.readDataset(datasetPath)

    # Plot letter count in dataset
    sns.countplot(dataset['letter']).set_title('Letter Count')
    plt.show()


    test_accuracy = np.empty(10)
    train_accuracy = np.empty(10)
    training_time = np.empty(10)
  
    # K-Fold Cross Validation
    # Train with 9 folds, test with 1 fold, 
    # Change test fold each iteration
    for i in range(0,10):
        print("*******************************************************")
        print("***************** Dataset: {} *************************".format(i))
        print("*******************************************************\n")

        # Extract and split data into 0 to 9 folds
        # Pass test fold index to function
        X_train, y_train, X_test, y_test = operation.extractData(dataset, i)


        # Plot image for given number of image (optional)
        # Image number is row number of image in the dataset
        #imageNum = 500
        #operation.plotImage(X_train.iloc[imageNum, :], y_train.iloc[imageNum])

        # Convert character labels into numbers
        le.fit(y_train)
        y_train = le.transform(y_train)
        le.fit(y_test)
        y_test = le.transform(y_test)


        #print("All possible letters in dataset: ",list(le.classes_))

    
        # Fit on training set only.
        scaler = StandardScaler()
        scaler.fit(X_train)
        scaler.fit(X_test)
        # Apply transform to both the training set and the test set.
        X_train = scaler.transform(X_train)
        X_test = scaler.transform(X_test)
        
        
        
        # Make an instance of the Model
        # PCA will keep that amount of data according to variance
        pca = PCA(.85)
        
        
        #pca.fit(X_train)
        
        X_train = pca.fit_transform(X_train)
        X_test = pca.transform(X_test)
        print("Number of Components after PCA: " , pca.n_components_)


        # Start timer to measure training time
        start = time.time()

        # Start training
        if(isKNN == True):
            test_accuracy[i], train_accuracy[i] = operation.kNN(X_train, X_test, y_train, y_test)

        elif(isSVM == True):
            test_accuracy[i], train_accuracy[i] = operation.svm(X_train, X_test, y_train, y_test, i)

        # Stop timer
        end = time.time()
        print("Elapsed time in seconds: ")
        print(end - start)

        training_time[i] = end - start


    # Plot Test Accuracy Results
    plt.title('K-Fold Cross Validation Testing Results')
    plt.plot(np.arange(0, 10), test_accuracy, label='Testing Accuracy(Overall: {:.4f})'.format(np.mean(test_accuracy)))
    plt.plot(np.arange(0, 10), train_accuracy, label='Training Accuracy(Overall: {:.4f})'.format(np.mean(train_accuracy)))
    plt.legend()
    plt.xlabel('Testing Fold Number')
    plt.ylabel('Testing Accuracy (Percent)')
    plt.show()
        
    # Plot Training Times
    plt.title('Training Time for Each Fold (Overall: {:.4f})'.format(np.mean(training_time)))
    plt.plot(np.arange(0, 10), training_time, label='Training Time')
    plt.legend()
    plt.xlabel('Testing Fold Number')
    plt.ylabel('Training Time (Seconds)')
    plt.show()
        
    print('Overall Testing Accuracy : {}'.format(np.mean(test_accuracy)))
    print('Overall Training Time : {}'.format(np.mean(training_time)))

if __name__ == "__main__":
    main()










