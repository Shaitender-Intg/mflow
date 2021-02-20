import datetime

import mlflow
import mlflow.sklearn
print(mlflow.__version__)

from sklearn.datasets import load_iris
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.model_selection import train_test_split

from sklearn.metrics import *

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from sklearn.decomposition import PCA


SEED = 101
data = load_iris()

def plot_graph():
    X =data.data
    y = data.target

    x_min,x_max=X[:,0].min() -.5,X[:,0].max() + .5
    y_min,y_max=X[:,1].min() -.5,X[:,1].max() + .5
    plt.figure(2, figsize=(10,8))
    plt.clf()

    #plot the training points
    plt.scatter(X[:,0], X[:,1],c=y , cmap="ocean" , edgecolors='k')
    plt.xlabel('Sepal length')
    plt.ylabel('Sepal width')

    plt.xlim(x_min,x_max)
    plt.ylim(x_min,x_max)
    plt.xticks()
    plt.yticks()

    fig =plt.figure(1,figsize=(10,8))
    ax= Axes3D(fig,elev=-150,azim=110)
    X_reduced = PCA(n_components=3).fit_transform(data.data)

    ax.scatter(X_reduced[:,0],X_reduced[:,1],X_reduced[:,2], c=y , cmap="jet" , edgecolors='k', s=40)

    ax.set_title('First three PCA Dimensions')
    ax.set_xlabel("1st EigenVector")
    ax.w_xaxis.set_ticklabels([])
    ax.set_ylabel("2nd EigenVector")
    ax.w_yaxis.set_ticklabels([])
    ax.set_zlabel("3rd EigenVector")
    ax.w_zaxis.set_ticklabels([])
    fig.savefig('iris.png')
    plt.show()
    plt.close(fig)


def decisionTreeClassifier_experiment(experiment_id,x_train, x_test, y_train, y_test):
    print('experiment started')
    with mlflow.start_run(experiment_id=experiment_id, run_name='DecisionTreeClassifier'):
        seed = 99
        dtc= DecisionTreeClassifier(random_state=seed, max_depth=5)
        dtc.fit(x_train,y_train)
        y_pred_class=dtc.predict(x_test)

        accuracy = accuracy_score(y_test,y_pred_class)
        print('accuracy',accuracy)

        #log experiment
        mlflow.log_param('random_state',seed)
        mlflow.log_param('test size',.33)
        mlflow.log_metric('accuracy',accuracy)

        mlflow.sklearn.log_model(dtc, 'model')
        model_path = f"Python/Models/DecisionTreeClassifier-{datetime.datetime.now().strftime('%Y_%m_%d-%I_%M_%S_%p')}"
        mlflow.sklearn.save_model(dtc,model_path)

        mlflow.log_artifact('iris.png')

    print('experiment ended')



if __name__ == '__main__':

    print('target -',data.target)
    print('feature_names ',data.feature_names)

    X= data.data
    y=data.target
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.33, random_state = SEED)

    #plot_graph()
    #experiment_id= mlflow.create_experiment('mlflow_demo')
    decisionTreeClassifier_experiment(1,X_train, X_test, y_train, y_test)





#pgrep gunicorn
#kill process id

