from sklearn.ensemble import RandomForestClassifier
from sqlalchemy import create_engine
import os
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score



def database_connect1():

    user="user"
    password="password"
    port="port"
    host="IP"
    database = "your_database"
    engine = create_engine('mysql+pymysql://%s:%s@%s:%s/%s?charset=utf8&autocommit=true' % (user, password, host, port, database))
    return engine

def execute_sql():
    connection = database_connect()

    sql = f"""
            select * from (select *, ROW_NUMBER() over(partition  by id order by date desc ) as num  from older_healthy) t  where t.num=1
            """
    df = pd.read_sql(sql, connection)
    df_x = df[["id", "feature1", "feature2", "feature3", "feature4", "feature5","gender"]]
    df_y = df[["risk_target"]]

    connection.connect().close()
    return df_x,df_y

def algorithm_rf(x,y):

    scaler = StandardScaler()
    x = pd.DataFrame(scaler.fit_transform(x))

    rfc = RandomForestClassifier(n_estimators=100, criterion="entropy",random_state=55,bootstrap=False)

    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3,shuffle=True, random_state=55)

    rfc.fit(x_train, y_train)
    rfc.feature_importances_
    y_pred = rfc.predict(x_test)
    score_pre = cross_val_score(rfc, x_test, y_test, cv=10).mean()
    print(f" rf {score_pre}")
    accuracy = accuracy_score(y_test, y_pred)

def algorithm_svm(X,y):
    scaler = StandardScaler()
    X = pd.DataFrame(scaler.fit_transform(X))

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.4, random_state=15)
    clf = SVC(kernel='rbf', random_state=1000)
    clf.fit(X_train, y_train)

    y_pred = clf.predict(X_test)

    print("svm:", accuracy_score(y_test, y_pred,normalize=True))

def rfc():
    x, y = execute_sql()
    algorithm_rf(x,y)

def svm():
    x, y = execute_sql()
    algorithm_svm(x,y)

if __name__=='__main__':
    rfc()

