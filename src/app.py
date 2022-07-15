from utils import db_connect
engine = db_connect()

url='https://raw.githubusercontent.com/4GeeksAcademy/regularized-linear-regression-project-tutorial/main/dataset.csv'
df=pd.read_csv(url, sep=',')

df.head(10)

pd.options.display.max_columns=200
pd.options.display.max_rows=200
df.sample(40)

df[['R_birth_2018', 'R_death_2018', 'Employed_2018','Unemployed_2018']].info()

df_frac.describe()

df_frac.corr().style.background_gradient(cmap='Blues')

# escalar los datos:
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.linear_model import Lasso

X=df_frac.drop(columns="Employed_2018")
y=df_frac["Employed_2018"]

X_train,X_test,y_train,y_test=train_test_split(X,y,random_state=57)

scaler=MinMaxScaler()
scaler.fit(df_frac)
scaler.transform(df_frac)

# modelo de regresion lineal 
modelo = LinearRegression(normalize=True)
modelo.fit(X_train,y_train)
score=modelo.score(X_train,y_train)
print(f'score is{score: .4f}')

from sklearn import linear_model
clf = linear_model.Lasso(alpha=3)
clf.fit(X,y)
Lasso(alpha=3)
print(clf.coef_)
print(clf.intercept_)

# aplicando lasso: no sería necesrio ya que sin la regularizacion me dio un score bueno.
clf.fit(X_train,y_train)
score=clf.score(X_train,y_train)
print(f'score is{score: .4f}')

