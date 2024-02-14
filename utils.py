# IMPORTOWANIE BIBLIOTEK

from datetime import datetime
import datetime as dt

import pickle
import itertools as it
from itertools import repeat

import numpy as np

import math
import matplotlib.pyplot as plt

import os

import pandas as pd

import scipy
from scipy import signal
from scipy import stats as ss

import seaborn as sns

import statistics as st
import statsmodels.api as sm

#Do otwarcia zip z url
import requests
import pandas as pd
from zipfile import ZipFile
from io import BytesIO

# Zbiór bibliotek sktime
import sktime
from sktime.forecasting.arima import AutoARIMA
from sktime.forecasting.base import ForecastingHorizon
from sktime.forecasting.compose import make_reduction
from sktime.forecasting.compose import TransformedTargetForecaster, EnsembleForecaster
from sktime.forecasting.fbprophet import Prophet
from sktime.forecasting.model_selection import (ForecastingRandomizedSearchCV, ForecastingGridSearchCV, SingleWindowSplitter)
from sktime.forecasting.model_selection import temporal_train_test_split
from sktime.forecasting.model_selection._split import temporal_train_test_split

# metryki
from sklearn.metrics import mean_squared_error
from sktime.performance_metrics.forecasting import mean_absolute_percentage_error, mean_squared_percentage_error,MeanSquaredError
from sktime.transformations.series.feature_selection import FeatureSelection
from sktime.transformations.series.outlier_detection import HampelFilter
from sktime.utils.plotting import plot_series
from sklearn.ensemble import RandomForestRegressor
import plotly.express as px
import plotly.graph_objects as go

from xgboost import XGBRegressor


import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)
from sklearn.exceptions import DataConversionWarning
warnings.filterwarnings(action='ignore', category=DataConversionWarning)

# SŁOWNIKI

# Słownik zawierający określone kolumny dla danych oczyszczonych do predykcji

AGG_DICT_OCZYSZCZONE = {
    ...
}

# Słownik zawierający określone kolumny dla danych odstających do predykcji

AGG_DICT_ODSTAJACE = {
    ...
}

# Słownik zawierający określone kolumny dla danych inwentaryzacyjnych do predykcji

AGG_DICT_INWENTARYZACJE = {
    ...
}


def wczytaj_dane_wszystkie(start_date, end_date, group_period='M', store="all", agg_dict=AGG_DICT_OCZYSZCZONE):
    
    """ Funkcja wczytuje określony plik danych oraz przeprowadza preprocessing na kolumnach w tym grupowanie dla podanego okresu z 
    zadanym słownikiem grupującym """
    
    # Wczytywanie
    dane_wszystkie = pd.read_csv('xxx.csv', dtype={'store': int}, low_memory=False)

    if store != "all":
        dane_wszystkie = dane_wszystkie.query("store == @store")
    
    # Preprocessing
    dane_wszystkie['CALDAY'] = pd.to_datetime(dane_wszystkie['day'])
    dane_wszystkie = dane_wszystkie.set_index('day')
    dane_wszystkie['INW'] = (dane_wszystkie.INVENTORY + dane_wszystkie.INVENTORY.shift(1).fillna(0))
    dane_wszystkie = dane_wszystkie[(dane_wszystkie.index > start_date) & (dane_wszystkie.index < end_date)]
    
    # Grupowanie
    y_wszystkie = dane_wszystkie.groupby([pd.Grouper(freq=group_period)]).agg({'VALUE': sum})
    X_wszystkie = dane_wszystkie.groupby([pd.Grouper(freq=group_period)]).aggregate(agg_dict)
    
    # Uzupełnianie braków
    for col in ['Wskaźnik_inflacji', 'SALES', 'VOLUME', 'UNEMP_RATE']:
        try:
            X_wszystkie[col] = X_wszystkie[col].replace(0, np.nan).fillna(method='ffill').fillna(
                method='bfill')
        except:
            pass

    return y_wszystkie, X_wszystkie


def wczytaj_dane_oczyszczone(start_date, end_date, group_period='W', store="all", agg_dict=AGG_DICT_OCZYSZCZONE):
    
    """ Funkcja wczytuje określony plik danych oraz przeprowadza preprocessing na kolumnach w tym grupowanie dla podanego okresu z 
    zadanym słownikiem grupującym """
    
    # Wczytywanie
    dane_oczyszczone = pd.read_csv('yyy.csv', dtype={'store': int}, low_memory=False)

    if store != "all":
        dane_oczyszczone = dane_oczyszczone.query("store == @store")
    
    # Preprocessing
    dane_oczyszczone['day'] = pd.to_datetime(dane_oczyszczone['day'])
    dane_oczyszczone = dane_oczyszczone.set_index('day')
    dane_oczyszczone['inw'] = (dane_oczyszczone.INVENTORY + dane_oczyszczone.INVENTORY.shift(1).fillna(0))
    dane_oczyszczone = dane_oczyszczone[(dane_oczyszczone.index > start_date) & (dane_oczyszczone.index < end_date)]
    
    # Grupowanie
    y_oczyszczone = dane_oczyszczone.groupby([pd.Grouper(freq=group_period)]).agg({'VALUE': sum})
    X_oczyszczone = dane_oczyszczone.groupby([pd.Grouper(freq=group_period)]).aggregate(agg_dict)
    
    # Uzupełnianie braków
    for col in ['Wskaźnik_inflacji', 'SALES', 'VOLUME', 'UNEMP_RATE']:
        try:
            X_oczyszczone[col] = X_oczyszczone[col].replace(0, np.nan).fillna(method='ffill').fillna(
                method='bfill')
        except:
            pass

    return y_oczyszczone, X_oczyszczone


def wczytaj_dane_odstajace(start_date, end_date, group_period='M', store="all", agg_dict=AGG_DICT_ODSTAJACE):
    
    """ Funkcja wczytuje określony plik danych oraz przeprowadza preprocessing na kolumnach w tym grupowanie dla podanego okresu z 
    zadanym słownikiem grupującym """
    
    # Wczytywanie
    dane_odstajace = pd.read_csv('zzz.csv', dtype={'store': int}, low_memory=False)

    if store != "all":
        dane_odstajace = dane_odstajace.query("store == @store")
    
    # Preprocessing
    dane_odstajace['day'] = pd.to_datetime(dane_odstajace['day'])
    dane_odstajace = dane_odstajace.set_index('day')
    dane_odstajace['inw'] = (dane_odstajace.INVENTORY + dane_odstajace.INVENTORY.shift(1).fillna(0))
    dane_odstajace = dane_odstajace[(dane_odstajace.index > start_date) & (dane_odstajace.index < end_date)]
    
    # Grupowanie
    y_odstajace = dane_odstajace.groupby([pd.Grouper(freq=group_period)]).agg({'VALUE': sum})
    X_odstajace = dane_odstajace.groupby([pd.Grouper(freq=group_period)]).aggregate(agg_dict)
    
    # Uzupełnianie braków
    for col in ['Wskaźnik_inflacji', 'SALES', 'VOLUME', 'UNEMP_RATE']:
        try:
            X_odstajace[col] = X_odstajace[col].replace(0, np.nan).fillna(method='ffill').fillna(
                method='bfill')
        except:
            pass

    return y_odstajace, X_odstajace


def wczytaj_dane_inwentaryzacje(start_date, end_date, group_period='M', store="all", agg_dict=AGG_DICT_INWENTARYZACJE):
    
    """ Funkcja wczytuje określony plik danych oraz przeprowadza preprocessing na kolumnach w tym grupowanie dla podanego okresu z 
    zadanym słownikiem grupującym """
    
    # Wczytywanie
    dane_inwentaryzacje = pd.read_csv('start.csv', dtype={'store': int}, low_memory=False)

    if store != "all":
        dane_inwentaryzacje = dane_inwentaryzacje.query("store == @store")
    
    # Preprocessing
    dane_inwentaryzacje['day'] = pd.to_datetime(dane_inwentaryzacje['day'])
    dane_inwentaryzacje = dane_inwentaryzacje.set_index('day')
    dane_inwentaryzacje['inw'] = (dane_inwentaryzacje.INVENTORY + dane_inwentaryzacje.INVENTORY.shift(1).fillna(0))
    dane_inwentaryzacje = dane_inwentaryzacje[(dane_inwentaryzacje.index > start_date) & (dane_inwentaryzacje.index < end_date)]
    
    # Grupowanie
    y_inwentaryzacje = dane_inwentaryzacje.groupby([pd.Grouper(freq=group_period)]).agg({'VALUE': sum})
    X_inwentaryzacje = dane_inwentaryzacje.groupby([pd.Grouper(freq=group_period)]).aggregate(agg_dict)
    
    # Uzupełnianie braków
    for col in ['Wskaźnik_inflacji', 'SALES', 'VOLUME', 'UNEMP_RATE']:
        try:
            X_inwentaryzacje[col] = X_inwentaryzacje[col].replace(0, np.nan).fillna(method='ffill').fillna(
                method='bfill')
        except:
            pass

    return y_inwentaryzacje, X_inwentaryzacje


def wczytaj_dane_oczyszczone_do_predykcji(group_period='W', store="all", agg_dict=AGG_DICT_OCZYSZCZONE, start_date = '' ):
    
    """ Funkcja wczytuje określony plik danych do predykcji oraz przeprowadza preprocessing na kolumnach w tym grupowanie dla podanego okresu z zadanym słownikiem grupującym """
    
    # Wczytywanie
    dane_oczyszczone = pd.read_csv('DANE_PRZYSZLE.csv', dtype={'STORE': int}, low_memory=False)

    if store != "all":
        dane_oczyszczone = dane_oczyszczone.query("STORE_FORCOM_ID == @store")
    
    # Preprocessing
    dane_oczyszczone['CALDAY'] = pd.to_datetime(dane_oczyszczone['CALDAY'])
    dane_oczyszczone = dane_oczyszczone.set_index('CALDAY')
    dane_oczyszczone['INWENTARYZACJA'] = (dane_oczyszczone.INVENTORY + dane_oczyszczone.INVENTORY.shift(1).fillna(0))
    
    # Grupowanie
    X_oczyszczone = dane_oczyszczone.groupby([pd.Grouper(freq=group_period)]).aggregate(agg_dict)
    
    # Uzupełnianie braków
    for col in ['Wskaźnik_inflacji', 'TOTAL_SALES', 'TOTAL_VOLUME_SOLD', 'UNEMP_RATE']:
        try:
            X_oczyszczone[col] = X_oczyszczone[col].replace(0, np.nan).fillna(method='ffill').fillna(
                method='bfill')
        except:
            pass

    return X_oczyszczone[X_oczyszczone.index >= start_date]


def wczytaj_dane_odstajace_do_predykcji(group_period='M', store="all", agg_dict=AGG_DICT_ODSTAJACE, start_date = ''):
    
    """ Funkcja wczytuje określony plik danych do predykcji oraz przeprowadza preprocessing na kolumnach w tym grupowanie dla podanego okresu z zadanym słownikiem grupującym """
    
    # Wczytywanie
    dane_odstajace = pd.read_csv('DANE_PRZYSZLE.csv', dtype={'STORE': int}, low_memory=False)

    if store != "all":
        dane_odstajace = dane_odstajace.query("STORE_FORCOM_ID == @store")
    
    # Preprocessing
    dane_odstajace['CALDAY'] = pd.to_datetime(dane_odstajace['CALDAY'])
    dane_odstajace = dane_odstajace.set_index('CALDAY')
    dane_odstajace['INWENTARYZACJA'] = (dane_odstajace.INVENTORY + dane_odstajace.INVENTORY.shift(1).fillna(0))
    
    # Grupowanie
    X_odstajace = dane_odstajace.groupby([pd.Grouper(freq=group_period)]).aggregate(agg_dict)
    
    # Uzupełnianie braków
    for col in ['Wskaźnik_inflacji', 'TOTAL_SALES', 'TOTAL_VOLUME_SOLD', 'UNEMP_RATE']:
        try:
            X_odstajace[col] = X_odstajace[col].replace(0, np.nan).fillna(method='ffill').fillna(
                method='bfill')
        except:
            pass

    return X_odstajace[X_odstajace.index >= start_date]


def wczytaj_dane_inwentaryzacje_do_predykcji(group_period='M', store="all", agg_dict=AGG_DICT_INWENTARYZACJE, start_date = ''):
    
    """ Funkcja wczytuje określony plik danych do predykcji oraz przeprowadza preprocessing na kolumnach w tym grupowanie dla podanego okresu z zadanym słownikiem grupującym """
    
    # Wczytywanie
    dane_inwentaryzacje = pd.read_csv('DANE_PRZYSZLE.csv', dtype={'STORE': int}, low_memory=False)

    if store != "all":
        dane_inwentaryzacje = dane_inwentaryzacje.query("STORE == @store")
    
    # Preprocessing
    dane_inwentaryzacje['CALDAY'] = pd.to_datetime(dane_inwentaryzacje['CALDAY'])
    dane_inwentaryzacje = dane_inwentaryzacje.set_index('CALDAY')
    dane_inwentaryzacje['INWENTARYZACJA'] = (dane_inwentaryzacje.INVENTORY + dane_inwentaryzacje.INVENTORY.shift(1).fillna(0))
    
    # Grupowanie
    X_inwentaryzacje = dane_inwentaryzacje.groupby([pd.Grouper(freq=group_period)]).aggregate(agg_dict)

    for col in ['Wskaźnik_inflacji', 'TOTAL_SALES', 'TOTAL_VOLUME_SOLD', 'UNEMP_RATE']:
        try:
            X_inwentaryzacje[col] = X_inwentaryzacje[col].replace(0, np.nan).fillna(method='ffill').fillna(
                method='bfill')
        except:
            pass
    
    # Uzupełnianie braków
    return X_inwentaryzacje[X_inwentaryzacje.index >= start_date]

def get_most_important_cols(X_train, y_train, X_test, y_test, fh, plot=False):
    
    
    """ Funkcja zwraca listę nazw kolumn, które minimalizują błąd predykcji, a tym samym stają się predyktorami do prognoz. Funkcja bazuje na szukanie optymalnych zmiennych za pomocą modelu ARIMA """
    
    # Wybór i przedstawienie na wykresie ważności zmiennych przy użyciu Random Forest
    rf = RandomForestRegressor(random_state=42)

    rf.fit(X_train, y_train)

    df_feature_importances_rf = pd.DataFrame(rf.feature_importances_)
    df_feature_importances_rf['cols_names'] = X_train.columns
    df_feature_importances_rf = df_feature_importances_rf.sort_values(0, ascending=False)

    # Tworzenie wykresów z pokazaniem 5 najbardziej wpływowych zmiennych
    importances = rf.feature_importances_
    std = np.std([tree.feature_importances_ for tree in rf.estimators_], axis=0)
    indices = np.argsort(importances)[::-1]

    num_feat = 5
    if plot:
        plt.figure(figsize=[15, 8])
        plt.title("Feature importances", fontsize=20)
        plt.bar(range(num_feat)[:num_feat], importances[indices][:num_feat],
                color="r", yerr=std[indices][:num_feat], align="center")
        plt.xticks(range(num_feat)[:num_feat], X_train.columns[indices[:num_feat]])
        plt.xlim([-1, num_feat])
        plt.ylabel("Impurity reduction", fontsize=15)
        plt.show()
    
    # Wybór optymalmiej liczby zmiennych stosując model ARIMA
    forecaster_arimax = AutoARIMA(sp=4, start_P=1, start_Q=1, max_P=8, max_Q=8, suppress_warnings=True)
    wyniki = []
    for i in range(1, 5, 1):
        transformer = FeatureSelection(method="feature-importances", n_columns=i)
        X_train_selected = transformer.fit_transform(X_train, y_train)
        train_kol = X_train_selected.columns
        forecaster_arimax.fit(y=y_train, X=X_train_selected)
        y_pred_arimax = forecaster_arimax.predict(X=X_test[train_kol], fh=fh)
        wynik = compute_metrics(y_test, y_pred_arimax)
        wynik.columns = [i]
        wyniki.append(wynik)
    wyniki = pd.concat(wyniki, axis=1)

    # Wskazanie liczby zmiennych zmiennych będących zmiennymi objaśniającymi w predykcjach
    try:
        liczba_zmiennych = int(wyniki.idxmin(axis=1).mode())
    except:
        liczba_zmiennych = int(wyniki.idxmin(axis=1).MAPE)
    
    # Wskazanie listy poszczególnych zmiennych będących zmiennymi objaśniającymi w predykcjach
    transformer = FeatureSelection(method="feature-importances", n_columns=liczba_zmiennych)
    X_train_selected = transformer.fit_transform(X_train, y_train)
    return X_train_selected.columns


def get_most_important_cols2(X_train, y_train, n_features=5):
    
    """ Funkcja zwraca listę nazw kolumn, które minimalizują błąd predykcji, a tym samym stają się predyktorami do prognoz. Liczba zmiennych jest tu zadana z góry przez użytkowanika """
        
    transformer = FeatureSelection(method="feature-importances", n_columns=n_features)
    X_train_selected = transformer.fit_transform(X_train, y_train)
    return X_train_selected.columns


def wykres(y_train ,y_test=None, y_pred=None, y_pred_intervals=None):
    
    """ Funkcja tworzy wykres dla zbioru uczącego, zbioru testowego, predykcji i przedziału ufności """
    
    # Wykres zbioru treningowego
    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=y_train.index,
            y=y_train['VALUE'],
            mode='lines+markers',
            name='TRAIN',
            line=dict(color='rgba(52,52,215,0.75)')

        )
    )
    
    # Wykres zbioru testowego
    if y_test is not None:
        fig.add_trace(
            go.Scatter(
                x=y_test.index,
                y=y_test['VALUE'],
                mode='lines+markers',
                name='TEST',
                line=dict(color='rgba(52,215,52,0.75)')
            )
        )
        
    # Wykres zbioru predykcji
    if y_pred is not None:
        fig.add_trace(
            go.Scatter(
                x=y_pred.index,
                y=y_pred['VALUE'],
                name='PRED',
                mode='lines+markers',
                line=dict(color='rgba(235,52,52,0.75)')
            )
        )
        
    # Wykres przedziałów ufności dla predykcji
    if y_pred_intervals is not None:
        fig.add_trace(
            go.Scatter(
                x=list(it.chain.from_iterable([y_pred_intervals.index, y_pred_intervals.index[::-1]])),
                # x, then x reversed
                y=list(it.chain.from_iterable([y_pred_intervals['upper'], y_pred_intervals['lower'][::-1]])),
                # upper, then lower reversed
                fill='toself',
                fillcolor='rgba(68, 68, 68, 0.3)',
                line=dict(color='rgba(255,255,255,0)'),
                name='CONFINT'
            )
        )
    return fig


def compute_metrics(y_true, y_pred):
    
    """ Funkcja tworząca ramkę danych z metrykami MAPE, MSPE i RMSE. Należy podać wektor prawdziwych wartości i wartości predykowane """
    
    mape = mean_absolute_percentage_error(y_true, y_pred, symmetric=False)
    mspe = mean_squared_percentage_error(y_true, y_pred, symmetric=False)
    _rmse = MeanSquaredError(square_root=True)
    rmse = _rmse(y_true, y_pred)
    ramka = pd.DataFrame([[mape], [mspe], [rmse]], index=['MAPE', 'MSPE', 'RMSE'])
    ramka.columns = ['wynik']
    return ramka


def personr_lagged(x, y, lag):
    
    """ Funkcja tworząca opóźnione korelacje Pearsona """
    
    return scipy.stats.pearsonr(x[(lag - 1):], y[:-(lag - 1)])[0]


def generate_confidence_intervals(preds, residuals, alpha=0.1):
    
    """ Funkcja tworząca obiekt DataFrame z kolumnami upper i lower będącymi górnymi i dolnymi krańcami przedziału ufności dla prognoz """
    
    ci = np.quantile(residuals, 1 - alpha)

    ci_interv = pd.DataFrame()
    if ci >= 0:
        ci_interv['upper'] = preds + ci
        ci_interv['lower'] = preds - ci
    else:
        ci_interv['upper'] = preds - ci
        ci_interv['lower'] = preds + ci
    return ci_interv


def quarter_result(pred):
    
    """ Funkcja zwaraca agregacje kwartalne dla podanej tabeli testowej lub predykcji"""
    
    kwartały = pred.groupby(pd.Grouper(freq='Q')).sum()
    suma = kwartały.sum()
    return kwartały, suma


def filtruj(wektor_y):
    
    """ Funkcja filtruje wartości odstające z szeregu i wypełnia braki wielomianem łączącym skrajne punkty"""
    
    transformer = HampelFilter(window_length=5)
    wektor_y = transformer.fit_transform(wektor_y)
    wektor_y = wektor_y.interpolate(method='polynomial', order=3)
    return wektor_y

# FUNKCJE KONWERTUJĄCE DANE Z JEDNEGO GRUPOWANIA WZGLĘDEM DATY NA INNE

def convert_weekly_to_daily(preds):
    _preds = preds.copy()
    _preds.index = _preds.index.shift(-6, freq='D')
    _preds = _preds.resample('D').ffill()
    try:
        _preds['VALUE'] = _preds['VALUE'] / 7
    except:
        _preds['upper'] = _preds['upper'] / 7
        _preds['lower'] = _preds['lower'] / 7
    _preds = pd.concat([_preds,pd.DataFrame(index = _preds.index[-6:].shift(6))]).ffill()
    return _preds


def convert_weekly_to_daily_xgb(preds):
    _preds = preds.copy()
    _preds = _preds.resample('D').ffill()
    try:
        _preds['VALUE'] = _preds['VALUE'] / 7
    except:
        _preds['upper'] = _preds['upper'] / 7
        _preds['lower'] = _preds['lower'] / 7
    _preds = pd.concat([_preds,pd.DataFrame(index = _preds.index[-6:].shift(6))]).ffill()
    return _preds


def convert_and_preprocess_daily_to_monthly(pred):
    try:
        selected_index = pred.groupby(pd.Grouper(freq='M')).count().query("VALUE > 25").index
    except:
        selected_index = pred.groupby(pd.Grouper(freq='M')).count().query("upper > 25").index
    return pred.groupby(pd.Grouper(freq='M')).sum().loc[selected_index,:]


def convert_weekly_to_monthly(pred):
    _pred = convert_and_preprocess_daily_to_monthly(convert_weekly_to_daily(pred))
    return _pred

def convert_weekly_to_monthly_xgb(pred):
    _pred = convert_and_preprocess_daily_to_monthly(convert_weekly_to_daily_xgb(pred))
    return _pred


def fh_oczyszczone(start = '', periods = 31, freq = 'W'):
    horyzont = ForecastingHorizon(pd.date_range(start=start, periods=periods, freq=freq), is_relative=False)
    return horyzont


def fh_odstajace(start = '', periods = 7, freq = 'M'):
    horyzont = ForecastingHorizon(pd.date_range(start=start, periods=periods, freq=freq), is_relative=False)
    return horyzont


def fh_inwentaryzacje(start = '', periods = 7, freq = 'M'):
    horyzont = ForecastingHorizon(pd.date_range(start=start, periods=periods, freq=freq), is_relative=False)
    return horyzont


def fh_wszystkie(start = '', periods = 7, freq = 'M'):
    horyzont = ForecastingHorizon(pd.date_range(start=start, periods=periods, freq=freq), is_relative=False)
    return horyzont


def fh_predykcje(start = '', freq = 'M'):
    if freq == 'M':
        periods = _
    elif freq == 'W':
        periods = _
    else:
        pass
    horyzont = ForecastingHorizon(pd.date_range(start=start, periods=periods, freq=freq), is_relative=False)
    return horyzont
