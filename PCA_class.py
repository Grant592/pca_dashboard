import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import dash
import dash_table
import dash_core_components as dcc
import dash_html_components as html
from dash.dependencies import Input, Output
import datetime
import plotly.graph_objs as go
from plotly.subplots import make_subplots
from IPython.display import HTML

class PCA_class:

    def __init__(self, csv_file, wellness_csv):
        self.df = pd.read_csv(csv_file)
        self.df['Date'] = pd.to_datetime(self.df['Date'] )
        self.df = self.df[self.df['Data Type'] == 'Whole Session']
        self.wellness_df = pd.read_csv(wellness_csv)
        self.wellness_df['Date'] = pd.to_datetime(self.wellness_df['Date'])
        self.players = self.df['Name'].unique()

    def create_EWM_(self,player_name):
        """Create EWM for given player for 7 and 28 day loads"""
        self.player_ = player_name
        self.df = self.df[self.df['Name'] == player_name]
        self.df['Total Distance (m)'] = pd.to_numeric(self.df['Total Distance (m)'], errors='coerce')
        df_daily = self.df.set_index('Date').groupby('Name').resample('D').sum()
        self.df.set_index(['Name', 'Date'], inplace=True)
        self.df.sort_index(level=1, inplace=True)
        df_daily7 = df_daily.groupby(level='Name').apply(lambda x: x.ewm(span=7).mean())
        df_daily28 = df_daily.groupby(level='Name').apply(lambda x: x.ewm(span=28).mean())
        column_map7 = {'Total Distance (m)':'distance_7_day',
                     '+5m/s Distance (m)':'hsr_7_day',
                      '+7m/s Distance (m)': 'vhsr_7_day',
                      'Player Load (2D)': 'pl_7_day'}
        column_map28 = {'Total Distance (m)':'distance_28_day',
                     '+5m/s Distance (m)':'hsr_28_day',
                       '+7m/s Distance (m)': 'vhsr_28_day',
                      'Player Load (2D)': 'pl_28_day'}

        df_daily7.rename(column_map7, axis=1, inplace=True)
        df_daily28.rename(column_map28, axis=1, inplace=True)
        all_df = self.df.join(df_daily7[['distance_7_day', 'hsr_7_day', 'vhsr_7_day', 'pl_7_day']], how='inner')
        all_df = all_df.join(df_daily28[['distance_28_day', 'hsr_28_day', 'vhsr_28_day', 'pl_28_day']], how='inner')

        pca_df = all_df.iloc[:,[0,1,2,3,4,5,6,7,8,9,10,11,12,13,96,97,98,99,100,101,102,103]]

        # Importing wellness data and calcultaing Z-score for each player for each day
        player_wellness = self.wellness_df.set_index('Date').groupby(['Name', 'Date']).sum()

        player_wellness = player_wellness.groupby(level=0).apply(lambda x: (x-x.mean())/x.std())

        pca_df = pca_df.iloc[:, -8:].groupby(level=0).apply(lambda x: (x-x.mean())/x.std())
        pca_df = pca_df.join(player_wellness['Wellness Total'], how='left')
        self.pca_df = pca_df

    def scale_fit_transform_(self, num_components=3):
        """Scale all data, center mean and map onto new principle components"""
        
        scaled_df = np.nan_to_num(self.pca_df.values)
        pca = PCA(n_components=num_components)
        pca.fit(scaled_df)
        self.pca_df_transformed = pca.transform(scaled_df)
        self.scaled_df = scaled_df

        n3 = pca.components_
        self.eig_vec = pd.DataFrame(n3).T
        self.eig_vec.set_index(keys=self.pca_df.columns, inplace=True)

        return self.eig_vec
        

    def plot_graphs(self, scree_on=True, biplot_on=True):
        
        combined_df_covar = np.cov(self.scaled_df, rowvar=False)
        values, vectors = np.linalg.eig(combined_df_covar)
        scree = (values/values.sum()).cumsum()
        scree_graph = go.Figure(data=go.Scatter(y=scree, x=list(range(1,4))))
        scree_graph.update_layout(title='Explained Variance', xaxis_title='Principle Component', yaxis_title="Cumulative Variance")

        if scree_on and biplot_on:
            fig = make_subplots(rows=3, cols=1)
            fig.add_trace(go.Scatter(x=self.pca_df_transformed[:,0],y=self.pca_df_transformed[:,1], mode='markers', marker=dict(opacity=0.2)), row=1, col=1)

            for i in range(9):
                fig.add_trace(go.Scatter(x=[0, self.eig_vec.iloc[i,0]], y=[0, self.eig_vec.iloc[i,1]], name=self.eig_vec.index[i], hovertext=self.eig_vec.index.values[i]),row=2, col=1)
            fig.add_trace(go.Scatter(y=scree, x=list(range(1,4))), row=3, col=1)
        return fig
        #return scree_graph

    def plot_annual(self):
        """Plot time series of PCA1 and 2 across all dates"""
        fig_annual = go.Figure()
        if self.eig_vec.iloc[:,0].sum() > 0:
            fig_annual.add_trace(go.Scatter(x=self.df.index.get_level_values(1), y=self.pca_df_transformed[:,0], name='PC1'))
        else:
            fig_annual.add_trace(go.Scatter(x=self.df.index.get_level_values(1), y=-self.pca_df_transformed[:,0], name='PC1'))
        if self.eig_vec.iloc[:,1].sum() > 0:
            fig_annual.add_trace(go.Scatter(x=self.df.index.get_level_values(1), y=self.pca_df_transformed[:,1], name='PC2'))
        else:
            fig_annual.add_trace(go.Scatter(x=self.df.index.get_level_values(1), y=-self.pca_df_transformed[:,1], name='PC2'))
        fig_annual.update_layout(title=self.player_)
        #fig_annual.show()
        return fig_annual
        

def main():
    PCA = PCA_class('whole_period_gps.csv', 'wellness_data.csv')
    player = input(f"Select Player from any of the following: {PCA.players}")
    PCA.create_EWM_(player)
    PCA.scale_fit_transform_()
    PCA.plot_graphs()
    PCA.plot_annual()


if __name__ == '__main__':
    main()


