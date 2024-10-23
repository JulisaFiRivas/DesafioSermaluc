import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
import pickle
import json

class LabelEncoderPersistence:
    def __init__(self):
        self.encoders = {}
        self.mappings = {}
    
    def fit_transform_columns(self, df, columns):
        """
        Ajusta y transforma las columnas especificadas usando LabelEncoder
        
        Args:
            df: DataFrame de pandas
            columns: Lista de nombres de columnas a transformar
        
        Returns:
            DataFrame con las columnas transformadas
        """
        df_copy = df.copy()
        
        for col in columns:
            # Crear y ajustar el encoder
            le = LabelEncoder()
            df_copy[col] = le.fit_transform(df[col].astype(str))
            
            # Guardar el encoder y sus mappings
            self.encoders[col] = le
            self.mappings[col] = dict(zip(le.classes_, le.transform(le.classes_)))
        
        return df_copy
    
    def transform_columns(self, df, columns):
        """
        Transforma nuevos datos usando los encoders previamente ajustados
        """
        df_copy = df.copy()
        
        for col in columns:
            if col in self.encoders:
                # Manejar valores nuevos no vistos durante el entrenamiento
                df_copy[col] = df[col].astype(str)
                df_copy[col] = df_copy[col].map(lambda x: -1 if x not in self.encoders[col].classes_ else self.encoders[col].transform([x])[0])
            
        return df_copy
    
    def save_transformations(self, filepath):
        """
        Guarda los encoders y mappings en archivo
        """
        # Guardar encoders usando pickle
        with open(f"{filepath}_encoders.pkl", 'wb') as f:
            pickle.dump(self.encoders, f)
        
        # Guardar mappings en formato JSON para fácil lectura
        with open(f"{filepath}_mappings.json", 'w', encoding='utf-8') as f:
            json.dump({k: {str(key): int(value) for key, value in v.items()} 
                      for k, v in self.mappings.items()}, f, indent=2)
    
    def load_transformations(self, filepath):
        """
        Carga los encoders y mappings desde archivo
        """
        # Cargar encoders
        with open(f"{filepath}_encoders.pkl", 'rb') as f:
            self.encoders = pickle.load(f)
        
        # Cargar mappings
        with open(f"{filepath}_mappings.json", 'r', encoding='utf-8') as f:
            self.mappings = json.load(f)