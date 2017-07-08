# pynoci
pynoci is a set of tools for Independent Component Analysis in Python 3. noci stands for optimal number of independent components in spanish

# Estudio de métodos de determinación del Número Óptimo de Componentes Independientes (NOCI) en señales multicanal.
# Trabajo Fin de Máster
# Daniel Mercader Rodríguez
# UNED 2017

En este trabajo fin de máster se presenta un estudio sobre diferentes métodos para la determinación del número óptimo de componentes independientes en señales multicanal.
Cuando se desarrolla un modelo de Análisis de Componentes Independientes (ICA) sobre un conjunto de señales multicanal, éstas se suponen formadas por diversas mezclas de otro conjunto de señales independientes entre sí. El Análisis de Componentes Independientes trata de extraer precisamente estas Componentes Independientes, de las que en numerosas ocasiones no se tiene ningún tipo de información a priori, hasta el punto de que ni siquiera seconoce el número de señales que forman este conjunto de señales independientes. Se han analizado diversos métodos para la determinación del Número Óptimo de Componentes Independientes (NOCI) a partir del conjunto de señales de mezcla y sin necesitar ningún conocimiento previo sobre las Componentes Independientes. Los principales métodos para determinación de NOCI son el Criterio de Durbin-Watson, el método ’ICA by blocks’, el método ’RV ICA by blocks’, método basado en Análisis de Componentes Principales (PCA) y recientemente el método basado en Correlación Lineal de Componentes (LCC).
En este trabajo se presentan dichos métodos, se han implementado en python 3 y se han aplicado sobre diferentes conjuntos de señales de prueba sintéticas y reales para demostrar su funcionamiento, comparar los métodos y adquirir experiencia en su utilización. Finalmente se presentan las principales características de cada uno de los métodos y una metodología de utilización de los mismos.
