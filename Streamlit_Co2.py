import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder, OrdinalEncoder
from sklearn.compose import ColumnTransformer, make_column_selector
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LinearRegression, LassoCV, Ridge, ElasticNetCV
from sklearn.model_selection import cross_validate, KFold, RandomizedSearchCV, GridSearchCV, cross_val_score
from sklearn.ensemble import RandomForestRegressor, HistGradientBoostingRegressor
from sklearn.model_selection import learning_curve
from scipy.stats import randint
import time
import pyarrow

st.set_page_config(layout="wide")
@st.cache_data(persist=True)

def load_data():
    data = pd.read_parquet("Co2_sample_v3_ST.parquet")

    return data

data = load_data()


data_metrics = data[["Pays", "Co2_Emission(WLTP)", "WLTP_poids", "Type_Carburant", "Puissance_KW"]]

data_electric_metrics = data_metrics[data_metrics["Type_Carburant"] == "Electric"]
data_thermique_metrics = data_metrics[~data_metrics["Type_Carburant"].isin(["Electric", "Hybride Essence", "Hybride Diesel","Autre"])]
data_hybride_metrics = data_metrics[data_metrics["Type_Carburant"].isin(["Hybride Essence", "Hybride Diesel"])]


avg_co2_global = data["Co2_Emission(WLTP)"].mean().round(2)

avg_co2_thermique = data_thermique_metrics.groupby("Pays")["Co2_Emission(WLTP)"].mean().round(2).reset_index()
# avg_co2_thermique["Co2_Emission(WLTP)"].mean().round(2)

avg_co2_hybrique = data_hybride_metrics.groupby("Pays")["Co2_Emission(WLTP)"].mean().round(2).reset_index()
# avg_co2_hybrique["Co2_Emission(WLTP)"].mean().round(2)


avg_poids_thermique = data_thermique_metrics.groupby("Pays")["WLTP_poids"].mean().round(2).reset_index()
#avg_poids_thermique["WLTP_poids"].mean().round(2)

avg_poids_hybride = data_hybride_metrics.groupby("Pays")["WLTP_poids"].mean().round(2).reset_index()
#avg_poids_hybride["WLTP_poids"].mean().round(2)


avg_poids_elec = data_electric_metrics.groupby("Pays")["WLTP_poids"].mean().round(2).reset_index()
#avg_poids_elec["WLTP_poids"].mean().round(2)


avg_power_thermique = data_thermique_metrics.groupby("Pays")["Puissance_KW"].mean().round(2).reset_index()
#avg_power_thermique["Puissance_KW"].mean().round(2)

avg_power_hybride = data_hybride_metrics.groupby("Pays")["Puissance_KW"].mean().round(2).reset_index()
#avg_power_hybride["Puissance_KW"].mean().round(2)

avg_power_electrique = data_electric_metrics.groupby("Pays")["Puissance_KW"].mean().round(2).reset_index()
#avg_power_electrique["Puissance_KW"].mean().round(2)


delta_poids = np.round(avg_poids_hybride["WLTP_poids"].mean().round(2) - avg_poids_thermique["WLTP_poids"].mean().round(2),2)
delta_poids_elec = np.round(avg_poids_elec["WLTP_poids"].mean().round(2) - avg_poids_thermique["WLTP_poids"].mean().round(2),2)

delta_power = np.round(avg_power_hybride["Puissance_KW"].mean().round(2) - avg_power_thermique["Puissance_KW"].mean().round(2),2)
delta_power_elec = np.round(avg_power_electrique["Puissance_KW"].mean().round(2) - avg_power_thermique["Puissance_KW"].mean().round(2),2)

total_elec = data_electric_metrics.shape[0]
total_hyb = data_hybride_metrics.shape[0]
total_therm = data_thermique_metrics.shape[0]

total = data.shape[0]
percent_elec = np.round((total_elec*100)/total,2)


carb_by_country = data.groupby("Pays")["Type_Carburant"].value_counts().reset_index(name = "count")


st.title("CO2 Emmission en Europe")

tab1, tab2, tab4 = st.tabs(["Général","Focus Thermique","Machine Learning"])

with tab1:

    st.header("Pays")

    options1 = sorted(data["Pays"].unique())

    select_pays = st.selectbox("Choisir un ou plusieurs pays :",options = options1, width=500,index = None, key = "test2")

    if select_pays:
        data = data[data["Pays"].str.contains(select_pays, case=False, na=False)]
    if select_pays:
        carb_by_country = carb_by_country[carb_by_country["Pays"].str.contains(select_pays, case=False, na=False)]
    if select_pays:
        avg_co2_thermique = avg_co2_thermique[avg_co2_thermique["Pays"].str.contains(select_pays, case = False, na = False)]
    if select_pays:
        avg_co2_hybrique = avg_co2_hybrique[avg_co2_hybrique["Pays"].str.contains(select_pays, case = False, na = False)]
    if select_pays:
        avg_poids_thermique = avg_poids_thermique[avg_poids_thermique["Pays"].str.contains(select_pays, case = False, na = False)]
    if select_pays:
        avg_poids_hybride = avg_poids_hybride[avg_poids_hybride["Pays"].str.contains(select_pays, case = False, na = False)]
    if select_pays:
        avg_poids_elec = avg_poids_elec[avg_poids_elec["Pays"].str.contains(select_pays, case = False, na = False)]
    if select_pays:
        avg_power_thermique = avg_power_thermique[avg_power_thermique["Pays"].str.contains(select_pays, case = False, na = False)]
    if select_pays:
        avg_power_hybride = avg_power_hybride[avg_power_hybride["Pays"].str.contains(select_pays, case = False, na = False)]
    if select_pays:
        avg_power_electrique = avg_power_electrique[avg_power_electrique["Pays"].str.contains(select_pays, case = False, na = False)]
    if select_pays:
        data_electric_metrics = data_electric_metrics[data_electric_metrics["Pays"].str.contains(select_pays, case = False, na = False)]


    col_emission, col_poids, col_power = st.columns([.33,.33,.33], gap= "small")

    with col_emission:

        st.markdown("<h3 style='text-align: center;'>Émissions de Co2</h3>", unsafe_allow_html=True)

        with st.container(border = True, height = 134):
           
            m1, m2, m3 = st.columns(3)

            m1.metric("Thermique",f"{avg_co2_thermique['Co2_Emission(WLTP)'].mean():.2f}")
    
            m2.metric("Hybride",f"{avg_co2_hybrique['Co2_Emission(WLTP)'].mean():.2f}" )
    
            m3.metric("Global",f"{avg_co2_global:.2f}" )
                   
    with col_poids:

        st.markdown("<h3 style='text-align: center;'>Poids</h3>", unsafe_allow_html=True)  

        with st.container(border = True):

            m4, m5, m6 = st.columns(3)

            m4.metric("Thermique",f"{avg_poids_thermique['WLTP_poids'].mean():.0f}" )
    
            m5.metric("Hybride",
        f"{avg_poids_hybride['WLTP_poids'].mean():.0f}",
        delta=f"{delta_poids:+.0f}",
        delta_color="inverse"
        )
    
            m6.metric("Électrique",
        f"{avg_poids_elec['WLTP_poids'].mean():.0f}",
        delta=f"{delta_poids_elec:+.0f}",
        delta_color="inverse"
        )
            
    with col_power:

        st.markdown("<h3 style='text-align: center;'>Puissance en Kw</h3>", unsafe_allow_html=True)  

        with st.container(border = True):
        
            m7, m8, m9 = st.columns(3)

            m7.metric("Thermique",f"{avg_power_thermique['Puissance_KW'].mean():.0f}")
    
            m8.metric("Hybride",
        f"{avg_power_hybride['Puissance_KW'].mean():.0f}",
        delta=f"{delta_power:+.0f}"
        )

            m9.metric("Électrique",f"{avg_power_electrique['Puissance_KW'].mean():.0f}",
        delta=f"{delta_power_elec:+.0f}"
        )

    col_pie2, col_map2 = st.columns([0.5,0.5], gap= "large")

    with col_pie2:
        st.subheader("Types de carburants")

        colors = ["#00b30f","#00b30fa7","#00b30f68","#00b30f2f","#00b32d1d","#02ab2c05"]
        pie2 = go.Figure()
        pie2.add_trace(go.Pie(
            values=carb_by_country["count"],
            labels=carb_by_country["Type_Carburant"],
            marker_line=dict(color="white", width=1.5),
            marker_colors=colors,
            textfont = dict(size = 14, color = "white"),
            pull=[0.05,0.05,0.05,0.25,0.25,0.25],
            hole=0.3
        ))
        pie2.update_layout(height=350, 
                        width = 350,
                        margin=dict(l=0, r=0, t=30, b=0),
                        legend=dict(
                        orientation="h",
                        y=1.1,
                        x=0.5,
                        xanchor="center")
                        )
        pie2.update_traces(textinfo='percent+label',textfont_size=12)
        st.plotly_chart(pie2, width="stretch", key = "pie2")

    with col_map2:
        
        st.subheader(f"CO₂ moyen par pays : {'Tous' if select_pays is None else select_pays}")

        avg_co2_pays = (
        data.groupby("Pays")["Co2_Emission(WLTP)"]
        .mean()
        .round(2)
        .reset_index(name="Avg_Co2")
        .sort_values(by="Avg_Co2", ascending=True)
        )

        fig_choro = go.Figure()
        fig_choro.add_trace(go.Choropleth(
        locations=avg_co2_pays["Pays"],
        locationmode="country names",
        z=avg_co2_pays["Avg_Co2"],
        text=avg_co2_pays["Pays"],
        colorscale="ylgn",
        marker_line_color="black",
        marker_line_width=0.5,
        showscale=False
        ))

        fig_choro.update_layout(paper_bgcolor="rgba(0,0,0,0)",
                            plot_bgcolor="rgba(0,0,0,0)") 
    
        fig_choro.update_geos(bgcolor="rgba(0,0,0,0)",
                          showland=True, landcolor="rgba(0,0,0,0)",
                          showocean= False,
                          showlakes= False,
                          showcountries=True,
                          countrycolor="rgba(150,150,150,0.3)",
                          center=dict(lat=50, lon=8),
                          projection_scale=1.8)

        fig_choro.update_layout(
        geo=dict(scope="europe", projection_type="natural earth"),
        height=560,width = 800,
        margin=dict(l=0, r=0, t=0, b=0),
    )

        st.plotly_chart(fig_choro, width="stretch", key = "map2")

with tab2:

    thermique = data[data["Type_Carburant"].isin(["Essence", "Diesel"])]

    thermique = thermique.drop(columns = ["Conso_Wh/km","Electric range (km)","ID"])

    def new_func(thermique):
        most_polluant_brand = thermique.groupby(["Pays","Constructeur","Model","Type_Carburant"])[["Co2_Emission(WLTP)","Fuel consumption"]].mean().round(2).reset_index()
        return most_polluant_brand

    most_polluant_brand = new_func(thermique)

    most_polluant_brand.sort_values(by = "Co2_Emission(WLTP)", ascending= False)    

    avg_global = most_polluant_brand.groupby("Pays")["Co2_Emission(WLTP)"].mean().reset_index().round(2).sort_values(by = "Co2_Emission(WLTP)", ascending = False)

    brands = np.round(most_polluant_brand.groupby(["Pays","Constructeur"])["Co2_Emission(WLTP)"].mean().reset_index().sort_values(by = "Co2_Emission(WLTP)", ascending=False),2)

    brand = brands["Constructeur"].unique()

    graph_tab2 = st.toggle("Afficher le graph")


    if graph_tab2:
        
        select_brand = st.selectbox(
        "Choisir un Constructeur :",
        options=brand,
        index=None,
        key="graph1"
    )

        if select_brand == None:
            fig_brand1 = go.Figure()

            avg_global1 = avg_global.copy()

            fig_brand1.add_trace(go.Bar(
                x = avg_global1["Pays"],
                y= avg_global1["Co2_Emission(WLTP)"],
                name = "Co2 Europe Moyen",
                text= avg_global1["Co2_Emission(WLTP)"]               
                            ))
            fig_brand1.update_layout(
                title = dict(
                text = f"Moyenne des emissions des Vehicules thermique en Europe",
                font = dict(size = 12)
                ))
        
            st.plotly_chart(fig_brand1)

        
        if select_brand:
            st.subheader(f"CO₂ moyen par pays, pour la marque : {"A selectionner" if select_brand is None else select_brand}")

            brands = brands[brands["Constructeur"].str.contains(select_brand, na = False, case = False)]

            fig_brand2 = go.Figure()

            fig_brand2.add_trace(go.Bar(
                x = avg_global["Pays"],
                y= avg_global["Co2_Emission(WLTP)"],
                name = "Co2 Europe Moyen",
                text= avg_global["Co2_Emission(WLTP)"],
                orientation = "v"
                            ))
            

            fig_brand2.add_trace(go.Bar(
                x = brands["Pays"],
                y = brands["Co2_Emission(WLTP)"],
                name = f"Co2 pour la Marque {select_brand}",
                text = brands["Co2_Emission(WLTP)"],
                orientation = "v"
                ))

            fig_brand2.update_layout(
            title = dict(
                text = f"Moyenne des emissions des Vehicules thermique  - {select_brand}",
                    font = dict(size = 12)),
                        legend = dict(x=0,y=1.18),
                            barmode = "group",
                                bargap = 0.15,
                                     bargroupgap = 0.1
        )
                                                        

            st.plotly_chart(fig_brand2)

        with st.expander("Voir les explications"):
            st.write(f"""le graphique montre les moyennes des emissions de Co2 pour chaque pays de l'Europe.
                     \n Nous pouvons aussi voir l'impact d'un constructeur, ici {select_brand}.""")

    graph1, graph2 = st.columns(2)
    
    with graph1:
        with st.container(border = True, height=650, width = 800):
            
            options2 = sorted(data["Pays"].unique())

            select_pays2 = st.selectbox("Choisir un plusieurs pays :",options = options2, width=500,index = None, key = "test3")
                        
            if select_pays2 == None:
                quartille_SANS_pays = (
                                data
                                .groupby("Poids_Quartile")["PuissanceKW_Quartile"]
                                .value_counts(normalize=True)   
                                .mul(100)                       
                                .reset_index(name="percent")
                                .round(2)
                                )
                
                quart_SP = px.bar(
                quartille_SANS_pays,
                x="Poids_Quartile",
                y="percent",
                color="PuissanceKW_Quartile",
                barmode="relative",
                text="percent"
                )

                quart_SP.update_layout(
                yaxis_title="Répartition (%)",
                xaxis_title="Quartile de poids",
                title = dict(
                        text = f"Distribution des quaritles de poids en fontion des quartiles de puissance.",
                                 font = dict(size=12)
                                 )
                )

                quart_SP.update_traces(
                texttemplate="%{text} %",
                textposition="inside"
                )

                st.plotly_chart(quart_SP)

            if select_pays2:
                quartille_avec_pays = (
                                 data
                                .groupby(["Pays","Poids_Quartile"])["PuissanceKW_Quartile"]
                                .value_counts(normalize=True)   
                                .mul(100)                       
                                .reset_index(name="percent")
                                .round(2)
                                )
                
                quartille_avec_pays = quartille_avec_pays[quartille_avec_pays["Pays"].str.contains(select_pays2, na = False, case = False)]
            
                quart_AP = px.bar(
                quartille_avec_pays,
                    x="Poids_Quartile",
                    y="percent",
                    color="PuissanceKW_Quartile",
                    barmode="relative",
                    text="percent"
                )

                quart_AP.update_layout(
                    yaxis_title="Répartition (%)",
                        xaxis_title="Quartile de poids",
                    title = dict(
                        text = f"Distribution des quaritles de poids en fontion des quartiles de puissance - {select_pays2}.",
                                 font = dict(size=12)
                                 )
                )

                quart_AP.update_traces(
                    texttemplate="%{text} %",
                        textposition="inside"
                )

                st.plotly_chart(quart_AP)

       
    with graph2:
        with st.container(border = True, height=650, width = 800):

            variables_1 = ["Puissance_KW", "WLTP_poids", "Fuel consumption","Co2_Emission(WLTP)"]
            variables_2 = ["Puissance_KW", "WLTP_poids", "Fuel consumption","Co2_Emission(WLTP)"]

            df_scatter = (thermique.groupby("Constructeur")
                          [["Puissance_KW","WLTP_poids","Fuel consumption","Co2_Emission(WLTP)"]]
                          .mean()
                          .reset_index()
                          .round(2))
            
            select_variable_1 = st.selectbox("Comparer la variable :", options = variables_1, index = None, key = "v1")
            select_variable_2 = st.selectbox("A la variable :", options = variables_2, index = None, key = "v2")
            
            if select_variable_1 is None or select_variable_2 is None:
                st.info("Sélectionner deux variables pour afficher le graphique.")

            elif select_variable_1 == select_variable_2:
                st.error("Veuillez choisir deux variables différentes.")
   
            else:
                fig_scatter = px.scatter(df_scatter,
                         x = select_variable_1,
                         y = select_variable_2,
                         hover_data="Constructeur",
                         trendline= "ols",
                         trendline_color_override= "red")
                
                fig_scatter.update_layout(
                    title = dict(
                        text = f"Corrélation entre la varaible {select_variable_1} et la variable {select_variable_2}",
                        font = dict(size = 12)
                        )
                )
            
                st.plotly_chart(fig_scatter)

    graph3, graph4 = st.columns(2)

    with graph3:
        with st.container(border = True, height=650, width = 800):
            nb_car_by_country = (data.groupby(["Pays","Type_Carburant"])["ID"]
                                 .size()
                                 .reset_index(name = "Nombre d'enregistrement")
                                 .sort_values(by = "Nombre d'enregistrement", ascending = False))
            
            top_10_enreg = nb_car_by_country.head(10)
            
            nb_car_by_country = nb_car_by_country.rename(columns = {"Type_Carburant" : "Type de Carburant"})

            options3 = sorted(data["Pays"].unique())

            select_pays3 = st.selectbox("Choisir un pays :",options = options3, width=500,index = None, key = "pays3")
            
            if select_pays3:
                nb_car_by_country = nb_car_by_country[nb_car_by_country["Pays"].str.contains(select_pays3, na = False, case = False)]
                
                st.data_editor(
                    nb_car_by_country,
                        column_config = {"Nombre d'enregistrement" :
                                        st.column_config.ProgressColumn("Nombre d'enregistrement",
                                                                     format = "%.0f",
                                                                     min_value = int(nb_car_by_country["Nombre d'enregistrement"].min()),
                                                                     max_value = int(nb_car_by_country["Nombre d'enregistrement"].max())
                                                                     )},
                                                                     hide_index = True,
                                                                     width="stretch",
                                                                     height = 248)
            else:
                st.data_editor(
                    top_10_enreg,
                        column_config = {"Nombre d'enregistrement" :
                                        st.column_config.ProgressColumn("Nombre d'enregistrement",
                                                                     format = "%.0f",
                                                                     min_value = int(nb_car_by_country["Nombre d'enregistrement"].min()),
                                                                     max_value = int(nb_car_by_country["Nombre d'enregistrement"].max())
                                                                     )},
                                                                     hide_index = True,
                                                                     width="stretch",
                                                                     height = 388)

if "results_df" not in st.session_state:
    st.session_state.results_df = None

with tab4:

    st.title("Machine Learning - Modèle prédictif -")

    st.markdown("""Problématique : 
                \n Prédire les émissions de CO₂ (WLTP) à partir des caractéristiques techniques des véhicules""")

    def load_data_ml():
        df = data[["Constructeur","WLTP_poids","Co2_Emission(WLTP)","Type_Carburant","Puissance_KW","Fuel consumption","Pays"]]

        df = df[df["Type_Carburant"] != "Electric"]

        df = df[~(
        df["Type_Carburant"].isin(["Autre"])
        & (df["Co2_Emission(WLTP)"] == 0.0)
        )]

        return df
    
    data_ML = load_data_ml()

    target = data_ML["Co2_Emission(WLTP)"]
    data_ML = data_ML.drop(columns="Co2_Emission(WLTP)")

    num_features = make_column_selector(dtype_exclude=["category","object"])(data_ML)
    cat_features = make_column_selector(dtype_include=["category","object"])(data_ML)

    kf = KFold(n_splits = 5, shuffle=True, random_state=42)

    num_prepocessor = Pipeline(
                steps = [
            ("impute",SimpleImputer(strategy="median")),
            ("scaler", StandardScaler())
        ]
    )

    cat_prepocessor = Pipeline(
                steps = [
            ("impute", SimpleImputer(strategy="constant", fill_value="Unknown")),
            ("encoder", OneHotEncoder(handle_unknown="ignore"))
        ]
    )

    preprocessors = ColumnTransformer(
                transformers = [
            ('num',num_prepocessor, num_features),
            ('cat',cat_prepocessor, cat_features)
        
        ]
    )
        
    model_LR = Pipeline(
                steps = [
            ("preprocessors", preprocessors),
            ("linreg", LinearRegression())
        ]
    )

    model_LR.fit(data_ML, target)

    feature_names = model_LR.named_steps["preprocessors"].get_feature_names_out()
    coefs = model_LR.named_steps["linreg"].coef_

    coef_df = pd.DataFrame({"feature": feature_names, "coef": coefs})

    coef_df["variables"] = (
            coef_df["feature"]
            .str.replace(r"^(num|cat)__", "", regex=True)   
            .str.split("_").str[0]                          
            )

    group_importance = (coef_df.groupby("variables")["coef"]
                    .apply(lambda s: np.sqrt(np.sum(s**2)))
                    .sort_values(ascending=False))

    rename_map = {
            "Type": "Type Carburant",
            "WLTP": "WLTP_poids",
            "Puissance": "Puissance_KW"
            }

    group_importance = group_importance.rename(index=rename_map)


    with st.expander(" 1 - Corrélation"):

        corr, scat = st.columns([4,6])

        with corr:

            with st.container(border = True):
            
                corr_matrix = data.corr(numeric_only=True)

                fig, ax = plt.subplots(figsize=(6, 4))
                sns.heatmap(
                corr_matrix,
                annot=True,
                cmap="coolwarm",
                center=0,
                ax=ax
                )

                ax.set_title("Feature Correlation Matrix")

                st.pyplot(fig)

                st.write(f"""La matrice de corrélation met en évidence des relations très fortes entre les émissions de CO₂ et certaines variables clés,
                     \n notamment la consommation de carburant et les indicateurs liés à l’électrification. 
                     \n Les corrélations négatives observées s’expliquent par la coexistence de différentes technologies de motorisation dans le jeu de données et ne traduisent pas une incohérence des données."""
                     )

        with scat:
            
            with st.container(border = True, height = 630, width = 700):
        
                scatter = data.copy()

                variables_ML = ["WLTP_poids","Puissance_KW","Fuel consumption"]

                select_variable_ML = st.selectbox("Choisir une variable:", index = None, options = variables_ML)

                if select_variable_ML:
                    fig, ax = plt.subplots(figsize=(10, 6))

                    sns.scatterplot(
                    data=data,
                    x=select_variable_ML,             
                    y="Co2_Emission(WLTP)",
                    hue="Type_Carburant",
                    ax=ax,
                    s=20,
                    alpha=0.5
                    )

                    ax.legend(title="Type_Carburant", fontsize=8, markerscale=0.7, loc="upper right")
                    ax.set_xlabel(select_variable_ML)
                    ax.set_ylabel("CO₂ (WLTP)")
                    st.pyplot(fig)

    with st.expander(" 2 - Préprocessing des données"):

        col_model, col_explication = st.columns([6,4])

        with col_model:

            with st.container():
                    
                    code_model =(
            """ 
num_features = make_column_selector(dtype_exclude=["category","object"])(data_ML)
cat_features = make_column_selector(dtype_include=["category","object"])(data_ML)

kf = KFold(n_splits = 10, shuffle=True, random_state=42)

num_prepocessor = Pipeline(
    steps = [
        ("impute",SimpleImputer(strategy="median")),
        ("scaler", StandardScaler())
    ]
)

cat_prepocessor = Pipeline(
    steps = [
        ("impute", SimpleImputer(strategy="constant", fill_value="Unknown")),
        ("encoder", OneHotEncoder(handle_unknown="ignore"))
    ]
)

preprocessors = ColumnTransformer(
    transformers = [
        ('num',num_prepocessor, num_features),
        ('cat',cat_prepocessor, cat_features)
        
    ]
)
        
model_LR = Pipeline(
    steps = [
        ("preprocessors", preprocessors),
        ("linreg", LinearRegression())
    ]
)"""
        )
                
            st.code(code_model, language = "python",height="content", width="stretch")

        with col_explication:

            with st.container(border = True):

                st.markdown("""Les variables numériques et catégorielles sont automatiquement identifiées à partir du jeu de données.
Les variables numériques sont imputées par la médiane puis standardisées, tandis que les variables catégorielles sont imputées par une modalité « Unknown » et encodées par one-hot encoding.
Ces transformations sont regroupées dans un préprocesseur unique appliqué via un ColumnTransformer.
L’ensemble du prétraitement est intégré dans un pipeline avec un modèle de régression linéaire, garantissant que les transformations sont correctement appliquées à chaque étape de la validation croisée (10 folds).""")

    with st.expander(" 3 - Choix du Modèle"):
        st.write("""

### Choix du modèle : Régression linéaire

La **régression linéaire** a été choisie comme modèle de prédiction en raison de sa simplicité, de sa robustesse et de son interprétabilité.  
Elle est particulièrement adaptée lorsque la variable cible peut être expliquée par une relation approximativement linéaire avec les variables explicatives, ce qui est pertinent dans le cadre de cette étude.

Ce modèle présente plusieurs avantages :
- il est facile à interpréter, chaque coefficient représentant l’influence d’une variable explicative sur la variable cible ;
- il constitue une référence solide (baseline) pour comparer des modèles plus complexes ;
- il est peu coûteux en temps de calcul, ce qui le rend adapté à une application interactive.

---

### Définition de la régression linéaire

La régression linéaire vise à modéliser la relation entre une variable cible continue et un ensemble de variables explicatives en ajustant une fonction linéaire aux données.  
L’objectif est de minimiser l’écart entre les valeurs observées et les valeurs prédites par le modèle, généralement à l’aide de la méthode des moindres carrés.

---

### Formule du modèle

La régression linéaire multiple s’écrit sous la forme :

 y = β0 + β1x1 + β2x2 +⋯+β9x9 + ε

où :
\n ▪ y : émissions de CO₂ (variable cible)
\n ▪ x1,x2,…,x9 : variables explicatives
    (consommation, poids, puissance, type de carburant, constructeur, etc.).
\n ▪ β1,…,β9 : coefficients estimés par le modèle, qui mesurent l’influence de chaque variable sur le CO₂, toutes choses égales par ailleurs.
\n ▪ β0 : intercept, (correspondant  avec standardisation) à la valeur moyenne du CO₂ prédite par le modèle.
\n ▪ ε/epsilone : erreur résiduelle, c’est-à-dire la part du CO₂ non expliquée par les variables du modèle (bruit)


Les coefficients sont estimés de manière à minimiser la somme des carrés des résidus, c’est-à-dire la différence entre les valeurs observées et les valeurs prédites.
""")

    with st.expander(" 4 - 1ère méthode de Scoring : R²"):

        col_r2, col_r2_explication = st.columns([6,4])

        with col_r2:               
            code_r2 = """
        
       def mean_r2_cv(model, X, y, cv):
            r2_scores = cross_validate(
            model,
            X,
            y,
            cv=cv,
            scoring="r2",
            return_train_score=True,
            n_jobs=-1
        )
            
            test_mean_r2 = r2_scores["test_score"].mean()
            train_mean_r2 = r2_scores["train_score"].mean()
        
            return train_mean_r2, test_mean_r2

    train_mean_r2, test_mean_r2 = mean_r2_cv(model_LR, data_ML, target, kf)"""
            
            st.code(code_r2, language = "python",height="content", width="stretch")

            with col_r2_explication:
                with st.container(border = True):
                    st.markdown("""R² :
                             \n Permet de vérifier si le modèle parvient à capter les relations entre les variables explicatives et la variable cible. Plus la valeur est élevée, plus le modèle traduit une meilleure capacité de prédiction.
                             \n * Un R² proche de 1 indique une très bonne qualité de prédiction, tandis qu’un R² proche de 0 indique que le modèle n’explique que peu la variabilité des données.
                             \n * Un R² négatif signifie que le modèle est moins performant qu’une prédiction basée sur la moyenne.""")
                    
                with st.container(border = True):

                    st.markdown("Résultats R² :")

                    kf2 = st.selectbox("Nombre de folds", [3, 5, 10], index=0, key = "r2_1")
                    kf = KFold(n_splits=kf2, shuffle=True, random_state=42)

                    if st.toggle("Mode rapide (échantillon) 5000 lignes", key = "r2_2"):

                        X_eval = data_ML.sample(n=min(5000, len(data_ML)), random_state=42)
                        y_eval = target.loc[X_eval.index]

                    else:
                        X_eval, y_eval = data_ML, target

                    def mean_r2_cv(model, X, y, cv):
                        r2_scores = cross_validate(
                        model,
                        X,
                        y,
                        cv=cv,
                        scoring="r2",
                        return_train_score=True,
                        n_jobs=-1
                        )

                        return r2_scores["train_score"].mean(), r2_scores["test_score"].mean()

                    if st.button("Run", key = "r2_3"):

                        start = time.time()

                        train_mean_r2, test_mean_r2 = mean_r2_cv(model_LR, X_eval, y_eval, kf2)
                        
                        time = time.time() - start
                        st.success(f"Le code à tourné pendant {np.round(time,2)} secondes.")

                        st.write(f"Train : {np.round(train_mean_r2,3)}")
                        st.write(f"Test : {np.round(test_mean_r2,3)}")

    with st.expander(" 5 - 2ème méthode de Scoring : Mean Squared Error (MSE)"):

        col_mse, col_mse_explication = st.columns([6,4])

        with col_mse:

            code_mse = """

        def mean_mse(model, X, y, cv):
            mse_scores = cross_validate(
            model,
            X,
            y,
            cv=cv,
            scoring = "neg_mean_squared_error",
            return_train_score = True,
            n_jobs=-1
        )

        test_mean_mse = -mse_scores["test_score"].mean()
        train_mean_mse = -mse_scores["train_score"].mean()

        return train_mean_mse, test_mean_mse
    
    train_mean_mse, test_mean_mse = mean_mse(model_LR, data_ML, target, kf)"""
            
            st.code(code_mse, language = "python", height="content", width="stretch")

        with col_mse_explication:
            
            with st.container(border = True):
                st.markdown("""La Mean Squared Error (MSE):
                            \n Mesure l’erreur moyenne entre les valeurs prédites par un modèle et les valeurs réelles.
                            Elle est calculée comme la moyenne des carrés des écarts entre prédictions et observations.
                            La MSE pénalise fortement les erreurs importantes, ce qui la rend particulièrement sensible aux valeurs aberrantes.
                            Une valeur de MSE faible indique une meilleure capacité du modèle à prédire correctement la variable cible.""")
                
            with st.container(border = True):
                st.markdown("Résultats MSE :")

                kfmse = st.selectbox("Nombre de folds", [3, 5, 10], index=0, key = "mse_1")
                kf = KFold(n_splits=kfmse, shuffle=True, random_state=42)

                if st.toggle("Mode rapide (échantillon) 5000 lignes", key = "mse_2"):

                    X_eval = data_ML.sample(n=min(5000, len(data_ML)), random_state=42)
                    y_eval = target.loc[X_eval.index]

                else:
                    X_eval, y_eval = data_ML, target

                    def mean_mse_cv(model, X, y, cv):
                        mse_scores = cross_validate(
                        model,
                        X,
                        y,
                        cv=cv,
                        scoring="neg_mean_squared_error",
                        return_train_score=True,
                        n_jobs=-1
                        )

                        return -mse_scores["train_score"].mean(), -mse_scores["test_score"].mean()

                    if st.button("Run", key = "mse_3"):

                        start = time.time()

                        train_mean_mse, test_mean_mse = mean_mse_cv(model_LR, X_eval, y_eval, kfmse)
                        
                        time = time.time() - start
                        st.success(f"Le code à tourné pendant {np.round(time,2)} secondes.")

                        st.write(f"Train : {np.round(train_mean_mse,3)}")
                        st.write(f"Test : {np.round(test_mean_mse,3)}")

    with st.expander(" 6 - Interprétation des Résultats"):
        if st.button("Lancer le code complet", key="all1"):
            kffinal = KFold(n_splits=5, shuffle=True, random_state=42)
            start = time.time()

            def evaluate_model_cv(model, X, y, cv):
                all_scores = cross_validate(
                model, X, y,
                cv=cv,
                scoring={"r2": "r2", "mse": "neg_mean_squared_error"},
                return_train_score=True,
                n_jobs=-1
            )
                return pd.DataFrame({
                "Metric": ["R2", "MSE"],
                "Train": [all_scores["train_r2"].mean(), -all_scores["train_mse"].mean()],
                "Test":  [all_scores["test_r2"].mean(),  -all_scores["test_mse"].mean()],
            })

            st.session_state.results_df = evaluate_model_cv(model_LR, X_eval, y_eval, kffinal)

            elapsed = time.time() - start
            st.success(f"Le code a tourné pendant {elapsed:.2f} secondes.")

    
        if st.session_state.results_df is None:
            st.info("Veuillez lancer le code pour obtenir les résultats finaux.")
        else:
            results_df = st.session_state.results_df
            st.dataframe(results_df, hide_index=True, width="stretch")

            r2_test  = results_df.loc[results_df.Metric=="R2",  "Test"].values[0]
            r2_train = results_df.loc[results_df.Metric=="R2",  "Train"].values[0]
            mse_test = results_df.loc[results_df.Metric=="MSE", "Test"].values[0]
            mse_train= results_df.loc[results_df.Metric=="MSE", "Train"].values[0]
            
            st.write(f"""
### Interprétation des résultats

**Coefficient de détermination (R²)**  
Le R² mesure la capacité du modèle à expliquer la variabilité de la variable cible à partir des variables explicatives.  
Une valeur proche de 1 indique que le modèle capture une grande partie de la structure présente dans les données.

- R² entraînement : **{r2_train:.3f}**
- R² validation   : **{r2_test:.3f}**

Les scores obtenus en entraînement et en validation (fold = 5) sont très proches, ce qui suggère une bonne stabilité du modèle
et l’absence de sur-apprentissage. Le modèle explique de manière cohérente la relation entre les caractéristiques des véhicules
et leurs émissions de CO₂.

---

**Erreur quadratique moyenne (MSE)**  
La MSE quantifie l’écart moyen entre les valeurs prédites par le modèle et les valeurs réelles observées.

- MSE entraînement : **{mse_train:.3f}**
- MSE validation   : **{mse_test:.3f}**

Les valeurs de MSE sont très similaires entre l’échantillon d’entraînement et l’échantillon de validation, ce qui indique que
la précision des prédictions reste stable lorsque le modèle est appliqué à des données non vues.
""")

    with st.expander(" 7 - Visualisation - Learning Curve"):

        # pills_LC = ("Calculer & Afficher la Learning Curve")

        lc_display, lc_inter = st.columns([7,3], gap = "small")

        with lc_display:

            with st.container(border = True, height = 600, width = 800):
                    
                           
                train_sizes, train_scores, test_scores = learning_curve(
                         estimator=model_LR,
                         X=data_ML,
                         y=target,
                         cv=kf,
                         scoring="r2",
                         train_sizes=np.linspace(0.1, 1.0, 10),
                         n_jobs=-1
                         )
                        
                train_mean = train_scores.mean(axis=1)
                test_mean = test_scores.mean(axis=1)

                lc = plt.figure(figsize=(8,5))
                     
                plt.plot(train_sizes, train_mean, label="Train R²")
                plt.plot(train_sizes, test_mean, label="Validation R²")

                plt.xlabel("Taille du jeu d'entraînement")
                plt.ylabel("R²")
                plt.title("Learning Curve")
                plt.legend()
                plt.grid(True)
                st.pyplot(lc)

            
                    # else:
                    #     lc_screen = r"C:\Users\alexd\Desktop\WorkSpace\NoteBook_Jupyter\Projet_DA_DST_Co2\lcStreamlit.png"

                    #     st.image(lc_screen)

            with lc_inter:

                with st.container(border = True):
                    st.write(f"""
                             
### Observations :
---
\n * Convergence des scores train et validation
\n * Stabilité du modèle sur de grands volumes de données
\n * Confirme une bonne généralisation du modèle
""")
                    
                    
                with st.container(border = True):
                    st.write(f"""Le Max train size (environ 550 000) est impacté par le KFold à 5, qui correspond à environ 80% du dataset.
                             Une validation croisée en KFold à 5 splits utilise environ 80 % des données pour l’entraînement à chaque itération, 
                             tout en garantissant que chaque observation est utilisée à la fois pour l’apprentissage et la validation.""")
                    
    with st.expander(" 8 - Visualisation des Prédictions"):

        viz_pred, pred_explication = st.columns([6,4], gap = "small")

        with viz_pred:

            with st.container(border = True, height = 647, width = 700):

                model_LR.fit(data_ML, target)
                train_preds = model_LR.predict(data_ML)

                nb_samples = st.slider("Choisir le nombre d'observations à afficher :", min_value=10,
                                       max_value=min(1000, len(target)),
                                       step=10,
                                       value=100)

                y_true_sample = target.iloc[:nb_samples]
                y_pred_sample = train_preds[:nb_samples]
                    
                pred_display = plt.figure(figsize=(12,8))

                plt.plot(y_true_sample.values, label="Vraies Emissions", marker='o')

                plt.plot(y_pred_sample, label="Prédictions", marker='x')

                plt.legend()
                plt.title(f"Comparaison vraies vs prédites (extrait des {nb_samples} premières valeurs)")
                plt.xlabel("Index")
                plt.ylabel("Emissions Co2")
                plt.grid(True)
                plt.legend(loc = "upper right")
        
                st.pyplot(pred_display)

        with pred_explication:

            with st.container(border = True):

                st.write("""
### Observation :
---
\n * Les prédictions suivent globalement bien la tendance des valeurs réelles
\n * Bonne capacité du modèle à reproduire les émissions de CO₂
\n * Les différences résiduelles sont cohérentes avec l’erreur moyenne observée (MSE)
""")
            with st.container(border = True):
                prediction = train_preds - target
                prediction = pd.DataFrame(prediction)
                prediction["Valeurs Réelles"] = target
                prediction["Prédictions"] = np.round(train_preds,3)

                prediction = prediction.rename(columns = ({"Co2_Emission(WLTP)" : "Erreur"}))

                prediction["Erreur"] = prediction["Erreur"].round(3)

                prediction.set_index("Valeurs Réelles")

                if nb_samples:
                    prediction = prediction.iloc[:nb_samples]

                    st.dataframe(prediction, hide_index = True, height = 280)

    with st.expander(" 9 - Visualisation - Régression Linéaire"):
        
        st.session_state.intercept = model_LR.named_steps["linreg"].intercept_

        viz_lr, explication_lr = st.columns([7,3], gap = "small")

        with viz_lr:
            with st.container(border = True, height = 600, width = 800):
                
                st.session_state.fig_lr = plt.figure(figsize=(7,5))

                plt.scatter(
                target,
                train_preds,
                alpha=0.6,
                label="Prédictions"
                )

                plt.plot(
                [target.min(), target.max()],
                [target.min(), target.max()],
                'r--',
                lw=2,
                label=f"Droite Linéaire (y = x)"
                )

                plt.xlabel("Valeurs réelles (CO₂)")
                plt.ylabel("Prédictions du modèle")
                plt.title("Prédictions vs Réalité – Jeu d'entraînement")

                plt.legend(
                title=f"Intercept du modèle = {st.session_state.intercept:.2f} g CO₂/Km"
                )


                plt.grid(True)
                st.pyplot(st.session_state.fig_lr)
        
        with explication_lr:
            with st.container(border = True):
                st.write(f"""
Le nuage de points montre une forte concordance entre les valeurs réelles et prédites,
confirmant la capacité du modèle à capturer la relation entre les caractéristiques des véhicules et leurs émissions de CO₂.
                         
Le modèle a tendance à :
                         
\n * Lisser les valeurs extrêmes
\n * Sous-estimer légèrement les très fortes émissions
\n * Surestimer certaines très faibles émissions
                         
Comportement classique d’un modèle linéaire.

""")

    with st.expander(" 10 - Métrics Complémentaires"):
        with st.expander("Root Mean Squared Error (RMSE)"):
                if st.session_state.results_df is None:
                    st.error("Calcule d’abord R²/MSE dans la section 6.")
                else:
                    df_metrics= st.session_state.results_df.set_index("Metric")
                    rmse = np.sqrt(df_metrics.loc["MSE", "Test"])
                    st.metric("RMSE", value=f"{rmse:.2f}g de CO₂/Km")
                    st.write(f"""
### Root Mean Squared Error (RMSE)

La **RMSE** permet de mesurer l’erreur moyenne du modèle en pénalisant davantage les écarts
importants entre les valeurs prédites et les valeurs réelles. Elle constitue ainsi un indicateur
pertinent de la **robustesse globale du modèle**.

---

La **Root Mean Squared Error (RMSE)** est estimée à **{rmse:.2f} g de CO₂ par kilomètre**.

C'est donc la moyenne des erreurs entre les valeurs prédites et les valeurs réelles.
""")

        with st.expander("Mean Absolute Error (MAE)"):
            mae = prediction["Erreur"].abs().mean().round(2)

            st.metric("MAE", value = f"{mae} g de CO₂/Km")
            st.write(f"""
la Mean Absolute Error (MAE) est égale à {mae} g de CO₂ par kilomètre. 
Cette métrique, plus intuitive, représente l’erreur moyenne absolue commise par le modèle et offre une lecture 
directe et facilement interprétable de la précision des prédictions.

""")

        with st.expander("Intercept du Modèle"):

            resultat_lr = model_LR.named_steps["linreg"]
            intercept_lr = np.round(resultat_lr.intercept_,2)

            avg_co2 = data["Co2_Emission(WLTP)"].mean().round(2)

            avgco2, intercept,aa = st.columns(3)

            with avgco2:
                st.metric("Moyenne Co2 du DataSet", value = f"{avg_co2} g de CO₂/Km", border = True, width = 380)

            with intercept:
                st.metric("Intercept du Modèle", value = f"{intercept_lr} g de CO₂/Km", border = True, width = 380)
            
            st.write("""
Dans le cadre d’un modèle utilisant des variables standardisées, 
l’intercept correspond à la valeur moyenne des émissions de CO₂ prédites par le modèle, 
c’est-à-dire la prédiction associée à un véhicule présentant des caractéristiques moyennes.

""")

    with st.expander(" 11 - Variables et Coefficient"):
        df_var, explication_var = st.columns([3,7])

        with df_var:
            with st.container(border = True):
                st.dataframe(group_importance, width="stretch")
        
        with explication_var:
            with st.container(border = True):
                st.write(f"""
La variable « Constructeur » présente la plus haute importance globale élevée dans le modèle. 
Cette importance ne traduit pas un effet direct, mais reflète des différences systématiques entre constructeurs liées à des choix technologiques, des stratégies de conception ou des caractéristiques non directement observées.
Le constructeur agit ainsi comme une variable capturant des effets résiduels non expliqués par les variables techniques.
Exemple des différents impacts en fonction des constructeurs :
                         
\n ● différences de technologie moteur
\n ● rendement réel des moteurs
\n ● stratégies d’optimisation WLTP
\n ● boîtes de vitesses
\n ● aérodynamique
\n ● calibration moteur

La variable “Fuel consumption” présente une importance globale plus modérée car son effet est capturé de manière directe par le modèle à travers un unique coefficient.
Contrairement aux variables catégorielles à nombreuses modalités, son influence n’est pas amplifiée par un mécanisme d’agrégation, 
ce qui explique son classement relatif tout en confirmant son rôle central dans la prédiction des émissions de CO₂.
""")
                
    with st.expander(" 12 - Conclusion"):
        st.write(f"""
L’utilisation d’un modèle de **régression linéaire multivariée** permet de prendre en compte simultanément l’ensemble des variables explicatives.  
Les performances obtenues sont **élevées et stables**, avec des scores de **R² très proches entre l’entraînement et la validation croisée**, indiquant une bonne capacité de généralisation.

L’analyse des **learning curves** confirme l’absence de sur-apprentissage : les performances convergent rapidement lorsque la taille du jeu d’entraînement augmente, suggérant que le modèle exploite efficacement l’information disponible.

L’étude des **coefficients du modèle** montre que les émissions de CO₂ sont expliquées à la fois par des variables techniques directes (consommation, masse, puissance) et par des variables induites, telles que le constructeur, qui capturent des effets technologiques ou structurels non explicitement mesurés.

Ainsi, le modèle fournit des **prédictions fiables et cohérentes** des émissions de CO₂ à partir des caractéristiques des véhicules.

""")

