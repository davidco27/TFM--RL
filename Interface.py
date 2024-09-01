#!/usr/bin/env python
# coding: utf-8

import os
import numpy as np
import streamlit as st
import pandas as pd
import ast
import altair as alt
import subprocess
import time
import matplotlib.pyplot as plt
from datetime import datetime

def get_highest_mean_rewards(folder_path):
    highest_mean_rewards = []
    algs = os.listdir(folder_path)
    for alg in algs:
        npz_file_path = os.path.join(folder_path, alg, 'evaluations.npz')
        if os.path.exists(npz_file_path):
            data = np.load(npz_file_path)
            rewards = data['results']
            mean_rewards = np.mean(rewards, axis=1)  # Calculate mean across episodes
            highest_mean_reward = np.max(mean_rewards)  # Find highest mean reward
            highest_mean_rewards.append({"Algoritmo": alg, "Mayor recompensa": round(highest_mean_reward)})
    return pd.DataFrame(highest_mean_rewards)
def get_reward_evolution(folder_path):
    npz_file_path = os.path.join(folder_path, 'evaluations.npz')
    if os.path.exists(npz_file_path):
        data = np.load(npz_file_path)
        rewards = data['results']
        mean_rewards = np.mean(rewards, axis=1)  # Calculate mean across episodes
        durations = data['ep_lengths']
        timesteps = data['timesteps']
        mean_durations = np.mean(durations, axis=1) 
    else:
        timesteps = []
        mean_rewards = []
        mean_durations = []
        
    return pd.DataFrame({"Iteraciones": timesteps, "Recompensas": mean_rewards, "Duraciones": mean_durations}) 

def read_txt_file(folder_path,file):
    txt_file_path = os.path.join(folder_path, file)
    if os.path.exists(txt_file_path):
        with open(txt_file_path, 'r',encoding='utf-8') as file:
            content = file.read()
        return content
    else:
        return "No se encontró ningún archivo de texto para este directorio."
def run_training(env,algo,selected_values,folder):
    command = f"python -m rl_zoo3.train --algo {algo} --env {env}"
    command += f" -n {selected_values['n_timesteps']} -f {folder}"
    command += " --eval-freq 10000 --eval-episodes 5 --n-eval-envs 2 --hyperparams"
    for parameter,value in selected_values.items():
        if parameter == "policy":
            value= f"'{value}'"
        command += f" {parameter}:{value}"
    process = subprocess.Popen(command)
    return process
def update_progress_bar(df):
    progress = len(df) / fixed_value
    return progress
current_directory = os.getcwd()
items_in_directory = os.listdir(current_directory)

# Filter out only the folders
folders_in_directory = [item for item in items_in_directory if os.path.isdir(os.path.join(current_directory, item))]


folders_in_directory.insert(0, "Sin selección")
selected_folder = st.sidebar.selectbox("Selecciona un entorno", folders_in_directory)

if selected_folder != "Sin selección":
    st.title(selected_folder)
    env_path = os.path.join(current_directory, selected_folder) 
    highest_mean_rewards_df = get_highest_mean_rewards(selected_folder).sort_values(by='Mayor recompensa',ascending=False).reset_index(drop=True)
    highest_mean_rewards_df.index += 1 
    highest_mean_rewards_df.rename_axis("Ranking", axis="index", inplace=True) 
    highest_mean_rewards_df["Mayor recompensa"] = highest_mean_rewards_df["Mayor recompensa"].apply(lambda x: str(x).replace(',', '.'))
    st.write(read_txt_file(env_path,'description.txt'))
    st.subheader("Caracterización del entorno")
    st.write(read_txt_file(env_path,'espacio_muestral.txt'))
    st.subheader("Agente sin entrenar")
    video_file_path = os.path.join(env_path, f'untrained-{selected_folder}.mp4')  # Adjust the file name and extension as needed
    if os.path.exists(video_file_path):
        st.video(video_file_path)
    st.divider()
    st.header("Aplicación de algoritmos de RL en este entorno")
    algs = os.listdir(env_path)
    algs = [item for item in algs if os.path.isdir(os.path.join(env_path, item))]
    option = st.radio("2 opciones disponibles:", ["Ver resultados de entrenamientos previos", "Entrena tu propio modelo"])
    if option == "Ver resultados de entrenamientos previos":
        st.header("Resultados de entrenamientos previos")
        st.write("El entrenamiento se ha realizado con 500000 iteraciones, haciendo una evaluación del modelo cada 10000 pasos.La evaluación se realiza con 5 episodios, quedando registrado la recompensa obtenida y la duración media de los episodios")
        st.subheader("Recompensas obtenidas por cada algoritmo")
        st.dataframe(highest_mean_rewards_df,width=600)
        st.divider()
        st.header("Comparación de las evoluciones")
        combined_data = pd.DataFrame()

        for alg in algs:
            alg_data = get_reward_evolution(os.path.join(selected_folder, alg))
            alg_data['Algoritmo'] = alg
            combined_data = pd.concat([combined_data, alg_data])

        # Crear un gráfico interactivo con Altair
        base = alt.Chart(combined_data).encode(
            x='Iteraciones',
            y='Recompensas',
            color='Algoritmo',
            tooltip=['Algoritmo', 'Iteraciones', 'Recompensas']
        )

        line = base.mark_line().interactive()
        points = base.mark_point().interactive()

        chart = line + points

        st.altair_chart(chart, use_container_width=True)
        st.divider()
        st.subheader("Selecciona un algoritmo")
        button_col = st.columns(len(algs))
        for i, alg in enumerate(algs):
            if button_col[i % len(algs)].button(alg):
                st.divider()
                st.header(alg.upper())
                code = f'''Mayor recompensa conseguida:  {highest_mean_rewards_df[highest_mean_rewards_df.Algoritmo == alg]['Mayor recompensa'].values[0]}'''
                st.code(code, language='matlab')
                st.subheader("Parámetros de entrenamiento usados:")
                with open(os.path.join(env_path,alg, 'params.txt') , "r") as file:
                    file_contents = file.read()
                code = f'''{file_contents}'''
                st.code(code, language='python')
                data = get_reward_evolution(os.path.join(selected_folder,alg))[:50]
                st.subheader("Datos de entrenamiento")
                st.dataframe(data)
                reward_chart = alt.Chart(data).mark_line().encode(
                x='Iteraciones',
                y='Recompensas',
                    color=alt.value('red')
                ).properties(
                    title='Recompensas medias obtenidas vs número de iteraciones'
                )
                st.altair_chart(reward_chart, use_container_width=True)
                st.divider()
                st.subheader(f"Agente entrenado con {alg}")
                video_file_path = os.path.join(env_path,alg, f'{alg}-{selected_folder}.mp4')  # Adjust the file name and extension as needed
                if os.path.exists(video_file_path):
                    st.video(video_file_path)

    if option == "Entrena tu propio modelo":
        st.header(f"Entrena tu modelo")
        st.subheader("1 - Elige el tipo de algoritmo")
        selected_algo = st.selectbox("Algoritmos disponibles :",algs)
        for i, alg in enumerate(algs):
            if selected_algo == alg:
                st.subheader("2 - Elige los parámetros")
                with open(os.path.join(env_path,selected_algo, 'params.txt') , "r") as file:
                    file_contents = file.read()
            
                with open(os.path.join(current_directory, f'params_{selected_algo}.txt') , "r") as file:
                    file_contents2 = file.read()

                st.write("Los parámetros que se usarán por defecto son:")
                dictionary_list = ast.literal_eval(file_contents)
                primary_values = {key: value for key, value in dictionary_list}

                dictionary_list = ast.literal_eval(file_contents2)
                default_values = {key: value for key, value in dictionary_list}
                code = {}
                for parameter, default_value in default_values.items():
                    value = primary_values.get(parameter, default_value)
                    code[parameter] = value
                st.code(code, language='python')

                st.write("Puedes personalizar ciertos parámetros a continuación")
                selected_params = {}
                for parameter, default_value in default_values.items():
                    value = primary_values.get(parameter, default_value)
                    if isinstance(value,str):
                        selected_params[parameter]  = st.selectbox("Policy:", ["CnnPolicy", "MlpPolicy"], index=0 if value == "CnnPolicy" else 1)
                    else:
                        value = str(value)
                        selected_params[parameter] = st.text_input(parameter.capitalize(), value=value)
                st.write(selected_params)
                st.subheader("3- Elige el número de iteraciones de entrenamiento")
                selected_params['n_timesteps'] = st.slider("Timesteps:", min_value=10000, max_value=1000000,step=10000,value=100000)
                btn_start = st.button("Empezar a entrenar!")
                graph_placeholder = st.empty()
                progress_placeholder = st.empty()
                timesteps_placeholder = st.empty()
                if btn_start:
                    st.write("¡Entrenamiento iniciado!")
                    env_dict = {'CarRacing': 'CarRacing-v2','MountainCarContinuous':'MountainCarContinuous-v0','HalfCheetah':'HalfCheetah-v4','Humanoid':'Humanoid-v4'}
                    current_time = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
                    folder_path = os.path.join(env_path,selected_algo,f"custom_training\{current_time}")
                    process = run_training(env_dict.get(selected_folder),selected_algo,selected_params,folder_path)
                    fixed_value = selected_params['n_timesteps']/10000
                    training = True
                    if st.button("Parar Entrenamiento",key="parar"):
                        process.terminate()
                        st.write("Entrenamiento terminado")
                        timesteps_placeholder.write(f"{len(results_df)*10000} / {fixed_value*10000} iteraciones ")
                        progress_placeholder.progress(update_progress_bar(results_df))
                        graph_placeholder.altair_chart(reward_chart, use_container_width=True)
                    path = os.path.join(folder_path,selected_algo, f"{env_dict.get(selected_folder)}_1")
                    while training:
                        results_df = get_reward_evolution(path)
                        timesteps_placeholder.write(f"{len(results_df)*10000} / {fixed_value*10000} iteraciones ")
                        progress_placeholder.progress(update_progress_bar(results_df))
                        reward_chart = alt.Chart(results_df).mark_line().encode(
                            x='Iteraciones',
                            y='Recompensas',
                                color=alt.value('red')
                            ).properties(
                                title='Recompensas medias obtenidas vs número de iteraciones'
                            )
                        reward_chart += alt.Chart(results_df).mark_point().encode(
                            x='Iteraciones',
                            y='Recompensas',
                            color=alt.value('red')
                         )
                        graph_placeholder.altair_chart(reward_chart, use_container_width=True)
                        time.sleep(1)

else:
    st.title("*Bienvenido a la plataforma interactiva de aprendizaje por refuerzo*")
    st.write("Esta plataforma te permite explorar los resultados de modelos preentrenados y experimentar entrenando nuevos modelos ajustando los parámetros a tu gusto. Contamos con cuatro entornos disponibles para tus experimentos: CarRacing, Mountain Car Continuous, Humanoid y HalfCheetah.")
    st.markdown("**Para empezar, simplemente elige uno de estos entornos y comienza a interactuar con los modelos.**")
    st.image("download.png")
