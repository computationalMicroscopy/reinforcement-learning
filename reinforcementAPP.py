import streamlit as st
import numpy as np
import pandas as pd
import time
import random

# --- SEITEN KONFIGURATION ---
st.set_page_config(page_title="KI Schulung: Reinforcement Learning", layout="wide")

st.title("🤖 Der lernende Saugroboter")
st.markdown("""
In dieser Demo lernst du das Prinzip von **Reinforcement Learning**. 
Der Roboter (S) versucht die Ladestation (L) zu finden. 
Er kennt den Weg anfangs nicht und lernt durch Belohnung und Bestrafung.
""")

# --- PARAMETER ---
GRID_SIZE = 5
ACTIONS = ['hoch', 'runter', 'links', 'rechts']
EPSILON = 0.2  # Exploration (Zufall)
ALPHA = 0.5    # Lernrate
GAMMA = 0.9    # Diskontierungsfaktor

# --- INITIALISIERUNG DES STATES ---
if 'q_table' not in st.session_state:
    # Q-Table: Zeilen sind Felder (0-24), Spalten sind Aktionen
    st.session_state.q_table = pd.DataFrame(
        np.zeros((GRID_SIZE*GRID_SIZE, len(ACTIONS))),
        columns=ACTIONS
    )
if 'robot_pos' not in st.session_state:
    st.session_state.robot_pos = 0
if 'steps' not in st.session_state:
    st.session_state.steps = 0
if 'score' not in st.session_state:
    st.session_state.score = 0

TARGET_POS = 24  # Unten rechts
OBSTACLE_POS = [12, 13] # Hindernisse in der Mitte

# --- FUNKTIONEN ---
def get_coords(state):
    return state // GRID_SIZE, state % GRID_SIZE

def move_robot(state, action_idx):
    r, c = get_coords(state)
    new_r, new_c = r, c
    
    if ACTIONS[action_idx] == 'hoch': new_r -= 1
    elif ACTIONS[action_idx] == 'runter': new_r += 1
    elif ACTIONS[action_idx] == 'links': new_c -= 1
    elif ACTIONS[action_idx] == 'rechts': new_c += 1
    
    # Wand-Kollision prüfen
    if new_r < 0 or new_r >= GRID_SIZE or new_c < 0 or new_c >= GRID_SIZE:
        return state, -10  # Starke Strafe für Wand-Kollision
    
    new_state = new_r * GRID_SIZE + new_c
    
    # Hindernis-Kollision prüfen
    if new_state in OBSTACLE_POS:
        return state, -10 # Strafe für Hindernis
    
    # Belohnung für Richtung
    old_dist = abs(r - (GRID_SIZE-1)) + abs(c - (GRID_SIZE-1))
    new_dist = abs(new_r - (GRID_SIZE-1)) + abs(new_c - (GRID_SIZE-1))
    
    if new_state == TARGET_POS:
        return new_state, 100
    elif new_dist < old_dist:
        return new_state, 1
    else:
        return new_state, -1

# --- UI LAYOUT ---
col1, col2 = st.columns([1, 1])

with col1:
    st.subheader("Spielfeld")
    grid_display = np.full((GRID_SIZE, GRID_SIZE), "⬜")
    for obs in OBSTACLE_POS:
        grid_display[get_coords(obs)] = "🚫"
    grid_display[get_coords(TARGET_POS)] = "🔋"
    grid_display[get_coords(st.session_state.robot_pos)] = "🤖"
    
    st.table(grid_display)
    st.write(f"**Schritte:** {st.session_state.steps} | **Gesamt-Score:** {st.session_state.score}")

with col2:
    st.subheader("KI-Gehirn (Q-Table)")
    st.write("Die Werte zeigen, welche Richtung in welcher Zelle als 'gut' empfunden wird.")
    st.dataframe(st.session_state.q_table.style.highlight_max(axis=1, color='lightgreen'))

# --- TRAININGSLOGIK ---
if st.button("Lernschritt ausführen (Auto-Play)"):
    for _ in range(10): # Führe 10 Schritte pro Klick aus für die Animation
        state = st.session_state.robot_pos
        
        # Epsilon-Greedy Strategie
        if random.uniform(0, 1) < EPSILON:
            action_idx = random.randint(0, 3)
        else:
            action_idx = np.argmax(st.session_state.q_table.iloc[state].values)
        
        new_state, reward = move_robot(state, action_idx)
        
        # Q-Learning Formel
        old_value = st.session_state.q_table.iloc[state, action_idx]
        next_max = np.max(st.session_state.q_table.iloc[new_state].values)
        
        new_value = (1 - ALPHA) * old_value + ALPHA * (reward + GAMMA * next_max)
        st.session_state.q_table.iloc[state, action_idx] = new_value
        
        st.session_state.robot_pos = new_state
        st.session_state.score += reward
        st.session_state.steps += 1
        
        if new_state == TARGET_POS:
            st.success("Ziel erreicht! Er startet neu.")
            time.sleep(1)
            st.session_state.robot_pos = 0
            break
            
        time.sleep(0.1)
    st.rerun()

if st.button("Reset"):
    st.session_state.robot_pos = 0
    st.session_state.score = 0
    st.session_state.steps = 0
    st.session_state.q_table = pd.DataFrame(np.zeros((GRID_SIZE*GRID_SIZE, len(ACTIONS))), columns=ACTIONS)
    st.rerun()
