import streamlit as st
import numpy as np
import pandas as pd
import base64
from io import BytesIO
import datetime
import io
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
import math
from PIL import Image
import matplotlib.patches as patches

st.set_page_config(page_title="Générateur de données minières synthétiques", layout="wide")

# Affichage du titre centré avec style amélioré
st.markdown(
    """
    <style>
    .title-container {
        display: flex;
        flex-direction: column;
        align-items: center;
        justify-content: center;
        text-align: center;
        margin-bottom: 30px;
        margin-top: 20px;
        background: linear-gradient(to right, #134e5e, #71b280);
        padding: 20px;
        border-radius: 10px;
        color: white;
    }
    .main-title {
        font-size: 36px;
        font-weight: bold;
        margin-bottom: 10px;
        text-shadow: 1px 1px 2px rgba(0,0,0,0.3);
    }
    .subtitle {
        font-size: 18px;
        opacity: 0.9;
    }
    </style>
    """,
    unsafe_allow_html=True
)

st.markdown(
    """
    <div class="title-container">
        <div class="main-title">Générateur de données minières synthétiques</div>
        <div class="subtitle">Développé par Didier Ouedraogo, P.Geo.</div>
    </div>
    """,
    unsafe_allow_html=True
)

# Sidebar pour les paramètres
st.sidebar.header("Paramètres")

# Section pour choisir le type de données
data_type = st.sidebar.radio(
    "Type de données à générer",
    ["Composites", "Modèle de bloc", "Données QAQC"]
)

# Liste des métaux disponibles
available_metals = ["Or", "Cuivre", "Zinc", "Manganèse", "Fer"]
selected_metals = st.sidebar.multiselect(
    "Sélectionnez les métaux",
    available_metals,
    default=["Or"]
)

# Fonction pour générer des forages avec orientation et répartition en grille
def generate_drillholes(num_drillholes, grid_spacing, azimuth, dip, min_depth, max_depth):
    # Calculer le nombre de rangées et colonnes pour la grille
    grid_size = math.ceil(math.sqrt(num_drillholes))
    
    # Coordonnées de départ des forages en grille
    hole_locations = []
    for i in range(grid_size):
        for j in range(grid_size):
            if len(hole_locations) < num_drillholes:
                # Ajouter un peu de bruit à la position exacte de la grille
                noise_x = np.random.normal(0, grid_spacing * 0.05)
                noise_y = np.random.normal(0, grid_spacing * 0.05)
                hole_locations.append((i * grid_spacing + noise_x, j * grid_spacing + noise_y))
    
    # Convertir azimuth et pendage en vecteurs de direction
    # Convention: azimuth 0 est Nord (Y+), 90 est Est (X+), etc. Dip est négatif vers le bas
    azimuth_rad = np.radians(azimuth)
    dip_rad = np.radians(dip)
    
    # Composantes du vecteur direction unitaire
    dx = np.sin(azimuth_rad) * np.cos(dip_rad)
    dy = np.cos(azimuth_rad) * np.cos(dip_rad)
    dz = -np.sin(dip_rad)  # Négatif car Z augmente vers le haut
    
    # Générer les trous de forage
    drillholes = {}
    
    for i, (x, y) in enumerate(hole_locations):
        # ID du trou de forage
        hole_id = f"DDH-{i+1:03d}"
        
        # Profondeur du trou (entre min_depth et max_depth)
        depth = np.random.uniform(min_depth, max_depth)
        
        # Coordonnées de début (collar)
        collar = (x, y, 0)  # Z = 0 pour le collet
        
        # Vecteur de direction (avec légères variations aléatoires pour plus de réalisme)
        var_azimuth = azimuth + np.random.normal(0, 2)  # Variation de ±2 degrés
        var_dip = dip + np.random.normal(0, 1)  # Variation de ±1 degré
        
        var_azimuth_rad = np.radians(var_azimuth)
        var_dip_rad = np.radians(var_dip)
        
        direction = (
            np.sin(var_azimuth_rad) * np.cos(var_dip_rad),
            np.cos(var_azimuth_rad) * np.cos(var_dip_rad),
            -np.sin(var_dip_rad)
        )
        
        drillholes[hole_id] = {
            'collar': collar,
            'direction': direction,
            'depth': depth
        }
    
    return drillholes

# Fonction pour générer des données de composites avec forages orientés
def generate_composite_data(drillholes, composite_size, mean_values, std_values, selected_metals):
    composite_data = []
    composite_id = 1
    
    for hole_id, hole_info in drillholes.items():
        # Extraire les informations du trou
        x_collar, y_collar, z_collar = hole_info['collar']
        dx, dy, dz = hole_info['direction']
        hole_depth = hole_info['depth']
        
        # Nombre de composites dans ce trou
        num_composites_in_hole = int(hole_depth / composite_size)
        
        # Générer les composites pour ce trou
        for i in range(num_composites_in_hole):
            # Profondeur de début et de fin du composite
            from_depth = i * composite_size
            to_depth = (i + 1) * composite_size
            
            # Coordonnées du milieu du composite
            mid_depth = (from_depth + to_depth) / 2
            x = x_collar + dx * mid_depth
            y = y_collar + dy * mid_depth
            z = z_collar + dz * mid_depth
            
            # Créer une ligne pour le composite
            composite = {
                'Composite_ID': composite_id,
                'Trou_ID': hole_id,
                'De': from_depth,
                'A': to_depth,
                'X': x,
                'Y': y,
                'Z': z,
                'Longueur': composite_size
            }
            
            # Ajouter les teneurs pour chaque métal sélectionné
            for i, metal in enumerate(selected_metals):
                # Distance au centre (pour simuler une minéralisation plus riche au centre)
                dist_to_center = np.sqrt((x - 500)**2 + (y - 500)**2)
                
                # Facteur d'enrichissement basé sur la distance au centre (décroît avec la distance)
                enrichment_factor = max(0.5, 2 - dist_to_center / 300)
                
                # Générer des données selon une distribution log-normale
                log_mean = np.log(mean_values[i] * enrichment_factor) - 0.5 * np.log(1 + (std_values[i]/mean_values[i])**2)
                log_std = np.sqrt(np.log(1 + (std_values[i]/mean_values[i])**2))
                
                composite[f'Teneur_{metal}'] = np.random.lognormal(log_mean, log_std)
            
            composite_data.append(composite)
            composite_id += 1
    
    return pd.DataFrame(composite_data)

# Fonction pour générer un modèle de bloc avec enrichissement au centre
def generate_block_model(nx, ny, nz, block_size_x, block_size_y, block_size_z, 
                         origin_x, origin_y, origin_z, means, stds, selected_metals):
    num_blocks = nx * ny * nz
    
    # Créer les indices de grille
    x_indices = np.repeat(np.arange(nx), ny * nz)
    y_indices = np.tile(np.repeat(np.arange(ny), nz), nx)
    z_indices = np.tile(np.arange(nz), nx * ny)
    
    # Calculer les coordonnées centrales
    x = origin_x + x_indices * block_size_x + block_size_x / 2
    y = origin_y + y_indices * block_size_y + block_size_y / 2
    z = origin_z + z_indices * block_size_z + block_size_z / 2
    
    # Créer le DataFrame du modèle de bloc
    data = {
        'Block_ID': np.arange(1, num_blocks + 1),
        'X': x,
        'Y': y,
        'Z': z,
        'X_size': np.ones(num_blocks) * block_size_x,
        'Y_size': np.ones(num_blocks) * block_size_y,
        'Z_size': np.ones(num_blocks) * block_size_z
    }
    
    # Calculer le centre du modèle
    center_x = origin_x + (nx * block_size_x) / 2
    center_y = origin_y + (ny * block_size_y) / 2
    center_z = origin_z + (nz * block_size_z) / 2
    
    # Générer des teneurs avec enrichissement au centre (forme de filon)
    for i, metal in enumerate(selected_metals):
        # Créer un champ de base avec une certaine corrélation spatiale
        base_field = np.zeros((nx, ny, nz))
        for dx in range(-2, 3):
            for dy in range(-2, 3):
                for dz in range(-2, 3):
                    weight = 1.0 / (1.0 + np.sqrt(dx**2 + dy**2 + dz**2))
                    random_field = np.random.normal(0, 1, (nx, ny, nz))
                    # Ajouter avec décalage et gestion des bords
                    x_start, x_end = max(0, dx), min(nx, nx + dx)
                    y_start, y_end = max(0, dy), min(ny, ny + dy)
                    z_start, z_end = max(0, dz), min(nz, nz + dz)
                    
                    x_target_start, x_target_end = max(0, -dx), min(nx, nx - dx)
                    y_target_start, y_target_end = max(0, -dy), min(ny, ny - dy)
                    z_target_start, z_target_end = max(0, -dz), min(nz, nz - dz)
                    
                    base_field[x_target_start:x_target_end, y_target_start:y_target_end, z_target_start:z_target_end] += (
                        weight * random_field[x_start:x_end, y_start:y_end, z_start:z_end]
                    )
        
        # Normaliser le champ de base
        base_field = (base_field - np.mean(base_field)) / np.std(base_field)
        
        # Créer un facteur d'enrichissement basé sur la distance au centre (forme de filon)
        enrichment_field = np.zeros((nx, ny, nz))
        for ix in range(nx):
            for iy in range(ny):
                for iz in range(nz):
                    # Coordonnées du bloc
                    block_x = origin_x + ix * block_size_x + block_size_x / 2
                    block_y = origin_y + iy * block_size_y + block_size_y / 2
                    block_z = origin_z + iz * block_size_z + block_size_z / 2
                    
                    # Distance au centre (composante radiale)
                    dist_xy = np.sqrt((block_x - center_x)**2 + (block_y - center_y)**2)
                    dist_z = abs(block_z - center_z)
                    
                    # Facteur d'enrichissement qui diminue avec la distance (forme de filon vertical)
                    # Plus élevé au centre, plus faible aux bords
                    r_max = min(nx * block_size_x, ny * block_size_y) / 2  # Rayon maximum
                    h_max = nz * block_size_z / 2  # Demi-hauteur
                    
                    # Terme d'enrichissement radial (diminue du centre vers l'extérieur)
                    radial_term = np.exp(-3 * (dist_xy / r_max)**2)
                    
                    # Terme d'enrichissement vertical (forme de filon)
                    vertical_term = np.exp(-2 * (dist_z / h_max)**2)
                    
                    # Enrichissement combiné
                    enrichment = 3 * radial_term * vertical_term + 0.1
                    
                    enrichment_field[ix, iy, iz] = enrichment
        
        # Combiner les champs (champ de base pour la variabilité + enrichissement pour la structure)
        combined_field = 0.3 * base_field + 0.7 * enrichment_field
        combined_field_flattened = combined_field.flatten()
        
        # Normaliser pour obtenir la distribution souhaitée
        final_values = combined_field_flattened * stds[i] + means[i]
        
        # S'assurer que toutes les valeurs sont positives
        final_values = np.maximum(0.001, final_values)
        
        data[f'Teneur_{metal}'] = final_values
    
    return pd.DataFrame(data)

# Fonction modifiée pour générer des données QAQC avec structure de duplicatas améliorée
def generate_qaqc_data(num_samples, crm_percent, duplicate_percent, blank_percent, crm_values, crm_std_values, selected_metals):
    # Calculer le nombre d'échantillons de chaque type
    num_crm = int(num_samples * crm_percent / 100)
    num_duplicates = int(num_samples * duplicate_percent / 100)
    num_blanks = int(num_samples * blank_percent / 100)
    num_regular = num_samples - num_crm - num_duplicates - num_blanks
    
    # Vérifier que nous avons au moins un échantillon régulier pour créer des duplicatas
    if num_regular <= 0:
        st.error("Les pourcentages sont trop élevés. Impossible de générer des échantillons réguliers.")
        return None, None, None, None
    
    # Générer des dates pour simulation d'une campagne de forage - CORRECTION
    start_date = datetime.datetime.now() - datetime.timedelta(days=180)
    # Utiliser tolist() pour convertir l'array numpy en liste Python
    random_days = np.sort(np.random.randint(0, 180, num_samples)).tolist()
    dates = [start_date + datetime.timedelta(days=day) for day in random_days]
    date_strings = [d.strftime("%Y-%m-%d") for d in dates]
    
    # Générer des échantillons réguliers
    regular_data = {
        'Sample_ID': [f"REG-{i+1:06d}" for i in range(num_regular)],
        'Batch_ID': np.random.randint(1000, 2000, num_regular),
        'Date': date_strings[:num_regular],
        'Type': ['Regular'] * num_regular,
        'Trou_ID': [f"DDH-{i+1:03d}" for i in np.random.randint(1, 51, num_regular)],
        'De': np.random.uniform(0, 500, num_regular),
    }
    
    # Ajouter la colonne "À" (profondeur de fin) basée sur "De" + une longueur aléatoire
    sample_lengths = np.random.uniform(0.5, 2.5, num_regular)
    regular_data['A'] = regular_data['De'] + sample_lengths
    
    # Générer des teneurs pour les échantillons réguliers
    for i, metal in enumerate(selected_metals):
        # Utiliser une distribution log-normale pour simuler des teneurs réalistes
        log_mean = np.log(crm_values[i]) - 0.5 * np.log(1 + (crm_std_values[i]/crm_values[i])**2)
        log_std = np.sqrt(np.log(1 + (crm_std_values[i]/crm_values[i])**2))
        
        # Simuler une variabilité spatiale en utilisant une tendance en fonction de la profondeur
        base_values = np.exp(log_mean + log_std * np.random.normal(0, 1, num_regular))
        depth_effect = 1 + 0.3 * np.sin(regular_data['De'] / 50)  # Une légère tendance cyclique avec la profondeur
        regular_data[f'Teneur_{metal}'] = base_values * depth_effect
    
    # Créer le DataFrame pour les échantillons réguliers
    regular_df = pd.DataFrame(regular_data)
    
    # Générer des CRM (matériaux de référence certifiés)
    crm_df = pd.DataFrame()
    if num_crm > 0:
        crm_types = [f"CRM-{chr(65+i)}" for i in range(min(3, len(selected_metals)))]  # Jusqu'à 3 types de CRM
        crm_data = {
            'Sample_ID': [f"CRM-{i+1:06d}" for i in range(num_crm)],
            'Batch_ID': np.random.randint(1000, 2000, num_crm),
            'Date': date_strings[num_regular:num_regular+num_crm],
            'Type': ['CRM'] * num_crm,
            'CRM_Type': np.random.choice(crm_types, num_crm),
            'Trou_ID': [f"QC" for _ in range(num_crm)],
            'De': [0] * num_crm,
            'A': [0] * num_crm
        }
        
        # Ajouter les teneurs certifiées pour chaque CRM avec une légère variation
        for i, metal in enumerate(selected_metals):
            crm_values_per_type = {
                crm_type: crm_values[i] * (0.8 + j*0.2)  # Différentes valeurs pour différents types de CRM
                for j, crm_type in enumerate(crm_types)
            }
            
            crm_data[f'Teneur_{metal}'] = [
                np.random.normal(
                    crm_values_per_type[crm_type], 
                    crm_std_values[i] * 0.5  # Précision plus élevée pour les CRM
                ) 
                for crm_type in crm_data['CRM_Type']
            ]
        
        crm_df = pd.DataFrame(crm_data)
    
    # Générer des duplicatas avec structure améliorée
    duplicates_df = pd.DataFrame()
    if num_duplicates > 0 and num_regular > 0:
        # Sélectionner aléatoirement des échantillons à dupliquer
        original_indices = np.random.choice(range(num_regular), min(num_duplicates, num_regular), replace=False)
        original_samples = regular_df.iloc[original_indices].copy()
        
        # Structure modifiée: un échantillon et son duplicata sur la même ligne
        duplicates_data = {
            'Pair_ID': [f"PAIR-{i+1:04d}" for i in range(len(original_indices))],
            'Original_Sample_ID': original_samples['Sample_ID'].values,
            'Duplicate_Sample_ID': [f"DUP-{i+1:06d}" for i in range(len(original_indices))],
            'Batch_ID': np.random.randint(1000, 2000, len(original_indices)),
            'Date': date_strings[num_regular+num_crm:num_regular+num_crm+len(original_indices)],
            'Trou_ID': original_samples['Trou_ID'].values,
            'De': original_samples['De'].values,
            'A': original_samples['A'].values
        }
        
        # Ajouter les teneurs originales et dupliquées côte à côte
        for i, metal in enumerate(selected_metals):
            original_values = original_samples[f'Teneur_{metal}'].values
            # Générer des duplicatas avec une légère variation
            duplicate_values = original_values * np.random.normal(1, 0.05, len(original_indices))
            
            duplicates_data[f'Teneur_Original_{metal}'] = original_values
            duplicates_data[f'Teneur_Duplicata_{metal}'] = duplicate_values
            # Ajouter aussi l'écart relatif (utile pour le QAQC)
            duplicates_data[f'Ecart_Relatif_{metal}'] = 100 * np.abs(duplicate_values - original_values) / original_values
        
        duplicates_df = pd.DataFrame(duplicates_data)
    
    # Générer des blancs
    blanks_df = pd.DataFrame()
    if num_blanks > 0:
        blank_data = {
            'Sample_ID': [f"BLK-{i+1:06d}" for i in range(num_blanks)],
            'Batch_ID': np.random.randint(1000, 2000, num_blanks),
            'Date': date_strings[num_regular+num_crm+num_duplicates:],
            'Type': ['Blank'] * num_blanks,
            'Trou_ID': [f"QC" for _ in range(num_blanks)],
            'De': [0] * num_blanks,
            'A': [0] * num_blanks
        }
        
        # Ajouter des teneurs très basses pour les blancs (avec occasionnellement une contamination)
        for metal in selected_metals:
            # La plupart des blancs auront des valeurs presque nulles, mais quelques-uns auront une légère contamination
            contamination = np.random.choice([0, 1], num_blanks, p=[0.95, 0.05])
            blank_data[f'Teneur_{metal}'] = np.random.lognormal(-4, 0.5, num_blanks) * (1 + contamination * 5)
        
        blanks_df = pd.DataFrame(blank_data)
    
    # Pour le DataFrame complet, nous devons transformer les duplicatas pour qu'ils correspondent au format des autres
    all_duplicates_data = []
    if not duplicates_df.empty:
        for _, row in duplicates_df.iterrows():
            # Ajouter l'échantillon original (déjà dans regular_df)
            # Ajouter le duplicat comme nouvel enregistrement
            duplicate = {
                'Sample_ID': row['Duplicate_Sample_ID'],
                'Batch_ID': row['Batch_ID'],
                'Date': row['Date'],
                'Type': 'Duplicate',
                'Original_Sample': row['Original_Sample_ID'],
                'Trou_ID': row['Trou_ID'],
                'De': row['De'],
                'A': row['A']
            }
            
            # Ajouter les teneurs du duplicata
            for metal in selected_metals:
                duplicate[f'Teneur_{metal}'] = row[f'Teneur_Duplicata_{metal}']
            
            all_duplicates_data.append(duplicate)
    
    # Créer le DataFrame des duplicatas pour le jeu de données complet
    full_duplicates_df = pd.DataFrame(all_duplicates_data) if all_duplicates_data else pd.DataFrame()
    
    # Combiner tous les types d'échantillons pour le jeu de données complet
    dfs_to_combine = [df for df in [regular_df, crm_df, full_duplicates_df, blanks_df] if not df.empty]
    qaqc_data = pd.concat(dfs_to_combine, ignore_index=True)
    
    return qaqc_data, crm_df, duplicates_df, blanks_df

# Fonction modifiée pour visualiser un modèle de bloc 2D avec des rectangles pleins
def visualize_block_model_xy(block_model, selected_metal, z_level, block_size_z, origin_z, block_size_x, block_size_y):
    # Filtrer les blocs à cette hauteur Z
    df_section = block_model[np.isclose(block_model['Z'], z_level, atol=block_size_z/2)]
    
    if df_section.empty:
        return "Aucun bloc trouvé à cette hauteur"
    
    # Créer la figure
    fig, ax = plt.subplots(figsize=(10, 8))
    
    # Définir la colormap et les normalisations
    cmap = plt.cm.viridis
    
    # Déterminer les valeurs min et max pour la barre de couleur
    vmin = df_section[f'Teneur_{selected_metal}'].min()
    vmax = df_section[f'Teneur_{selected_metal}'].max()
    
    # Utiliser LogNorm sans notation scientifique
    norm = LogNorm(vmin=vmin, vmax=vmax)
    
    # Dessiner chaque bloc comme un rectangle
    for _, block in df_section.iterrows():
        # Coordonnées du coin inférieur gauche du bloc
        x_min = block['X'] - block_size_x/2
        y_min = block['Y'] - block_size_y/2
        
        # Teneur
        grade = block[f'Teneur_{selected_metal}']
        
        # Ajouter un rectangle pour le bloc
        rect = patches.Rectangle(
            (x_min, y_min), block_size_x, block_size_y,
            facecolor=cmap(norm(grade)),
            edgecolor='black',
            linewidth=0.5,
            alpha=0.8
        )
        ax.add_patch(rect)
    
    # Configurer les limites du graphique
    x_min = df_section['X'].min() - block_size_x
    x_max = df_section['X'].max() + block_size_x
    y_min = df_section['Y'].min() - block_size_y
    y_max = df_section['Y'].max() + block_size_y
    
    ax.set_xlim(x_min, x_max)
    ax.set_ylim(y_min, y_max)
    
    # Ajouter la barre de couleur avec format spécifique (pas de notation scientifique)
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])
    cbar = plt.colorbar(sm, ax=ax)
    
    # Formater les étiquettes sans notation scientifique
    from matplotlib.ticker import ScalarFormatter
    formatter = ScalarFormatter(useOffset=False)
    formatter.set_scientific(False)
    cbar.ax.yaxis.set_major_formatter(formatter)
    
    cbar.set_label(f'Teneur {selected_metal}')
    
    # Étiquettes des axes
    ax.set_xlabel('X (m)')
    ax.set_ylabel('Y (m)')
    ax.set_title(f'Section XY à Z = {z_level:.1f} m - Teneur en {selected_metal}')
    
    # Garder les ratios d'aspect égaux
    ax.set_aspect('equal')
    
    # Convertir la figure en image
    buf = io.BytesIO()
    plt.tight_layout()
    fig.savefig(buf, format='png', dpi=150)
    buf.seek(0)
    plt.close(fig)
    
    return buf

# Fonction pour visualiser une section de forage 3D
def visualize_drillholes_3d(drillholes):
    from mpl_toolkits.mplot3d import Axes3D
    
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')
    
    for hole_id, hole_info in drillholes.items():
        # Extraire les informations du trou
        x_collar, y_collar, z_collar = hole_info['collar']
        dx, dy, dz = hole_info['direction']
        hole_depth = hole_info['depth']
        
        # Calculer le point final du trou
        x_end = x_collar + dx * hole_depth
        y_end = y_collar + dy * hole_depth
        z_end = z_collar + dz * hole_depth
        
        # Tracer le trou
        ax.plot([x_collar, x_end], [y_collar, y_end], [z_collar, z_end], 'r-', linewidth=1)
        ax.scatter(x_collar, y_collar, z_collar, color='blue', s=30, label='_nolegend_')
    
    # Étiquettes des axes
    ax.set_xlabel('X (m)')
    ax.set_ylabel('Y (m)')
    ax.set_zlabel('Z (m)')
    ax.set_title('Visualisation 3D des forages')
    
    # Limites d'axes
    max_range = max([
        np.max([hole_info['collar'][0] + hole_info['direction'][0] * hole_info['depth'] for hole_id, hole_info in drillholes.items()]),
        np.max([hole_info['collar'][1] + hole_info['direction'][1] * hole_info['depth'] for hole_id, hole_info in drillholes.items()]),
        0  # Z max (surface)
    ])
    
    min_z = min([hole_info['collar'][2] + hole_info['direction'][2] * hole_info['depth'] for hole_id, hole_info in drillholes.items()])
    
    ax.set_xlim(0, max_range)
    ax.set_ylim(0, max_range)
    ax.set_zlim(min_z, 0)  # Z négatif car on fore vers le bas
    
    # Égaliser les échelles
    ax.set_box_aspect([1, 1, 0.7])  # ratio légèrement différent pour Z pour mieux voir les forages
    
    # Convertir la figure en image
    buf = io.BytesIO()
    plt.tight_layout()
    fig.savefig(buf, format='png')
    buf.seek(0)
    plt.close(fig)
    
    return buf

# Fonction pour télécharger le DataFrame en CSV
def get_csv_download_link(df, filename="data.csv"):
    csv = df.to_csv(index=False)
    b64 = base64.b64encode(csv.encode()).decode()
    href = f'<a href="data:file/csv;base64,{b64}" download="{filename}" class="download-link">Télécharger {filename}</a>'
    return href

# Ajouter un peu de CSS pour styliser les liens de téléchargement
st.markdown("""
<style>
.download-link {
    display: inline-block;
    padding: 8px 16px;
    background-color: #4CAF50;
    color: white !important;
    text-decoration: none;
    font-weight: bold;
    border-radius: 4px;
    margin: 5px 0;
    transition: background-color 0.3s;
}
.download-link:hover {
    background-color: #45a049;
}
</style>
""", unsafe_allow_html=True)

# Interface utilisateur en fonction du type de données choisi
if data_type == "Composites":
    st.header("Génération de données de composites")
    
    # Paramètres de forage
    st.subheader("Paramètres des forages")
    
    col1, col2 = st.columns(2)
    
    with col1:
        num_drillholes = st.slider("Nombre de forages", 5, 100, 20)
        grid_spacing = st.slider("Espacement de la grille (m)", 10.0, 200.0, 50.0)
        min_depth = st.slider("Profondeur minimale (m)", 50.0, 300.0, 100.0)
    
    with col2:
        azimuth = st.slider("Azimut des forages (°)", 0, 360, 270)
        dip = st.slider("Inclinaison des forages (°)", -90, 0, -60)
        max_depth = st.slider("Profondeur maximale (m)", 100.0, 1000.0, 500.0)
    
    composite_size = st.slider("Taille des composites (m)", 0.5, 10.0, 2.0)
    
    # Paramètres pour chaque métal sélectionné
    st.subheader("Paramètres des teneurs")
    mean_values = []
    std_values = []
    
    metal_units = {
        "Or": "g/t",
        "Cuivre": "%",
        "Zinc": "%",
        "Manganèse": "%",
        "Fer": "%"
    }
    
    default_values = {
        "Or": (1.5, 0.8),
        "Cuivre": (0.5, 0.2),
        "Zinc": (2.0, 1.0),
        "Manganèse": (1.2, 0.5),
        "Fer": (30.0, 10.0)
    }
    
    # Création d'une mise en page en colonnes pour les paramètres des métaux
    cols = st.columns(len(selected_metals) if selected_metals else 1)
    
    for i, metal in enumerate(selected_metals):
        with cols[i]:
            st.write(f"**{metal}** ({metal_units[metal]})")
            mean = st.number_input(f"Teneur moyenne - {metal}", 
                                 min_value=0.001, 
                                 max_value=100.0, 
                                 value=float(default_values[metal][0]),
                                 step=0.1,
                                 key=f"mean_{metal}")
            std = st.number_input(f"Écart-type - {metal}", 
                                min_value=0.001, 
                                max_value=50.0, 
                                value=float(default_values[metal][1]),
                                step=0.1,
                                key=f"std_{metal}")
            mean_values.append(mean)
            std_values.append(std)
    
    if st.button("Générer les données de composites"):
        if selected_metals:
            # Générer les forages
            st.write("Génération des forages...")
            drillholes = generate_drillholes(num_drillholes, grid_spacing, azimuth, dip, min_depth, max_depth)
            
            # Afficher la visualisation 3D des forages
            st.subheader("Visualisation 3D des forages")
            drillhole_viz = visualize_drillholes_3d(drillholes)
            st.image(drillhole_viz, caption="Visualisation 3D des forages")
            
            # Générer les données de composites
            st.write("Génération des composites...")
            composite_data = generate_composite_data(drillholes, composite_size, mean_values, std_values, selected_metals)
            
            # Afficher les statistiques des données
            st.subheader("Statistiques des données générées")
            st.dataframe(composite_data.describe())
            
            # Afficher un échantillon des données
            st.subheader("Aperçu des données")
            st.dataframe(composite_data.head(10))
            
            # Statistiques par forage
            st.subheader("Statistiques par forage")
            trou_stats = composite_data.groupby('Trou_ID')[
                [f'Teneur_{metal}' for metal in selected_metals]
            ].agg(['mean', 'min', 'max', 'count'])
            st.dataframe(trou_stats)
            
            # Lien de téléchargement
            st.markdown(get_csv_download_link(composite_data, "composites_data.csv"), unsafe_allow_html=True)
        else:
            st.error("Veuillez sélectionner au moins un métal.")

elif data_type == "Modèle de bloc":
    st.header("Génération de modèle de bloc")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Dimensions du modèle")
        nx = st.slider("Nombre de blocs en X", 5, 100, 20)
        ny = st.slider("Nombre de blocs en Y", 5, 100, 20)
        nz = st.slider("Nombre de blocs en Z", 5, 50, 10)
    
    with col2:
        st.subheader("Taille des blocs (m)")
        block_size_x = st.slider("Taille du bloc en X", 1.0, 50.0, 10.0)
        block_size_y = st.slider("Taille du bloc en Y", 1.0, 50.0, 10.0)
        block_size_z = st.slider("Taille du bloc en Z", 1.0, 25.0, 5.0)
    
    # Origine du modèle de bloc
    st.subheader("Origine du modèle de bloc")
    origin_cols = st.columns(3)
    with origin_cols[0]:
        origin_x = st.number_input("Origine X", value=0.0, step=100.0)
    with origin_cols[1]:
        origin_y = st.number_input("Origine Y", value=0.0, step=100.0)
    with origin_cols[2]:
        origin_z = st.number_input("Origine Z", value=-500.0, step=100.0)
    
    # Calculer et afficher le nombre total de blocs
    total_blocks = nx * ny * nz
    st.info(f"Nombre total de blocs: {total_blocks:,}")
    
    # Paramètres pour chaque métal sélectionné
    st.subheader("Paramètres des teneurs")
    mean_values = []
    std_values = []
    
    metal_units = {
        "Or": "g/t",
        "Cuivre": "%",
        "Zinc": "%",
        "Manganèse": "%",
        "Fer": "%"
    }
    
    default_values = {
        "Or": (1.0, 0.5),
        "Cuivre": (0.4, 0.15),
        "Zinc": (1.5, 0.7),
        "Manganèse": (1.0, 0.4),
        "Fer": (25.0, 8.0)
    }
    
    # Création d'une mise en page en colonnes pour les paramètres des métaux
    cols = st.columns(len(selected_metals) if selected_metals else 1)
    
    for i, metal in enumerate(selected_metals):
        with cols[i]:
            st.write(f"**{metal}** ({metal_units[metal]})")
            mean = st.number_input(f"Teneur moyenne - {metal}", 
                                 min_value=0.001, 
                                 max_value=100.0, 
                                 value=float(default_values[metal][0]),
                                 step=0.1,
                                 key=f"bm_mean_{metal}")
            std = st.number_input(f"Écart-type - {metal}", 
                                min_value=0.001, 
                                max_value=50.0, 
                                value=float(default_values[metal][1]),
                                step=0.1,
                                key=f"bm_std_{metal}")
            mean_values.append(mean)
            std_values.append(std)
    
    if st.button("Générer le modèle de bloc"):
        if selected_metals:
            if total_blocks > 500000:
                if not st.warning("Le modèle va générer un grand nombre de blocs, ce qui peut prendre du temps. Continuer?"):
                    st.stop()
            
            # Afficher un indicateur de progression
            progress_bar = st.progress(0)
            st.write("Génération du modèle de bloc en cours...")
            
            # Générer les données
            block_model = generate_block_model(nx, ny, nz, block_size_x, block_size_y, block_size_z, 
                                             origin_x, origin_y, origin_z, mean_values, std_values, selected_metals)
            
            progress_bar.progress(100)
            
            # Afficher les statistiques des données
            st.subheader("Statistiques des données générées")
            st.dataframe(block_model.describe())
            
            # Visualisation des sections
            st.subheader("Visualisation des sections")
            
            # Sélectionner un métal pour la visualisation
            viz_metal = st.selectbox("Métal à visualiser", selected_metals)
            
            # Sélectionner une section Z
            z_levels = sorted(block_model['Z'].unique())
            selected_z = st.slider(
                "Sélectionner une section Z", 
                min_value=float(min(z_levels)), 
                max_value=float(max(z_levels)), 
                value=float(z_levels[len(z_levels)//2]),
                step=float(block_size_z)
            )
            
            # Afficher la section avec des rectangles pleins
            section_image = visualize_block_model_xy(
                block_model, viz_metal, selected_z, block_size_z, origin_z, block_size_x, block_size_y
            )
            if isinstance(section_image, str):
                st.error(section_image)
            else:
                st.image(section_image, caption=f"Section XY à Z = {selected_z} m")
            
            # Afficher un échantillon des données
            st.subheader("Aperçu des données")
            st.dataframe(block_model.head(10))
            
            # Lien de téléchargement
            st.markdown(get_csv_download_link(block_model, "block_model_data.csv"), unsafe_allow_html=True)
        else:
            st.error("Veuillez sélectionner au moins un métal.")

elif data_type == "Données QAQC":
    st.header("Génération de données QAQC")
    
    num_samples = st.slider("Nombre total d'échantillons", 50, 5000, 500)
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        crm_percent = st.slider("Pourcentage de CRM (%)", 0, 30, 5)
    
    with col2:
        duplicate_percent = st.slider("Pourcentage de duplicatas (%)", 0, 30, 5)
    
    with col3:
        blank_percent = st.slider("Pourcentage de blancs (%)", 0, 30, 3)
    
    # Vérifier que le total ne dépasse pas 100%
    total_percent = crm_percent + duplicate_percent + blank_percent
    if total_percent > 50:
        st.warning(f"Le total des pourcentages de QAQC est de {total_percent}%. Il est recommandé de ne pas dépasser 50%.")
    
    # Paramètres pour les CRM (pour chaque métal)
    st.subheader("Paramètres des CRM")
    crm_values = []
    crm_std_values = []
    
    metal_units = {
        "Or": "g/t",
        "Cuivre": "%",
        "Zinc": "%",
        "Manganèse": "%",
        "Fer": "%"
    }
    
    default_values = {
        "Or": (2.5, 0.1),
        "Cuivre": (0.7, 0.03),
        "Zinc": (3.0, 0.15),
        "Manganèse": (1.5, 0.07),
        "Fer": (40.0, 2.0)
    }
    
    # Création d'une mise en page en colonnes pour les paramètres des métaux
    cols = st.columns(len(selected_metals) if selected_metals else 1)
    
    for i, metal in enumerate(selected_metals):
        with cols[i]:
            st.write(f"**{metal}** ({metal_units[metal]})")
            value = st.number_input(f"Valeur certifiée - {metal}", 
                                   min_value=0.001, 
                                   max_value=100.0, 
                                   value=float(default_values[metal][0]),
                                   step=0.1,
                                   key=f"crm_val_{metal}")
            std = st.number_input(f"Écart-type toléré - {metal}", 
                                min_value=0.001, 
                                max_value=10.0, 
                                value=float(default_values[metal][1]),
                                step=0.01,
                                key=f"crm_std_{metal}")
            crm_values.append(value)
            crm_std_values.append(std)
    
    if st.button("Générer les données QAQC"):
        if selected_metals:
            # Générer les données
            with st.spinner("Génération des données QAQC en cours..."):
                qaqc_data, crm_df, duplicates_df, blanks_df = generate_qaqc_data(
                    num_samples, crm_percent, duplicate_percent, blank_percent, 
                    crm_values, crm_std_values, selected_metals
                )
            
            if qaqc_data is None:
                st.error("La génération des données a échoué. Veuillez ajuster les paramètres.")
            else:
                # Afficher les statistiques des données complètes
                st.subheader("Statistiques des données générées")
                st.dataframe(qaqc_data.describe())
                
                # Afficher la distribution des types d'échantillons
                st.subheader("Distribution des types d'échantillons")
                type_counts = qaqc_data['Type'].value_counts()
                st.write(type_counts)
                
                # Afficher un échantillon des données par type
                st.subheader("Aperçu des échantillons réguliers")
                st.dataframe(qaqc_data[qaqc_data['Type'] == 'Regular'].head(5))
                
                # CRM
                if not crm_df.empty:
                    st.subheader("Aperçu des CRM")
                    st.dataframe(crm_df.head(5))
                    st.markdown(get_csv_download_link(crm_df, "crm_data.csv"), unsafe_allow_html=True)
                
                # Duplicatas (avec nouveau format)
                if not duplicates_df.empty:
                    st.subheader("Aperçu des duplicatas (format côte à côte)")
                    st.dataframe(duplicates_df.head(5))
                    
                    # Afficher les statistiques d'écart pour les duplicatas
                    st.subheader("Statistiques des écarts relatifs entre duplicatas (%)")
                    ecart_columns = [col for col in duplicates_df.columns if col.startswith('Ecart_Relatif')]
                    st.dataframe(duplicates_df[ecart_columns].describe())
                    
                    # Téléchargement du fichier de duplicatas
                    st.markdown(get_csv_download_link(duplicates_df, "duplicates_data.csv"), unsafe_allow_html=True)
                
                # Blancs
                if not blanks_df.empty:
                    st.subheader("Aperçu des blancs")
                    st.dataframe(blanks_df.head(5))
                    st.markdown(get_csv_download_link(blanks_df, "blanks_data.csv"), unsafe_allow_html=True)
                
                # Lien de téléchargement pour l'ensemble complet des données
                st.subheader("Téléchargement des données")
                st.markdown("Télécharger l'ensemble complet des données:")
                st.markdown(get_csv_download_link(qaqc_data, "qaqc_data_complete.csv"), unsafe_allow_html=True)
                
                st.success("Données QAQC générées avec succès!")
        else:
            st.error("Veuillez sélectionner au moins un métal.")

# Footer
st.markdown("---")
st.markdown("<div style='text-align: center;'>Générateur de données minières synthétiques | Développé par Didier Ouedraogo, P.Geo.</div>", unsafe_allow_html=True)