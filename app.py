# app.py
import streamlit as st
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.datasets import make_moons
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
import time
import pandas as pd

# --- CONFIG STREAMLIT ---
st.set_page_config(page_title="Optimiseurs - Visualisation temps réel", layout="wide")
st.title("🧠 Visualisation temps réel : Optimiseurs avec métriques avancées")

# --- PARAMÈTRES UTILISATEUR ---
st.sidebar.header("⚙️ Configuration")

# Mode selection
analysis_mode = st.sidebar.radio(
    "Mode d'analyse",
    ["Entraînement Simple", "Comparaison Optimiseurs", "Sensibilité Hyperparamètres", "Analyse Robustesse"],
    index=0
)

compare_mode = (analysis_mode == "Comparaison Optimiseurs")

if analysis_mode == "Entraînement Simple":
    opt_name = st.sidebar.selectbox("Optimiseur", ["SGD", "RMSprop", "Adam"])
    learning_rate = st.sidebar.slider("Learning rate", 0.0001, 0.1, 0.01, step=0.001)
    epochs = st.sidebar.slider("Époques", 10, 200, 50)
    batch_size = st.sidebar.slider("Batch size", 8, 256, 64)
elif analysis_mode == "Comparaison Optimiseurs":
    opt_name = "Comparison"
    learning_rate = st.sidebar.slider("Learning rate", 0.0001, 0.1, 0.01, step=0.001)
    epochs = st.sidebar.slider("Époques", 10, 200, 50)
    batch_size = st.sidebar.slider("Batch size", 8, 256, 64)
elif analysis_mode == "Sensibilité Hyperparamètres":
    opt_name = "All"  # Compare all optimizers
    st.sidebar.markdown("**Grid Search Parameters:**")
    lr_min = st.sidebar.slider("LR min", 0.0001, 0.01, 0.001, step=0.0001, format="%.4f")
    lr_max = st.sidebar.slider("LR max", 0.01, 0.1, 0.05, step=0.001, format="%.4f")
    lr_steps = st.sidebar.slider("Nombre de LR à tester", 3, 8, 5)
    
    bs_min = st.sidebar.selectbox("Batch size min", [8, 16, 32], index=1)
    bs_max = st.sidebar.selectbox("Batch size max", [64, 128, 256], index=1)
    bs_steps = st.sidebar.slider("Nombre de BS à tester", 3, 5, 3)
    
    epochs = st.sidebar.slider("Époques par config", 20, 100, 30)
    learning_rate = lr_min  # Default for consistency
    batch_size = bs_min
else:  # Analyse Robustesse
    opt_name = "All"  # Compare all optimizers
    learning_rate = st.sidebar.slider("Learning rate", 0.0001, 0.1, 0.01, step=0.001)
    epochs = st.sidebar.slider("Époques", 10, 200, 50)
    batch_size = st.sidebar.slider("Batch size", 8, 256, 64)
    n_runs = st.sidebar.slider("Nombre de runs", 3, 10, 5)

# Learning rate scheduling
use_scheduler = st.sidebar.checkbox("Utiliser LR Scheduler", value=False)
if use_scheduler:
    scheduler_type = st.sidebar.selectbox("Type de scheduler", ["StepLR", "ExponentialLR", "CosineAnnealingLR"])
    if scheduler_type == "StepLR":
        step_size = st.sidebar.slider("Step size", 5, 50, 20)
        gamma = st.sidebar.slider("Gamma", 0.1, 0.9, 0.5, step=0.1)
    elif scheduler_type == "ExponentialLR":
        gamma = st.sidebar.slider("Gamma", 0.9, 0.99, 0.95, step=0.01)
    else:  # CosineAnnealing
        T_max = st.sidebar.slider("T_max", 10, 100, 50)

# Interactive controls
st.sidebar.markdown("---")
st.sidebar.header("🎮 Contrôles")
col_start, col_stop = st.sidebar.columns(2)
start_training = col_start.button("▶️ Démarrer", type="primary")
stop_training = col_stop.button("⏸️ Pause")

st.markdown("---")

# --- DONNÉES SYNTHÉTIQUES ---
X, y = make_moons(n_samples=800, noise=0.25, random_state=0)
X = StandardScaler().fit_transform(X)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

X_train_tensor = torch.tensor(X_train, dtype=torch.float32)
y_train_tensor = torch.tensor(y_train, dtype=torch.long)
X_test_tensor = torch.tensor(X_test, dtype=torch.float32)
y_test_tensor = torch.tensor(y_test, dtype=torch.long)

# --- FONCTION POUR CRÉER MODÈLE ET OPTIMISEUR ---
def create_model():
    return nn.Sequential(
        nn.Linear(2, 16),
        nn.ReLU(),
        nn.Linear(16, 8),
        nn.ReLU(),
        nn.Linear(8, 2)
    )

def create_optimizer(model, opt_type, lr):
    if opt_type == "SGD":
        return optim.SGD(model.parameters(), lr=lr)
    elif opt_type == "RMSprop":
        return optim.RMSprop(model.parameters(), lr=lr)
    else:  # Adam
        return optim.Adam(model.parameters(), lr=lr)

def create_scheduler(optimizer, scheduler_type):
    if scheduler_type == "StepLR":
        return optim.lr_scheduler.StepLR(optimizer, step_size=step_size, gamma=gamma)
    elif scheduler_type == "ExponentialLR":
        return optim.lr_scheduler.ExponentialLR(optimizer, gamma=gamma)
    else:  # CosineAnnealingLR
        return optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=T_max)

def compute_metrics(model, X, y):
    """Compute classification metrics"""
    model.eval()
    with torch.no_grad():
        outputs = model(X)
        _, predicted = torch.max(outputs, 1)
        y_pred = predicted.cpu().numpy()
        y_true = y.cpu().numpy()
        
        acc = accuracy_score(y_true, y_pred)
        prec = precision_score(y_true, y_pred, zero_division=0)
        rec = recall_score(y_true, y_pred, zero_division=0)
        f1 = f1_score(y_true, y_pred, zero_division=0)
        cm = confusion_matrix(y_true, y_pred)
        
    return acc, prec, rec, f1, cm

def compute_gradient_norms(model):
    """Compute gradient norms for each layer"""
    grad_norms = []
    for name, param in model.named_parameters():
        if param.grad is not None:
            grad_norms.append(param.grad.norm().item())
    return grad_norms

def detect_convergence(losses, window=5, threshold=0.005):
    """Detect when model has converged (loss changes < threshold)"""
    if len(losses) < window:
        return False, -1
    
    recent_losses = losses[-window:]
    loss_change = np.std(recent_losses)
    
    if loss_change < threshold:
        # Find the epoch where it first converged
        for i in range(len(losses) - window + 1):
            window_losses = losses[i:i+window]
            if np.std(window_losses) < threshold:
                return True, i + window
    
    return False, -1

def compute_convergence_speed(losses, target_percentile=0.95):
    """Compute epochs needed to reach X% of final performance"""
    if len(losses) == 0:
        return -1
    
    final_loss = losses[-1]
    initial_loss = losses[0]
    target_loss = initial_loss - (initial_loss - final_loss) * target_percentile
    
    for epoch, loss in enumerate(losses):
        if loss <= target_loss:
            return epoch + 1
    
    return len(losses)

def compute_loss_smoothness(losses, window=5):
    """Compute variance in loss over sliding window (stability metric)"""
    if len(losses) < window:
        return []
    
    smoothness = []
    for i in range(len(losses) - window + 1):
        window_losses = losses[i:i+window]
        smoothness.append(np.std(window_losses))
    
    return smoothness

criterion = nn.CrossEntropyLoss()

# --- GRILLE POUR FRONTIÈRE DE DÉCISION ---
x_min, x_max = X[:, 0].min() - 0.5, X[:, 0].max() + 0.5
y_min, y_max = X[:, 1].min() - 0.5, X[:, 1].max() + 0.5
xx, yy = np.meshgrid(np.linspace(x_min, x_max, 200), np.linspace(y_min, y_max, 200))
grid_points = np.c_[xx.ravel(), yy.ravel()]
grid_tensor = torch.tensor(grid_points, dtype=torch.float32)

# --- FONCTION D'ENTRAÎNEMENT ---
def train_model(model, optimizer, scheduler, opt_name, color):
    """Train a single model and return all metrics"""
    train_losses = []
    val_losses = []
    train_accs, test_accs = [], []
    train_precs, test_precs = [], []
    train_recs, test_recs = [], []
    train_f1s, test_f1s = [], []
    learning_rates = []
    gradient_norms_history = []
    
    start_time = time.time()
    
    for epoch in range(epochs):
        model.train()
        permutation = torch.randperm(X_train_tensor.size(0))
        total_loss = 0.0
        
        for i in range(0, X_train_tensor.size(0), batch_size):
            indices = permutation[i:i+batch_size]
            batch_x, batch_y = X_train_tensor[indices], y_train_tensor[indices]
            
            optimizer.zero_grad()
            outputs = model(batch_x)
            loss = criterion(outputs, batch_y)
            loss.backward()
            
            # Capture gradient norms
            grad_norms = compute_gradient_norms(model)
            
            optimizer.step()
            total_loss += loss.item()
        
        # Update learning rate
        current_lr = optimizer.param_groups[0]['lr']
        learning_rates.append(current_lr)
        
        if scheduler:
            scheduler.step()
        
        avg_train_loss = total_loss / (X_train_tensor.size(0) / batch_size)
        train_losses.append(avg_train_loss)
        
        # Compute validation loss
        model.eval()
        with torch.no_grad():
            val_outputs = model(X_test_tensor)
            val_loss = criterion(val_outputs, y_test_tensor)
            val_losses.append(val_loss.item())
        
        gradient_norms_history.append(np.mean(grad_norms) if grad_norms else 0)
        
        # Compute metrics
        train_acc, train_prec, train_rec, train_f1, _ = compute_metrics(model, X_train_tensor, y_train_tensor)
        test_acc, test_prec, test_rec, test_f1, cm = compute_metrics(model, X_test_tensor, y_test_tensor)
        
        train_accs.append(train_acc)
        test_accs.append(test_acc)
        train_precs.append(train_prec)
        test_precs.append(test_prec)
        train_recs.append(train_rec)
        test_recs.append(test_rec)
        train_f1s.append(train_f1)
        test_f1s.append(test_f1)
    
    training_time = time.time() - start_time
    
    # Decision boundary
    model.eval()
    with torch.no_grad():
        Z = model(grid_tensor)
        Z_probs = torch.softmax(Z, dim=1)[:, 1].reshape(xx.shape).numpy()
    
    # Convergence analysis (based on validation loss)
    converged, convergence_epoch = detect_convergence(val_losses)
    convergence_speed_95 = compute_convergence_speed(val_losses, 0.95)
    convergence_speed_99 = compute_convergence_speed(val_losses, 0.99)
    loss_smoothness = compute_loss_smoothness(val_losses)
    
    return {
        'train_losses': train_losses,
        'val_losses': val_losses,
        'train_accs': train_accs,
        'test_accs': test_accs,
        'train_precs': train_precs,
        'test_precs': test_precs,
        'train_recs': train_recs,
        'test_recs': test_recs,
        'train_f1s': train_f1s,
        'test_f1s': test_f1s,
        'learning_rates': learning_rates,
        'gradient_norms': gradient_norms_history,
        'confusion_matrix': cm,
        'decision_boundary': Z_probs,
        'training_time': training_time,
        'final_test_acc': test_accs[-1],
        'final_train_loss': train_losses[-1],
        'final_val_loss': val_losses[-1],
        'color': color,
        'converged': converged,
        'convergence_epoch': convergence_epoch,
        'convergence_speed_95': convergence_speed_95,
        'convergence_speed_99': convergence_speed_99,
        'loss_smoothness': loss_smoothness
    }

# --- ENTRAÎNEMENT ---
if start_training:
    if analysis_mode == "Sensibilité Hyperparamètres":
        st.info(f"🔍 Analyse de sensibilité pour tous les optimiseurs...")
        
        # Generate grid
        learning_rates_grid = np.logspace(np.log10(lr_min), np.log10(lr_max), lr_steps)
        batch_sizes_grid = np.logspace(np.log2(bs_min), np.log2(bs_max), bs_steps, base=2).astype(int)
        
        optimizers_list = ['SGD', 'RMSprop', 'Adam']
        colors_opt = {'SGD': '#FF6B6B', 'RMSprop': '#4ECDC4', 'Adam': '#45B7D1'}
        
        total_runs = len(learning_rates_grid) * len(batch_sizes_grid) * len(optimizers_list)
        progress_bar = st.progress(0)
        status_text = st.empty()
        
        # Results per optimizer
        results_per_opt = {}
        
        run_idx = 0
        
        for opt_type in optimizers_list:
            results_grid = np.zeros((len(batch_sizes_grid), len(learning_rates_grid)))
            convergence_grid = np.zeros((len(batch_sizes_grid), len(learning_rates_grid)))
            time_grid = np.zeros((len(batch_sizes_grid), len(learning_rates_grid)))
            detailed_results = {}
            
            for i, bs in enumerate(batch_sizes_grid):
                for j, lr in enumerate(learning_rates_grid):
                    status_text.text(f"Test {run_idx+1}/{total_runs}: {opt_type} - LR={lr:.4f}, BS={bs}")
                    
                    model = create_model()
                    optimizer = create_optimizer(model, opt_type, lr)
                    scheduler = create_scheduler(optimizer, scheduler_type) if use_scheduler else None
                    
                    # Train with this configuration
                    train_losses_temp = []
                    val_losses_temp = []
                    test_accs_temp = []
                    start_time = time.time()
                    
                    for epoch in range(epochs):
                        model.train()
                        permutation = torch.randperm(X_train_tensor.size(0))
                        total_loss = 0.0
                        
                        for k in range(0, X_train_tensor.size(0), bs):
                            indices = permutation[k:k+bs]
                            batch_x, batch_y = X_train_tensor[indices], y_train_tensor[indices]
                            
                            optimizer.zero_grad()
                            outputs = model(batch_x)
                            loss = criterion(outputs, batch_y)
                            loss.backward()
                            optimizer.step()
                            
                            total_loss += loss.item()
                        
                        if scheduler:
                            scheduler.step()
                        
                        avg_train_loss = total_loss / (X_train_tensor.size(0) / bs)
                        train_losses_temp.append(avg_train_loss)
                        
                        # Validation loss
                        model.eval()
                        with torch.no_grad():
                            val_outputs = model(X_test_tensor)
                            val_loss = criterion(val_outputs, y_test_tensor)
                            val_losses_temp.append(val_loss.item())
                        
                        test_acc, _, _, _, _ = compute_metrics(model, X_test_tensor, y_test_tensor)
                        test_accs_temp.append(test_acc)
                    
                    training_time = time.time() - start_time
                    convergence_speed = compute_convergence_speed(val_losses_temp, 0.95)
                    
                    results_grid[i, j] = test_accs_temp[-1]
                    convergence_grid[i, j] = convergence_speed
                    time_grid[i, j] = training_time
                    
                    detailed_results[f"LR={lr:.4f}_BS={bs}"] = {
                        'train_losses': train_losses_temp,
                        'val_losses': val_losses_temp,
                        'test_accs': test_accs_temp,
                        'final_acc': test_accs_temp[-1],
                        'convergence_speed': convergence_speed,
                        'time': training_time
                    }
                    
                    run_idx += 1
                    progress_bar.progress(min(run_idx / total_runs, 1.0))
            
            # Store results for this optimizer
            results_per_opt[opt_type] = {
                'results_grid': results_grid.copy(),
                'convergence_grid': convergence_grid.copy(),
                'time_grid': time_grid.copy(),
                'detailed_results': detailed_results.copy(),
                'color': colors_opt[opt_type]
            }
        
        status_text.empty()
        progress_bar.empty()
        
        # --- VISUALISATIONS SENSIBILITÉ ---
        st.markdown("## 🎯 Analyse de Sensibilité aux Hyperparamètres (Tous Optimiseurs)")
        
        # Best configurations per optimizer
        st.markdown("### 🏆 Meilleures Configurations par Optimiseur")
        
        best_configs_data = []
        for opt_type, data in results_per_opt.items():
            results_grid = data['results_grid']
            convergence_grid = data['convergence_grid']
            time_grid = data['time_grid']
            
            # Find best configuration
            best_idx = np.unravel_index(np.argmax(results_grid), results_grid.shape)
            best_lr = learning_rates_grid[best_idx[1]]
            best_bs = batch_sizes_grid[best_idx[0]]
            best_acc = results_grid[best_idx]
            
            # Fastest convergence
            min_conv_idx = np.unravel_index(np.argmin(convergence_grid), convergence_grid.shape)
            conv_speed = convergence_grid[min_conv_idx]
            
            # Average time
            avg_time = np.mean(time_grid)
            
            best_configs_data.append({
                'Optimiseur': opt_type,
                'Best LR': f"{best_lr:.4f}",
                'Best BS': best_bs,
                'Best Accuracy': f"{best_acc:.4f}",
                'Conv. Speed (95%)': f"{int(conv_speed)} ép.",
                'Temps Moyen': f"{avg_time:.2f}s"
            })
        
        st.dataframe(pd.DataFrame(best_configs_data), use_container_width=True)
        
        st.markdown("---")
        
        # Heatmaps - Comparison across optimizers
        st.markdown("### 🔥 Heatmaps de Performance par Optimiseur")
        
        # Accuracy heatmaps
        st.markdown("**Accuracy Finale**")
        cols = st.columns(3)
        for idx, (opt_type, data) in enumerate(results_per_opt.items()):
            with cols[idx]:
                fig_heatmap_acc = go.Figure(data=go.Heatmap(
                    z=data['results_grid'],
                    x=[f"{lr:.4f}" for lr in learning_rates_grid],
                    y=[f"{bs}" for bs in batch_sizes_grid],
                    colorscale='Viridis',
                    text=np.round(data['results_grid'], 4),
                    texttemplate='%{text}',
                    textfont={"size": 9},
                    colorbar=dict(title="Acc")
                ))
                fig_heatmap_acc.update_layout(
                    title=f"{opt_type}",
                    xaxis_title="LR",
                    yaxis_title="BS",
                    height=350
                )
                st.plotly_chart(fig_heatmap_acc, use_container_width=True)
        
        # Convergence speed heatmaps
        st.markdown("**Vitesse de Convergence (95%)**")
        cols = st.columns(3)
        for idx, (opt_type, data) in enumerate(results_per_opt.items()):
            with cols[idx]:
                fig_heatmap_conv = go.Figure(data=go.Heatmap(
                    z=data['convergence_grid'],
                    x=[f"{lr:.4f}" for lr in learning_rates_grid],
                    y=[f"{bs}" for bs in batch_sizes_grid],
                    colorscale='RdYlGn_r',
                    text=np.round(data['convergence_grid'], 0).astype(int),
                    texttemplate='%{text}',
                    textfont={"size": 9},
                    colorbar=dict(title="Ép.")
                ))
                fig_heatmap_conv.update_layout(
                    title=f"{opt_type}",
                    xaxis_title="LR",
                    yaxis_title="BS",
                    height=350
                )
                st.plotly_chart(fig_heatmap_conv, use_container_width=True)
        
        # 3D Surfaces - Side by side
        st.markdown("### 📊 Surfaces 3D de Performance")
        
        cols = st.columns(3)
        for idx, (opt_type, data) in enumerate(results_per_opt.items()):
            with cols[idx]:
                fig_3d = go.Figure(data=[go.Surface(
                    z=data['results_grid'],
                    x=learning_rates_grid,
                    y=batch_sizes_grid,
                    colorscale='Viridis',
                    colorbar=dict(title="Acc")
                )])
                
                fig_3d.update_layout(
                    title=f"{opt_type}",
                    scene=dict(
                        xaxis_title="LR",
                        yaxis_title="BS",
                        zaxis_title="Accuracy",
                        xaxis_type="log"
                    ),
                    height=500
                )
                st.plotly_chart(fig_3d, use_container_width=True)
        
        # Detailed comparison - best config per optimizer
        st.markdown("### 📈 Comparaison des Meilleures Configurations par Optimiseur")
        
        fig_comparison = make_subplots(
            rows=1, cols=2,
            subplot_titles=('Loss (Train & Val)', 'Accuracy Test')
        )
        
        for opt_type, data in results_per_opt.items():
            results_grid = data['results_grid']
            detailed_results = data['detailed_results']
            color = data['color']
            
            # Get best config for this optimizer
            best_idx = np.unravel_index(np.argmax(results_grid), results_grid.shape)
            best_lr = learning_rates_grid[best_idx[1]]
            best_bs = batch_sizes_grid[best_idx[0]]
            key = f"LR={best_lr:.4f}_BS={best_bs}"
            res = detailed_results[key]
            
            # Train loss (solid)
            fig_comparison.add_trace(
                go.Scatter(y=res['train_losses'], mode='lines', 
                          name=f"{opt_type} (Train)",
                          line=dict(color=color, width=2)),
                row=1, col=1
            )
            # Val loss (dashed)
            fig_comparison.add_trace(
                go.Scatter(y=res['val_losses'], mode='lines', 
                          name=f"{opt_type} (Val)",
                          line=dict(color=color, width=2, dash='dash'),
                          showlegend=False),
                row=1, col=1
            )
            # Accuracy
            fig_comparison.add_trace(
                go.Scatter(y=res['test_accs'], mode='lines',
                          name=f"{opt_type}",
                          line=dict(color=color, width=2),
                          showlegend=False),
                row=1, col=2
            )
        
        fig_comparison.update_xaxes(title_text="Époque", row=1, col=1)
        fig_comparison.update_xaxes(title_text="Époque", row=1, col=2)
        fig_comparison.update_yaxes(title_text="Loss", row=1, col=1)
        fig_comparison.update_yaxes(title_text="Accuracy", row=1, col=2)
        fig_comparison.update_layout(height=400, template="plotly_white")
        
        st.plotly_chart(fig_comparison, use_container_width=True)
        
        # Efficiency Frontier - Compare optimizers across hyperparams
        st.markdown("### ⚡ Frontière d'Efficacité (Temps vs Performance)")
        
        fig_pareto = go.Figure()
        
        for opt_type, data in results_per_opt.items():
            results_grid = data['results_grid']
            time_grid = data['time_grid']
            color = data['color']
            
            # Get all configs as points
            times_list = []
            accs_list = []
            
            for i in range(len(batch_sizes_grid)):
                for j in range(len(learning_rates_grid)):
                    times_list.append(time_grid[i, j])
                    accs_list.append(results_grid[i, j])
            
            fig_pareto.add_trace(go.Scatter(
                x=times_list,
                y=accs_list,
                mode='markers',
                marker=dict(size=10, color=color, opacity=0.6),
                name=opt_type
            ))
        
        fig_pareto.update_layout(
            title="Trade-off Temps vs Performance (toutes configs)",
            xaxis_title="Temps d'entraînement (s)",
            yaxis_title="Accuracy finale",
            height=500,
            template="plotly_white"
        )
        st.plotly_chart(fig_pareto, use_container_width=True)
        
        # Summary comparison
        st.markdown("### 📊 Résumé de la Sensibilité")
        
        col1, col2 = st.columns(2)
        
        with col1:
            # Variance of accuracy across hyperparameters
            fig_variance = go.Figure()
            
            for opt_type, data in results_per_opt.items():
                variance = np.std(data['results_grid'].flatten())
                mean_acc = np.mean(data['results_grid'].flatten())
                
                fig_variance.add_trace(go.Bar(
                    x=[opt_type],
                    y=[variance],
                    name=opt_type,
                    marker_color=data['color'],
                    text=[f"σ={variance:.4f}"],
                    textposition='auto'
                ))
            
            fig_variance.update_layout(
                title="Sensibilité aux Hyperparamètres (Écart-type Accuracy)",
                yaxis_title="Écart-type",
                template="plotly_white",
                showlegend=False,
                height=400
            )
            st.plotly_chart(fig_variance, use_container_width=True)
            st.caption("Plus l'écart-type est faible, moins l'optimiseur est sensible aux hyperparamètres")
        
        with col2:
            # Average performance
            fig_avg = go.Figure()
            
            for opt_type, data in results_per_opt.items():
                mean_acc = np.mean(data['results_grid'].flatten())
                
                fig_avg.add_trace(go.Bar(
                    x=[opt_type],
                    y=[mean_acc],
                    name=opt_type,
                    marker_color=data['color'],
                    text=[f"{mean_acc:.4f}"],
                    textposition='auto'
                ))
            
            fig_avg.update_layout(
                title="Performance Moyenne (toutes configs)",
                yaxis_title="Accuracy Moyenne",
                template="plotly_white",
                showlegend=False,
                height=400
            )
            st.plotly_chart(fig_avg, use_container_width=True)
            st.caption("Performance moyenne sur toutes les combinaisons de hyperparamètres")
        
        st.success("✅ Analyse de sensibilité terminée !")
    
    elif analysis_mode == "Analyse Robustesse":
        st.info(f"🔄 Analyse de robustesse pour tous les optimiseurs ({n_runs} runs chacun)...")
        
        optimizers_list = ['SGD', 'RMSprop', 'Adam']
        colors_opt = {'SGD': '#FF6B6B', 'RMSprop': '#4ECDC4', 'Adam': '#45B7D1'}
        
        total_runs = n_runs * len(optimizers_list)
        progress_bar = st.progress(0)
        status_text = st.empty()
        
        results_per_optimizer = {}
        run_idx = 0
        
        for opt_type in optimizers_list:
            all_runs = []
            
            for run in range(n_runs):
                status_text.text(f"Run {run_idx+1}/{total_runs}: {opt_type} - Run {run+1}/{n_runs}")
                
                # Reinitialize model for each run
                model = create_model()
                optimizer = create_optimizer(model, opt_type, learning_rate)
                scheduler = create_scheduler(optimizer, scheduler_type) if use_scheduler else None
                
                result = train_model(model, optimizer, scheduler, opt_type, colors_opt[opt_type])
                all_runs.append(result)
                
                run_idx += 1
                progress_bar.progress(min(run_idx / total_runs, 1.0))
            
            results_per_optimizer[opt_type] = {
                'runs': all_runs,
                'color': colors_opt[opt_type]
            }
        
        status_text.empty()
        progress_bar.empty()
        
        # --- ANALYSE ROBUSTESSE ---
        st.markdown("## 🎲 Analyse de Robustesse (Tous Optimiseurs)")
        
        # Statistics comparison
        st.markdown("### 📊 Statistiques de Robustesse par Optimiseur")
        
        stats_data = []
        for opt_type, data in results_per_optimizer.items():
            all_runs = data['runs']
            
            final_accs = [run['final_test_acc'] for run in all_runs]
            final_val_losses = [run['final_val_loss'] for run in all_runs]
            training_times = [run['training_time'] for run in all_runs]
            conv_speeds = [run['convergence_speed_95'] for run in all_runs]
            
            stats_data.append({
                'Optimiseur': opt_type,
                'Acc Moyenne': f"{np.mean(final_accs):.4f} ± {np.std(final_accs):.4f}",
                'Val Loss Moyenne': f"{np.mean(final_val_losses):.4f} ± {np.std(final_val_losses):.4f}",
                'Temps Moyen': f"{np.mean(training_times):.2f}s ± {np.std(training_times):.2f}s",
                'Conv. Moyenne': f"{np.mean(conv_speeds):.1f} ± {np.std(conv_speeds):.1f} ép.",
                'CV Acc (%)': f"{(np.std(final_accs) / np.mean(final_accs) * 100):.2f}%"
            })
        
        st.dataframe(pd.DataFrame(stats_data), use_container_width=True)
        st.caption("CV = Coefficient de Variation (plus faible = plus stable)")
        
        st.markdown("---")
        
        # Visualizations - Compare optimizers
        st.markdown("### 📈 Courbes d'Entraînement (Tous Runs)")
        
        col1, col2 = st.columns(2)
        
        with col1:
            # Loss curves for all runs (Train and Val) - All optimizers
            fig_losses = go.Figure()
            
            for opt_type, data in results_per_optimizer.items():
                all_runs = data['runs']
                color = data['color']
                
                # Add individual runs (light)
                for idx, run in enumerate(all_runs):
                    fig_losses.add_trace(go.Scatter(
                        y=run['val_losses'],
                        mode='lines',
                        name=f"{opt_type} Run {idx+1}" if idx == 0 else None,
                        opacity=0.2,
                        line=dict(width=1, color=color),
                        showlegend=(idx == 0),
                        legendgroup=opt_type
                    ))
                
                # Add mean curve (bold)
                max_len = max(len(run['val_losses']) for run in all_runs)
                all_val_losses_padded = []
                for run in all_runs:
                    val_padded = list(run['val_losses']) + [run['val_losses'][-1]] * (max_len - len(run['val_losses']))
                    all_val_losses_padded.append(val_padded)
                
                mean_val_losses = np.mean(all_val_losses_padded, axis=0)
                
                fig_losses.add_trace(go.Scatter(
                    y=mean_val_losses,
                    mode='lines',
                    name=f'{opt_type} Moyenne',
                    line=dict(color=color, width=3),
                    legendgroup=opt_type
                ))
            
            fig_losses.update_layout(
                title=f"Validation Loss - {n_runs} Runs par Optimiseur",
                xaxis_title="Époque",
                yaxis_title="Loss",
                template="plotly_white",
                height=400
            )
            st.plotly_chart(fig_losses, use_container_width=True)
        
        with col2:
            # Accuracy curves for all runs - All optimizers
            fig_accs = go.Figure()
            
            for opt_type, data in results_per_optimizer.items():
                all_runs = data['runs']
                color = data['color']
                
                # Add individual runs (light)
                for idx, run in enumerate(all_runs):
                    fig_accs.add_trace(go.Scatter(
                        y=run['test_accs'],
                        mode='lines',
                        name=f"{opt_type} Run {idx+1}" if idx == 0 else None,
                        opacity=0.2,
                        line=dict(width=1, color=color),
                        showlegend=(idx == 0),
                        legendgroup=opt_type
                    ))
                
                # Mean accuracy
                max_len = max(len(run['test_accs']) for run in all_runs)
                all_accs_padded = []
                for run in all_runs:
                    padded = list(run['test_accs']) + [run['test_accs'][-1]] * (max_len - len(run['test_accs']))
                    all_accs_padded.append(padded)
                
                mean_accs = np.mean(all_accs_padded, axis=0)
                
                fig_accs.add_trace(go.Scatter(
                    y=mean_accs,
                    mode='lines',
                    name=f'{opt_type} Moyenne',
                    line=dict(color=color, width=3),
                    legendgroup=opt_type
                ))
            
            fig_accs.update_layout(
                title=f"Accuracy Test - {n_runs} Runs par Optimiseur",
                xaxis_title="Époque",
                yaxis_title="Accuracy",
                template="plotly_white",
                height=400
            )
            st.plotly_chart(fig_accs, use_container_width=True)
        
        # Distribution plots
        st.markdown("### 📊 Distributions des Métriques Finales")
        
        col1, col2 = st.columns(2)
        
        with col1:
            # Accuracy distribution
            fig_dist_acc = go.Figure()
            
            for opt_type, data in results_per_optimizer.items():
                all_runs = data['runs']
                final_accs = [run['final_test_acc'] for run in all_runs]
                
                fig_dist_acc.add_trace(go.Box(
                    y=final_accs,
                    name=opt_type,
                    marker_color=data['color']
                ))
            
            fig_dist_acc.update_layout(
                title="Distribution de l'Accuracy Finale",
                yaxis_title="Accuracy",
                template="plotly_white",
                height=400
            )
            st.plotly_chart(fig_dist_acc, use_container_width=True)
        
        with col2:
            # Val Loss distribution
            fig_dist_loss = go.Figure()
            
            for opt_type, data in results_per_optimizer.items():
                all_runs = data['runs']
                final_val_losses = [run['final_val_loss'] for run in all_runs]
                
                fig_dist_loss.add_trace(go.Box(
                    y=final_val_losses,
                    name=opt_type,
                    marker_color=data['color']
                ))
            
            fig_dist_loss.update_layout(
                title="Distribution de la Val Loss Finale",
                yaxis_title="Val Loss",
                template="plotly_white",
                height=400
            )
            st.plotly_chart(fig_dist_loss, use_container_width=True)
        
        # Stability over time
        st.markdown("### 📉 Stabilité de l'Entraînement")
        
        fig_stability = go.Figure()
        
        for opt_type, data in results_per_optimizer.items():
            all_runs = data['runs']
            color = data['color']
            
            # Compute std at each epoch for val loss
            max_len = max(len(run['val_losses']) for run in all_runs)
            all_val_losses_padded = []
            for run in all_runs:
                val_padded = list(run['val_losses']) + [run['val_losses'][-1]] * (max_len - len(run['val_losses']))
                all_val_losses_padded.append(val_padded)
            
            epoch_val_stds = np.std(all_val_losses_padded, axis=0)
            
            fig_stability.add_trace(go.Scatter(
                y=epoch_val_stds,
                mode='lines',
                name=opt_type,
                line=dict(color=color, width=2)
            ))
        
        fig_stability.update_layout(
            title="Variabilité de la Val Loss entre les Runs",
            xaxis_title="Époque",
            yaxis_title="Écart-type",
            template="plotly_white",
            height=400
        )
        st.plotly_chart(fig_stability, use_container_width=True)
        st.caption("Plus la courbe est basse, plus l'optimiseur est stable (variance faible entre les runs)")
        
        # Coefficient of variation comparison
        st.markdown("### 📊 Comparaison de la Stabilité (Coefficient de Variation)")
        
        cv_data = []
        for opt_type, data in results_per_optimizer.items():
            all_runs = data['runs']
            
            final_accs = [run['final_test_acc'] for run in all_runs]
            final_val_losses = [run['final_val_loss'] for run in all_runs]
            
            cv_acc = (np.std(final_accs) / np.mean(final_accs)) * 100
            cv_val_loss = (np.std(final_val_losses) / np.mean(final_val_losses)) * 100
            
            cv_data.append({
                'Optimiseur': opt_type,
                'CV Accuracy (%)': f"{cv_acc:.2f}%",
                'CV Val Loss (%)': f"{cv_val_loss:.2f}%"
            })
        
        st.dataframe(pd.DataFrame(cv_data), use_container_width=True)
        st.caption("⚠️ Plus le coefficient de variation (CV) est faible, plus l'optimiseur est robuste et stable")
        
        st.success("✅ Analyse de robustesse terminée !")
    
    elif compare_mode:
        st.info("🔄 Mode comparaison : entraînement de SGD, RMSprop et Adam...")
        
        # Create models and optimizers for all three
        results = {}
        colors = {'SGD': '#FF6B6B', 'RMSprop': '#4ECDC4', 'Adam': '#45B7D1'}
        
        progress_bar = st.progress(0)
        status_text = st.empty()
        
        for idx, opt_type in enumerate(['SGD', 'RMSprop', 'Adam']):
            status_text.text(f"Entraînement de {opt_type}...")
            model = create_model()
            optimizer = create_optimizer(model, opt_type, learning_rate)
            scheduler = create_scheduler(optimizer, scheduler_type) if use_scheduler else None
            
            results[opt_type] = train_model(model, optimizer, scheduler, opt_type, colors[opt_type])
            progress_bar.progress((idx + 1) / 3)
        
        status_text.empty()
        progress_bar.empty()
        
        # --- VISUALISATIONS COMPARATIVES ---
        st.markdown("## 📊 Comparaison des Optimiseurs")
        
        # Performance Summary Table
        summary_data = []
        for opt_type, res in results.items():
                summary_data.append({
                'Optimiseur': opt_type,
                'Temps (s)': f"{res['training_time']:.2f}",
                'Accuracy finale': f"{res['final_test_acc']:.4f}",
                'Train Loss': f"{res['final_train_loss']:.4f}",
                'Val Loss': f"{res['final_val_loss']:.4f}",
                'F1 Score': f"{res['test_f1s'][-1]:.4f}",
                'Convergence (95%)': f"{res['convergence_speed_95']} époques",
                'Convergence (99%)': f"{res['convergence_speed_99']} époques",
                'Converged': '✅' if res['converged'] else '❌'
            })
        
        st.dataframe(pd.DataFrame(summary_data), use_container_width=True)
        
        # Loss comparison
        col1, col2 = st.columns(2)
        
        with col1:
            fig_loss = go.Figure()
            for opt_type, res in results.items():
                # Training loss
                fig_loss.add_trace(go.Scatter(
                    y=res['train_losses'],
                    mode='lines',
                    name=f"{opt_type} (Train)",
                    line=dict(color=res['color'], width=2)
                ))
                # Validation loss
                fig_loss.add_trace(go.Scatter(
                    y=res['val_losses'],
                    mode='lines',
                    name=f"{opt_type} (Val)",
                    line=dict(color=res['color'], width=2, dash='dash')
                ))
            fig_loss.update_layout(
                title="Loss par époque (Train & Validation)",
                xaxis_title="Époque",
                yaxis_title="Loss",
                template="plotly_white",
                height=400
            )
            st.plotly_chart(fig_loss, use_container_width=True)
        
        with col2:
            fig_acc = go.Figure()
            for opt_type, res in results.items():
                fig_acc.add_trace(go.Scatter(
                    y=res['test_accs'],
                    mode='lines',
                    name=opt_type,
                    line=dict(color=res['color'], width=2)
                ))
            fig_acc.update_layout(
                title="Accuracy Test par époque",
                xaxis_title="Époque",
                yaxis_title="Accuracy",
                template="plotly_white",
                height=400
            )
            st.plotly_chart(fig_acc, use_container_width=True)
        
        # Decision boundaries comparison
        st.markdown("### 🎯 Frontières de Décision")
        cols = st.columns(3)
        for idx, (opt_type, res) in enumerate(results.items()):
            with cols[idx]:
                fig = go.Figure()
                fig.add_trace(go.Contour(
                    x=np.linspace(x_min, x_max, 200),
                    y=np.linspace(y_min, y_max, 200),
                    z=res['decision_boundary'],
                    colorscale="RdBu",
                    opacity=0.7,
                    showscale=False
                ))
                fig.add_trace(go.Scatter(
                    x=X_test[:, 0],
                    y=X_test[:, 1],
                    mode="markers",
                    marker=dict(
                        color=y_test,
                        colorscale="RdBu",
                        line=dict(width=1, color="black"),
                        size=6
                    ),
                    showlegend=False
                ))
                fig.update_layout(
                    title=f"{opt_type}",
                    xaxis_title="x₁",
                    yaxis_title="x₂",
                    height=350
                )
                st.plotly_chart(fig, use_container_width=True)
        
        # Convergence Analysis
        st.markdown("### ⚡ Analyse de Convergence")
        
        col1, col2 = st.columns(2)
        
        with col1:
            # Convergence speed comparison
            fig_conv = go.Figure()
            
            conv_data = []
            for opt_type, res in results.items():
                conv_data.append({
                    'Optimiseur': opt_type,
                    '95% Performance': res['convergence_speed_95'],
                    '99% Performance': res['convergence_speed_99']
                })
            
            conv_df = pd.DataFrame(conv_data)
            
            fig_conv.add_trace(go.Bar(
                x=conv_df['Optimiseur'],
                y=conv_df['95% Performance'],
                name='95% de la perf. finale',
                marker_color='#4ECDC4'
            ))
            fig_conv.add_trace(go.Bar(
                x=conv_df['Optimiseur'],
                y=conv_df['99% Performance'],
                name='99% de la perf. finale',
                marker_color='#FF6B6B'
            ))
            
            fig_conv.update_layout(
                title="Vitesse de Convergence (époques nécessaires)",
                xaxis_title="Optimiseur",
                yaxis_title="Nombre d'époques",
                barmode='group',
                template="plotly_white",
                height=400
            )
            st.plotly_chart(fig_conv, use_container_width=True)
        
        with col2:
            # Stability comparison
            fig_stability = go.Figure()
            
            for opt_type, res in results.items():
                if len(res['loss_smoothness']) > 0:
                    fig_stability.add_trace(go.Scatter(
                        y=res['loss_smoothness'],
                        mode='lines',
                        name=opt_type,
                        line=dict(color=res['color'], width=2)
                    ))
            
            fig_stability.update_layout(
                title="Stabilité de l'Entraînement (variance de la loss)",
                xaxis_title="Époque",
                yaxis_title="Écart-type (fenêtre de 5)",
                template="plotly_white",
                height=400
            )
            st.plotly_chart(fig_stability, use_container_width=True)
        
        # Metrics comparison
        col1, col2 = st.columns(2)
        
        with col1:
            fig_metrics = go.Figure()
            for opt_type, res in results.items():
                fig_metrics.add_trace(go.Scatter(
                    y=res['test_f1s'],
                    mode='lines',
                    name=opt_type,
                    line=dict(color=res['color'], width=2)
                ))
            fig_metrics.update_layout(
                title="F1 Score (Test) par époque",
                xaxis_title="Époque",
                yaxis_title="F1 Score",
                template="plotly_white",
                height=400
            )
            st.plotly_chart(fig_metrics, use_container_width=True)
        
        with col2:
            if use_scheduler:
                fig_lr = go.Figure()
                for opt_type, res in results.items():
                    fig_lr.add_trace(go.Scatter(
                        y=res['learning_rates'],
                        mode='lines',
                        name=opt_type,
                        line=dict(color=res['color'], width=2)
                    ))
                fig_lr.update_layout(
                    title="Learning Rate par époque",
                    xaxis_title="Époque",
                    yaxis_title="Learning Rate",
                    template="plotly_white",
                    height=400
                )
                st.plotly_chart(fig_lr, use_container_width=True)
            else:
                fig_grad = go.Figure()
                for opt_type, res in results.items():
                    fig_grad.add_trace(go.Scatter(
                        y=res['gradient_norms'],
                        mode='lines',
                        name=opt_type,
                        line=dict(color=res['color'], width=2)
                    ))
                fig_grad.update_layout(
                    title="Gradient Norms (moyenne) par époque",
                    xaxis_title="Époque",
                    yaxis_title="Gradient Norm",
                    template="plotly_white",
                    height=400
                )
                st.plotly_chart(fig_grad, use_container_width=True)
        
        # Efficiency Analysis
        st.markdown("### ⚡ Analyse d'Efficacité")
        
        fig_efficiency = go.Figure()
        
        for opt_type, res in results.items():
            fig_efficiency.add_trace(go.Scatter(
                x=[res['training_time']],
                y=[res['final_test_acc']],
                mode='markers+text',
                marker=dict(size=20, color=res['color']),
                text=[opt_type],
                textposition="top center",
                textfont=dict(size=14),
                name=opt_type
            ))
        
        fig_efficiency.update_layout(
            title="Frontière d'Efficacité: Temps vs Performance",
            xaxis_title="Temps d'entraînement (s)",
            yaxis_title="Accuracy finale",
            template="plotly_white",
            height=400
        )
        st.plotly_chart(fig_efficiency, use_container_width=True)
        
    else:
        # Single optimizer mode
        st.info(f"🔄 Entraînement avec {opt_name}...")
        
        model = create_model()
        optimizer = create_optimizer(model, opt_name, learning_rate)
        scheduler = create_scheduler(optimizer, scheduler_type) if use_scheduler else None
        
        color = '#45B7D1'
        result = train_model(model, optimizer, scheduler, opt_name, color)
        
        # --- VISUALISATIONS DÉTAILLÉES ---
        st.markdown("## 📊 Métriques d'Entraînement")
        
        # Performance summary
        col1, col2, col3, col4, col5 = st.columns(5)
        col1.metric("Temps d'entraînement", f"{result['training_time']:.2f}s")
        col2.metric("Accuracy finale (test)", f"{result['final_test_acc']:.4f}")
        col3.metric("Train Loss finale", f"{result['final_train_loss']:.4f}")
        col4.metric("Val Loss finale", f"{result['final_val_loss']:.4f}")
        col5.metric("Convergence (95%)", f"{result['convergence_speed_95']} ép.", 
                   delta="Converged ✅" if result['converged'] else "Non convergé ❌")
        
        # Loss and Accuracy
        col1, col2 = st.columns(2)
        
        with col1:
            fig_loss = go.Figure()
            fig_loss.add_trace(go.Scatter(
                y=result['train_losses'],
                mode='lines+markers',
                name='Train Loss',
                line=dict(color='#FF6B6B', width=2)
            ))
            fig_loss.add_trace(go.Scatter(
                y=result['val_losses'],
                mode='lines+markers',
                name='Validation Loss',
                line=dict(color='#FFA500', width=2, dash='dash')
            ))
            fig_loss.update_layout(
                title="Loss par époque (Train & Validation)",
                xaxis_title="Époque",
                yaxis_title="Loss",
                template="plotly_white",
                height=350,
                legend=dict(x=0.7, y=0.95)
            )
            st.plotly_chart(fig_loss, use_container_width=True)
        
        with col2:
            fig_acc = go.Figure()
            fig_acc.add_trace(go.Scatter(
                y=result['train_accs'],
                mode='lines',
                name='Train',
                line=dict(color='#4ECDC4', width=2)
            ))
            fig_acc.add_trace(go.Scatter(
                y=result['test_accs'],
                mode='lines',
                name='Test',
                line=dict(color='#FF6B6B', width=2)
            ))
            fig_acc.update_layout(
                title="Accuracy par époque",
                xaxis_title="Époque",
                yaxis_title="Accuracy",
                template="plotly_white",
                height=350
            )
            st.plotly_chart(fig_acc, use_container_width=True)
        
        # Classification metrics
        st.markdown("### 📈 Métriques de Classification")
        
        col1, col2 = st.columns(2)
        
        with col1:
            fig_metrics = make_subplots(
                rows=2, cols=2,
                subplot_titles=('Precision', 'Recall', 'F1 Score', 'All Metrics'),
                specs=[[{}, {}], [{'colspan': 2}, None]]
            )
            
            # Precision
            fig_metrics.add_trace(go.Scatter(y=result['test_precs'], mode='lines', name='Precision', 
                                            line=dict(color='#FF6B6B')), row=1, col=1)
            
            # Recall
            fig_metrics.add_trace(go.Scatter(y=result['test_recs'], mode='lines', name='Recall',
                                            line=dict(color='#4ECDC4')), row=1, col=2)
            
            # All metrics combined
            fig_metrics.add_trace(go.Scatter(y=result['test_accs'], mode='lines', name='Accuracy',
                                            line=dict(color='#45B7D1')), row=2, col=1)
            fig_metrics.add_trace(go.Scatter(y=result['test_precs'], mode='lines', name='Precision',
                                            line=dict(color='#FF6B6B')), row=2, col=1)
            fig_metrics.add_trace(go.Scatter(y=result['test_recs'], mode='lines', name='Recall',
                                            line=dict(color='#4ECDC4')), row=2, col=1)
            fig_metrics.add_trace(go.Scatter(y=result['test_f1s'], mode='lines', name='F1',
                                            line=dict(color='#95E1D3')), row=2, col=1)
            
            fig_metrics.update_xaxes(title_text="Époque")
            fig_metrics.update_yaxes(title_text="Score")
            fig_metrics.update_layout(height=600, template="plotly_white", showlegend=True)
            st.plotly_chart(fig_metrics, use_container_width=True)
        
        with col2:
            # Confusion Matrix
            cm = result['confusion_matrix']
            fig_cm = go.Figure(data=go.Heatmap(
                z=cm,
                x=['Classe 0', 'Classe 1'],
                y=['Classe 0', 'Classe 1'],
                colorscale='Blues',
                text=cm,
                texttemplate='%{text}',
                textfont={"size": 20}
            ))
            fig_cm.update_layout(
                title="Matrice de Confusion (Test)",
                xaxis_title="Prédiction",
                yaxis_title="Vérité",
                height=300
            )
            st.plotly_chart(fig_cm, use_container_width=True)
            
            # Learning Rate / Gradient Norms
            if use_scheduler:
                fig_lr = go.Figure()
                fig_lr.add_trace(go.Scatter(
                    y=result['learning_rates'],
                    mode='lines+markers',
                    name='Learning Rate',
                    line=dict(color='#FF6B6B', width=2)
                ))
                fig_lr.update_layout(
                    title="Learning Rate par époque",
                    xaxis_title="Époque",
                    yaxis_title="LR",
                    template="plotly_white",
                    height=250
                )
                st.plotly_chart(fig_lr, use_container_width=True)
            
            fig_grad = go.Figure()
            fig_grad.add_trace(go.Scatter(
                y=result['gradient_norms'],
                mode='lines+markers',
                name='Gradient Norm',
                line=dict(color='#4ECDC4', width=2)
            ))
            fig_grad.update_layout(
                title="Gradient Norms (moyenne) par époque",
                xaxis_title="Époque",
                yaxis_title="Norm",
                template="plotly_white",
                height=250
            )
            st.plotly_chart(fig_grad, use_container_width=True)
        
        # Convergence Analysis
        st.markdown("### ⚡ Analyse de Convergence et Stabilité")
        
        col1, col2 = st.columns(2)
        
        with col1:
            # Convergence info
            st.markdown(f"""
            **Vitesse de Convergence:**
            - 95% de la performance finale: **{result['convergence_speed_95']} époques**
            - 99% de la performance finale: **{result['convergence_speed_99']} époques**
            - Convergence détectée: **{'✅ Oui' if result['converged'] else '❌ Non'}**
            {f"- Époque de convergence: **{result['convergence_epoch']}**" if result['converged'] else ""}
            """)
            
            # Mark convergence points on loss curve
            fig_conv_markers = go.Figure()
            fig_conv_markers.add_trace(go.Scatter(
                y=result['train_losses'],
                mode='lines',
                name='Train Loss',
                line=dict(color='#FF6B6B', width=2)
            ))
            fig_conv_markers.add_trace(go.Scatter(
                y=result['val_losses'],
                mode='lines',
                name='Val Loss',
                line=dict(color='#FFA500', width=2, dash='dash')
            ))
            
            # Add vertical lines for convergence points (based on val loss)
            if result['convergence_speed_95'] < len(result['val_losses']):
                fig_conv_markers.add_vline(
                    x=result['convergence_speed_95'],
                    line_dash="dash",
                    line_color="green",
                    annotation_text="95%"
                )
            
            if result['convergence_speed_99'] < len(result['val_losses']):
                fig_conv_markers.add_vline(
                    x=result['convergence_speed_99'],
                    line_dash="dash",
                    line_color="orange",
                    annotation_text="99%"
                )
            
            if result['converged']:
                fig_conv_markers.add_vline(
                    x=result['convergence_epoch'],
                    line_dash="dot",
                    line_color="blue",
                    annotation_text="Plateau"
                )
            
            fig_conv_markers.update_layout(
                title="Points de Convergence (Train & Val Loss)",
                xaxis_title="Époque",
                yaxis_title="Loss",
                template="plotly_white",
                height=350
            )
            st.plotly_chart(fig_conv_markers, use_container_width=True)
        
        with col2:
            # Stability analysis
            if len(result['loss_smoothness']) > 0:
                fig_stability = go.Figure()
                fig_stability.add_trace(go.Scatter(
                    y=result['loss_smoothness'],
                    mode='lines+markers',
                    name='Variance de la Loss',
                    line=dict(color='#45B7D1', width=2),
                    fill='tozeroy'
                ))
                
                fig_stability.update_layout(
                    title="Stabilité de l'Entraînement",
                    xaxis_title="Époque",
                    yaxis_title="Écart-type (fenêtre de 5)",
                    template="plotly_white",
                    height=350
                )
                st.plotly_chart(fig_stability, use_container_width=True)
                
                avg_stability = np.mean(result['loss_smoothness'])
                st.markdown(f"""
                **Métriques de Stabilité:**
                - Variance moyenne: **{avg_stability:.6f}**
                - Variance minimale: **{np.min(result['loss_smoothness']):.6f}**
                - Variance maximale: **{np.max(result['loss_smoothness']):.6f}**
                
                ℹ️ Plus la variance est faible, plus l'entraînement est stable.
                """)
        
        # Decision Boundary with Confidence
        st.markdown("### 🎯 Frontière de Décision avec Confiance")
        
        fig_decision = go.Figure()
        
        # Contour with confidence levels
        fig_decision.add_trace(go.Contour(
            x=np.linspace(x_min, x_max, 200),
            y=np.linspace(y_min, y_max, 200),
            z=result['decision_boundary'],
            colorscale="RdBu",
            opacity=0.6,
            contours=dict(
                start=0,
                end=1,
                size=0.1,
                showlabels=True
            ),
            colorbar=dict(title="Probabilité<br>Classe 1")
        ))
        
        # Test points
        fig_decision.add_trace(go.Scatter(
            x=X_test[:, 0],
            y=X_test[:, 1],
            mode="markers",
            marker=dict(
                color=y_test,
                colorscale="RdBu",
                line=dict(width=1, color="black"),
                size=8
            ),
            name="Test data"
        ))
        
        # Decision boundary line at 0.5
        fig_decision.add_trace(go.Contour(
            x=np.linspace(x_min, x_max, 200),
            y=np.linspace(y_min, y_max, 200),
            z=result['decision_boundary'],
            contours=dict(
                start=0.5,
                end=0.5,
                size=0,
                coloring='lines'
            ),
            line=dict(width=3, color='yellow'),
            showscale=False,
            name='Frontière 0.5'
        ))
        
        fig_decision.update_layout(
            title=f"Frontière de décision avec niveaux de confiance ({opt_name})",
            xaxis_title="x₁",
            yaxis_title="x₂",
            height=500,
            template="plotly_white"
        )
        st.plotly_chart(fig_decision, use_container_width=True)
    
    st.success("✅ Entraînement terminé !")
else:
    st.info("👆 Cliquez sur 'Démarrer' dans la barre latérale pour lancer l'entraînement")
