# t_visual_utils.py
import matplotlib.pyplot as plt

def plot_loss_curve(loss_history, save_path=None):
    """
    Affiche la courbe de perte pendant l'entraînement.

    Args:
        loss_history (list of float): Les pertes enregistrées au cours du training.
        save_path (str, optional): Si spécifié, sauvegarde le graphique à ce chemin.
    """
    plt.figure(figsize=(10, 5))
    plt.plot(loss_history, label="Training Loss", color="blue")
    plt.xlabel("Batch")
    plt.ylabel("Loss")
    plt.title("Courbe de convergence de l'entraînement (LoRA)")
    plt.grid(True)
    plt.legend()
    if save_path:
        plt.savefig(save_path)
    plt.show()

def smooth_curve(points, factor=0.9):
    """
    Applique un lissage exponentiel à une courbe.

    Args:
        points (list): Liste des valeurs originales.
        factor (float): Coefficient de lissage (0.0 = pas de lissage).

    Returns:
        list: courbe lissée
    """
    smoothed = []
    for point in points:
        if smoothed:
            previous = smoothed[-1]
            smoothed.append(previous * factor + point * (1 - factor))
        else:
            smoothed.append(point)
    return smoothed
