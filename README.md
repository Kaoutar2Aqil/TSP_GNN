# TSP_GNN
Le Problème du Voyageur de Commerce (Travelling Salesperson Problem, TSP) est un problème classique d'optimisation combinatoire. L'objectif est de trouver le plus court chemin qui permet de visiter une liste de villes exactement une fois, en revenant à la ville de départ.
# Représentation du problème sous forme de graphe
Avant d'utiliser un GNN, on doit transformer le TSP en un graphe :
•	Nœuds : Chaque nœud représente une ville.
•	Arêtes : Chaque arête correspond à la distance entre deux villes (souvent un coût pondéré).
•	Caractéristiques des nœuds : Les coordonnées (x, y) des villes sont souvent utilisées comme attributs.
•	Caractéristiques des arêtes : La distance entre les villes peut être ajoutée comme poids.
