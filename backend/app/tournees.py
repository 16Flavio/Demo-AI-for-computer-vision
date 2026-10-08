import math
from ortools.constraint_solver import pywrapcp, routing_enums_pb2

MAX_POINTS = 50
MAX_VEHICULES = 10
LIMITE_SECONDES = 1  # recherche locale : on s'arrête après ce délai (5 s au grand maximum)


def haversine(a, b):
    """Distance à vol d'oiseau en mètres entre deux points (lat, lon)."""
    lat1, lon1, lat2, lon2 = map(math.radians, (a[0], a[1], b[0], b[1]))
    h = math.sin((lat2 - lat1) / 2) ** 2 + math.cos(lat1) * math.cos(lat2) * math.sin((lon2 - lon1) / 2) ** 2
    return 2 * 6_371_000 * math.asin(math.sqrt(h))


def matrice_distances(points):
    return [[round(haversine(a, b)) for b in points] for a in points]


def longueur(tournee, distances):
    """Longueur d'une tournée (liste d'indices) qui part du dépôt (0) et y revient."""
    etapes = [0, *tournee, 0]
    return sum(distances[i][j] for i, j in zip(etapes, etapes[1:]))


def trajet_naif(nb_livraisons, capacite):
    """Points pris dans l'ordre de saisie : chaque véhicule se remplit à tour de rôle."""
    indices = list(range(1, nb_livraisons + 1))
    return [indices[i:i + capacite] for i in range(0, nb_livraisons, capacite)]


def optimiser(points, nb_vehicules, capacite=None, limite_secondes=LIMITE_SECONDES):
    """points[0] est le dépôt, les autres sont les livraisons (1 colis chacune).

    Renvoie les tournées optimisées et la comparaison avec le trajet naïf.
    """
    nb_livraisons = len(points) - 1
    if capacite is None:
        capacite = math.ceil(nb_livraisons / nb_vehicules)
    if capacite * nb_vehicules < nb_livraisons:
        raise ValueError(f"{nb_vehicules} véhicule(s) de capacité {capacite} ne suffisent pas "
                         f"pour {nb_livraisons} livraisons.")

    distances = matrice_distances(points)
    gestion = pywrapcp.RoutingIndexManager(len(points), nb_vehicules, 0)
    modele = pywrapcp.RoutingModel(gestion)

    def cout(depuis, vers):
        return distances[gestion.IndexToNode(depuis)][gestion.IndexToNode(vers)]

    modele.SetArcCostEvaluatorOfAllVehicles(modele.RegisterTransitCallback(cout))

    def demande(index):
        return 0 if gestion.IndexToNode(index) == 0 else 1

    modele.AddDimensionWithVehicleCapacity(
        modele.RegisterUnaryTransitCallback(demande), 0, [capacite] * nb_vehicules, True, "Capacite")

    parametres = pywrapcp.DefaultRoutingSearchParameters()
    parametres.first_solution_strategy = routing_enums_pb2.FirstSolutionStrategy.PATH_CHEAPEST_ARC
    parametres.local_search_metaheuristic = routing_enums_pb2.LocalSearchMetaheuristic.GUIDED_LOCAL_SEARCH
    parametres.time_limit.FromMilliseconds(int(limite_secondes * 1000))

    solution = modele.SolveWithParameters(parametres)
    if solution is None:
        raise ValueError("Aucune solution trouvée.")

    tournees = []
    for v in range(nb_vehicules):
        index = solution.Value(modele.NextVar(modele.Start(v)))
        tournee = []
        while not modele.IsEnd(index):
            tournee.append(gestion.IndexToNode(index))
            index = solution.Value(modele.NextVar(index))
        if tournee:
            tournees.append(tournee)

    naif = trajet_naif(nb_livraisons, capacite)
    distance_optimisee = sum(longueur(t, distances) for t in tournees)
    distance_naive = sum(longueur(t, distances) for t in naif)
    gain = 100 * (distance_naive - distance_optimisee) / distance_naive if distance_naive else 0

    return {
        "tournees": [{"points": t, "distance_m": longueur(t, distances)} for t in tournees],
        "naif": [{"points": t, "distance_m": longueur(t, distances)} for t in naif],
        "distance_optimisee_m": distance_optimisee,
        "distance_naive_m": distance_naive,
        "gain_pourcent": round(gain, 1),
        "capacite": capacite,
    }


if __name__ == "__main__":
    # Test rapide : 10 points dans Bruxelles, 1 véhicule
    import time
    points = [
        (50.8466, 4.3528), (50.8503, 4.3517), (50.8389, 4.3755), (50.8263, 4.3725),
        (50.8619, 4.3441), (50.8336, 4.3302), (50.8530, 4.3800), (50.8146, 4.3546),
        (50.8442, 4.3990), (50.8686, 4.3712),
    ]
    debut = time.perf_counter()
    resultat = optimiser(points, nb_vehicules=1)
    print(f"Calcul : {time.perf_counter() - debut:.2f} s")
    for t in resultat["tournees"]:
        print("Ordre de passage :", [0, *t["points"], 0], f"{t['distance_m'] / 1000:.2f} km")
    print(f"Naïf : {resultat['distance_naive_m'] / 1000:.2f} km, gain : {resultat['gain_pourcent']} %")
