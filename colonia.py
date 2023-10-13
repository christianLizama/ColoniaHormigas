import numpy as np
import matplotlib.pyplot as plt
import tsplib95

# Cargar el archivo TSP
problem = tsplib95.load("a280.tsp")

# Obtener el número de ciudades y las coordenadas
num_ciudades = len(problem.node_coords)
coordenadas = np.array([problem.node_coords[i] for i in range(1, num_ciudades + 1)])

# Calcular las distancias a partir de las coordenadas
distancias = np.linalg.norm(coordenadas[:, np.newaxis] - coordenadas, axis=2)

# Definir los parámetros del algoritmo
num_hormigas = 30
evaporacion = 0.5
alpha = 1.0
beta = 3.0
iteraciones = 1000

# Algoritmo de Colonia de Hormigas
def colonia_de_hormigas(distancias, num_hormigas, evaporacion, alpha, beta, iteraciones):
    num_ciudades = len(distancias)
    feromonas = np.ones((num_ciudades, num_ciudades))

    mejor_ruta = None
    mejor_distancia = float('inf')

    for iteracion in range(iteraciones):
        rutas = []  # Lista de rutas de cada hormiga
        for hormiga in range(num_hormigas):
            ciudad_actual = np.random.randint(0, num_ciudades)
            ruta = [ciudad_actual]
            ciudades_no_visitadas = set(range(num_ciudades))
            ciudades_no_visitadas.remove(ciudad_actual)

            while ciudades_no_visitadas:
                probabilidades = np.zeros(num_ciudades)
                for ciudad in ciudades_no_visitadas:
                    distancia = distancias[ciudad_actual][ciudad]
                    if distancia != 0:
                        probabilidad = (feromonas[ciudad_actual][ciudad] ** alpha) * (1.0 / distancia ** beta)
                        probabilidades[ciudad] = probabilidad

                # Normalizar las probabilidades para que sumen 1
                probabilidades /= probabilidades.sum()

                # Verificar si hay ciudades no visitadas con probabilidades
                ciudades_con_probabilidades = [ciudad for ciudad, prob in enumerate(probabilidades) if prob > 0]

                if not ciudades_con_probabilidades:
                    break  # Todas las ciudades se han visitado, termina el ciclo

                # Elegir la siguiente ciudad según las probabilidades
                siguiente_ciudad = np.random.choice(ciudades_con_probabilidades, p=probabilidades[ciudades_con_probabilidades])
                ruta.append(siguiente_ciudad)
                ciudades_no_visitadas.remove(siguiente_ciudad)
                ciudad_actual = siguiente_ciudad

            rutas.append(ruta)  # Agregar la ruta a la lista

            distancia_ruta = distancias[ruta, np.roll(ruta, -1)].sum()

            if distancia_ruta < mejor_distancia:
                mejor_ruta = ruta
                mejor_distancia = distancia_ruta

        # Actualizar las feromonas según las rutas
        feromonas *= (1 - evaporacion)
        for ruta in rutas:
            for i, j in zip(ruta, np.roll(ruta, -1)):
                feromonas[i][j] += evaporacion / distancia_ruta

        print(f"Iteración {iteracion + 1}: Mejor distancia = {mejor_distancia}")

    return mejor_ruta, mejor_distancia

# Llamar al algoritmo con los parámetros definidos
mejor_ruta, mejor_distancia = colonia_de_hormigas(distancias, num_hormigas, evaporacion, alpha, beta, iteraciones)

# Imprimir la mejor ruta y la mejor distancia
print("Mejor ruta encontrada:", mejor_ruta)
print("Distancia de la mejor ruta:", mejor_distancia)

# Graficar las ciudades y la mejor ruta
plt.figure(figsize=(10, 10))
plt.scatter(coordenadas[:, 0], coordenadas[:, 1], s=100, c='#5599FF')
for i in range(len(mejor_ruta)):
    ciudad1 = coordenadas[mejor_ruta[i]]
    ciudad2 = coordenadas[mejor_ruta[(i + 1) % len(mejor_ruta)]]
    plt.plot([ciudad1[0], ciudad2[0]], [ciudad1[1], ciudad2[1]], c='k')
plt.title('Mejor ruta encontrada')
plt.show()
