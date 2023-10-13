import numpy as np
import matplotlib.pyplot as plt
import tsplib95

# Cargar el archivo TSP
problem = tsplib95.load("a280.tsp")

# Obtener el número de ciudades y las coordenadas
num_ciudades = len(problem.node_coords)
coordenadas = [problem.node_coords[i] for i in range(1, num_ciudades + 1)]

# Calcular las distancias a partir de las coordenadas
distancias = np.zeros((num_ciudades, num_ciudades))
for i in range(num_ciudades):
    for j in range(num_ciudades):
        distancias[i][j] = np.linalg.norm(np.array(coordenadas[i]) - np.array(coordenadas[j]))

# Definir los parámetros del algoritmo
num_hormigas = 30
evaporacion = 0.5
alpha = 1.0
beta = 3.0
iteraciones = 1000

# Función para calcular la distancia total de una ruta
def calcular_distancia(ruta, distancias):
    distancia_total = 0
    for i in range(len(ruta) - 1):
        distancia_total += distancias[ruta[i]][ruta[i + 1]]
    distancia_total += distancias[ruta[-1]][ruta[0]]  # Volver al punto de partida
    return distancia_total

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
            ciudades_no_visitadas = list(range(num_ciudades))
            ciudades_no_visitadas.remove(ciudad_actual)
            
            while ciudades_no_visitadas:
                probabilidades = []
                denominador_probabilidad = 0
                
                for ciudad in ciudades_no_visitadas:
                    distancia = distancias[ciudad_actual][ciudad]
                    probabilidad = 0  # Inicializar probabilidad en cero
                    
                    if distancia != 0:
                        probabilidad = (feromonas[ciudad_actual][ciudad] ** alpha) * (1.0 / distancia ** beta)
                    
                    denominador_probabilidad += probabilidad
                    probabilidades.append(probabilidad)
                
                # Normalizar las probabilidades para que sumen 1
                probabilidades = [p / denominador_probabilidad for p in probabilidades]
                # Elegir la siguiente ciudad según las probabilidades
                siguiente_ciudad = np.random.choice(ciudades_no_visitadas, p=probabilidades)
                ruta.append(siguiente_ciudad)
                ciudades_no_visitadas.remove(siguiente_ciudad)
                ciudad_actual = siguiente_ciudad
            
            rutas.append(ruta)  # Agregar la ruta a la lista
            
            distancia_ruta = calcular_distancia(ruta, distancias)
            
            if distancia_ruta < mejor_distancia:
                mejor_ruta = ruta
                mejor_distancia = distancia_ruta
        
        # Actualizar las feromonas según las rutas
        for i in range(num_ciudades):
            for j in range(num_ciudades):
                feromonas[i][j] = (1 - evaporacion) * feromonas[i][j]  # Evaporación
                
                for ruta in rutas:
                    if j in ruta and ruta.index(j) == (ruta.index(i) + 1) % num_ciudades:  # Si la ciudad j es la siguiente a la ciudad i en la ruta
                        feromonas[i][j] += evaporacion / calcular_distancia(ruta, distancias)  # Depósito de feromonas
        
        print(f"Iteración {iteracion + 1}: Mejor distancia = {mejor_distancia}")
    
    return mejor_ruta, mejor_distancia

# Llamar al algoritmo con los parámetros definidos
mejor_ruta, mejor_distancia = colonia_de_hormigas(distancias, num_hormigas, evaporacion, alpha, beta, iteraciones)

# Imprimir la mejor ruta y la mejor distancia
print("Mejor ruta encontrada:", mejor_ruta)
print("Distancia de la mejor ruta:", mejor_distancia)

# Graficar las ciudades y la mejor ruta
plt.figure(figsize=(10,10))
plt.scatter(coordenadas[0], coordenadas[1], s=100, c='#5599FF')
for i in range(len(mejor_ruta)):
    ciudad1 = coordenadas[mejor_ruta[i]]
    ciudad2 = coordenadas[mejor_ruta[(i+1) % len(mejor_ruta)]]
    plt.plot([ciudad1[0], ciudad2[0]], [ciudad1[1], ciudad2[1]], c='k')
plt.title('Mejor ruta encontrada')
plt.show()
